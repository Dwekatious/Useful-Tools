from __future__ import annotations

import importlib
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

try:
    import yt_dlp  # type: ignore
except Exception:  # pragma: no cover - handled at runtime
    yt_dlp = None  # type: ignore[assignment]


APP_TITLE = "YouTube Downloader Studio"
POLL_INTERVAL_MS = 80
MAX_EVENTS_PER_TICK = 200
MAX_LOG_LINES = 800
STANDARD_RESOLUTION_STEPS = [144, 240, 360, 480, 720, 1080, 1440, 2160, 4320, 8640]

PALETTE = {
    "bg": "#edf2f7",
    "card": "#ffffff",
    "header_bg": "#0f172a",
    "header_subtle": "#1e293b",
    "text": "#0f172a",
    "muted": "#475569",
    "accent": "#0ea5e9",
    "accent_hover": "#0284c7",
    "success": "#16a34a",
    "danger": "#dc2626",
    "input_border": "#cbd5e1",
    "progress_trough": "#dbeafe",
    "progress_bg": "#38bdf8",
    "log_bg": "#0b1324",
    "log_fg": "#e2e8f0",
}


class DownloadCancelled(Exception):
    pass


@dataclass(frozen=True)
class VideoProfile:
    label: str
    height: Optional[int]
    requires_ffmpeg: bool = False


def format_bytes(value: Optional[float]) -> str:
    if value is None:
        return "-"
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(value)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def format_eta(seconds: Optional[float]) -> str:
    if seconds is None:
        return "-"
    total = max(0, int(seconds))
    minutes, sec = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


def format_duration(seconds: Optional[int]) -> str:
    if seconds is None:
        return "-"
    total = max(0, int(seconds))
    minutes, sec = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {sec:02d}s"
    return f"{minutes}m {sec:02d}s"


def resolution_label(height: int) -> str:
    if height >= 2160:
        return f"{height}p (4K)"
    if height >= 1440:
        return f"{height}p (2K)"
    if height >= 1080:
        return f"{height}p (Full HD)"
    if height >= 720:
        return f"{height}p (HD)"
    return f"{height}p"


def extract_height(fmt: dict[str, Any]) -> Optional[int]:
    raw_height = fmt.get("height")
    if isinstance(raw_height, (int, float)):
        return int(raw_height)

    resolution = fmt.get("resolution")
    if isinstance(resolution, str):
        match = re.search(r"(\d+)x(\d+)", resolution)
        if match:
            return int(match.group(2))
        match = re.search(r"(\d{3,4})p", resolution)
        if match:
            return int(match.group(1))

    format_note = fmt.get("format_note")
    if isinstance(format_note, str):
        match = re.search(r"(\d{3,4})p", format_note)
        if match:
            return int(match.group(1))

    format_label = fmt.get("format")
    if isinstance(format_label, str):
        match = re.search(r"(\d{3,4})p", format_label)
        if match:
            return int(match.group(1))

    return None


class ProbeWorker(threading.Thread):
    def __init__(
        self,
        url: str,
        event_queue: "queue.Queue[tuple[str, Any]]",
        stop_event: threading.Event,
    ) -> None:
        super().__init__(daemon=True)
        self.url = url
        self.event_queue = event_queue
        self.stop_event = stop_event

    def _emit(self, event: str, payload: Any) -> None:
        self.event_queue.put((event, payload))

    def run(self) -> None:
        if yt_dlp is None:
            self._emit(
                "probe_error",
                "Missing dependency: yt-dlp. Install with: pip install yt-dlp",
            )
            return

        self._emit("probe_started", None)
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
            "skip_download": True,
            "noplaylist": True,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(self.url, download=False)
        except Exception as exc:
            self._emit("probe_error", str(exc))
            return

        if self.stop_event.is_set():
            self._emit("probe_cancelled", None)
            return

        if isinstance(info, dict) and isinstance(info.get("entries"), list):
            first_entry = next((entry for entry in info["entries"] if entry), None)
            if first_entry:
                info = first_entry

        if not isinstance(info, dict):
            self._emit("probe_error", "Could not read video metadata for this URL.")
            return

        formats = info.get("formats") or []
        heights = sorted(
            {
                height
                for fmt in formats
                for height in [extract_height(fmt)]
                if height is not None and fmt.get("vcodec") not in (None, "none")
            },
            reverse=True,
        )
        muxed_heights = sorted(
            {
                height
                for fmt in formats
                for height in [extract_height(fmt)]
                if height is not None
                and fmt.get("vcodec") not in (None, "none")
                and fmt.get("acodec") not in (None, "none")
            },
            reverse=True,
        )
        payload = {
            "title": info.get("title") or "-",
            "uploader": info.get("uploader") or info.get("channel") or "-",
            "duration": info.get("duration"),
            "heights": heights,
            "muxed_heights": muxed_heights,
        }
        self._emit("probe_done", payload)


class DownloadWorker(threading.Thread):
    def __init__(
        self,
        url: str,
        output_dir: Path,
        mode: str,
        resolution_height: Optional[int],
        audio_format: str,
        ffmpeg_available: bool,
        event_queue: "queue.Queue[tuple[str, Any]]",
        stop_event: threading.Event,
    ) -> None:
        super().__init__(daemon=True)
        self.url = url
        self.output_dir = output_dir
        self.mode = mode
        self.resolution_height = resolution_height
        self.audio_format = audio_format
        self.ffmpeg_available = ffmpeg_available
        self.event_queue = event_queue
        self.stop_event = stop_event

    def _emit(self, event: str, payload: Any) -> None:
        self.event_queue.put((event, payload))

    def _select_video_format(self) -> str:
        if self.ffmpeg_available:
            if self.resolution_height is None:
                return (
                    "bestvideo[vcodec!=none]+bestaudio[ext=m4a]"
                    "/bestvideo[vcodec!=none]+bestaudio[acodec!=none]"
                    "/best[vcodec!=none][acodec!=none]"
                    "/bestvideo+bestaudio/best"
                )
            return (
                f"bestvideo[height={self.resolution_height}][vcodec!=none]+bestaudio[ext=m4a]"
                f"/bestvideo[height={self.resolution_height}][vcodec!=none]+bestaudio[acodec!=none]"
                f"/best[height={self.resolution_height}][vcodec!=none][acodec!=none]"
                f"/bestvideo[height<={self.resolution_height}][vcodec!=none]+bestaudio[ext=m4a]"
                f"/bestvideo[height<={self.resolution_height}][vcodec!=none]+bestaudio[acodec!=none]"
                f"/best[height<={self.resolution_height}][vcodec!=none][acodec!=none]"
                "/bestvideo[vcodec!=none]+bestaudio[ext=m4a]"
                "/bestvideo[vcodec!=none]+bestaudio[acodec!=none]"
                "/best[vcodec!=none][acodec!=none]"
                "/best"
            )

        if self.resolution_height is None:
            return "best[vcodec!=none][acodec!=none]/best"
        return (
            f"best[height={self.resolution_height}][vcodec!=none][acodec!=none]"
            f"/best[height<={self.resolution_height}][vcodec!=none][acodec!=none]"
            "/best[vcodec!=none][acodec!=none]/best"
        )

    def _summarize_download_quality(self, info: dict[str, Any]) -> str:
        requested_formats = info.get("requested_formats")
        if isinstance(requested_formats, list) and requested_formats:
            heights = [
                extract_height(fmt)
                for fmt in requested_formats
                if isinstance(fmt, dict)
            ]
            heights = [h for h in heights if isinstance(h, int)]
            ext = info.get("ext")
            if heights:
                best_h = max(heights)
                return f"{best_h}p ({ext or 'merged'})"
            if isinstance(ext, str):
                return ext

        requested_downloads = info.get("requested_downloads")
        if isinstance(requested_downloads, list) and requested_downloads:
            heights: list[int] = []
            exts: list[str] = []
            for download in requested_downloads:
                if not isinstance(download, dict):
                    continue
                h = extract_height(download)
                if isinstance(h, int):
                    heights.append(h)
                ext = download.get("ext")
                if isinstance(ext, str):
                    exts.append(ext)
            if heights:
                best_h = max(heights)
                ext_label = exts[0] if exts else (info.get("ext") or "container")
                return f"{best_h}p ({ext_label})"

        single_h = extract_height(info)
        if isinstance(single_h, int):
            ext = info.get("ext")
            if isinstance(ext, str):
                return f"{single_h}p ({ext})"
            return f"{single_h}p"

        ext = info.get("ext")
        if isinstance(ext, str):
            return ext
        return "unknown"

    def _progress_hook(self, data: dict[str, Any]) -> None:
        if self.stop_event.is_set():
            raise DownloadCancelled("Download cancelled by user.")

        status = data.get("status")
        if status == "downloading":
            downloaded = float(data.get("downloaded_bytes") or 0.0)
            total = data.get("total_bytes") or data.get("total_bytes_estimate")
            total_value = float(total) if total else None
            percent: Optional[float] = None
            has_total = bool(total_value and total_value > 0)
            if has_total and total_value is not None:
                percent = downloaded / total_value * 100.0
            else:
                percent_str = data.get("_percent_str")
                if isinstance(percent_str, str):
                    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", percent_str)
                    if match:
                        try:
                            percent = float(match.group(1))
                        except ValueError:
                            percent = None
            payload = {
                "percent": min(100.0, max(0.0, percent)) if isinstance(percent, (int, float)) else None,
                "has_total": has_total,
                "downloaded": downloaded,
                "total": total_value,
                "speed": data.get("speed"),
                "speed_str": data.get("_speed_str"),
                "eta": data.get("eta"),
                "eta_str": data.get("_eta_str"),
            }
            self._emit("progress", payload)
            return

        if status == "finished":
            self._emit("stage", "Download complete. Finalizing media...")

    def _postprocessor_hook(self, data: dict[str, Any]) -> None:
        if self.stop_event.is_set():
            raise DownloadCancelled("Download cancelled by user.")
        status = data.get("status")
        processor = data.get("postprocessor", "post-processing")
        if status == "started":
            self._emit("stage", f"Running {processor}...")
        elif status == "finished":
            self._emit("stage", f"{processor} complete.")

    def run(self) -> None:
        if yt_dlp is None:
            self._emit("error", "Missing dependency: yt-dlp. Install with: pip install yt-dlp")
            return

        self._emit("download_started", None)
        ydl_opts: dict[str, Any] = {
            "outtmpl": str(self.output_dir / "%(title).180s [%(id)s].%(ext)s"),
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "prefer_ffmpeg": True,
            "retries": 8,
            "continuedl": True,
            "nopart": False,
            "progress_hooks": [self._progress_hook],
            "postprocessor_hooks": [self._postprocessor_hook],
            "concurrent_fragment_downloads": 4,
        }

        if self.mode == "audio":
            ydl_opts["format"] = "bestaudio/best"
            if self.ffmpeg_available:
                ydl_opts["postprocessors"] = [
                    {
                        "key": "FFmpegExtractAudio",
                        "preferredcodec": self.audio_format,
                        "preferredquality": "192",
                    }
                ]
            else:
                self._emit(
                    "log",
                    "FFmpeg not detected. Downloading original audio stream without format conversion.",
                )
        else:
            ydl_opts["format"] = self._select_video_format()
            if self.ffmpeg_available:
                ydl_opts["merge_output_format"] = "mkv"
                self._emit(
                    "log",
                    "Using FFmpeg merge to produce highest-quality video with audio (MKV container).",
                )
            else:
                self._emit(
                    "log",
                    "FFmpeg not detected. Using progressive video files that already include audio.",
                )

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(self.url, download=True)
        except DownloadCancelled:
            self._emit("cancelled", None)
            return
        except Exception as exc:
            if self.stop_event.is_set() or "cancel" in str(exc).lower():
                self._emit("cancelled", None)
                return
            self._emit("error", str(exc))
            return

        if isinstance(info, dict) and isinstance(info.get("entries"), list):
            first_entry = next((entry for entry in info["entries"] if entry), None)
            if first_entry:
                info = first_entry

        title = "-"
        quality = "unknown"
        if isinstance(info, dict):
            title = str(info.get("title") or "-")
            quality = self._summarize_download_quality(info)

        self._emit("completed", {"title": title, "quality": quality})


@dataclass(frozen=True)
class DependencyInstallResult:
    key: str
    label: str
    success: bool
    details: str


class DependencyInstallWorker(threading.Thread):
    def __init__(
        self,
        dependency_keys: list[str],
        can_install_ffmpeg: bool,
        event_queue: "queue.Queue[tuple[str, Any]]",
    ) -> None:
        super().__init__(daemon=True)
        self.dependency_keys = dependency_keys
        self.can_install_ffmpeg = can_install_ffmpeg
        self.event_queue = event_queue

    def _emit(self, event: str, payload: Any) -> None:
        self.event_queue.put((event, payload))

    def _run_command(self, label: str, command: list[str]) -> DependencyInstallResult:
        self._emit("dep_install_step", f"Installing {label}...")
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
            )
        except Exception as exc:
            return DependencyInstallResult(
                key=label.lower(),
                label=label,
                success=False,
                details=f"{label} installer failed to start: {exc}",
            )

        output_parts = []
        if completed.stdout:
            output_parts.append(completed.stdout.strip())
        if completed.stderr:
            output_parts.append(completed.stderr.strip())
        combined_output = "\n".join(part for part in output_parts if part)
        if combined_output:
            tail = "\n".join(combined_output.splitlines()[-10:])
            self._emit("dep_install_log", f"{label} installer output:\n{tail}")

        success = completed.returncode == 0
        if success:
            details = f"{label} installed successfully."
        else:
            details = f"{label} install returned exit code {completed.returncode}."
        return DependencyInstallResult(
            key=label.lower(),
            label=label,
            success=success,
            details=details,
        )

    def run(self) -> None:
        self._emit("dep_install_started", {"keys": list(self.dependency_keys)})
        results: list[DependencyInstallResult] = []

        for key in self.dependency_keys:
            if key == "yt-dlp":
                result = self._run_command(
                    "yt-dlp",
                    [sys.executable, "-m", "pip", "install", "--upgrade", "yt-dlp"],
                )
                results.append(result)
                continue

            if key == "ffmpeg":
                if not self.can_install_ffmpeg:
                    results.append(
                        DependencyInstallResult(
                            key="ffmpeg",
                            label="FFmpeg",
                            success=False,
                            details=(
                                "Automatic FFmpeg install is unavailable on this machine. "
                                "Install FFmpeg manually and ensure it is in PATH."
                            ),
                        )
                    )
                    continue
                result = self._run_command(
                    "FFmpeg",
                    [
                        "winget",
                        "install",
                        "--id",
                        "Gyan.FFmpeg",
                        "-e",
                        "--accept-source-agreements",
                        "--accept-package-agreements",
                    ],
                )
                results.append(result)
                continue

            results.append(
                DependencyInstallResult(
                    key=key,
                    label=key,
                    success=False,
                    details=f"No installer configured for dependency: {key}",
                )
            )

        self._emit("dep_install_done", results)


class YouTubeDownloaderApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("640x640")
        self.minsize(640, 640)
        self.configure(bg=PALETTE["bg"])

        self.ffmpeg_available = False
        self._event_queue: "queue.Queue[tuple[str, Any]]" = queue.Queue()
        self._stop_event = threading.Event()
        self._probe_worker: Optional[ProbeWorker] = None
        self._download_worker: Optional[DownloadWorker] = None
        self._dep_install_worker: Optional[DependencyInstallWorker] = None
        self._analyzing = False
        self._downloading = False
        self._installing_dependencies = False
        self._progress_is_indeterminate = False
        self._notice_auto_hide_job: Optional[str] = None
        self._last_output_dir: Optional[Path] = None
        self._compact_layout = False
        self._layout_after_id: Optional[str] = None

        self.url_var = tk.StringVar()
        self.output_var = tk.StringVar(value=str(Path.home() / "Downloads"))
        self.mode_var = tk.StringVar(value="video")
        self.resolution_var = tk.StringVar(value="Best available")
        self.audio_format_var = tk.StringVar(value="mp3")
        self.title_var = tk.StringVar(value="Title: -")
        self.channel_var = tk.StringVar(value="Uploader: -")
        self.duration_var = tk.StringVar(value="Duration: -")
        self.ffmpeg_var = tk.StringVar(value="FFmpeg: checking...")
        self.dependency_var = tk.StringVar(value="Dependencies: checking...")
        self.status_var = tk.StringVar(value="Ready")
        self.metrics_var = tk.StringVar(value="Waiting for download...")
        self.progress_var = tk.DoubleVar(value=0.0)
        self._missing_dependency_keys: list[str] = []
        self._installable_dependency_keys: list[str] = []

        self._video_profiles = [VideoProfile(label="Best available", height=None)]

        self._init_styles()
        self._build_ui()
        dep_state = self._refresh_dependency_state(log_result=False)
        self._refresh_mode_controls()
        self._refresh_buttons()

        if "yt-dlp" in dep_state["missing_keys"]:
            self.after(
                300,
                lambda: messagebox.showwarning(
                    APP_TITLE,
                    "The 'yt-dlp' package is not installed.\nRun: pip install yt-dlp",
                ),
            )
            self._add_log("yt-dlp is missing. Install with: pip install yt-dlp")
        if "ffmpeg" in dep_state["missing_keys"]:
            self._add_log(
                "FFmpeg not found in PATH. High-resolution merge and audio conversion options may be limited."
            )

        self.after(POLL_INTERVAL_MS, self._process_events)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.bind("<Configure>", self._on_window_configure)

    def _refresh_ffmpeg_status(self) -> None:
        ffmpeg_path = shutil.which("ffmpeg")
        ffmpeg_ok = False
        if ffmpeg_path:
            try:
                check = subprocess.run(
                    [ffmpeg_path, "-version"],
                    capture_output=True,
                    text=True,
                    timeout=4,
                    check=False,
                )
                ffmpeg_ok = check.returncode == 0
            except Exception:
                ffmpeg_ok = False
        self.ffmpeg_available = ffmpeg_ok
        self.ffmpeg_var.set(
            "FFmpeg: Detected"
            if self.ffmpeg_available
            else "FFmpeg: Not detected (high-res video+audio needs FFmpeg)"
        )

    def _refresh_yt_dlp_status(self) -> tuple[bool, str]:
        global yt_dlp
        try:
            module = importlib.import_module("yt_dlp")
            yt_dlp = module
            version = getattr(module, "__version__", None)
            if not version:
                version_module = getattr(module, "version", None)
                version = getattr(version_module, "__version__", None) if version_module else None
            return True, str(version) if version else "installed"
        except Exception:
            yt_dlp = None  # type: ignore[assignment]
            return False, "not installed"

    def _can_auto_install_ffmpeg(self) -> bool:
        return sys.platform.startswith("win") and (shutil.which("winget") is not None)

    def _dependency_label(self, key: str) -> str:
        if key == "ffmpeg":
            return "FFmpeg"
        return "yt-dlp"

    def _max_supported_standard_resolution(self, max_height: int) -> int:
        lower_steps = [step for step in STANDARD_RESOLUTION_STEPS if step <= max_height]
        if lower_steps:
            return lower_steps[-1]
        return STANDARD_RESOLUTION_STEPS[0]

    def _build_video_profiles(
        self,
        heights: list[int],
        muxed_heights: list[int],
    ) -> list[VideoProfile]:
        clean_heights = sorted({h for h in heights if isinstance(h, int) and h > 0}, reverse=True)
        if not clean_heights:
            return [VideoProfile(label="Best available", height=None)]

        max_height = clean_heights[0]
        max_muxed = max((h for h in muxed_heights if isinstance(h, int)), default=0)
        best_cap = max_height if self.ffmpeg_available else (max_muxed if max_muxed > 0 else max_height)
        best_label = f"Best available (up to {best_cap}p)"
        max_supported_step = self._max_supported_standard_resolution(max_height)
        standard_steps = [step for step in STANDARD_RESOLUTION_STEPS if step <= max_supported_step]
        if not standard_steps:
            standard_steps = [STANDARD_RESOLUTION_STEPS[0]]

        profiles = [VideoProfile(label=best_label, height=None)]

        nearest_step = min(STANDARD_RESOLUTION_STEPS, key=lambda step: abs(step - max_height))
        add_native_max = abs(nearest_step - max_height) > 60
        if add_native_max:
            needs_ffmpeg = (not self.ffmpeg_available) and (
                max_muxed == 0 or max_height > int(max_muxed * 1.12)
            )
            native_label = f"Native max ({max_height}p)"
            if needs_ffmpeg:
                native_label = f"{native_label} (needs FFmpeg)"
            profiles.append(
                VideoProfile(label=native_label, height=max_height, requires_ffmpeg=needs_ffmpeg)
            )

        for step in sorted(standard_steps, reverse=True):
            needs_ffmpeg = (not self.ffmpeg_available) and (
                max_muxed == 0 or step > int(max_muxed * 1.12)
            )
            label = resolution_label(step)
            if needs_ffmpeg:
                label = f"{label} (needs FFmpeg)"
            profiles.append(VideoProfile(label=label, height=step, requires_ffmpeg=needs_ffmpeg))

        return profiles

    def _refresh_dependency_state(self, log_result: bool = True) -> dict[str, Any]:
        self._refresh_ffmpeg_status()
        yt_installed, yt_version = self._refresh_yt_dlp_status()

        missing_keys: list[str] = []
        installable_keys: list[str] = []
        installed_lines: list[str] = []
        missing_lines: list[str] = []

        if yt_installed:
            installed_lines.append(f"yt-dlp ({yt_version})")
        else:
            missing_keys.append("yt-dlp")
            installable_keys.append("yt-dlp")
            missing_lines.append("yt-dlp")

        if self.ffmpeg_available:
            installed_lines.append("FFmpeg")
        else:
            missing_keys.append("ffmpeg")
            missing_lines.append("FFmpeg")
            if self._can_auto_install_ffmpeg():
                installable_keys.append("ffmpeg")

        self._missing_dependency_keys = missing_keys
        self._installable_dependency_keys = installable_keys

        if missing_lines:
            summary = f"Dependencies missing: {', '.join(missing_lines)}"
        else:
            summary = "Dependencies: all installed"
        self.dependency_var.set(summary)

        if log_result:
            if missing_lines:
                self._add_log(summary)
            else:
                self._add_log("Dependency check: all required dependencies are installed.")

        return {
            "missing_keys": missing_keys,
            "installable_keys": installable_keys,
            "installed_lines": installed_lines,
            "missing_lines": missing_lines,
        }

    def _start_indeterminate_progress(self, status_text: Optional[str] = None) -> None:
        if status_text:
            self.status_var.set(status_text)
        if str(self.progress.cget("mode")) != "indeterminate":
            self.progress.configure(mode="indeterminate")
        if not self._progress_is_indeterminate:
            self.progress.start(12)
            self._progress_is_indeterminate = True

    def _start_determinate_progress(self) -> None:
        if self._progress_is_indeterminate:
            self.progress.stop()
            self._progress_is_indeterminate = False
        if str(self.progress.cget("mode")) != "determinate":
            self.progress.configure(mode="determinate")

    def _stop_progress_animation(self) -> None:
        if self._progress_is_indeterminate:
            self.progress.stop()
            self._progress_is_indeterminate = False
        if str(self.progress.cget("mode")) != "determinate":
            self.progress.configure(mode="determinate")

    def _show_notice(
        self,
        message: str,
        level: str = "info",
        action_text: Optional[str] = None,
        action_command: Optional[Callable[[], None]] = None,
        auto_hide_ms: Optional[int] = 9000,
    ) -> None:
        themes = {
            "success": {
                "bg": "#dcfce7",
                "fg": "#14532d",
                "button_bg": "#166534",
                "button_active": "#14532d",
                "dismiss_bg": "#bbf7d0",
                "dismiss_active": "#86efac",
            },
            "warning": {
                "bg": "#fef9c3",
                "fg": "#713f12",
                "button_bg": "#a16207",
                "button_active": "#854d0e",
                "dismiss_bg": "#fde68a",
                "dismiss_active": "#facc15",
            },
            "error": {
                "bg": "#fee2e2",
                "fg": "#7f1d1d",
                "button_bg": "#b91c1c",
                "button_active": "#991b1b",
                "dismiss_bg": "#fecaca",
                "dismiss_active": "#fca5a5",
            },
            "info": {
                "bg": "#dbeafe",
                "fg": "#1e3a8a",
                "button_bg": "#1d4ed8",
                "button_active": "#1e40af",
                "dismiss_bg": "#bfdbfe",
                "dismiss_active": "#93c5fd",
            },
        }
        palette = themes.get(level, themes["info"])
        self.notice_frame.configure(background=palette["bg"])
        self.notice_label.configure(background=palette["bg"], foreground=palette["fg"])
        self.notice_close_button.configure(
            background=palette["dismiss_bg"],
            foreground=palette["fg"],
            activebackground=palette["dismiss_active"],
            activeforeground=palette["fg"],
        )
        self.notice_label_var.set(message)

        if action_text and action_command:
            self.notice_action_button.configure(
                text=action_text,
                command=action_command,
                background=palette["button_bg"],
                foreground="#ffffff",
                activebackground=palette["button_active"],
                activeforeground="#ffffff",
            )
            self.notice_action_button.pack(side="right", padx=(6, 6), pady=6)
        else:
            self.notice_action_button.pack_forget()

        if self._notice_auto_hide_job is not None:
            self.after_cancel(self._notice_auto_hide_job)
            self._notice_auto_hide_job = None

        self.notice_frame.pack(fill="x", pady=(0, 6), before=self.source_card)

        if auto_hide_ms and auto_hide_ms > 0:
            self._notice_auto_hide_job = self.after(auto_hide_ms, self._hide_notice)

    def _hide_notice(self) -> None:
        if self._notice_auto_hide_job is not None:
            self.after_cancel(self._notice_auto_hide_job)
            self._notice_auto_hide_job = None
        self.notice_action_button.pack_forget()
        self.notice_frame.pack_forget()

    def _open_last_output_dir(self) -> None:
        if self._last_output_dir is None:
            return
        target = self._last_output_dir
        if not target.exists():
            self._show_notice(
                f"Output folder not found: {target}",
                level="warning",
                auto_hide_ms=6000,
            )
            return
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(target))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(target)])
            else:
                subprocess.Popen(["xdg-open", str(target)])
        except Exception as exc:
            self._add_log(f"Failed to open folder: {exc}")
            self._show_notice(
                "Could not open output folder. See log for details.",
                level="error",
                auto_hide_ms=7000,
            )

    def _init_styles(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        ui_font = "Segoe UI"
        style.configure("Root.TFrame", background=PALETTE["bg"])
        style.configure("Card.TFrame", background=PALETTE["card"])
        style.configure("Header.TFrame", background=PALETTE["header_bg"])

        style.configure(
            "HeaderTitle.TLabel",
            background=PALETTE["header_bg"],
            foreground="#f8fafc",
            font=(ui_font, 16, "bold"),
        )
        style.configure(
            "HeaderSub.TLabel",
            background=PALETTE["header_bg"],
            foreground="#bfdbfe",
            font=(ui_font, 9),
        )
        style.configure(
            "SectionTitle.TLabel",
            background=PALETTE["card"],
            foreground=PALETTE["text"],
            font=(ui_font, 11, "bold"),
        )
        style.configure(
            "Field.TLabel",
            background=PALETTE["card"],
            foreground=PALETTE["muted"],
            font=(ui_font, 10),
        )
        style.configure(
            "Meta.TLabel",
            background=PALETTE["card"],
            foreground=PALETTE["text"],
            font=(ui_font, 10, "bold"),
        )
        style.configure(
            "Status.TLabel",
            background=PALETTE["card"],
            foreground=PALETTE["text"],
            font=(ui_font, 10, "bold"),
        )
        style.configure(
            "Metrics.TLabel",
            background=PALETTE["card"],
            foreground=PALETTE["muted"],
            font=(ui_font, 10),
        )

        style.configure(
            "Accent.TButton",
            background=PALETTE["accent"],
            foreground="#ffffff",
            borderwidth=0,
            focuscolor=PALETTE["accent"],
            font=(ui_font, 10, "bold"),
            padding=(10, 6),
        )
        style.map(
            "Accent.TButton",
            background=[("active", PALETTE["accent_hover"]), ("disabled", "#93c5fd")],
            foreground=[("disabled", "#e2e8f0")],
        )
        style.configure(
            "Neutral.TButton",
            background="#e2e8f0",
            foreground=PALETTE["text"],
            borderwidth=0,
            font=(ui_font, 10, "bold"),
            padding=(10, 6),
        )
        style.map(
            "Neutral.TButton",
            background=[("active", "#cbd5e1"), ("disabled", "#e2e8f0")],
            foreground=[("disabled", "#94a3b8")],
        )
        style.configure(
            "Danger.TButton",
            background=PALETTE["danger"],
            foreground="#ffffff",
            borderwidth=0,
            font=(ui_font, 10, "bold"),
            padding=(10, 6),
        )
        style.map(
            "Danger.TButton",
            background=[("active", "#b91c1c"), ("disabled", "#fecaca")],
            foreground=[("disabled", "#ffffff")],
        )

        style.configure(
            "TCombobox",
            fieldbackground="#ffffff",
            background="#ffffff",
            bordercolor=PALETTE["input_border"],
            lightcolor=PALETTE["input_border"],
            darkcolor=PALETTE["input_border"],
            arrowcolor=PALETTE["muted"],
            padding=5,
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", "#ffffff")],
            selectbackground=[("readonly", "#ffffff")],
            selectforeground=[("readonly", PALETTE["text"])],
        )
        style.configure("TRadiobutton", background=PALETTE["card"], foreground=PALETTE["text"])
        style.configure(
            "Accent.Horizontal.TProgressbar",
            troughcolor=PALETTE["progress_trough"],
            background=PALETTE["progress_bg"],
            bordercolor=PALETTE["progress_trough"],
            lightcolor=PALETTE["progress_bg"],
            darkcolor=PALETTE["progress_bg"],
            thickness=15,
        )

    def _build_ui(self) -> None:
        self.root_frame = ttk.Frame(self, style="Root.TFrame")
        self.root_frame.pack(fill="both", expand=True)

        header = ttk.Frame(self.root_frame, style="Header.TFrame", padding=(14, 10))
        header.pack(fill="x")
        ttk.Label(header, text=APP_TITLE, style="HeaderTitle.TLabel").pack(anchor="w")
        ttk.Label(header, text="Local downloader with quality selection.", style="HeaderSub.TLabel").pack(
            anchor="w", pady=(1, 0)
        )

        self.content_container = ttk.Frame(self.root_frame, style="Root.TFrame")
        self.content_container.pack(fill="both", expand=True)
        self.surface = ttk.Frame(self.content_container, style="Root.TFrame")
        self.surface.pack(fill="both", expand=True, padx=10, pady=8)

        surface = self.surface

        self.notice_frame = tk.Frame(surface, background="#dcfce7", bd=1, relief="solid")
        self.notice_frame.pack(fill="x", pady=(0, 6))
        self.notice_label_var = tk.StringVar(value="")
        self.notice_label = tk.Label(
            self.notice_frame,
            textvariable=self.notice_label_var,
            anchor="w",
            justify="left",
            font=("Segoe UI", 10, "bold"),
            background="#dcfce7",
            foreground="#14532d",
            padx=10,
            pady=8,
        )
        self.notice_label.pack(side="left", fill="x", expand=True)
        self.notice_action_button = tk.Button(
            self.notice_frame,
            text="Open Folder",
            command=self._open_last_output_dir,
            borderwidth=0,
            padx=10,
            pady=6,
            font=("Segoe UI", 9, "bold"),
            cursor="hand2",
            background="#166534",
            foreground="#ffffff",
            activebackground="#14532d",
            activeforeground="#ffffff",
        )
        self.notice_action_button.pack(side="right", padx=(6, 6), pady=6)
        self.notice_action_button.pack_forget()
        self.notice_close_button = tk.Button(
            self.notice_frame,
            text="Dismiss",
            command=self._hide_notice,
            borderwidth=0,
            padx=8,
            pady=6,
            font=("Segoe UI", 9),
            cursor="hand2",
            background="#bbf7d0",
            foreground="#14532d",
            activebackground="#86efac",
            activeforeground="#14532d",
        )
        self.notice_close_button.pack(side="right", padx=(0, 8), pady=6)
        self.notice_frame.pack_forget()

        self.source_card = ttk.Frame(surface, style="Card.TFrame", padding=(10, 8))
        self.source_card.pack(fill="x", pady=(0, 6))
        self.source_card.columnconfigure(1, weight=1)

        ttk.Label(self.source_card, text="Source", style="SectionTitle.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 6)
        )

        self.source_url_label = ttk.Label(self.source_card, text="Video URL", style="Field.TLabel")
        self.source_url_label.grid(row=1, column=0, sticky="w")
        self.url_entry = ttk.Entry(self.source_card, textvariable=self.url_var)
        self.url_entry.grid(row=1, column=1, sticky="ew", padx=(10, 8), pady=4)
        self.paste_button = ttk.Button(
            self.source_card,
            text="Paste",
            style="Neutral.TButton",
            command=self._paste_url,
        )
        self.paste_button.grid(
            row=1, column=2, sticky="ew", padx=(0, 8), pady=4
        )
        self.analyze_button = ttk.Button(
            self.source_card,
            text="Analyze URL",
            style="Accent.TButton",
            command=self._start_probe,
        )
        self.analyze_button.grid(row=1, column=3, sticky="ew", pady=4)

        self.source_output_label = ttk.Label(self.source_card, text="Save folder", style="Field.TLabel")
        self.source_output_label.grid(row=2, column=0, sticky="w")
        self.output_entry = ttk.Entry(self.source_card, textvariable=self.output_var)
        self.output_entry.grid(row=2, column=1, sticky="ew", padx=(10, 8), pady=4)
        self.browse_button = ttk.Button(
            self.source_card,
            text="Browse...",
            style="Neutral.TButton",
            command=self._choose_output_dir,
        )
        self.browse_button.grid(row=2, column=2, sticky="ew", padx=(0, 8), pady=4)

        self.options_card = ttk.Frame(surface, style="Card.TFrame", padding=(10, 8))
        self.options_card.pack(fill="x", pady=(0, 6))
        self.options_card.columnconfigure(1, weight=1)

        ttk.Label(self.options_card, text="Download Options", style="SectionTitle.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 6)
        )

        self.mode_label = ttk.Label(self.options_card, text="Mode", style="Field.TLabel")
        self.mode_label.grid(row=1, column=0, sticky="w")
        self.video_radio = ttk.Radiobutton(
            self.options_card,
            text="Video with sound",
            variable=self.mode_var,
            value="video",
            command=self._refresh_mode_controls,
        )
        self.video_radio.grid(row=1, column=1, sticky="w", pady=4)
        self.audio_radio = ttk.Radiobutton(
            self.options_card,
            text="Audio only",
            variable=self.mode_var,
            value="audio",
            command=self._refresh_mode_controls,
        )
        self.audio_radio.grid(row=1, column=2, sticky="w", pady=4)

        self.resolution_label_widget = ttk.Label(self.options_card, text="Resolution", style="Field.TLabel")
        self.resolution_label_widget.grid(row=2, column=0, sticky="w")
        self.resolution_combo = ttk.Combobox(
            self.options_card,
            textvariable=self.resolution_var,
            values=[profile.label for profile in self._video_profiles],
            state="readonly",
            width=24,
        )
        self.resolution_combo.grid(row=2, column=1, sticky="w", pady=4)

        self.audio_format_label = ttk.Label(self.options_card, text="Audio format", style="Field.TLabel")
        self.audio_format_label.grid(row=2, column=2, sticky="w")
        self.audio_combo = ttk.Combobox(
            self.options_card,
            textvariable=self.audio_format_var,
            values=["mp3", "m4a", "wav", "flac", "aac"],
            state="readonly",
            width=12,
        )
        self.audio_combo.grid(row=2, column=3, sticky="w", pady=4)

        meta_card = ttk.Frame(surface, style="Card.TFrame", padding=(10, 8))
        meta_card.pack(fill="x", pady=(0, 6))

        ttk.Label(meta_card, text="Video Metadata", style="SectionTitle.TLabel").pack(anchor="w", pady=(0, 4))
        ttk.Label(meta_card, textvariable=self.title_var, style="Meta.TLabel").pack(anchor="w", pady=(0, 1))
        ttk.Label(meta_card, textvariable=self.channel_var, style="Field.TLabel").pack(anchor="w", pady=(0, 1))
        ttk.Label(meta_card, textvariable=self.duration_var, style="Field.TLabel").pack(anchor="w", pady=(0, 1))
        ttk.Label(meta_card, textvariable=self.ffmpeg_var, style="Field.TLabel").pack(anchor="w")

        self.actions_card = ttk.Frame(surface, style="Card.TFrame", padding=(10, 8))
        self.actions_card.pack(fill="x", pady=(0, 6))
        self.actions_card.columnconfigure(0, weight=1)
        ttk.Label(self.actions_card, text="Actions", style="SectionTitle.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 6)
        )

        self.download_button = ttk.Button(
            self.actions_card,
            text="Start Download",
            style="Accent.TButton",
            command=self._start_download,
        )
        self.download_button.grid(row=1, column=0, sticky="w")
        self.cancel_button = ttk.Button(
            self.actions_card,
            text="Cancel",
            style="Danger.TButton",
            command=self._cancel_active_job,
        )
        self.cancel_button.grid(row=1, column=1, sticky="w", padx=(10, 0))

        self.check_deps_button = ttk.Button(
            self.actions_card,
            text="Check Dependencies",
            style="Neutral.TButton",
            command=self._check_dependencies,
        )
        self.check_deps_button.grid(row=2, column=0, sticky="w", pady=(6, 0))

        self.install_deps_button = ttk.Button(
            self.actions_card,
            text="Install Missing",
            style="Neutral.TButton",
            command=self._install_missing_dependencies,
        )
        self.install_deps_button.grid(row=2, column=1, sticky="w", padx=(10, 0), pady=(6, 0))
        self.dependency_status_label = ttk.Label(
            self.actions_card,
            textvariable=self.dependency_var,
            style="Field.TLabel",
        )
        self.dependency_status_label.grid(
            row=3, column=0, columnspan=2, sticky="w", pady=(4, 0)
        )

        progress_card = ttk.Frame(surface, style="Card.TFrame", padding=(10, 8))
        progress_card.pack(fill="x", pady=(0, 6))

        ttk.Label(progress_card, text="Progress", style="SectionTitle.TLabel").pack(anchor="w", pady=(0, 4))
        self.progress = ttk.Progressbar(
            progress_card,
            style="Accent.Horizontal.TProgressbar",
            variable=self.progress_var,
            maximum=100.0,
        )
        self.progress.pack(fill="x")
        ttk.Label(progress_card, textvariable=self.status_var, style="Status.TLabel").pack(anchor="w", pady=(4, 1))
        ttk.Label(progress_card, textvariable=self.metrics_var, style="Metrics.TLabel").pack(anchor="w")

        log_card = ttk.Frame(surface, style="Card.TFrame", padding=(10, 8))
        log_card.pack(fill="both", expand=True)
        ttk.Label(log_card, text="Activity Log", style="SectionTitle.TLabel").pack(anchor="w", pady=(0, 4))
        self.log_text = ScrolledText(
            log_card,
            height=2,
            wrap="word",
            font=("Consolas", 10),
            background=PALETTE["log_bg"],
            foreground=PALETTE["log_fg"],
            insertbackground=PALETTE["log_fg"],
            relief="flat",
            borderwidth=0,
            padx=10,
            pady=8,
        )
        self.log_text.pack(fill="both", expand=True)
        self.log_text.configure(state="disabled")
        self.update_idletasks()
        self._apply_responsive_layout(self.winfo_width())

    def _on_surface_configure(self, _event: tk.Event) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event: tk.Event) -> None:
        self.canvas.itemconfigure(self.canvas_window, width=event.width)

    def _on_mouse_wheel(self, event: tk.Event) -> None:
        try:
            delta = int(event.delta)
        except Exception:
            delta = 0
        if delta == 0:
            return
        self.canvas.yview_scroll(int(-delta / 120), "units")

    def _on_window_configure(self, event: tk.Event) -> None:
        if event.widget is not self:
            return
        if int(event.width) < 200:
            return
        if self._layout_after_id is not None:
            self.after_cancel(self._layout_after_id)
            self._layout_after_id = None
        width = int(event.width)
        self._layout_after_id = self.after(50, lambda w=width: self._apply_responsive_layout(w))

    def _apply_responsive_layout(self, width: int) -> None:
        compact = width < 600
        if compact == self._compact_layout:
            return
        self._compact_layout = compact
        self._layout_source_card(compact)
        self._layout_options_card(compact)
        self._layout_actions_card(compact)
        self._refresh_mode_controls()

    def _layout_source_card(self, compact: bool) -> None:
        for column in range(4):
            self.source_card.columnconfigure(column, weight=0)

        if compact:
            self.source_card.columnconfigure(1, weight=1)
            self.source_url_label.grid_configure(row=1, column=0, padx=(0, 8), pady=4, sticky="w")
            self.url_entry.grid_configure(row=1, column=1, columnspan=2, padx=(0, 0), pady=4, sticky="ew")
            self.paste_button.grid_configure(row=2, column=1, padx=(0, 8), pady=4, sticky="ew")
            self.analyze_button.grid_configure(row=2, column=2, padx=(0, 0), pady=4, sticky="ew")
            self.source_output_label.grid_configure(row=3, column=0, padx=(0, 8), pady=4, sticky="w")
            self.output_entry.grid_configure(row=3, column=1, columnspan=2, padx=(0, 0), pady=4, sticky="ew")
            self.browse_button.grid_configure(row=4, column=2, padx=(0, 0), pady=4, sticky="ew")
            return

        self.source_card.columnconfigure(1, weight=1)
        self.source_url_label.grid_configure(row=1, column=0, padx=(0, 0), pady=4, sticky="w")
        self.url_entry.grid_configure(row=1, column=1, columnspan=1, padx=(10, 8), pady=4, sticky="ew")
        self.paste_button.grid_configure(row=1, column=2, padx=(0, 8), pady=4, sticky="ew")
        self.analyze_button.grid_configure(row=1, column=3, padx=(0, 0), pady=4, sticky="ew")
        self.source_output_label.grid_configure(row=2, column=0, padx=(0, 0), pady=4, sticky="w")
        self.output_entry.grid_configure(row=2, column=1, columnspan=1, padx=(10, 8), pady=4, sticky="ew")
        self.browse_button.grid_configure(row=2, column=2, padx=(0, 8), pady=4, sticky="ew")

    def _layout_options_card(self, compact: bool) -> None:
        for column in range(4):
            self.options_card.columnconfigure(column, weight=0)

        if compact:
            self.options_card.columnconfigure(1, weight=1)
            self.mode_label.grid_configure(row=1, column=0, padx=(0, 8), pady=4, sticky="w")
            self.video_radio.grid_configure(row=1, column=1, padx=(0, 0), pady=4, sticky="w")
            self.audio_radio.grid_configure(row=2, column=1, padx=(0, 0), pady=4, sticky="w")
            self.resolution_label_widget.grid_configure(row=3, column=0, padx=(0, 8), pady=4, sticky="w")
            self.resolution_combo.grid_configure(row=3, column=1, padx=(0, 0), pady=4, sticky="ew")
            self.audio_format_label.grid_configure(row=4, column=0, padx=(0, 8), pady=4, sticky="w")
            self.audio_combo.grid_configure(row=4, column=1, padx=(0, 0), pady=4, sticky="ew")
            return

        self.options_card.columnconfigure(1, weight=1)
        self.mode_label.grid_configure(row=1, column=0, padx=(0, 0), pady=4, sticky="w")
        self.video_radio.grid_configure(row=1, column=1, padx=(0, 0), pady=4, sticky="w")
        self.audio_radio.grid_configure(row=1, column=2, padx=(0, 0), pady=4, sticky="w")
        self.resolution_label_widget.grid_configure(row=2, column=0, padx=(0, 0), pady=4, sticky="w")
        self.resolution_combo.grid_configure(row=2, column=1, padx=(0, 0), pady=4, sticky="w")
        self.audio_format_label.grid_configure(row=2, column=2, padx=(0, 0), pady=4, sticky="w")
        self.audio_combo.grid_configure(row=2, column=3, padx=(0, 0), pady=4, sticky="w")

    def _layout_actions_card(self, compact: bool) -> None:
        for column in range(2):
            self.actions_card.columnconfigure(column, weight=0)

        if compact:
            self.actions_card.columnconfigure(0, weight=1)
            self.actions_card.columnconfigure(1, weight=1)
            self.download_button.grid_configure(row=1, column=0, columnspan=2, padx=(0, 0), pady=(0, 6), sticky="ew")
            self.cancel_button.grid_configure(row=2, column=0, columnspan=2, padx=(0, 0), pady=(0, 6), sticky="ew")
            self.check_deps_button.grid_configure(
                row=3,
                column=0,
                columnspan=2,
                padx=(0, 0),
                pady=(4, 6),
                sticky="ew",
            )
            self.install_deps_button.grid_configure(
                row=4,
                column=0,
                columnspan=2,
                padx=(0, 0),
                pady=(0, 6),
                sticky="ew",
            )
            self.dependency_status_label.grid_configure(
                row=5,
                column=0,
                columnspan=2,
                padx=(0, 0),
                pady=(4, 0),
                sticky="w",
            )
            return

        self.actions_card.columnconfigure(0, weight=1)
        self.download_button.grid_configure(row=1, column=0, columnspan=1, padx=(0, 0), pady=(0, 0), sticky="w")
        self.cancel_button.grid_configure(row=1, column=1, columnspan=1, padx=(10, 0), pady=(0, 0), sticky="w")
        self.check_deps_button.grid_configure(
            row=2,
            column=0,
            columnspan=1,
            padx=(0, 0),
            pady=(10, 0),
            sticky="w",
        )
        self.install_deps_button.grid_configure(
            row=2,
            column=1,
            columnspan=1,
            padx=(10, 0),
            pady=(10, 0),
            sticky="w",
        )
        self.dependency_status_label.grid_configure(
            row=3,
            column=0,
            columnspan=2,
            padx=(0, 0),
            pady=(8, 0),
            sticky="w",
        )

    def _on_close(self) -> None:
        if self._analyzing or self._downloading or self._installing_dependencies:
            if not messagebox.askyesno(
                APP_TITLE, "A task is still running. Do you want to exit and stop it?"
            ):
                return
            self._stop_event.set()
        if self._layout_after_id is not None:
            try:
                self.after_cancel(self._layout_after_id)
            except Exception:
                pass
            self._layout_after_id = None
        if hasattr(self, "canvas"):
            try:
                self.canvas.unbind_all("<MouseWheel>")
            except Exception:
                pass
        self.destroy()

    def _process_events(self) -> None:
        for _ in range(MAX_EVENTS_PER_TICK):
            try:
                event, payload = self._event_queue.get_nowait()
            except queue.Empty:
                break
            self._handle_event(event, payload)
        self.after(POLL_INTERVAL_MS, self._process_events)

    def _handle_event(self, event: str, payload: Any) -> None:
        if event == "log":
            self._add_log(str(payload))
            return

        if event == "probe_started":
            self.status_var.set("Analyzing URL...")
            self._add_log("Analyzing URL for metadata and quality options...")
            return

        if event == "probe_done":
            self._analyzing = False
            self._apply_probe_data(payload if isinstance(payload, dict) else {})
            self.status_var.set("URL analyzed. Ready to download.")
            self._refresh_buttons()
            return

        if event == "probe_error":
            self._analyzing = False
            self.status_var.set("Analyze failed.")
            self._refresh_buttons()
            err = str(payload)
            self._add_log(f"Analyze failed: {err}")
            messagebox.showerror(APP_TITLE, f"Could not analyze URL:\n\n{err}")
            return

        if event == "probe_cancelled":
            self._analyzing = False
            self.status_var.set("Analyze cancelled.")
            self._refresh_buttons()
            self._add_log("Analyze cancelled.")
            return

        if event == "download_started":
            self._start_indeterminate_progress("Downloading...")
            self._add_log("Download started.")
            return

        if event == "progress":
            data = payload if isinstance(payload, dict) else {}
            percent_value = data.get("percent")
            has_total = bool(data.get("has_total"))
            downloaded = data.get("downloaded")
            total = data.get("total")
            speed = data.get("speed")
            speed_str = data.get("speed_str")
            eta = data.get("eta")
            eta_str = data.get("eta_str")

            speed_text = f"{format_bytes(speed)}/s" if isinstance(speed, (int, float)) else (
                str(speed_str).strip() if isinstance(speed_str, str) and speed_str.strip() else "-"
            )
            eta_text = (
                format_eta(eta)
                if isinstance(eta, (int, float))
                else (str(eta_str).strip() if isinstance(eta_str, str) and eta_str.strip() else "-")
            )

            if isinstance(percent_value, (int, float)):
                self._start_determinate_progress()
                percent = min(100.0, max(0.0, float(percent_value)))
                if percent < float(self.progress_var.get()):
                    percent = float(self.progress_var.get())
                self.progress_var.set(percent)
                if has_total:
                    metrics = (
                        f"{percent:5.1f}% | Speed: {speed_text} | ETA: {eta_text} | "
                        f"Downloaded: {format_bytes(downloaded)} / {format_bytes(total)}"
                    )
                else:
                    metrics = (
                        f"{percent:5.1f}% | Speed: {speed_text} | ETA: {eta_text} | "
                        f"Downloaded: {format_bytes(downloaded)}"
                    )
            else:
                self._start_indeterminate_progress()
                metrics = (
                    f"Working... | Speed: {speed_text} | ETA: {eta_text} | "
                    f"Downloaded: {format_bytes(downloaded)}"
                )
            self.metrics_var.set(metrics)
            return

        if event == "stage":
            self._start_indeterminate_progress(str(payload))
            return

        if event == "dep_install_started":
            self.status_var.set("Installing dependencies...")
            self._add_log("Dependency installation started.")
            return

        if event == "dep_install_step":
            self.status_var.set(str(payload))
            self._add_log(str(payload))
            return

        if event == "dep_install_log":
            self._add_log(str(payload))
            return

        if event == "dep_install_done":
            self._installing_dependencies = False
            results = payload if isinstance(payload, list) else []
            ok_results = [
                result for result in results if isinstance(result, DependencyInstallResult) and result.success
            ]
            failed_results = [
                result for result in results if isinstance(result, DependencyInstallResult) and not result.success
            ]
            for result in ok_results:
                self._add_log(result.details)
            for result in failed_results:
                self._add_log(result.details)

            dep_state = self._refresh_dependency_state(log_result=False)
            self._refresh_buttons()
            if failed_results:
                failed_text = "\n".join(f"- {result.label}: {result.details}" for result in failed_results)
                messagebox.showwarning(
                    APP_TITLE,
                    "Some dependencies could not be installed:\n\n" + failed_text,
                )
                self.status_var.set("Dependency install finished with errors.")
                return

            installed_text = "\n".join(f"- {item}" for item in dep_state["installed_lines"])
            messagebox.showinfo(
                APP_TITLE,
                "Dependency installation complete.\n\nInstalled dependencies:\n" + installed_text,
            )
            self.status_var.set("Dependencies installed.")
            return

        if event == "completed":
            self._downloading = False
            self._stop_progress_animation()
            self.progress_var.set(100.0)
            self.status_var.set("Download complete.")
            title = "-"
            quality = "-"
            if isinstance(payload, dict):
                title = str(payload.get("title") or "-")
                quality = str(payload.get("quality") or "-")
            self.metrics_var.set("100.0% | Completed successfully")
            self._add_log(f"Completed: {title}")
            self._add_log(f"Actual downloaded quality: {quality}")
            self._show_notice(
                f"Download complete: {title} ({quality})",
                level="success",
                action_text="Open Folder",
                action_command=self._open_last_output_dir,
                auto_hide_ms=12000,
            )
            self._refresh_buttons()
            return

        if event == "cancelled":
            self._downloading = False
            self._analyzing = False
            self._stop_progress_animation()
            self.status_var.set("Task cancelled.")
            self.metrics_var.set("0.0% | Cancelled")
            self._refresh_buttons()
            self._add_log("Task cancelled by user.")
            self._show_notice("Download cancelled.", level="warning", auto_hide_ms=7000)
            return

        if event == "error":
            self._downloading = False
            self._analyzing = False
            self._stop_progress_animation()
            self.status_var.set("Task failed.")
            self._refresh_buttons()
            err = str(payload)
            self._add_log(f"Task failed: {err}")
            self._show_notice("Download failed. Check activity log for details.", level="error", auto_hide_ms=10000)
            messagebox.showerror(APP_TITLE, f"Task failed:\n\n{err}")
            return

    def _apply_probe_data(self, data: dict[str, Any]) -> None:
        title = str(data.get("title") or "-")
        uploader = str(data.get("uploader") or "-")
        duration = data.get("duration")
        heights = data.get("heights") if isinstance(data.get("heights"), list) else []
        muxed_heights = data.get("muxed_heights") if isinstance(data.get("muxed_heights"), list) else []

        self.title_var.set(f"Title: {title}")
        self.channel_var.set(f"Uploader: {uploader}")
        self.duration_var.set(f"Duration: {format_duration(duration if isinstance(duration, int) else None)}")

        profiles = self._build_video_profiles(
            [h for h in heights if isinstance(h, int)],
            [h for h in muxed_heights if isinstance(h, int)],
        )
        self._video_profiles = profiles
        labels = [profile.label for profile in profiles]
        self.resolution_combo["values"] = labels
        if labels:
            self.resolution_var.set(labels[0])

        self._add_log(f"Metadata loaded: {title}")
        if len(profiles) > 1:
            self._add_log(f"Detected {len(profiles) - 1} video resolution options.")
        else:
            self._add_log("No explicit quality list returned. Best-quality fallback will be used.")
        if any(h not in STANDARD_RESOLUTION_STEPS for h in heights if isinstance(h, int)):
            self._add_log(
                "Source returned non-standard stream heights. Showing normalized quality tiers (144p/240p/360p/etc.)."
            )
        if not self.ffmpeg_available and any(p.requires_ffmpeg for p in profiles):
            self._add_log(
                "FFmpeg missing: some higher resolutions are unavailable because separate video/audio streams cannot be merged."
            )

    def _start_probe(self) -> None:
        if self._analyzing or self._downloading or self._installing_dependencies:
            return
        self._refresh_dependency_state(log_result=False)
        url = self.url_var.get().strip()
        if not url:
            messagebox.showwarning(APP_TITLE, "Please paste a YouTube URL first.")
            return
        self._stop_event = threading.Event()
        self._analyzing = True
        self._refresh_buttons()
        self._probe_worker = ProbeWorker(url=url, event_queue=self._event_queue, stop_event=self._stop_event)
        self._probe_worker.start()

    def _start_download(self) -> None:
        if self._analyzing or self._downloading or self._installing_dependencies:
            return
        self._refresh_dependency_state(log_result=False)

        url = self.url_var.get().strip()
        if not url:
            messagebox.showwarning(APP_TITLE, "Please paste a YouTube URL first.")
            return

        output_path = Path(self.output_var.get().strip()).expanduser()
        try:
            output_path.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"Cannot use output directory:\n\n{exc}")
            return

        mode = self.mode_var.get()
        selected_profile = next(
            (profile for profile in self._video_profiles if profile.label == self.resolution_var.get()),
            self._video_profiles[0],
        )
        if (
            mode == "video"
            and selected_profile.requires_ffmpeg
            and not self.ffmpeg_available
        ):
            messagebox.showerror(
                APP_TITLE,
                "Selected quality needs FFmpeg to merge video + audio.\n\n"
                "Install FFmpeg and make sure 'ffmpeg' is in PATH, then try again.",
            )
            self._add_log(
                "Blocked download: selected resolution requires FFmpeg for combined video+audio output."
            )
            return

        self._stop_event = threading.Event()
        self._downloading = True
        self._last_output_dir = output_path
        self._hide_notice()
        self.progress_var.set(0.0)
        self._start_indeterminate_progress("Preparing download...")
        self.metrics_var.set("Connecting... | Speed: - | ETA: - | Downloaded: 0 B")
        self._refresh_buttons()

        self._add_log(f"Output directory: {output_path}")
        if mode == "video":
            quality_label = selected_profile.label
            self._add_log(f"Mode: video with sound ({quality_label})")
        else:
            self._add_log(f"Mode: audio only ({self.audio_format_var.get()})")

        self._download_worker = DownloadWorker(
            url=url,
            output_dir=output_path,
            mode=mode,
            resolution_height=selected_profile.height,
            audio_format=self.audio_format_var.get(),
            ffmpeg_available=self.ffmpeg_available,
            event_queue=self._event_queue,
            stop_event=self._stop_event,
        )
        self._download_worker.start()

    def _check_dependencies(self) -> None:
        if self._analyzing or self._downloading or self._installing_dependencies:
            return
        dep_state = self._refresh_dependency_state(log_result=True)
        self._refresh_buttons()

        installed_lines = dep_state["installed_lines"]
        missing_lines = dep_state["missing_lines"]
        installable_keys = dep_state["installable_keys"]

        if not missing_lines:
            messagebox.showinfo(
                APP_TITLE,
                "All required dependencies are installed:\n\n"
                + "\n".join(f"- {item}" for item in installed_lines),
            )
            return

        message_lines = ["Missing dependencies:"]
        message_lines.extend(f"- {item}" for item in missing_lines)
        if "ffmpeg" in dep_state["missing_keys"] and "ffmpeg" not in installable_keys:
            message_lines.append("")
            message_lines.append(
                "Automatic FFmpeg install is unavailable (winget not found)."
            )
        if installable_keys:
            message_lines.append("")
            message_lines.append("Install all missing dependencies now?")
            if messagebox.askyesno(APP_TITLE, "\n".join(message_lines)):
                self._start_dependency_install(installable_keys)
                return
            return
        messagebox.showwarning(APP_TITLE, "\n".join(message_lines))

    def _start_dependency_install(self, dependency_keys: list[str]) -> None:
        if self._analyzing or self._downloading or self._installing_dependencies:
            return
        keys = [key for key in dependency_keys if key in {"yt-dlp", "ffmpeg"}]
        if not keys:
            messagebox.showinfo(APP_TITLE, "There are no auto-installable dependencies to install.")
            return

        self._installing_dependencies = True
        self.status_var.set("Installing dependencies...")
        labels = ", ".join(self._dependency_label(key) for key in keys)
        self._add_log(f"Starting dependency install: {labels}")
        self._refresh_buttons()

        self._dep_install_worker = DependencyInstallWorker(
            dependency_keys=keys,
            can_install_ffmpeg=self._can_auto_install_ffmpeg(),
            event_queue=self._event_queue,
        )
        self._dep_install_worker.start()

    def _install_missing_dependencies(self) -> None:
        dep_state = self._refresh_dependency_state(log_result=False)
        self._refresh_buttons()
        missing_lines = dep_state["missing_lines"]
        installable_keys = dep_state["installable_keys"]

        if not missing_lines:
            messagebox.showinfo(APP_TITLE, "All dependencies are already installed.")
            return
        if not installable_keys:
            messagebox.showwarning(
                APP_TITLE,
                "Missing dependencies cannot be auto-installed on this machine.\n"
                "Install them manually and make sure they are in PATH.",
            )
            return
        labels = ", ".join(self._dependency_label(key) for key in installable_keys)
        if not messagebox.askyesno(APP_TITLE, f"Install missing dependencies now?\n\n{labels}"):
            return
        self._start_dependency_install(installable_keys)

    def _cancel_active_job(self) -> None:
        if not (self._analyzing or self._downloading):
            return
        self._stop_event.set()
        self.status_var.set("Cancelling...")
        self._add_log("Cancellation requested...")

    def _refresh_mode_controls(self) -> None:
        mode = self.mode_var.get()
        if mode == "audio":
            self.resolution_combo.configure(state="disabled")
            self.audio_combo.configure(state="readonly")
        else:
            self.resolution_combo.configure(state="readonly")
            self.audio_combo.configure(state="disabled")

    def _refresh_buttons(self) -> None:
        busy = self._analyzing or self._downloading or self._installing_dependencies
        self.analyze_button.configure(state="disabled" if busy else "normal")
        self.download_button.configure(state="disabled" if busy else "normal")
        self.cancel_button.configure(state="normal" if (self._analyzing or self._downloading) else "disabled")
        self.check_deps_button.configure(state="disabled" if busy else "normal")
        self.install_deps_button.configure(
            state="normal" if (not busy and bool(self._installable_dependency_keys)) else "disabled"
        )
        self.browse_button.configure(state="disabled" if busy else "normal")
        self.url_entry.configure(state="disabled" if busy else "normal")
        self.output_entry.configure(state="disabled" if busy else "normal")
        self.video_radio.configure(state="disabled" if busy else "normal")
        self.audio_radio.configure(state="disabled" if busy else "normal")
        if busy:
            self.resolution_combo.configure(state="disabled")
            self.audio_combo.configure(state="disabled")
        else:
            self._refresh_mode_controls()

    def _choose_output_dir(self) -> None:
        initial = self.output_var.get().strip() or str(Path.home())
        chosen = filedialog.askdirectory(title="Select download folder", initialdir=initial)
        if chosen:
            self.output_var.set(chosen)

    def _paste_url(self) -> None:
        try:
            clipboard_text = self.clipboard_get().strip()
        except tk.TclError:
            clipboard_text = ""
        if clipboard_text:
            self.url_var.set(clipboard_text)
            self._add_log("Pasted URL from clipboard.")

    def _add_log(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        line = f"[{timestamp}] {message}\n"
        self.log_text.configure(state="normal")
        self.log_text.insert("end", line)
        total_lines = int(self.log_text.index("end-1c").split(".")[0])
        if total_lines > MAX_LOG_LINES:
            self.log_text.delete("1.0", f"{total_lines - MAX_LOG_LINES + 1}.0")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")


def main() -> None:
    app = YouTubeDownloaderApp()
    app.mainloop()


if __name__ == "__main__":
    main()
