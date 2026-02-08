from __future__ import annotations

import importlib
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None  # type: ignore[assignment]
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

UI_POLL_MS = 40
MAX_EVENTS_PER_TICK = 200
SCAN_PROGRESS_EMIT_EVERY = 2000
MAX_LOG_LINES = 2500
FILE_LOG_EVERY_N = 25

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD  # type: ignore

    DND_AVAILABLE = True
except Exception:
    DND_AVAILABLE = False


@dataclass
class ConversionSummary:
    total_pdfs: int
    converted_pdfs: int
    failed_pdfs: int
    images_written: int
    output_dir: Path
    duration_s: float


@dataclass(frozen=True)
class DependencyInstallResult:
    package: str
    success: bool
    details: str


class PDFToImagesWorker(threading.Thread):
    def __init__(
        self,
        source_path: Path,
        output_dir: Path,
        dpi: int,
        image_format: str,
        jpg_quality: int,
        stop_event: threading.Event,
        event_queue: "queue.Queue[tuple[str, object]]",
    ) -> None:
        super().__init__(daemon=True)
        self.source_path = source_path
        self.source_is_file = source_path.is_file()
        self.output_dir = output_dir
        self.dpi = dpi
        self.image_format = image_format
        self.jpg_quality = jpg_quality
        self.stop_event = stop_event
        self.event_queue = event_queue

    def _list_pdfs(self) -> List[Path]:
        if self.source_is_file:
            if self.source_path.suffix.lower() == ".pdf":
                return [self.source_path]
            return []

        pdf_files: List[Path] = []
        to_scan: List[Path] = [self.source_path]
        entries_seen = 0

        while to_scan:
            if self.stop_event.is_set():
                return pdf_files

            current = to_scan.pop()
            try:
                with os.scandir(current) as it:
                    for entry in it:
                        entries_seen += 1
                        if entries_seen % SCAN_PROGRESS_EMIT_EVERY == 0:
                            self._emit("scan_progress", {"entries_seen": entries_seen})

                        try:
                            if entry.is_dir(follow_symlinks=False):
                                to_scan.append(Path(entry.path))
                                continue
                            if entry.is_file(follow_symlinks=False) and entry.name.lower().endswith(".pdf"):
                                pdf_files.append(Path(entry.path))
                        except OSError:
                            continue
            except OSError:
                continue

        pdf_files.sort()
        return pdf_files

    def _emit(self, event: str, payload: object) -> None:
        self.event_queue.put((event, payload))

    def _save_page_image(self, page, out_path: Path) -> None:
        scale = self.dpi / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        if self.image_format == "png":
            pix.save(str(out_path))
            return
        pix.save(str(out_path), jpg_quality=self.jpg_quality)

    def run(self) -> None:
        started = time.time()
        converted_pdfs = 0
        failed_pdfs = 0
        images_written = 0

        try:
            if fitz is None:
                self._emit("fatal_error", "PyMuPDF is not installed. Install with: pip install pymupdf")
                return

            pdf_files = self._list_pdfs()
            if self.stop_event.is_set():
                self._emit("cancelled", {"index": 0, "total": 0})
                return
            total_pdfs = len(pdf_files)
            self._emit("scan_complete", {"total_pdfs": total_pdfs})

            if total_pdfs == 0:
                summary = ConversionSummary(
                    total_pdfs=0,
                    converted_pdfs=0,
                    failed_pdfs=0,
                    images_written=0,
                    output_dir=self.output_dir,
                    duration_s=time.time() - started,
                )
                self._emit("done", summary)
                return

            self.output_dir.mkdir(parents=True, exist_ok=True)

            for index, pdf_path in enumerate(pdf_files, start=1):
                if self.stop_event.is_set():
                    self._emit("cancelled", {"index": index, "total": total_pdfs})
                    return

                if self.source_is_file:
                    out_pdf_dir = self.output_dir / pdf_path.stem
                else:
                    rel = pdf_path.relative_to(self.source_path)
                    out_pdf_dir = self.output_dir / rel.parent / pdf_path.stem
                out_pdf_dir.mkdir(parents=True, exist_ok=True)

                self._emit(
                    "file_start",
                    {
                        "index": index,
                        "total": total_pdfs,
                        "pdf_path": str(pdf_path),
                    },
                )

                try:
                    with fitz.open(pdf_path) as doc:
                        page_count = len(doc)
                        self._emit("pdf_pages", {"pdf_path": str(pdf_path), "pages": page_count})

                        for page_no in range(page_count):
                            if self.stop_event.is_set():
                                self._emit("cancelled", {"index": index, "total": total_pdfs})
                                return

                            page = doc[page_no]
                            suffix = "png" if self.image_format == "png" else "jpg"
                            out_name = f"page_{page_no + 1:04d}.{suffix}"
                            out_path = out_pdf_dir / out_name
                            self._save_page_image(page, out_path)
                            images_written += 1
                            if page_no % 8 == 0:
                                time.sleep(0)

                    converted_pdfs += 1

                except Exception as exc:
                    failed_pdfs += 1
                    self._emit(
                        "file_error",
                        {
                            "pdf_path": str(pdf_path),
                            "error": str(exc),
                        },
                    )

                percent = (index / total_pdfs) * 100.0
                self._emit(
                    "progress",
                    {
                        "index": index,
                        "total": total_pdfs,
                        "percent": percent,
                        "converted_pdfs": converted_pdfs,
                        "failed_pdfs": failed_pdfs,
                        "images_written": images_written,
                    },
                )

            summary = ConversionSummary(
                total_pdfs=total_pdfs,
                converted_pdfs=converted_pdfs,
                failed_pdfs=failed_pdfs,
                images_written=images_written,
                output_dir=self.output_dir,
                duration_s=time.time() - started,
            )
            self._emit("done", summary)

        except Exception as exc:
            self._emit("fatal_error", str(exc))


class DependencyInstallWorker(threading.Thread):
    def __init__(
        self,
        packages: list[str],
        event_queue: "queue.Queue[tuple[str, object]]",
    ) -> None:
        super().__init__(daemon=True)
        self.packages = packages
        self.event_queue = event_queue

    def _emit(self, event: str, payload: object) -> None:
        self.event_queue.put((event, payload))

    def run(self) -> None:
        self._emit("dep_install_started", {"packages": list(self.packages)})
        results: list[DependencyInstallResult] = []
        for package in self.packages:
            self._emit("dep_install_step", {"package": package})
            try:
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "--upgrade",
                        "--disable-pip-version-check",
                        package,
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
            except Exception as exc:
                results.append(
                    DependencyInstallResult(
                        package=package,
                        success=False,
                        details=f"Failed to run installer: {exc}",
                    )
                )
                continue

            out = (completed.stdout or "").strip()
            err = (completed.stderr or "").strip()
            output_tail = "\n".join([line for line in (out + "\n" + err).splitlines() if line][-12:])
            if output_tail:
                self._emit("dep_install_log", {"package": package, "text": output_tail})

            if completed.returncode == 0:
                results.append(
                    DependencyInstallResult(
                        package=package,
                        success=True,
                        details=f"{package} installed successfully.",
                    )
                )
            else:
                results.append(
                    DependencyInstallResult(
                        package=package,
                        success=False,
                        details=f"{package} install failed with exit code {completed.returncode}.",
                    )
                )

        self._emit("dep_install_done", results)


class PDFToImagesApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("PDF to Images Converter Pro")
        self.root.geometry("1120x760")
        self.root.minsize(980, 680)

        self.source_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.dpi_var = tk.IntVar(value=200)
        self.format_var = tk.StringVar(value="png")
        self.jpg_quality_var = tk.IntVar(value=92)
        self.progress_var = tk.DoubleVar(value=0.0)
        self.status_var = tk.StringVar(value="Ready.")
        self.percent_var = tk.StringVar(value="0.0%")
        self.current_file_var = tk.StringVar(value="No file in progress.")
        self.verbose_log_var = tk.BooleanVar(value=False)
        self.total_var = tk.StringVar(value="0")
        self.converted_var = tk.StringVar(value="0")
        self.failed_var = tk.StringVar(value="0")
        self.images_var = tk.StringVar(value="0")
        self.elapsed_var = tk.StringVar(value="00:00:00")
        self._log_write_count = 0
        self._run_started_at: Optional[float] = None
        self._elapsed_after_id: Optional[str] = None

        self.event_queue: "queue.Queue[tuple[str, object]]" = queue.Queue()
        self.stop_event = threading.Event()
        self.worker: Optional[PDFToImagesWorker] = None
        self.dep_worker: Optional[DependencyInstallWorker] = None
        self.last_output_dir: Optional[Path] = None
        self._installing_deps = False
        self._has_fitz = fitz is not None
        self._has_tkdnd_runtime = DND_AVAILABLE
        self._missing_install_packages: list[str] = []
        self.dep_status_var = tk.StringVar(value="Dependencies: checking...")
        self.dep_hint_var = tk.StringVar(value="")

        self.start_btn: ttk.Button
        self.cancel_btn: ttk.Button
        self.open_btn: ttk.Button
        self.zip_btn: ttk.Button
        self.check_deps_btn: ttk.Button
        self.install_deps_btn: ttk.Button
        self.log_text: ScrolledText
        self.jpg_quality_spin: ttk.Spinbox
        self.progress: ttk.Progressbar
        self.drop_area: tk.Label

        self._apply_theme()
        self._build_ui()
        self._refresh_dependency_state(log_result=False)
        self._set_running_state(False)

    def _apply_theme(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except Exception:
            pass

        self.root.configure(bg="#e9eff6")
        style.configure("App.TFrame", background="#e9eff6")
        style.configure("Card.TLabelframe", background="#ffffff", borderwidth=1, relief="solid")
        style.configure(
            "Card.TLabelframe.Label",
            background="#ffffff",
            foreground="#1b2c40",
            font=("Segoe UI Semibold", 10),
        )
        style.configure(
            "HeaderTitle.TLabel",
            background="#0f1f3a",
            foreground="#f8fafc",
            font=("Segoe UI Semibold", 19),
        )
        style.configure(
            "HeaderSub.TLabel",
            background="#0f1f3a",
            foreground="#c9d8f0",
            font=("Segoe UI", 10),
        )
        style.configure("Muted.TLabel", background="#e9eff6", foreground="#4f6074", font=("Segoe UI", 10))
        style.configure("CardLabel.TLabel", background="#ffffff", foreground="#1f2a38", font=("Segoe UI", 10))
        style.configure("SmallMuted.TLabel", background="#ffffff", foreground="#647488", font=("Segoe UI", 9))
        style.configure(
            "Accent.TButton",
            background="#1e88e5",
            foreground="#ffffff",
            borderwidth=0,
            font=("Segoe UI Semibold", 10),
            padding=(10, 8),
        )
        style.map(
            "Accent.TButton",
            background=[("active", "#1976d2"), ("disabled", "#9ecaf2")],
            foreground=[("disabled", "#e8f2fd")],
        )
        style.configure(
            "Neutral.TButton",
            background="#e2e8f0",
            foreground="#1f2a38",
            borderwidth=0,
            font=("Segoe UI Semibold", 10),
            padding=(10, 8),
        )
        style.map(
            "Neutral.TButton",
            background=[("active", "#cfd8e3"), ("disabled", "#ecf1f7")],
            foreground=[("disabled", "#8ca0b9")],
        )
        style.configure(
            "Danger.TButton",
            background="#dc2626",
            foreground="#ffffff",
            borderwidth=0,
            font=("Segoe UI Semibold", 10),
            padding=(10, 8),
        )
        style.map(
            "Danger.TButton",
            background=[("active", "#b91c1c"), ("disabled", "#f3b8b8")],
            foreground=[("disabled", "#ffffff")],
        )
        style.configure(
            "Success.TButton",
            background="#059669",
            foreground="#ffffff",
            borderwidth=0,
            font=("Segoe UI Semibold", 10),
            padding=(10, 8),
        )
        style.map(
            "Success.TButton",
            background=[("active", "#047857"), ("disabled", "#9ad7c5")],
            foreground=[("disabled", "#eefcf7")],
        )
        style.configure("Accent.Horizontal.TProgressbar", troughcolor="#d8e3f0", background="#1e88e5")
        style.configure("Dark.TFrame", background="#0f1f3a")

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=14, style="App.TFrame")
        outer.pack(fill="both", expand=True)

        header = ttk.Frame(outer, style="Dark.TFrame", padding=(16, 14))
        header.pack(fill="x", pady=(0, 12))
        ttk.Label(header, text="PDF Dataset to Images", style="HeaderTitle.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="Convert PDFs into mirrored image folders with progress, logs, and ZIP export.",
            style="HeaderSub.TLabel",
        ).pack(anchor="w", pady=(2, 0))

        content = ttk.Panedwindow(outer, orient="horizontal")
        content.pack(fill="both", expand=True)

        left = ttk.Frame(content, style="App.TFrame")
        right = ttk.Frame(content, style="App.TFrame")
        content.add(left, weight=46)
        content.add(right, weight=54)

        input_card = ttk.LabelFrame(left, text="Input (Folder or PDF)", style="Card.TLabelframe", padding=12)
        input_card.pack(fill="x")
        input_row = ttk.Frame(input_card, style="App.TFrame")
        input_row.pack(fill="x")
        source_entry = ttk.Entry(input_row, textvariable=self.source_var)
        source_entry.pack(side="left", fill="x", expand=True)
        ttk.Button(input_row, text="Browse Folder...", command=self.pick_source, style="Neutral.TButton").pack(
            side="left", padx=(8, 0)
        )
        ttk.Button(input_row, text="Browse PDF...", command=self.pick_source_file, style="Neutral.TButton").pack(
            side="left", padx=(8, 0)
        )

        drop_hint = (
            "Drop a folder or a single PDF here"
            if DND_AVAILABLE
            else "Drag and drop unavailable (install tkinterdnd2). Use Browse Folder/PDF."
        )
        self.drop_area = tk.Label(
            input_card,
            text=drop_hint,
            relief="groove",
            bd=1,
            height=3,
            bg="#f1f7ff",
            fg="#224164",
            font=("Segoe UI", 10),
        )
        self.drop_area.pack(fill="x", pady=(10, 0))
        if DND_AVAILABLE:
            self.drop_area.drop_target_register(DND_FILES)  # type: ignore[attr-defined]
            self.drop_area.dnd_bind("<<Drop>>", self.on_drop)  # type: ignore[attr-defined]

        output_card = ttk.LabelFrame(left, text="Output Folder", style="Card.TLabelframe", padding=12)
        output_card.pack(fill="x", pady=(10, 0))
        output_row = ttk.Frame(output_card, style="App.TFrame")
        output_row.pack(fill="x")
        out_entry = ttk.Entry(output_row, textvariable=self.output_var)
        out_entry.pack(side="left", fill="x", expand=True)
        ttk.Button(output_row, text="Browse...", command=self.pick_output, style="Neutral.TButton").pack(
            side="left", padx=(8, 0)
        )

        settings = ttk.LabelFrame(left, text="Settings", style="Card.TLabelframe", padding=12)
        settings.pack(fill="x", pady=(10, 0))
        settings.grid_columnconfigure(1, weight=1)
        settings.grid_columnconfigure(3, weight=1)

        ttk.Label(settings, text="DPI", style="CardLabel.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(settings, from_=72, to=600, increment=10, textvariable=self.dpi_var, width=8).grid(
            row=0, column=1, sticky="w", padx=(8, 14)
        )
        ttk.Label(settings, text="Format", style="CardLabel.TLabel").grid(row=0, column=2, sticky="w")
        fmt = ttk.Combobox(
            settings,
            textvariable=self.format_var,
            state="readonly",
            values=["png", "jpg"],
            width=10,
        )
        fmt.grid(row=0, column=3, sticky="w", padx=(8, 14))
        fmt.bind("<<ComboboxSelected>>", lambda _: self._toggle_quality_state())

        ttk.Label(settings, text="JPG Quality", style="CardLabel.TLabel").grid(row=1, column=0, sticky="w", pady=(10, 0))
        self.jpg_quality_spin = ttk.Spinbox(
            settings,
            from_=30,
            to=100,
            increment=1,
            textvariable=self.jpg_quality_var,
            width=8,
        )
        self.jpg_quality_spin.grid(row=1, column=1, sticky="w", padx=(8, 14), pady=(10, 0))
        self._toggle_quality_state()

        ttk.Checkbutton(settings, text="Verbose log", variable=self.verbose_log_var).grid(
            row=1, column=2, columnspan=2, sticky="w", padx=(0, 0), pady=(10, 0)
        )

        actions = ttk.LabelFrame(left, text="Actions", style="Card.TLabelframe", padding=12)
        actions.pack(fill="x", pady=(10, 0))
        self.start_btn = ttk.Button(actions, text="Start Conversion", command=self.start_conversion, style="Accent.TButton")
        self.start_btn.pack(fill="x")
        self.cancel_btn = ttk.Button(actions, text="Cancel", command=self.cancel_conversion, state="disabled", style="Danger.TButton")
        self.cancel_btn.pack(fill="x", pady=(8, 0))
        self.open_btn = ttk.Button(
            actions,
            text="Open Output Folder",
            command=self.open_output,
            state="disabled",
            style="Success.TButton",
        )
        self.open_btn.pack(fill="x", pady=(8, 0))
        self.zip_btn = ttk.Button(
            actions,
            text="Create ZIP (Download)",
            command=self.create_zip,
            state="disabled",
            style="Neutral.TButton",
        )
        self.zip_btn.pack(fill="x", pady=(8, 0))

        dep_card = ttk.LabelFrame(left, text="Dependencies", style="Card.TLabelframe", padding=12)
        dep_card.pack(fill="x", pady=(10, 0))
        ttk.Label(dep_card, textvariable=self.dep_status_var, style="CardLabel.TLabel", wraplength=420).pack(anchor="w")
        ttk.Label(dep_card, textvariable=self.dep_hint_var, style="SmallMuted.TLabel", wraplength=420).pack(
            anchor="w", pady=(4, 0)
        )
        dep_actions = ttk.Frame(dep_card, style="App.TFrame")
        dep_actions.pack(fill="x", pady=(8, 0))
        self.check_deps_btn = ttk.Button(
            dep_actions,
            text="Check Dependencies",
            command=self.check_dependencies,
            style="Neutral.TButton",
        )
        self.check_deps_btn.pack(side="left", fill="x", expand=True)
        self.install_deps_btn = ttk.Button(
            dep_actions,
            text="Install Missing",
            command=self.install_missing_dependencies,
            style="Accent.TButton",
        )
        self.install_deps_btn.pack(side="left", fill="x", expand=True, padx=(8, 0))

        stats_card = ttk.LabelFrame(right, text="Overview", style="Card.TLabelframe", padding=12)
        stats_card.pack(fill="x")
        stats_grid = ttk.Frame(stats_card, style="App.TFrame")
        stats_grid.pack(fill="x")
        stats_grid.grid_columnconfigure((0, 1), weight=1)
        self._build_stat_tile(stats_grid, "PDFs Found", self.total_var, 0, 0)
        self._build_stat_tile(stats_grid, "Converted", self.converted_var, 0, 1)
        self._build_stat_tile(stats_grid, "Failed", self.failed_var, 1, 0)
        self._build_stat_tile(stats_grid, "Images", self.images_var, 1, 1)

        progress_card = ttk.LabelFrame(right, text="Progress", style="Card.TLabelframe", padding=12)
        progress_card.pack(fill="x", pady=(10, 0))
        top_prog = ttk.Frame(progress_card, style="App.TFrame")
        top_prog.pack(fill="x")
        ttk.Label(top_prog, text="Overall", style="CardLabel.TLabel").pack(side="left")
        ttk.Label(top_prog, textvariable=self.percent_var, style="CardLabel.TLabel").pack(side="right")
        self.progress = ttk.Progressbar(
            progress_card,
            variable=self.progress_var,
            maximum=100.0,
            mode="determinate",
            style="Accent.Horizontal.TProgressbar",
        )
        self.progress.pack(fill="x", pady=(8, 8))
        ttk.Label(progress_card, text="Current File", style="SmallMuted.TLabel").pack(anchor="w")
        ttk.Label(progress_card, textvariable=self.current_file_var, style="CardLabel.TLabel", wraplength=500).pack(anchor="w")

        log_card = ttk.LabelFrame(right, text="Debug Log", style="Card.TLabelframe", padding=10)
        log_card.pack(fill="both", expand=True, pady=(10, 0))
        log_tools = ttk.Frame(log_card, style="App.TFrame")
        log_tools.pack(fill="x", pady=(0, 6))
        ttk.Button(log_tools, text="Clear Log", command=self.clear_log, style="Neutral.TButton").pack(side="left")
        ttk.Label(log_tools, text="Consolas-style activity stream", style="SmallMuted.TLabel").pack(side="right")

        self.log_text = ScrolledText(
            log_card,
            height=14,
            wrap="word",
            font=("Consolas", 10),
            bg="#0f1722",
            fg="#cfe2ff",
            insertbackground="#cfe2ff",
            relief="flat",
            bd=0,
        )
        self.log_text.pack(fill="both", expand=True)
        self.log_text.configure(state="disabled")

        status_bar = tk.Frame(outer, bg="#dbe8f6", height=28)
        status_bar.pack(fill="x", pady=(10, 0))
        status_bar.pack_propagate(False)
        tk.Label(
            status_bar,
            textvariable=self.status_var,
            bg="#dbe8f6",
            fg="#223246",
            anchor="w",
            padx=10,
            font=("Segoe UI", 9),
        ).pack(side="left", fill="x", expand=True)
        tk.Label(
            status_bar,
            textvariable=self.elapsed_var,
            bg="#dbe8f6",
            fg="#223246",
            anchor="e",
            padx=10,
            font=("Segoe UI", 9, "bold"),
        ).pack(side="right")

    def _build_stat_tile(self, parent: ttk.Frame, title: str, var: tk.StringVar, row: int, col: int) -> None:
        tile = tk.Frame(parent, bg="#f4f8ff", bd=1, relief="solid", highlightthickness=0)
        tile.grid(row=row, column=col, sticky="nsew", padx=4, pady=4)
        tk.Label(tile, text=title, bg="#f4f8ff", fg="#4f6074", font=("Segoe UI", 9)).pack(anchor="w", padx=10, pady=(8, 2))
        tk.Label(tile, textvariable=var, bg="#f4f8ff", fg="#17263a", font=("Segoe UI Semibold", 18)).pack(
            anchor="w", padx=10, pady=(0, 8)
        )

    def _toggle_quality_state(self) -> None:
        if self.format_var.get() == "jpg":
            self.jpg_quality_spin.configure(state="normal")
        else:
            self.jpg_quality_spin.configure(state="disabled")

    @staticmethod
    def _fmt_elapsed(seconds: int) -> str:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def _start_elapsed_timer(self) -> None:
        self._run_started_at = time.time()
        self.elapsed_var.set("00:00:00")
        self._schedule_elapsed_tick()

    def _schedule_elapsed_tick(self) -> None:
        if self._run_started_at is None:
            return
        elapsed = int(time.time() - self._run_started_at)
        self.elapsed_var.set(self._fmt_elapsed(elapsed))
        if self.worker and self.worker.is_alive():
            self._elapsed_after_id = self.root.after(1000, self._schedule_elapsed_tick)
        else:
            self._elapsed_after_id = None

    def _stop_elapsed_timer(self) -> None:
        if self._elapsed_after_id:
            try:
                self.root.after_cancel(self._elapsed_after_id)
            except Exception:
                pass
            self._elapsed_after_id = None
        if self._run_started_at is not None:
            elapsed = int(time.time() - self._run_started_at)
            self.elapsed_var.set(self._fmt_elapsed(elapsed))
            self._run_started_at = None

    def _log(self, message: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{stamp}] {message}\n"
        self.log_text.configure(state="normal")
        self.log_text.insert("end", line)
        self._log_write_count += 1
        if self._log_write_count % 25 == 0:
            total_lines = int(float(self.log_text.index("end-1c").split(".")[0]))
            if total_lines > MAX_LOG_LINES:
                trim_to_line = total_lines - MAX_LOG_LINES
                self.log_text.delete("1.0", f"{trim_to_line}.0")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _set_status(self, text: str) -> None:
        self.status_var.set(text)

    def clear_log(self) -> None:
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")
        self._log_write_count = 0

    def pick_source(self) -> None:
        path = filedialog.askdirectory(title="Select Source Folder")
        if not path:
            return
        self.source_var.set(path)
        self._suggest_output(Path(path))
        self._log(f"Source selected: {path}")

    def pick_source_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Select PDF File",
            filetypes=[("PDF files", "*.pdf")],
        )
        if not path:
            return
        self.source_var.set(path)
        self._suggest_output(Path(path))
        self._log(f"Source file selected: {path}")

    def pick_output(self) -> None:
        path = filedialog.askdirectory(title="Select Output Folder")
        if not path:
            return
        self.output_var.set(path)
        self._log(f"Output selected: {path}")

    def _suggest_output(self, source: Path) -> None:
        if self.output_var.get().strip():
            return
        if source.is_file():
            suggestion = source.parent / f"{source.stem}_images"
        else:
            suggestion = source.parent / f"{source.name}_images"
        self.output_var.set(str(suggestion))

    def on_drop(self, event) -> None:
        dropped = self.root.tk.splitlist(event.data)
        if not dropped:
            return
        candidate = dropped[0].strip("{}")
        path = Path(candidate)
        if not path.exists():
            self._log(f"Drop ignored (path not found): {candidate}")
            return
        if path.is_file() and path.suffix.lower() != ".pdf":
            self._log(f"Drop ignored (not a PDF file): {candidate}")
            return
        self.source_var.set(str(path))
        self._suggest_output(path)
        if path.is_file():
            self._log(f"PDF dropped: {path}")
        else:
            self._log(f"Folder dropped: {path}")

    def _set_running_state(self, running: bool) -> None:
        has_active_conversion = running
        busy = has_active_conversion or self._installing_deps
        self.start_btn.configure(state="disabled" if (busy or not self._has_fitz) else "normal")
        self.cancel_btn.configure(state="normal" if has_active_conversion else "disabled")
        self.open_btn.configure(state="disabled" if busy else ("normal" if self.last_output_dir else "disabled"))
        self.zip_btn.configure(state="disabled" if busy else ("normal" if self.last_output_dir else "disabled"))
        self.check_deps_btn.configure(state="disabled" if busy else "normal")
        self.install_deps_btn.configure(
            state="normal" if (not busy and bool(self._missing_install_packages)) else "disabled"
        )

    def _refresh_dependency_state(self, log_result: bool = True) -> dict[str, object]:
        global fitz
        fitz_ok = False
        tkdnd_ok = False

        try:
            fitz_module = importlib.import_module("fitz")
            fitz = fitz_module  # type: ignore[assignment]
            fitz_ok = True
        except Exception:
            fitz_ok = False

        try:
            importlib.import_module("tkinterdnd2")
            tkdnd_ok = True
        except Exception:
            tkdnd_ok = False

        self._has_fitz = fitz_ok
        self._has_tkdnd_runtime = tkdnd_ok
        missing_packages: list[str] = []
        missing_labels: list[str] = []
        if not fitz_ok:
            missing_packages.append("pymupdf")
            missing_labels.append("PyMuPDF (required)")
        if not tkdnd_ok:
            missing_packages.append("tkinterdnd2")
            missing_labels.append("tkinterdnd2 (optional drag/drop)")

        self._missing_install_packages = missing_packages

        if not missing_labels:
            self.dep_status_var.set("All dependencies are installed.")
            if DND_AVAILABLE:
                self.dep_hint_var.set("Drag and drop is active.")
                self.drop_area.configure(text="Drop a folder or a single PDF here")
            elif tkdnd_ok:
                self.dep_hint_var.set("tkinterdnd2 is installed. Restart app to activate drag and drop.")
            else:
                self.dep_hint_var.set("Core dependencies are ready.")
            if log_result:
                self._log("Dependency check: all dependencies are installed.")
        else:
            self.dep_status_var.set("Missing: " + ", ".join(missing_labels))
            hints: list[str] = []
            if not fitz_ok:
                hints.append("PyMuPDF is required for conversion.")
            if not tkdnd_ok:
                hints.append("Install tkinterdnd2 to enable drag and drop.")
            elif tkdnd_ok and not DND_AVAILABLE:
                hints.append("tkinterdnd2 installed. Restart app to enable drag and drop.")
            self.dep_hint_var.set(" ".join(hints))
            if log_result:
                self._log("Dependency check: " + ", ".join(missing_labels))

        self._set_running_state(bool(self.worker and self.worker.is_alive()))
        return {
            "fitz_ok": fitz_ok,
            "tkdnd_ok": tkdnd_ok,
            "missing_packages": missing_packages,
            "missing_labels": missing_labels,
        }

    def check_dependencies(self) -> None:
        if (self.worker and self.worker.is_alive()) or self._installing_deps:
            return
        state = self._refresh_dependency_state(log_result=True)
        missing_labels = state["missing_labels"] if isinstance(state, dict) else []
        if not isinstance(missing_labels, list) or not missing_labels:
            messagebox.showinfo("Dependencies", "All dependencies are installed.")
            return
        prompt = "Missing dependencies:\n\n" + "\n".join(f"- {item}" for item in missing_labels)
        if self._missing_install_packages:
            prompt += "\n\nInstall missing dependencies now?"
            if messagebox.askyesno("Dependencies", prompt):
                self.install_missing_dependencies()
        else:
            messagebox.showwarning("Dependencies", prompt)

    def install_missing_dependencies(self) -> None:
        if (self.worker and self.worker.is_alive()) or self._installing_deps:
            return
        state = self._refresh_dependency_state(log_result=False)
        missing_packages = state["missing_packages"] if isinstance(state, dict) else []
        if not isinstance(missing_packages, list) or not missing_packages:
            messagebox.showinfo("Dependencies", "No missing dependencies to install.")
            return

        self._installing_deps = True
        self._set_status("Installing dependencies...")
        self._log("Installing missing dependencies: " + ", ".join(missing_packages))
        self._set_running_state(bool(self.worker and self.worker.is_alive()))

        self.dep_worker = DependencyInstallWorker(
            packages=missing_packages,
            event_queue=self.event_queue,
        )
        self.dep_worker.start()
        self.root.after(UI_POLL_MS, self._pump_events)

    def start_conversion(self) -> None:
        if self._installing_deps:
            messagebox.showinfo("Dependencies", "Please wait for dependency installation to finish.")
            return
        dep_state = self._refresh_dependency_state(log_result=False)
        if isinstance(dep_state, dict) and not bool(dep_state.get("fitz_ok")):
            messagebox.showerror(
                "Missing Dependency",
                "PyMuPDF is required for conversion.\nUse 'Install Missing' in the Dependencies section.",
            )
            return
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("Busy", "Conversion is already running.")
            return

        source_raw = self.source_var.get().strip()
        output_raw = self.output_var.get().strip()
        if not source_raw:
            messagebox.showerror("Missing Input", "Please choose or drop a source folder or PDF file.")
            return

        source_path = Path(source_raw)
        if not source_path.exists():
            messagebox.showerror("Invalid Input", "Source path does not exist.")
            return
        if source_path.is_file() and source_path.suffix.lower() != ".pdf":
            messagebox.showerror("Invalid Input", "When selecting a file, it must be a .pdf file.")
            return

        if not output_raw:
            if source_path.is_file():
                output_dir = source_path.parent / f"{source_path.stem}_images"
            else:
                output_dir = source_path.parent / f"{source_path.name}_images"
            self.output_var.set(str(output_dir))
        else:
            output_dir = Path(output_raw)

        if output_dir.resolve() == source_path.resolve():
            messagebox.showerror("Invalid Output", "Output folder must be different from input folder.")
            return

        if output_dir.exists():
            if any(output_dir.iterdir()):
                overwrite = messagebox.askyesno(
                    "Output Not Empty",
                    "Output folder is not empty. Continue and merge/overwrite generated image files?",
                )
                if not overwrite:
                    return
        else:
            output_dir.mkdir(parents=True, exist_ok=True)

        self.progress_var.set(0.0)
        self.percent_var.set("0.0%")
        self.current_file_var.set("Scanning source folder...")
        self.total_var.set("0")
        self.converted_var.set("0")
        self.failed_var.set("0")
        self.images_var.set("0")
        self.stop_event.clear()
        self.last_output_dir = output_dir
        self._set_running_state(True)
        self._start_elapsed_timer()
        self._set_status("Running conversion...")
        self._log(f"Starting conversion: source={source_path} output={output_dir}")
        self._log(
            f"Settings: dpi={self.dpi_var.get()}, format={self.format_var.get()}, jpg_quality={self.jpg_quality_var.get()}"
        )

        self.worker = PDFToImagesWorker(
            source_path=source_path,
            output_dir=output_dir,
            dpi=max(72, int(self.dpi_var.get())),
            image_format=self.format_var.get(),
            jpg_quality=max(30, min(100, int(self.jpg_quality_var.get()))),
            stop_event=self.stop_event,
            event_queue=self.event_queue,
        )
        self.worker.start()
        self.root.after(UI_POLL_MS, self._pump_events)

    def cancel_conversion(self) -> None:
        if self.worker and self.worker.is_alive():
            self.stop_event.set()
            self._set_status("Cancelling...")
            self._log("Cancel requested by user.")

    def _pump_events(self) -> None:
        processed = 0
        while processed < MAX_EVENTS_PER_TICK:
            try:
                event, payload = self.event_queue.get_nowait()
            except queue.Empty:
                break
            self._handle_event(event, payload)
            processed += 1

        if (
            (self.worker and self.worker.is_alive())
            or (self.dep_worker and self.dep_worker.is_alive())
            or (not self.event_queue.empty())
        ):
            self.root.after(UI_POLL_MS, self._pump_events)
        else:
            self._set_running_state(False)
            self._stop_elapsed_timer()

    def _handle_event(self, event: str, payload: object) -> None:
        if event == "dep_install_started":
            self._set_status("Installing dependencies...")
            return

        if event == "dep_install_step":
            data = payload if isinstance(payload, dict) else {}
            package = str(data.get("package", "dependency"))
            self._set_status(f"Installing {package}...")
            self._log(f"Installing: {package}")
            return

        if event == "dep_install_log":
            data = payload if isinstance(payload, dict) else {}
            package = str(data.get("package", "dependency"))
            text = str(data.get("text", "")).strip()
            if text:
                self._log(f"{package} installer output:")
                for line in text.splitlines():
                    self._log("  " + line)
            return

        if event == "dep_install_done":
            self._installing_deps = False
            results = payload if isinstance(payload, list) else []
            fail_count = 0
            for result in results:
                if isinstance(result, DependencyInstallResult):
                    self._log(result.details)
                    if not result.success:
                        fail_count += 1

            state = self._refresh_dependency_state(log_result=False)
            missing_labels = state.get("missing_labels", []) if isinstance(state, dict) else []

            if fail_count > 0:
                self._set_status("Dependency install completed with errors.")
                messagebox.showwarning(
                    "Dependencies",
                    "Some dependencies could not be installed automatically.\n"
                    "Review the log and install remaining items manually if needed.",
                )
            else:
                self._set_status("Dependency installation complete.")
                if isinstance(missing_labels, list) and missing_labels:
                    messagebox.showinfo(
                        "Dependencies",
                        "Installation finished, but some optional features may still need a restart.",
                    )
                else:
                    messagebox.showinfo("Dependencies", "All dependencies are now installed.")

            if self._has_tkdnd_runtime and not DND_AVAILABLE:
                self._log("tkinterdnd2 installed. Restart app to enable drag-and-drop support.")
            self._set_running_state(bool(self.worker and self.worker.is_alive()))
            return

        if event == "scan_progress":
            data = payload if isinstance(payload, dict) else {}
            entries = int(data.get("entries_seen", 0))
            self._set_status(f"Scanning source folder... entries checked: {entries:,}")
            self.current_file_var.set(f"Scanning... {entries:,} entries checked")
            if self.verbose_log_var.get():
                self._log(f"Scan progress: {entries:,} entries checked")
            return

        if event == "scan_complete":
            data = payload if isinstance(payload, dict) else {}
            total = int(data.get("total_pdfs", 0))
            self.total_var.set(f"{total:,}")
            self._log(f"Scan complete: {total} PDF file(s) found.")
            if total == 0:
                self._set_status("No PDFs found in selected folder.")
                self.current_file_var.set("No PDFs found.")
            return

        if event == "file_start":
            data = payload if isinstance(payload, dict) else {}
            idx = data.get("index", "?")
            total = data.get("total", "?")
            pdf_path = data.get("pdf_path", "")
            self._set_status(f"Converting {idx}/{total}: {pdf_path}")
            self.current_file_var.set(str(pdf_path))
            try:
                idx_int = int(idx)
                total_int = int(total)
            except Exception:
                idx_int = -1
                total_int = -1
            if self.verbose_log_var.get() or idx_int in {1, total_int} or (
                idx_int > 0 and idx_int % FILE_LOG_EVERY_N == 0
            ):
                self._log(f"Converting {idx}/{total}: {pdf_path}")
            return

        if event == "pdf_pages":
            data = payload if isinstance(payload, dict) else {}
            if self.verbose_log_var.get():
                self._log(f"Pages: {data.get('pages', '?')} in {data.get('pdf_path', '')}")
            return

        if event == "progress":
            data = payload if isinstance(payload, dict) else {}
            percent = float(data.get("percent", 0.0))
            self.progress_var.set(percent)
            self.percent_var.set(f"{percent:.1f}%")
            self.converted_var.set(f"{int(data.get('converted_pdfs', 0)):,}")
            self.failed_var.set(f"{int(data.get('failed_pdfs', 0)):,}")
            self.images_var.set(f"{int(data.get('images_written', 0)):,}")
            self._set_status(
                f"{data.get('index', '?')}/{data.get('total', '?')} PDFs | "
                f"Converted: {data.get('converted_pdfs', 0)} | "
                f"Failed: {data.get('failed_pdfs', 0)} | "
                f"Images: {data.get('images_written', 0)}"
            )
            return

        if event == "file_error":
            data = payload if isinstance(payload, dict) else {}
            self._log(f"ERROR {data.get('pdf_path', '')}: {data.get('error', 'unknown')}")
            return

        if event == "cancelled":
            self._set_status("Conversion cancelled.")
            self.current_file_var.set("Cancelled by user.")
            self._log("Conversion cancelled.")
            return

        if event == "fatal_error":
            err = str(payload)
            self._set_status("Fatal error.")
            self.current_file_var.set("Fatal error encountered.")
            self._log(f"FATAL ERROR: {err}")
            messagebox.showerror("Fatal Error", err)
            return

        if event == "done":
            if isinstance(payload, ConversionSummary):
                summary = payload
                self.progress_var.set(100.0)
                self.percent_var.set("100.0%")
                self.total_var.set(f"{summary.total_pdfs:,}")
                self.converted_var.set(f"{summary.converted_pdfs:,}")
                self.failed_var.set(f"{summary.failed_pdfs:,}")
                self.images_var.set(f"{summary.images_written:,}")
                self._set_status(
                    f"Done. PDFs: {summary.total_pdfs}, converted: {summary.converted_pdfs}, "
                    f"failed: {summary.failed_pdfs}, images: {summary.images_written}"
                )
                self.current_file_var.set("Completed successfully.")
                self._log(
                    "Done "
                    f"(duration={summary.duration_s:.1f}s): "
                    f"total_pdfs={summary.total_pdfs}, converted={summary.converted_pdfs}, "
                    f"failed={summary.failed_pdfs}, images={summary.images_written}, "
                    f"output={summary.output_dir}"
                )
                self.last_output_dir = summary.output_dir
                self.open_btn.configure(state="normal")
                self.zip_btn.configure(state="normal")
            return

    def open_output(self) -> None:
        if not self.last_output_dir or not self.last_output_dir.exists():
            messagebox.showinfo("No Output", "No output folder is available yet.")
            return
        os.startfile(str(self.last_output_dir))  # type: ignore[attr-defined]

    def create_zip(self) -> None:
        if not self.last_output_dir or not self.last_output_dir.exists():
            messagebox.showinfo("No Output", "Run a conversion first.")
            return

        default_zip_name = f"{self.last_output_dir.name}.zip"
        save_path = filedialog.asksaveasfilename(
            title="Save ZIP",
            defaultextension=".zip",
            filetypes=[("ZIP files", "*.zip")],
            initialfile=default_zip_name,
        )
        if not save_path:
            return

        try:
            zip_path = Path(save_path)
            base_name = str(zip_path.with_suffix(""))
            created = Path(shutil.make_archive(base_name, "zip", root_dir=str(self.last_output_dir)))
            if created.resolve() != zip_path.resolve():
                if zip_path.exists():
                    zip_path.unlink()
                created.replace(zip_path)
            self._log(f"ZIP created: {zip_path}")
            self._set_status(f"ZIP ready: {zip_path}")
            messagebox.showinfo("ZIP Created", f"Saved:\n{zip_path}")
        except Exception as exc:
            self._log(f"ZIP ERROR: {exc}")
            messagebox.showerror("ZIP Error", str(exc))


def build_root() -> tk.Tk:
    if DND_AVAILABLE:
        return TkinterDnD.Tk()  # type: ignore[no-any-return]
    return tk.Tk()


def main() -> None:
    root = build_root()
    app = PDFToImagesApp(root)
    _ = app
    root.mainloop()


if __name__ == "__main__":
    main()
