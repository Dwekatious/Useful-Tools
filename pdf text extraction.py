from __future__ import annotations

import argparse
import csv
import importlib
import os
import queue
import subprocess
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None  # type: ignore[assignment]
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

UI_POLL_MS = 50
MAX_EVENTS_PER_TICK = 250
SCAN_PROGRESS_EMIT_EVERY = 2000
MAX_LOG_LINES = 3000

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD  # type: ignore

    DND_AVAILABLE = True
except Exception:
    DND_AVAILABLE = False


@dataclass
class FileResult:
    pdf_path: Path
    txt_path: Path
    success: bool
    pages: int
    images: int
    chars: int
    error: str = ""


@dataclass
class RunSummary:
    source: Path
    output_dir: Path
    total_pdfs: int
    succeeded: int
    failed: int
    pages: int
    images: int
    chars: int
    duration_s: float
    report_path: Optional[Path]


@dataclass(frozen=True)
class DependencyInstallResult:
    package: str
    success: bool
    details: str


def build_root() -> tk.Tk:
    if DND_AVAILABLE:
        return TkinterDnD.Tk()  # type: ignore[no-any-return]
    return tk.Tk()


def normalize_text(text: str) -> str:
    return (
        text.replace("\u00a0", " ")
        .replace("\u200b", "")
        .replace("\xad", "")
        .replace("\ufeff", "")
    )


def render_line_from_spans(spans: list[dict]) -> str:
    spans_sorted = sorted(spans, key=lambda s: (s.get("bbox", [0, 0, 0, 0])[0],))
    pieces: list[str] = []
    last_x1: Optional[float] = None
    avg_char_width = 6.0

    for span in spans_sorted:
        raw = span.get("text", "")
        if not raw:
            continue
        text = normalize_text(raw)
        if not text:
            continue
        bbox = span.get("bbox", [0.0, 0.0, 0.0, 0.0])
        x0 = float(bbox[0])
        x1 = float(bbox[2])

        width = max(0.0, x1 - x0)
        if len(text) > 0 and width > 0.0:
            char_w = width / max(len(text), 1)
            if 0.5 <= char_w <= 40.0:
                avg_char_width = (avg_char_width * 0.7) + (char_w * 0.3)

        if last_x1 is not None:
            gap = max(0.0, x0 - last_x1)
            if gap > (avg_char_width * 0.5):
                spaces = int(round(gap / max(avg_char_width, 1.0)))
                if spaces > 0:
                    pieces.append(" " * min(spaces, 48))

        pieces.append(text)
        last_x1 = x1

    return "".join(pieces).rstrip()


def render_text_block(block: dict) -> str:
    lines = block.get("lines", [])
    if not lines:
        return ""

    lines_sorted = sorted(lines, key=lambda l: (l.get("bbox", [0, 0, 0, 0])[1], l.get("bbox", [0, 0, 0, 0])[0]))
    out_lines: list[str] = []
    prev_bottom: Optional[float] = None
    prev_height = 12.0

    for line in lines_sorted:
        lb = line.get("bbox", [0.0, 0.0, 0.0, 0.0])
        y0 = float(lb[1])
        y1 = float(lb[3])
        height = max(1.0, y1 - y0)

        if prev_bottom is not None:
            gap = y0 - prev_bottom
            if gap > max(prev_height, height) * 0.85:
                out_lines.append("")

        rendered = render_line_from_spans(line.get("spans", []))
        if rendered:
            out_lines.append(rendered)
        else:
            out_lines.append("")

        prev_bottom = y1
        prev_height = height

    while out_lines and not out_lines[-1].strip():
        out_lines.pop()

    return "\n".join(out_lines).rstrip()


def extract_page_text(
    page: fitz.Page,
    image_counter_start: int,
    include_image_markers: bool,
) -> tuple[str, int, int]:
    # dict + sorted blocks gives reliable reading order for most digital PDFs.
    data = page.get_text("dict", sort=True)
    blocks = data.get("blocks", [])
    blocks_sorted = sorted(
        blocks,
        key=lambda b: (
            float(b.get("bbox", [0.0, 0.0, 0.0, 0.0])[1]),
            float(b.get("bbox", [0.0, 0.0, 0.0, 0.0])[0]),
        ),
    )

    out: list[str] = []
    chars = 0
    image_counter = image_counter_start
    prev_bottom: Optional[float] = None
    prev_height = 14.0

    for block in blocks_sorted:
        btype = int(block.get("type", -1))
        bb = block.get("bbox", [0.0, 0.0, 0.0, 0.0])
        y0 = float(bb[1])
        y1 = float(bb[3])
        height = max(1.0, y1 - y0)

        if prev_bottom is not None:
            gap = y0 - prev_bottom
            if gap > max(prev_height, height) * 0.95:
                if out and out[-1] != "":
                    out.append("")

        if btype == 0:
            rendered = render_text_block(block)
            if rendered:
                out.append(rendered)
                chars += len(rendered)
        elif btype == 1 and include_image_markers:
            image_counter += 1
            out.append(f"(img{image_counter})")

        prev_bottom = y1
        prev_height = height

    while out and out[-1] == "":
        out.pop()

    return "\n".join(out), image_counter, chars


def iter_pdf_files(source: Path, recursive: bool, stop_event: Optional[threading.Event] = None) -> Iterable[Path]:
    if source.is_file():
        if source.suffix.lower() == ".pdf":
            yield source
        return

    if recursive:
        stack = [source]
        while stack:
            if stop_event and stop_event.is_set():
                return
            current = stack.pop()
            try:
                with os.scandir(current) as it:
                    for entry in it:
                        if stop_event and stop_event.is_set():
                            return
                        try:
                            if entry.is_dir(follow_symlinks=False):
                                stack.append(Path(entry.path))
                                continue
                            if entry.is_file(follow_symlinks=False) and entry.name.lower().endswith(".pdf"):
                                yield Path(entry.path)
                        except OSError:
                            continue
            except OSError:
                continue
        return

    try:
        with os.scandir(source) as it:
            for entry in it:
                if stop_event and stop_event.is_set():
                    return
                try:
                    if entry.is_file(follow_symlinks=False) and entry.name.lower().endswith(".pdf"):
                        yield Path(entry.path)
                except OSError:
                    continue
    except OSError:
        return


def resolve_output_path(source: Path, pdf_path: Path, output_dir: Path) -> Path:
    if source.is_file():
        return output_dir / f"{source.stem}.txt"

    rel = pdf_path.relative_to(source)
    return (output_dir / rel).with_suffix(".txt")


def extract_pdf_to_txt(
    pdf_path: Path,
    txt_path: Path,
    include_image_markers: bool,
    include_page_headers: bool,
    overwrite_existing: bool,
    stop_event: Optional[threading.Event] = None,
) -> FileResult:
    if not overwrite_existing and txt_path.exists():
        return FileResult(
            pdf_path=pdf_path,
            txt_path=txt_path,
            success=True,
            pages=0,
            images=0,
            chars=0,
            error="skipped_existing",
        )

    txt_path.parent.mkdir(parents=True, exist_ok=True)

    pages = 0
    image_counter = 0
    total_chars = 0

    try:
        fitz.TOOLS.set_small_glyph_heights(True)
        with fitz.open(pdf_path) as doc, txt_path.open("w", encoding="utf-8", newline="\n") as out:
            for page_index in range(len(doc)):
                if stop_event and stop_event.is_set():
                    raise RuntimeError("cancelled")

                page = doc[page_index]
                page_text, image_counter, chars = extract_page_text(
                    page=page,
                    image_counter_start=image_counter,
                    include_image_markers=include_image_markers,
                )
                pages += 1
                total_chars += chars

                if include_page_headers:
                    out.write(f"===== Page {page_index + 1} =====\n")

                if page_text.strip():
                    out.write(page_text.rstrip())
                out.write("\n\n")

        return FileResult(
            pdf_path=pdf_path,
            txt_path=txt_path,
            success=True,
            pages=pages,
            images=image_counter,
            chars=total_chars,
        )
    except Exception as exc:
        return FileResult(
            pdf_path=pdf_path,
            txt_path=txt_path,
            success=False,
            pages=pages,
            images=image_counter,
            chars=total_chars,
            error=str(exc),
        )


class ExtractionWorker(threading.Thread):
    def __init__(
        self,
        source: Path,
        output_dir: Path,
        recursive: bool,
        max_workers: int,
        include_image_markers: bool,
        include_page_headers: bool,
        overwrite_existing: bool,
        stop_event: threading.Event,
        event_queue: "queue.Queue[tuple[str, object]]",
    ) -> None:
        super().__init__(daemon=True)
        self.source = source
        self.output_dir = output_dir
        self.recursive = recursive
        self.max_workers = max(1, max_workers)
        self.include_image_markers = include_image_markers
        self.include_page_headers = include_page_headers
        self.overwrite_existing = overwrite_existing
        self.stop_event = stop_event
        self.event_queue = event_queue

    def _emit(self, event: str, payload: object) -> None:
        self.event_queue.put((event, payload))

    def _scan(self) -> list[Path]:
        pdfs: list[Path] = []
        seen = 0
        for path in iter_pdf_files(self.source, self.recursive, self.stop_event):
            if self.stop_event.is_set():
                return pdfs
            pdfs.append(path)
            seen += 1
            if seen % SCAN_PROGRESS_EMIT_EVERY == 0:
                self._emit("scan_progress", {"seen": seen})
        pdfs.sort()
        return pdfs

    def _write_report(self, rows: list[FileResult]) -> Optional[Path]:
        if not rows:
            return None
        report = self.output_dir / "_extraction_report.csv"
        try:
            with report.open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["pdf_path", "txt_path", "success", "pages", "images", "chars", "error"]
                )
                for row in rows:
                    writer.writerow(
                        [
                            str(row.pdf_path),
                            str(row.txt_path),
                            str(row.success),
                            row.pages,
                            row.images,
                            row.chars,
                            row.error,
                        ]
                    )
            return report
        except Exception:
            return None

    def run(self) -> None:
        started = time.time()
        all_rows: list[FileResult] = []
        total_pages = 0
        total_images = 0
        total_chars = 0
        succeeded = 0
        failed = 0

        try:
            pdf_files = self._scan()
            if self.stop_event.is_set():
                self._emit("cancelled", {"processed": 0, "total": 0})
                return

            total = len(pdf_files)
            self._emit("scan_complete", {"total": total})
            if total == 0:
                summary = RunSummary(
                    source=self.source,
                    output_dir=self.output_dir,
                    total_pdfs=0,
                    succeeded=0,
                    failed=0,
                    pages=0,
                    images=0,
                    chars=0,
                    duration_s=time.time() - started,
                    report_path=None,
                )
                self._emit("done", summary)
                return

            self.output_dir.mkdir(parents=True, exist_ok=True)
            targets = [(pdf, resolve_output_path(self.source, pdf, self.output_dir)) for pdf in pdf_files]
            total_targets = len(targets)

            self._emit(
                "status",
                {
                    "message": f"Running extraction with {self.max_workers} worker(s)...",
                },
            )

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                pending: dict[Future[FileResult], Path] = {}
                idx = 0

                def submit_next() -> bool:
                    nonlocal idx
                    if self.stop_event.is_set():
                        return False
                    if idx >= total_targets:
                        return False
                    pdf_path, txt_path = targets[idx]
                    idx += 1
                    self._emit(
                        "file_start",
                        {"index": idx, "total": total_targets, "pdf_path": str(pdf_path)},
                    )
                    fut = executor.submit(
                        extract_pdf_to_txt,
                        pdf_path,
                        txt_path,
                        self.include_image_markers,
                        self.include_page_headers,
                        self.overwrite_existing,
                        self.stop_event,
                    )
                    pending[fut] = pdf_path
                    return True

                for _ in range(min(self.max_workers * 2, total_targets)):
                    if not submit_next():
                        break

                processed = 0
                while pending:
                    if self.stop_event.is_set():
                        for fut in list(pending.keys()):
                            fut.cancel()
                        self._emit("cancelled", {"processed": processed, "total": total_targets})
                        return

                    done, _ = wait(pending.keys(), timeout=0.2, return_when=FIRST_COMPLETED)
                    if not done:
                        continue

                    for fut in done:
                        pdf_path = pending.pop(fut)
                        try:
                            result = fut.result()
                        except Exception as exc:
                            result = FileResult(
                                pdf_path=pdf_path,
                                txt_path=resolve_output_path(self.source, pdf_path, self.output_dir),
                                success=False,
                                pages=0,
                                images=0,
                                chars=0,
                                error=str(exc),
                            )

                        all_rows.append(result)
                        processed += 1
                        total_pages += result.pages
                        total_images += result.images
                        total_chars += result.chars

                        if result.success:
                            succeeded += 1
                            if result.error == "skipped_existing":
                                self._emit(
                                    "file_skipped",
                                    {"pdf_path": str(result.pdf_path), "txt_path": str(result.txt_path)},
                                )
                            else:
                                self._emit(
                                    "file_done",
                                    {
                                        "pdf_path": str(result.pdf_path),
                                        "txt_path": str(result.txt_path),
                                        "pages": result.pages,
                                        "images": result.images,
                                    },
                                )
                        else:
                            failed += 1
                            self._emit(
                                "file_error",
                                {"pdf_path": str(result.pdf_path), "error": result.error},
                            )

                        percent = (processed / total_targets) * 100.0
                        self._emit(
                            "progress",
                            {
                                "processed": processed,
                                "total": total_targets,
                                "percent": percent,
                                "succeeded": succeeded,
                                "failed": failed,
                                "pages": total_pages,
                                "images": total_images,
                                "chars": total_chars,
                            },
                        )

                        submit_next()

            report_path = self._write_report(all_rows)
            summary = RunSummary(
                source=self.source,
                output_dir=self.output_dir,
                total_pdfs=total_targets,
                succeeded=succeeded,
                failed=failed,
                pages=total_pages,
                images=total_images,
                chars=total_chars,
                duration_s=time.time() - started,
                report_path=report_path,
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


class PDFTextExtractorApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("PDF Text Extractor Pro")
        self.root.geometry("1120x760")
        self.root.minsize(980, 680)

        self.source_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.recursive_var = tk.BooleanVar(value=True)
        self.include_image_markers_var = tk.BooleanVar(value=True)
        self.include_page_headers_var = tk.BooleanVar(value=True)
        self.overwrite_existing_var = tk.BooleanVar(value=True)
        self.workers_var = tk.IntVar(value=max(1, min(8, (os.cpu_count() or 4))))
        self.verbose_log_var = tk.BooleanVar(value=False)

        self.progress_var = tk.DoubleVar(value=0.0)
        self.percent_var = tk.StringVar(value="0.0%")
        self.status_var = tk.StringVar(value="Ready.")
        self.current_file_var = tk.StringVar(value="No file in progress.")
        self.elapsed_var = tk.StringVar(value="00:00:00")

        self.total_var = tk.StringVar(value="0")
        self.ok_var = tk.StringVar(value="0")
        self.fail_var = tk.StringVar(value="0")
        self.page_var = tk.StringVar(value="0")
        self.image_var = tk.StringVar(value="0")
        self.char_var = tk.StringVar(value="0")

        self.event_queue: "queue.Queue[tuple[str, object]]" = queue.Queue()
        self.stop_event = threading.Event()
        self.worker: Optional[ExtractionWorker] = None
        self.dep_worker: Optional[DependencyInstallWorker] = None
        self.last_output_dir: Optional[Path] = None
        self._run_started_at: Optional[float] = None
        self._elapsed_after_id: Optional[str] = None
        self._log_write_count = 0
        self._installing_deps = False
        self._has_fitz = fitz is not None
        self._has_tkdnd_runtime = DND_AVAILABLE
        self._missing_install_packages: list[str] = []
        self.dep_status_var = tk.StringVar(value="Dependencies: checking...")
        self.dep_hint_var = tk.StringVar(value="")

        self.start_btn: ttk.Button
        self.cancel_btn: ttk.Button
        self.open_btn: ttk.Button
        self.check_deps_btn: ttk.Button
        self.install_deps_btn: ttk.Button
        self.progress: ttk.Progressbar
        self.log_text: ScrolledText
        self.drop_area: tk.Label

        self._apply_theme()
        self._build_ui()
        self._refresh_dependency_state(log_result=False)
        self._set_running(False)

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
        style.configure("Muted.TLabel", background="#e9eff6", foreground="#516279", font=("Segoe UI", 10))
        style.configure("CardLabel.TLabel", background="#ffffff", foreground="#1f2b3a", font=("Segoe UI", 10))
        style.configure("SmallMuted.TLabel", background="#ffffff", foreground="#647386", font=("Segoe UI", 9))
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
            foreground="#1f2b3a",
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
        style.configure("Accent.Horizontal.TProgressbar", troughcolor="#dbe7f5", background="#1e88e5")
        style.configure("Dark.TFrame", background="#0f1f3a")

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=14, style="App.TFrame")
        outer.pack(fill="both", expand=True)

        header = ttk.Frame(outer, style="Dark.TFrame", padding=(16, 14))
        header.pack(fill="x", pady=(0, 12))
        ttk.Label(header, text="PDF to Text Extractor", style="HeaderTitle.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="Extract text from single PDFs or datasets with clean mirrored output and live progress.",
            style="HeaderSub.TLabel",
        ).pack(anchor="w", pady=(2, 0))

        content = ttk.Panedwindow(outer, orient="horizontal")
        content.pack(fill="both", expand=True)

        left = ttk.Frame(content, style="App.TFrame")
        right = ttk.Frame(content, style="App.TFrame")
        content.add(left, weight=46)
        content.add(right, weight=54)

        source_card = ttk.LabelFrame(left, text="Input (Folder or PDF)", style="Card.TLabelframe", padding=12)
        source_card.pack(fill="x")
        row = ttk.Frame(source_card, style="App.TFrame")
        row.pack(fill="x")
        entry = ttk.Entry(row, textvariable=self.source_var)
        entry.pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="Browse Folder...", command=self.pick_source_folder, style="Neutral.TButton").pack(
            side="left", padx=(8, 0)
        )
        ttk.Button(row, text="Browse PDF...", command=self.pick_source_file, style="Neutral.TButton").pack(
            side="left", padx=(8, 0)
        )

        drop_hint = (
            "Drop a folder or PDF file here"
            if DND_AVAILABLE
            else "Drag and drop unavailable (install tkinterdnd2). Use Browse buttons."
        )
        self.drop_area = tk.Label(
            source_card,
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
        out_row = ttk.Frame(output_card, style="App.TFrame")
        out_row.pack(fill="x")
        out_entry = ttk.Entry(out_row, textvariable=self.output_var)
        out_entry.pack(side="left", fill="x", expand=True)
        ttk.Button(out_row, text="Browse...", command=self.pick_output, style="Neutral.TButton").pack(
            side="left", padx=(8, 0)
        )

        settings_card = ttk.LabelFrame(left, text="Settings", style="Card.TLabelframe", padding=12)
        settings_card.pack(fill="x", pady=(10, 0))
        settings_card.grid_columnconfigure(1, weight=1)
        settings_card.grid_columnconfigure(3, weight=1)

        ttk.Label(settings_card, text="Workers", style="CardLabel.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(settings_card, from_=1, to=32, increment=1, textvariable=self.workers_var, width=8).grid(
            row=0, column=1, sticky="w", padx=(8, 16)
        )
        ttk.Checkbutton(settings_card, text="Recursive folder scan", variable=self.recursive_var).grid(
            row=0, column=2, columnspan=2, sticky="w"
        )

        ttk.Checkbutton(
            settings_card,
            text="Insert image markers like (img1)",
            variable=self.include_image_markers_var,
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(10, 0))
        ttk.Checkbutton(
            settings_card,
            text="Add page headers",
            variable=self.include_page_headers_var,
        ).grid(row=1, column=2, columnspan=2, sticky="w", pady=(10, 0))
        ttk.Checkbutton(
            settings_card,
            text="Overwrite existing .txt files",
            variable=self.overwrite_existing_var,
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Checkbutton(
            settings_card,
            text="Verbose log",
            variable=self.verbose_log_var,
        ).grid(row=2, column=2, columnspan=2, sticky="w", pady=(8, 0))

        action_card = ttk.LabelFrame(left, text="Actions", style="Card.TLabelframe", padding=12)
        action_card.pack(fill="x", pady=(10, 0))
        self.start_btn = ttk.Button(action_card, text="Start Extraction", command=self.start_run, style="Accent.TButton")
        self.start_btn.pack(fill="x")
        self.cancel_btn = ttk.Button(action_card, text="Cancel", command=self.cancel_run, state="disabled", style="Danger.TButton")
        self.cancel_btn.pack(fill="x", pady=(8, 0))
        self.open_btn = ttk.Button(
            action_card,
            text="Open Output Folder",
            command=self.open_output,
            state="disabled",
            style="Success.TButton",
        )
        self.open_btn.pack(fill="x", pady=(8, 0))

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
        stats_grid.grid_columnconfigure((0, 1, 2), weight=1)
        self._stat_tile(stats_grid, "PDFs", self.total_var, 0, 0)
        self._stat_tile(stats_grid, "Succeeded", self.ok_var, 0, 1)
        self._stat_tile(stats_grid, "Failed", self.fail_var, 0, 2)
        self._stat_tile(stats_grid, "Pages", self.page_var, 1, 0)
        self._stat_tile(stats_grid, "Images", self.image_var, 1, 1)
        self._stat_tile(stats_grid, "Chars", self.char_var, 1, 2)

        progress_card = ttk.LabelFrame(right, text="Progress", style="Card.TLabelframe", padding=12)
        progress_card.pack(fill="x", pady=(10, 0))
        top = ttk.Frame(progress_card, style="App.TFrame")
        top.pack(fill="x")
        ttk.Label(top, text="Overall", style="CardLabel.TLabel").pack(side="left")
        ttk.Label(top, textvariable=self.percent_var, style="CardLabel.TLabel").pack(side="right")
        self.progress = ttk.Progressbar(
            progress_card,
            variable=self.progress_var,
            maximum=100.0,
            mode="determinate",
            style="Accent.Horizontal.TProgressbar",
        )
        self.progress.pack(fill="x", pady=(8, 8))
        ttk.Label(progress_card, text="Current File", style="SmallMuted.TLabel").pack(anchor="w")
        ttk.Label(progress_card, textvariable=self.current_file_var, style="CardLabel.TLabel", wraplength=520).pack(anchor="w")

        log_card = ttk.LabelFrame(right, text="Log", style="Card.TLabelframe", padding=10)
        log_card.pack(fill="both", expand=True, pady=(10, 0))
        log_top = ttk.Frame(log_card, style="App.TFrame")
        log_top.pack(fill="x", pady=(0, 6))
        ttk.Button(log_top, text="Clear Log", command=self.clear_log, style="Neutral.TButton").pack(side="left")
        ttk.Label(log_top, text="Live extraction events", style="SmallMuted.TLabel").pack(side="right")

        self.log_text = ScrolledText(
            log_card,
            height=14,
            wrap="word",
            font=("Consolas", 10),
            bg="#0f1722",
            fg="#d5e5ff",
            insertbackground="#d5e5ff",
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
            bg="#dce7f4",
            fg="#223246",
            anchor="w",
            padx=10,
            font=("Segoe UI", 9),
        ).pack(side="left", fill="x", expand=True)
        tk.Label(
            status_bar,
            textvariable=self.elapsed_var,
            bg="#dce7f4",
            fg="#223246",
            anchor="e",
            padx=10,
            font=("Segoe UI", 9, "bold"),
        ).pack(side="right")

    def _stat_tile(self, parent: ttk.Frame, title: str, var: tk.StringVar, row: int, col: int) -> None:
        tile = tk.Frame(parent, bg="#f4f8ff", bd=1, relief="solid", highlightthickness=0)
        tile.grid(row=row, column=col, sticky="nsew", padx=4, pady=4)
        tk.Label(tile, text=title, bg="#f4f8ff", fg="#4f6074", font=("Segoe UI", 9)).pack(anchor="w", padx=10, pady=(8, 2))
        tk.Label(tile, textvariable=var, bg="#f4f8ff", fg="#17263a", font=("Segoe UI Semibold", 16)).pack(
            anchor="w", padx=10, pady=(0, 8)
        )

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

    def _set_running(self, running: bool) -> None:
        has_active_extraction = running
        busy = has_active_extraction or self._installing_deps
        self.start_btn.configure(state="disabled" if (busy or not self._has_fitz) else "normal")
        self.cancel_btn.configure(state="normal" if has_active_extraction else "disabled")
        self.open_btn.configure(
            state="disabled" if busy else ("normal" if self.last_output_dir else "disabled")
        )
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
                self.drop_area.configure(text="Drop a folder or PDF file here")
            elif tkdnd_ok:
                self.dep_hint_var.set("tkinterdnd2 is installed. Restart app to activate drag and drop.")
            else:
                self.dep_hint_var.set("Core dependencies are ready.")
            if log_result:
                self._log("Dependency check: all dependencies are installed.")
        else:
            self.dep_status_var.set("Missing: " + ", ".join(missing_labels))
            hint_parts: list[str] = []
            if not fitz_ok:
                hint_parts.append("PyMuPDF is required to run extraction.")
            if not tkdnd_ok:
                hint_parts.append("Install tkinterdnd2 to enable drag and drop.")
            elif tkdnd_ok and not DND_AVAILABLE:
                hint_parts.append("tkinterdnd2 installed. Restart app to enable drag and drop.")
            self.dep_hint_var.set(" ".join(hint_parts))
            if log_result:
                self._log("Dependency check: " + ", ".join(missing_labels))

        self._set_running(bool(self.worker and self.worker.is_alive()))
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
        self._set_running(bool(self.worker and self.worker.is_alive()))

        self.dep_worker = DependencyInstallWorker(
            packages=missing_packages,
            event_queue=self.event_queue,
        )
        self.dep_worker.start()
        self.root.after(UI_POLL_MS, self._pump_events)

    def _set_status(self, text: str) -> None:
        self.status_var.set(text)

    def _log(self, message: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{stamp}] {message}\n"
        self.log_text.configure(state="normal")
        self.log_text.insert("end", line)
        self._log_write_count += 1
        if self._log_write_count % 25 == 0:
            total_lines = int(float(self.log_text.index("end-1c").split(".")[0]))
            if total_lines > MAX_LOG_LINES:
                trim_to = total_lines - MAX_LOG_LINES
                self.log_text.delete("1.0", f"{trim_to}.0")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def clear_log(self) -> None:
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")
        self._log_write_count = 0

    def pick_source_folder(self) -> None:
        path = filedialog.askdirectory(title="Select Source Folder")
        if not path:
            return
        src = Path(path)
        self.source_var.set(str(src))
        self._suggest_output(src)
        self._log(f"Source folder selected: {src}")

    def pick_source_file(self) -> None:
        path = filedialog.askopenfilename(title="Select PDF File", filetypes=[("PDF files", "*.pdf")])
        if not path:
            return
        src = Path(path)
        self.source_var.set(str(src))
        self._suggest_output(src)
        self._log(f"Source file selected: {src}")

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
            suggestion = source.parent / f"{source.stem}_text"
        else:
            suggestion = source.parent / f"{source.name}_text"
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
            self._log(f"Drop ignored (not a PDF): {candidate}")
            return
        self.source_var.set(str(path))
        self._suggest_output(path)
        self._log(f"Dropped: {path}")

    def start_run(self) -> None:
        if self._installing_deps:
            messagebox.showinfo("Dependencies", "Please wait for dependency installation to finish.")
            return
        dep_state = self._refresh_dependency_state(log_result=False)
        if isinstance(dep_state, dict) and not bool(dep_state.get("fitz_ok")):
            messagebox.showerror(
                "Missing Dependency",
                "PyMuPDF is required for extraction.\nUse 'Install Missing' in the Dependencies section.",
            )
            return
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("Busy", "Extraction is already running.")
            return

        source_raw = self.source_var.get().strip()
        output_raw = self.output_var.get().strip()

        if not source_raw:
            messagebox.showerror("Missing Input", "Please choose a source folder or source PDF file.")
            return
        source = Path(source_raw)
        if not source.exists():
            messagebox.showerror("Invalid Input", "Source path does not exist.")
            return
        if source.is_file() and source.suffix.lower() != ".pdf":
            messagebox.showerror("Invalid Input", "When selecting a file, it must be a .pdf file.")
            return

        if not output_raw:
            if source.is_file():
                output_dir = source.parent / f"{source.stem}_text"
            else:
                output_dir = source.parent / f"{source.name}_text"
            self.output_var.set(str(output_dir))
        else:
            output_dir = Path(output_raw)

        try:
            source_resolved = source.resolve()
            output_resolved = output_dir.resolve()
            if source_resolved == output_resolved:
                messagebox.showerror("Invalid Output", "Output folder must be different from source.")
                return
        except Exception:
            pass

        if output_dir.exists() and any(output_dir.iterdir()):
            ok = messagebox.askyesno(
                "Output Not Empty",
                "Output folder is not empty. Continue and merge/overwrite generated files?",
            )
            if not ok:
                return

        output_dir.mkdir(parents=True, exist_ok=True)

        workers = max(1, int(self.workers_var.get()))
        workers = min(workers, 32)

        self.progress_var.set(0.0)
        self.percent_var.set("0.0%")
        self.current_file_var.set("Scanning source...")
        self.total_var.set("0")
        self.ok_var.set("0")
        self.fail_var.set("0")
        self.page_var.set("0")
        self.image_var.set("0")
        self.char_var.set("0")
        self.stop_event.clear()
        self.last_output_dir = output_dir
        self._set_running(True)
        self._start_elapsed_timer()
        self._set_status("Running extraction...")
        self._log(f"Starting: source={source} output={output_dir}")
        self._log(
            "Settings: "
            f"workers={workers}, recursive={self.recursive_var.get()}, "
            f"image_markers={self.include_image_markers_var.get()}, "
            f"page_headers={self.include_page_headers_var.get()}, "
            f"overwrite={self.overwrite_existing_var.get()}"
        )

        self.worker = ExtractionWorker(
            source=source,
            output_dir=output_dir,
            recursive=self.recursive_var.get(),
            max_workers=workers,
            include_image_markers=self.include_image_markers_var.get(),
            include_page_headers=self.include_page_headers_var.get(),
            overwrite_existing=self.overwrite_existing_var.get(),
            stop_event=self.stop_event,
            event_queue=self.event_queue,
        )
        self.worker.start()
        self.root.after(UI_POLL_MS, self._pump_events)

    def cancel_run(self) -> None:
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
            self._set_running(False)
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
            ok_count = 0
            fail_count = 0
            for result in results:
                if isinstance(result, DependencyInstallResult):
                    if result.success:
                        ok_count += 1
                    else:
                        fail_count += 1
                    self._log(result.details)

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
            self._set_running(bool(self.worker and self.worker.is_alive()))
            if self._has_tkdnd_runtime and not DND_AVAILABLE:
                self._log("tkinterdnd2 installed. Restart app to enable drag-and-drop support.")
            return

        if event == "scan_progress":
            data = payload if isinstance(payload, dict) else {}
            seen = int(data.get("seen", 0))
            self._set_status(f"Scanning source... {seen:,} PDFs found so far")
            if self.verbose_log_var.get():
                self._log(f"Scan progress: {seen:,} PDF files found")
            return

        if event == "scan_complete":
            data = payload if isinstance(payload, dict) else {}
            total = int(data.get("total", 0))
            self.total_var.set(f"{total:,}")
            self._log(f"Scan complete: {total:,} PDF file(s) found.")
            if total == 0:
                self.current_file_var.set("No PDFs found.")
                self._set_status("No PDFs found in input.")
            return

        if event == "status":
            data = payload if isinstance(payload, dict) else {}
            msg = str(data.get("message", ""))
            if msg:
                self._set_status(msg)
            return

        if event == "file_start":
            data = payload if isinstance(payload, dict) else {}
            idx = data.get("index", "?")
            total = data.get("total", "?")
            pdf_path = str(data.get("pdf_path", ""))
            self.current_file_var.set(pdf_path)
            self._set_status(f"Processing {idx}/{total}: {pdf_path}")
            if self.verbose_log_var.get():
                self._log(f"Start {idx}/{total}: {pdf_path}")
            return

        if event == "file_done":
            data = payload if isinstance(payload, dict) else {}
            if self.verbose_log_var.get():
                self._log(
                    f"Done: {data.get('pdf_path', '')} -> {data.get('txt_path', '')} "
                    f"(pages={data.get('pages', 0)}, images={data.get('images', 0)})"
                )
            return

        if event == "file_skipped":
            data = payload if isinstance(payload, dict) else {}
            self._log(f"Skipped existing: {data.get('txt_path', '')}")
            return

        if event == "file_error":
            data = payload if isinstance(payload, dict) else {}
            self._log(f"ERROR {data.get('pdf_path', '')}: {data.get('error', 'unknown')}")
            return

        if event == "progress":
            data = payload if isinstance(payload, dict) else {}
            percent = float(data.get("percent", 0.0))
            processed = int(data.get("processed", 0))
            total = int(data.get("total", 0))
            succeeded = int(data.get("succeeded", 0))
            failed = int(data.get("failed", 0))
            pages = int(data.get("pages", 0))
            images = int(data.get("images", 0))
            chars = int(data.get("chars", 0))

            self.progress_var.set(percent)
            self.percent_var.set(f"{percent:.1f}%")
            self.ok_var.set(f"{succeeded:,}")
            self.fail_var.set(f"{failed:,}")
            self.page_var.set(f"{pages:,}")
            self.image_var.set(f"{images:,}")
            self.char_var.set(f"{chars:,}")
            self._set_status(
                f"{processed:,}/{total:,} PDFs | ok={succeeded:,} failed={failed:,} "
                f"pages={pages:,} images={images:,}"
            )
            return

        if event == "cancelled":
            data = payload if isinstance(payload, dict) else {}
            self._set_status("Extraction cancelled.")
            self.current_file_var.set("Cancelled by user.")
            self._log(
                f"Cancelled after processing {data.get('processed', 0)} of {data.get('total', 0)} PDF files."
            )
            return

        if event == "fatal_error":
            err = str(payload)
            self._set_status("Fatal error.")
            self.current_file_var.set("Fatal error encountered.")
            self._log(f"FATAL ERROR: {err}")
            messagebox.showerror("Fatal Error", err)
            return

        if event == "done":
            if isinstance(payload, RunSummary):
                summary = payload
                self.progress_var.set(100.0 if summary.total_pdfs > 0 else 0.0)
                self.percent_var.set("100.0%" if summary.total_pdfs > 0 else "0.0%")
                self.total_var.set(f"{summary.total_pdfs:,}")
                self.ok_var.set(f"{summary.succeeded:,}")
                self.fail_var.set(f"{summary.failed:,}")
                self.page_var.set(f"{summary.pages:,}")
                self.image_var.set(f"{summary.images:,}")
                self.char_var.set(f"{summary.chars:,}")
                self.current_file_var.set("Completed.")
                self._set_status(
                    f"Done in {summary.duration_s:.1f}s | PDFs={summary.total_pdfs:,} "
                    f"ok={summary.succeeded:,} failed={summary.failed:,}"
                )
                self._log(
                    "Done: "
                    f"total={summary.total_pdfs}, ok={summary.succeeded}, failed={summary.failed}, "
                    f"pages={summary.pages}, images={summary.images}, chars={summary.chars}, "
                    f"output={summary.output_dir}"
                )
                if summary.report_path:
                    self._log(f"Report written: {summary.report_path}")
                self.last_output_dir = summary.output_dir
                self.open_btn.configure(state="normal")
            return

    def open_output(self) -> None:
        if not self.last_output_dir or not self.last_output_dir.exists():
            messagebox.showinfo("No Output", "No output folder is available yet.")
            return
        os.startfile(str(self.last_output_dir))  # type: ignore[attr-defined]


def run_cli(args: argparse.Namespace) -> int:
    if fitz is None:
        print("ERROR: PyMuPDF is not installed. Install with: pip install pymupdf")
        return 2

    source = Path(args.source).expanduser()
    output_dir = Path(args.output).expanduser()

    if not source.exists():
        print(f"ERROR: source does not exist: {source}")
        return 2
    if source.is_file() and source.suffix.lower() != ".pdf":
        print("ERROR: source file must be a .pdf")
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    stop_event = threading.Event()

    started = time.time()
    pdfs = sorted(iter_pdf_files(source, recursive=args.recursive, stop_event=stop_event))
    total = len(pdfs)
    if total == 0:
        print("No PDF files found.")
        return 0

    print(f"Found {total} PDF files.")
    workers = max(1, min(32, int(args.workers)))

    rows: list[FileResult] = []
    processed = 0
    ok = 0
    fail = 0
    total_pages = 0
    total_images = 0
    total_chars = 0

    targets = [(pdf, resolve_output_path(source, pdf, output_dir)) for pdf in pdfs]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending: dict[Future[FileResult], Path] = {}
        idx = 0

        def submit_next() -> bool:
            nonlocal idx
            if idx >= len(targets):
                return False
            pdf_path, txt_path = targets[idx]
            idx += 1
            fut = executor.submit(
                extract_pdf_to_txt,
                pdf_path,
                txt_path,
                args.image_markers,
                args.page_headers,
                args.overwrite,
                stop_event,
            )
            pending[fut] = pdf_path
            return True

        for _ in range(min(workers * 2, len(targets))):
            if not submit_next():
                break

        while pending:
            done, _ = wait(pending.keys(), return_when=FIRST_COMPLETED)
            for fut in done:
                pdf_path = pending.pop(fut)
                try:
                    result = fut.result()
                except Exception as exc:
                    result = FileResult(
                        pdf_path=pdf_path,
                        txt_path=resolve_output_path(source, pdf_path, output_dir),
                        success=False,
                        pages=0,
                        images=0,
                        chars=0,
                        error=str(exc),
                    )

                rows.append(result)
                processed += 1
                total_pages += result.pages
                total_images += result.images
                total_chars += result.chars
                if result.success:
                    ok += 1
                else:
                    fail += 1
                    print(f"ERROR: {result.pdf_path} -> {result.error}")

                print(
                    f"{processed}/{total} | ok={ok} fail={fail} "
                    f"pages={total_pages} images={total_images}",
                    end="\r",
                    flush=True,
                )
                submit_next()

    report = output_dir / "_extraction_report.csv"
    with report.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pdf_path", "txt_path", "success", "pages", "images", "chars", "error"])
        for row in rows:
            writer.writerow(
                [
                    str(row.pdf_path),
                    str(row.txt_path),
                    str(row.success),
                    row.pages,
                    row.images,
                    row.chars,
                    row.error,
                ]
            )

    print()
    print(
        "Done "
        f"in {time.time() - started:.1f}s | total={total} ok={ok} fail={fail} "
        f"pages={total_pages} images={total_images} chars={total_chars}"
    )
    print(f"Output: {output_dir}")
    print(f"Report: {report}")
    return 0 if fail == 0 else 1


def make_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract text from a PDF file or folder of PDFs into .txt files with optional image markers.",
    )
    parser.add_argument("--source", help="Input PDF file or folder path")
    parser.add_argument("--output", help="Destination folder for generated .txt files")
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 4))),
        help="Parallel worker count for batch processing (default: min(8, CPU cores))",
    )
    parser.add_argument("--recursive", action="store_true", help="Recursively scan folders for PDFs")
    parser.add_argument("--no-recursive", dest="recursive", action="store_false", help="Only scan top folder")
    parser.set_defaults(recursive=True)

    parser.add_argument("--image-markers", action="store_true", help="Insert (imgX) marker when image blocks are found")
    parser.add_argument("--no-image-markers", dest="image_markers", action="store_false", help="Disable image markers")
    parser.set_defaults(image_markers=True)

    parser.add_argument("--page-headers", action="store_true", help="Include page separators in output text")
    parser.add_argument("--no-page-headers", dest="page_headers", action="store_false", help="Disable page separators")
    parser.set_defaults(page_headers=True)

    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing text outputs")
    parser.add_argument("--no-overwrite", dest="overwrite", action="store_false", help="Skip existing text outputs")
    parser.set_defaults(overwrite=True)
    return parser


def main() -> None:
    parser = make_arg_parser()
    args = parser.parse_args()

    if args.source and args.output:
        raise SystemExit(run_cli(args))

    root = build_root()
    app = PDFTextExtractorApp(root)
    _ = app
    root.mainloop()


if __name__ == "__main__":
    main()
