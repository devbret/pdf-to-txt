from __future__ import annotations
import argparse
import concurrent.futures as futures
import hashlib
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple, Callable
from itertools import repeat

_LOG_PATH: Optional[Path] = None
_VERBOSE: bool = False

def _maybe_setup_child_logging() -> None:
    global _LOG_PATH, _VERBOSE
    root = logging.getLogger()
    if root.handlers:
        return
    handlers = []
    if _LOG_PATH is not None:
        handlers.append(logging.FileHandler(_LOG_PATH, encoding="utf-8"))
    if _VERBOSE:
        handlers.append(logging.StreamHandler())
    level = logging.DEBUG if _VERBOSE else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(processName)s %(levelname)s] %(message)s", handlers=handlers or [logging.StreamHandler()])
    logging.debug("Child logging initialized verbose=%s path=%s", _VERBOSE, _LOG_PATH)

PdfReader = None
try:
    from pypdf import PdfReader as _PdfReader
    PdfReader = _PdfReader
except Exception:
    try:
        from PyPDF2 import PdfReader as _PdfReader
        PdfReader = _PdfReader
    except Exception:
        PdfReader = None

_pdfminer_extract_text: Optional[Callable[..., str]] = None
try:
    from pdfminer.high_level import extract_text as _pdfminer_extract_text
except Exception:
    _pdfminer_extract_text = None

_ocr_available = True
try:
    from pdf2image import convert_from_path
    import pytesseract
    from PIL import Image
except Exception:
    _ocr_available = False

_use_tqdm = False
try:
    from tqdm import tqdm
    _use_tqdm = True
except Exception:
    _use_tqdm = False

LOG_DEFAULT = "pdf_extraction.log"
OUTPUT_DEFAULT = "combined_output.txt"

@dataclass
class ExtractionResult:
    path: str
    ok: bool
    pages: int
    chars: int
    sha256: Optional[str]
    error: Optional[str]
    engine: str
    ocr_used: bool

@dataclass
class WorkerConfig:
    engine: str
    keep_formfeed: bool
    try_decrypt_empty: bool
    requested_pages_1based: Optional[List[int]]
    enable_ocr: bool
    per_file_out_dir: Optional[str]

def setup_logging(log_file: Path, verbose: bool) -> None:
    global _LOG_PATH, _VERBOSE
    _LOG_PATH = log_file
    _VERBOSE = verbose
    log_file.parent.mkdir(parents=True, exist_ok=True)
    handlers = [logging.FileHandler(log_file, encoding="utf-8")]
    if verbose:
        handlers.append(logging.StreamHandler())
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format="%(asctime)s [%(processName)s %(levelname)s] %(message)s", handlers=handlers)
    logging.info("Logging initialized -> %s (verbose=%s)", log_file, verbose)

def discover_pdfs(root: Path, recursive: bool, pattern: str) -> List[Path]:
    t0 = time.perf_counter()
    if recursive:
        result = sorted(p for p in root.rglob(pattern) if p.suffix.lower() == ".pdf")
    else:
        result = sorted(p for p in root.glob(pattern) if p.suffix.lower() == ".pdf")
    dt = (time.perf_counter() - t0) * 1000
    logging.debug("Discovered %d PDFs in %.1f ms (root=%s recursive=%s pattern=%s)", len(result), dt, root, recursive, pattern)
    return result

def parse_pages_arg(pages_arg: Optional[str]) -> Optional[List[int]]:
    if not pages_arg:
        return None
    parts = [p.strip() for p in pages_arg.split(",") if p.strip()]
    page_set = set()
    for part in parts:
        m = re.match(r"^(\d+)-(\d+)$", part)
        if m:
            start, end = int(m.group(1)), int(m.group(2))
            if end < start:
                start, end = end, start
            page_set.update(range(start, end + 1))
            continue
        m = re.match(r"^(\d+)-$", part)
        if m:
            page_set.add(-int(m.group(1)))
            continue
        m = re.match(r"^(\d+)$", part)
        if m:
            page_set.add(int(m.group(1)))
            continue
        raise ValueError(f"Invalid pages spec: {part}")
    resolved = sorted(page_set)
    logging.debug("Parsed pages arg '%s' -> %s", pages_arg, resolved)
    return resolved

def resolve_page_numbers(requested_1based: Optional[List[int]], total_pages: int) -> Optional[List[int]]:
    if requested_1based is None:
        return None
    resolved = set()
    for n in requested_1based:
        if n < 0:
            resolved.update(range(-n, total_pages + 1))
        elif 1 <= n <= total_pages:
            resolved.add(n)
    out = sorted(resolved)
    logging.debug("Resolved page numbers %s with total=%d -> %s", requested_1based, total_pages, out)
    return out

def normalize_text(s: str, keep_formfeed: bool) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    if not keep_formfeed:
        s = s.replace("\f", "\n")
    lines = [line.strip() for line in s.splitlines()]
    return "\n".join(lines)

def atomic_write(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(data)
    tmp.replace(path)
    logging.debug("Wrote file atomically -> %s (%d bytes)", path, len(data.encode('utf-8')))

def extract_with_pypdf(pdf_path: Path, page_numbers_1based: Optional[List[int]], keep_formfeed: bool, try_decrypt_empty: bool) -> str:
    if PdfReader is None:
        raise RuntimeError("Neither pypdf nor PyPDF2 is available.")
    t0 = time.perf_counter()
    text_parts = []
    with pdf_path.open("rb") as fh:
        reader = PdfReader(fh)
        is_encrypted = bool(getattr(reader, "is_encrypted", False))
        if is_encrypted and try_decrypt_empty:
            try:
                reader.decrypt("")
                logging.debug("Tried empty-password decrypt for %s", pdf_path.name)
            except Exception as e:
                logging.debug("Empty-password decrypt failed for %s: %s", pdf_path.name, e)
        total = len(reader.pages)
        pages = [p - 1 for p in page_numbers_1based if 1 <= p <= total] if page_numbers_1based else range(total)
        for idx in pages:
            try:
                t = reader.pages[idx].extract_text() or ""
                text_parts.append(t + ("\f" if keep_formfeed else ""))
            except Exception as e:
                logging.warning("Failed to extract page %s of %s: %s", idx + 1, pdf_path.name, e)
    dt = (time.perf_counter() - t0) * 1000
    logging.debug("pypdf extracted %s in %.1f ms", pdf_path.name, dt)
    return "".join(text_parts)

def extract_with_pdfminer(pdf_path: Path, page_numbers_1based: Optional[List[int]], keep_formfeed: bool) -> str:
    if _pdfminer_extract_text is None:
        raise RuntimeError("pdfminer.six not installed.")
    t0 = time.perf_counter()
    page_numbers = [p - 1 for p in page_numbers_1based] if page_numbers_1based else None
    txt = _pdfminer_extract_text(str(pdf_path), page_numbers=page_numbers)
    if keep_formfeed and "\f" not in txt:
        txt += "\f"
    dt = (time.perf_counter() - t0) * 1000
    logging.debug("pdfminer extracted %s in %.1f ms", pdf_path.name, dt)
    return txt

def ocr_pdf_to_text(pdf_path: Path, page_numbers_1based: Optional[List[int]], keep_formfeed: bool, dpi: int = 300) -> str:
    if not _ocr_available:
        raise RuntimeError("OCR stack not available.")
    t0 = time.perf_counter()
    images = convert_from_path(str(pdf_path), dpi=dpi)
    total = len(images)
    indices = [p - 1 for p in page_numbers_1based if 1 <= p <= total] if page_numbers_1based else range(total)
    text_parts = [pytesseract.image_to_string(images[i]) + ("\f" if keep_formfeed else "") for i in indices]
    dt = (time.perf_counter() - t0) * 1000
    logging.debug("OCR extracted %s in %.1f ms (dpi=%d, pages=%d)", pdf_path.name, dt, dpi, len(list(indices)))
    return "".join(text_parts)

def extract_text_from_pdf(pdf_path: Path, engine: str, keep_formfeed: bool, try_decrypt_empty: bool, requested_pages_1based: Optional[List[int]], enable_ocr: bool) -> Tuple[str, ExtractionResult]:
    ocr_used = False
    error_msg = None
    extracted = ""
    pages = 0
    logging.info("Starting extraction: %s engine=%s ocr=%s", pdf_path.name, engine, enable_ocr)
    t0 = time.perf_counter()
    try:
        total_pages_known = None
        if PdfReader:
            try:
                with pdf_path.open("rb") as fh:
                    reader = PdfReader(fh)
                    total_pages_known = len(reader.pages)
            except Exception as e:
                logging.debug("Failed to pre-read page count for %s: %s", pdf_path.name, e)
        pages = total_pages_known or 0
        pages_1based = resolve_page_numbers(requested_pages_1based, pages) if pages else requested_pages_1based
        if engine == "pdfminer":
            extracted = extract_with_pdfminer(pdf_path, pages_1based, keep_formfeed)
            if not pages and extracted:
                pages = extracted.count("\f") or 1
        else:
            extracted = extract_with_pypdf(pdf_path, pages_1based, keep_formfeed, try_decrypt_empty)
            if not pages:
                pages = extracted.count("\f") or 1
        if enable_ocr and (not extracted or not extracted.strip()):
            try:
                logging.info("Falling back to OCR for %s", pdf_path.name)
                extracted = ocr_pdf_to_text(pdf_path, pages_1based, keep_formfeed)
                ocr_used = True
            except Exception as e:
                error_msg = f"OCR failed: {e}"
                logging.warning("OCR failed for %s: %s", pdf_path.name, e)
        norm = normalize_text(extracted, keep_formfeed)
        sha256 = hashlib.sha256(norm.encode("utf-8")).hexdigest() if norm else None
        meta = ExtractionResult(str(pdf_path), True, pages, len(norm), sha256, None if norm.strip() else error_msg or "Empty extraction", engine, ocr_used)
        dt = (time.perf_counter() - t0) * 1000
        logging.info("Finished extraction: %s pages=%d chars=%d ocr=%s ms=%.1f", pdf_path.name, pages, len(norm), ocr_used, dt)
        if not norm.strip():
            logging.warning("Empty text for %s (engine=%s ocr=%s)", pdf_path.name, engine, ocr_used)
        return norm, meta
    except Exception as e:
        dt = (time.perf_counter() - t0) * 1000
        logging.error("Extraction error for %s after %.1f ms: %s", pdf_path.name, dt, e)
        meta = ExtractionResult(str(pdf_path), False, pages, 0, None, str(e), engine, ocr_used)
        return "", meta

def handle_one(pdf_path: str, config: WorkerConfig) -> Tuple[str, ExtractionResult, Optional[str]]:
    _maybe_setup_child_logging()
    logging.debug("Worker handling %s", pdf_path)
    text, meta = extract_text_from_pdf(Path(pdf_path), config.engine, config.keep_formfeed, config.try_decrypt_empty, config.requested_pages_1based, config.enable_ocr)
    per_file = None
    if config.per_file_out_dir:
        per_file = str(Path(config.per_file_out_dir) / (Path(pdf_path).stem + ".txt"))
    return text, meta, per_file

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Extract text from PDFs.")
    parser.add_argument("-i", "--input", default="./input")
    parser.add_argument("-o", "--output", default=OUTPUT_DEFAULT)
    parser.add_argument("--per-file-out", default=None)
    parser.add_argument("-r", "--recursive", action="store_true")
    parser.add_argument("--pattern", default="*")
    parser.add_argument("--keep-formfeed", action="store_true")
    parser.add_argument("--index-jsonl", default=None)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--log-file", default=LOG_DEFAULT)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--try-decrypt-empty", action="store_true")
    parser.add_argument("--engine", choices=["pypdf", "pdfminer"], default="pypdf")
    parser.add_argument("--ocr", action="store_true")
    parser.add_argument("--pages", default=None)
    args = parser.parse_args(argv)

    input_dir = Path(args.input)
    output_file = Path(args.output)
    per_file_dir = Path(args.per_file_out) if args.per_file_out else None
    index_path = Path(args.index_jsonl) if args.index_jsonl else None

    setup_logging(Path(args.log_file), args.verbose)
    logging.info("Args: %s", vars(args))

    if args.engine == "pdfminer" and _pdfminer_extract_text is None:
        logging.warning("pdfminer not available, switching engine to pypdf")
        args.engine = "pypdf"
    if args.ocr and not _ocr_available:
        logging.warning("OCR stack not available, disabling --ocr")
        args.ocr = False
    if not input_dir.exists():
        logging.error("Input directory missing: %s", input_dir)
        print("Input directory missing.")
        return 2

    t_discover = time.perf_counter()
    pdfs = discover_pdfs(input_dir, args.recursive, args.pattern)
    logging.info("Discovery complete: %d PDF(s) found in %.1f ms", len(pdfs), (time.perf_counter() - t_discover) * 1000)
    if not pdfs:
        logging.info("No PDFs found. Exiting.")
        print("No PDFs found.")
        return 0

    try:
        requested_pages = parse_pages_arg(args.pages)
    except ValueError as ve:
        logging.error("Invalid --pages: %s", ve)
        print(f"Invalid --pages: {ve}")
        return 2

    print(f"Found {len(pdfs)} PDF(s). Starting extraction...\n")
    progress = tqdm(total=len(pdfs), unit="pdf") if _use_tqdm else None

    combined_chunks = []
    index_lines = []
    config = WorkerConfig(args.engine, args.keep_formfeed, args.try_decrypt_empty, requested_pages, args.ocr and _ocr_available, str(per_file_dir) if per_file_dir else None)
    worker_count = max(1, (os.cpu_count() or 4) if args.workers == 0 else args.workers)
    logging.info("Workers: requested=%s effective=%d", args.workers, worker_count)

    start_all = time.perf_counter()
    if worker_count == 1:
        iterator = (handle_one(str(p), config) for p in pdfs)
    else:
        try:
            pool = futures.ProcessPoolExecutor(max_workers=worker_count)
            iterator = pool.map(handle_one, map(str, pdfs), repeat(config))
        except Exception as e:
            logging.error("Failed to start ProcessPoolExecutor: %s. Falling back to single worker.", e)
            worker_count = 1
            iterator = (handle_one(str(p), config) for p in pdfs)

    processed = 0
    for text, meta, per_path in iterator:
        processed += 1
        logging.info("Processed %d/%d: %s ok=%s pages=%d chars=%d engine=%s ocr=%s", processed, len(pdfs), Path(meta.path).name, meta.ok, meta.pages, meta.chars, meta.engine, meta.ocr_used)
        if per_path:
            path = Path(per_path)
            try:
                if meta.ok and text:
                    atomic_write(path, text)
                    logging.debug("Per-file output saved: %s", path)
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.touch()
                    logging.debug("Per-file placeholder created (empty or failed): %s", path)
            except Exception as e:
                logging.error("Failed to write per-file output %s: %s", path, e)
        if meta.ok and text:
            combined_chunks.append(f"\n\n==== {Path(meta.path).name} ({meta.pages} pages) ====\n\n" + text)
        index_lines.append(json.dumps(asdict(meta), ensure_ascii=False))
        if progress:
            progress.update(1)

    if progress:
        progress.close()

    try:
        atomic_write(output_file, "".join(combined_chunks))
        print(f"Saved to: {output_file}")
        logging.info("Combined output written: %s", output_file)
    except Exception as e:
        print(f"Failed to write combined: {e}")
        logging.error("Failed to write combined output %s: %s", output_file, e)

    if index_path:
        try:
            atomic_write(index_path, "\n".join(index_lines) + "\n")
            logging.info("Index JSONL written: %s lines=%d", index_path, len(index_lines))
        except Exception as e:
            print(f"Failed to write index: {e}")
            logging.error("Failed to write index %s: %s", index_path, e)

    total_ms = (time.perf_counter() - start_all) * 1000
    logging.info("All done. PDFs=%d total_ms=%.1f", len(pdfs), total_ms)
    return 0

if __name__ == "__main__":
    sys.exit(main())
