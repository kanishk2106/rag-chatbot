from __future__ import annotations
from pathlib import Path
import argparse
import json
import sys
import re
import shutil
from typing import Iterator, Tuple, List, Optional

# ---------- Core text extraction ----------
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------- Optional OCR deps ----------
try:
    from pdf2image import convert_from_path
    import pytesseract
    OCR_IMPORTS_OK = True
except Exception:
    OCR_IMPORTS_OK = False

# ---------- Config ----------
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# OCR tuning
OCR_DPI = 300
OCR_LANG = "eng"            # change if needed (e.g., "eng+spa")
MIN_PAGE_ALNUM = 25         # threshold to decide a page is "too empty" and needs OCR
POPPLER_PATH: Optional[str] = None  # not required; auto-detected if possible

# ---------- Cleaning & helpers ----------
def clean_text(t: str) -> str:
    """Light cleanup to make form-like PDFs more readable."""
    t = re.sub(r"\.{3,}", " … ", t)    # dotted leaders …… -> ellipsis
    t = re.sub(r"_{3,}", " ", t)       # long underscores -> space
    t = re.sub(r"[•●◦·]", "-", t)     # bullets -> dashes
    t = re.sub(r"[ \t]+\n", "\n", t)   # trim trailing spaces before newline
    t = re.sub(r"\n{3,}", "\n\n", t)   # collapse >2 blank lines
    t = re.sub(r"[ ]{2,}", " ", t)     # collapse multiple spaces
    return t.strip()

def _alnum_count(s: str) -> int:
    return sum(c.isalnum() for c in s)

def _detect_binaries(verbose: bool = False) -> tuple[bool, Optional[str], Optional[str]]:
    """
    Detect system binaries for Tesseract and Poppler (pdftoppm).
    Returns (ocr_available, tesseract_cmd, poppler_dir)
    """
    tesseract_cmd = shutil.which("tesseract")
    pdftoppm_cmd = shutil.which("pdftoppm")

    # Common Homebrew locations on macOS if PATH isn't visible to PyCharm/venv
    if not tesseract_cmd:
        for cand in ("/opt/homebrew/bin/tesseract", "/usr/local/bin/tesseract"):
            if Path(cand).exists():
                tesseract_cmd = cand
                break

    if not pdftoppm_cmd:
        for cand in ("/opt/homebrew/bin/pdftoppm", "/usr/local/bin/pdftoppm"):
            if Path(cand).exists():
                pdftoppm_cmd = cand
                break

    ocr_available = OCR_IMPORTS_OK and bool(tesseract_cmd) and bool(pdftoppm_cmd)

    if verbose:
        print(f"[CHECK] OCR_IMPORTS_OK={OCR_IMPORTS_OK}", file=sys.stderr)
        print(f"[CHECK] tesseract: {tesseract_cmd or 'NOT FOUND'}", file=sys.stderr)
        print(f"[CHECK] pdftoppm:  {pdftoppm_cmd or 'NOT FOUND'}", file=sys.stderr)

    # Configure pytesseract if we found the binary
    if OCR_IMPORTS_OK and tesseract_cmd:
        try:
            import pytesseract as _pt
            _pt.pytesseract.tesseract_cmd = tesseract_cmd
        except Exception:
            pass

    poppler_dir = str(Path(pdftoppm_cmd).parent) if pdftoppm_cmd else None
    return ocr_available, tesseract_cmd, poppler_dir

# Set at runtime in main()
_OCR_AVAILABLE: bool = False
_POPPLER_DIR: Optional[str] = None

def _ocr_single_page(pdf_path: Path, page_index_0based: int, verbose: bool = False) -> str:
    """OCR just one page (0-based index)."""
    if not _OCR_AVAILABLE:
        if verbose:
            print(f"[PAGE {page_index_0based+1}] OCR unavailable; skipping.", file=sys.stderr)
        return ""
    kwargs = {"dpi": OCR_DPI}
    # pdf2image expects the directory containing Poppler binaries
    if _POPPLER_DIR:
        kwargs["poppler_path"] = _POPPLER_DIR
    try:
        # pdf2image uses 1-based page indices
        imgs = convert_from_path(
            str(pdf_path),
            first_page=page_index_0based + 1,
            last_page=page_index_0based + 1,
            **kwargs
        )
        if not imgs:
            if verbose:
                print(f"[PAGE {page_index_0based+1}] pdf2image returned no image.", file=sys.stderr)
            return ""
        txt = pytesseract.image_to_string(imgs[0], lang=OCR_LANG) or ""
        return txt
    except Exception as e:
        if verbose:
            print(f"[PAGE {page_index_0based+1}] OCR error: {e}", file=sys.stderr)
        return ""

# ---------- Readers ----------
def read_pdf_with_ocr_fallback(path: Path, verbose: bool = False) -> str:
    """
    Extract text with pypdf; if a page has too little text, try OCR for that page.
    Logs per-page decisions when verbose=True.
    """
    reader = PdfReader(str(path))
    parts: List[str] = []

    if verbose:
        print(
            f"[INFO] Reading PDF: {path} | pages={len(reader.pages)} | OCR_AVAILABLE={_OCR_AVAILABLE}",
            file=sys.stderr
        )

    for i, page in enumerate(reader.pages):
        # 1) Try direct text extract
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        direct_count = _alnum_count(txt)
        if verbose:
            print(f"[PAGE {i+1}] pypdf alnum={direct_count}", file=sys.stderr)

        # 2) If page looks empty/low-info, try OCR
        if direct_count < MIN_PAGE_ALNUM:
            ocr_txt = _ocr_single_page(path, i, verbose=verbose)
            ocr_count = _alnum_count(ocr_txt)
            if verbose:
                print(f"[PAGE {i+1}] OCR attempted: alnum={ocr_count}", file=sys.stderr)
            if ocr_count >= direct_count:
                txt = ocr_txt
                if verbose:
                    print(f"[PAGE {i+1}] Using OCR text (better or equal).", file=sys.stderr)
            else:
                if verbose:
                    print(f"[PAGE {i+1}] Keeping pypdf text (OCR worse).", file=sys.stderr)

        parts.append(txt)

    full = "\n".join(parts).strip()
    if verbose:
        print(f"[INFO] Finished {path.name}: total_chars={len(full)}", file=sys.stderr)
    return full

def read_md(path: Path) -> str:
    """Read Markdown/MD-like text as UTF-8."""
    return path.read_text(encoding="utf-8", errors="ignore")

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks for retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_text(text)

def iter_documents(input_dir: Path, verbose: bool = False) -> Iterator[Tuple[str, str]]:
    """
    Recursively yield (source_path, cleaned_text) for .pdf and .md files under input_dir.
    Streaming/generator version: does not accumulate in memory.
    """
    for p in input_dir.rglob("*"):
        if not p.is_file():
            continue
        suf = p.suffix.lower()
        try:
            if suf == ".pdf":
                txt = read_pdf_with_ocr_fallback(p, verbose=verbose)
            elif suf in {".md", ".markdown"}:
                txt = read_md(p)
            else:
                continue
        except Exception as e:
            if verbose:
                print(f"[WARN] Skipping unreadable {p}: {e}", file=sys.stderr)
            continue

        txt = clean_text(txt)
        if txt and txt.strip():
            yield (str(p), txt)
        elif verbose:
            print(f"[WARN] Empty after cleaning: {p}", file=sys.stderr)

# ---------- Post-run sampler ----------
def preview_chunks(
    jsonl_path: str,
    n: int = 3,
    only_source: str | None = None,  # e.g., "/path/to/file.pdf"
    clean: bool = True
) -> None:
    """
    Print the first `n` chunks from `jsonl_path`.
    If `only_source` is provided, only show chunks from that source file.
    """
    p = Path(jsonl_path)
    if not p.exists():
        print(f"⚠️ File not found: {p}")
        return

    shown = 0
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue

            if only_source and rec.get("source") != only_source:
                continue

            text = rec.get("text", "")
            if clean:
                text = clean_text(text)

            print(f"\n=== Chunk {rec.get('chunk_id')} | Source: {rec.get('source')} ===")
            print(text[:600])  # show up to 600 chars
            shown += 1
            if shown >= n:
                break

    if shown == 0:
        msg = "No chunks found"
        if only_source:
            msg += f" for source: {only_source}"
        print(f"ℹ️ {msg} in {jsonl_path}")

# ---------- Main ----------
def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest PDFs/MD → chunk → JSONL (streaming, OCR fallback)")
    ap.add_argument("--input", default="data", help="Input folder containing PDFs/MD")
    ap.add_argument("--out", default="data/chunks.jsonl", help="Output JSONL path")
    ap.add_argument("--size", type=int, default=CHUNK_SIZE, help="Chunk size")
    ap.add_argument("--overlap", type=int, default=CHUNK_OVERLAP, help="Chunk overlap")
    ap.add_argument("--verbose", action="store_true", help="Print per-page extraction/OCR logs")
    ap.add_argument("--preview", action="store_true", help="Preview first few chunks after write")
    ap.add_argument("--preview_n", type=int, default=3, help="How many chunks to preview")
    ap.add_argument("--preview_only_source", type=str, default=None, help="Preview only chunks from this exact source path")
    args, _ = ap.parse_known_args()  # ignore unexpected args from IDEs

    input_dir = Path(args.input)
    out_path = Path(args.out)

    if not input_dir.exists():
        raise SystemExit(f"Input folder not found: {input_dir}")

    # Ensure output directory exists
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Detect OCR binaries (works nicely from PyCharm/venv)
    global _OCR_AVAILABLE, _POPPLER_DIR
    _OCR_AVAILABLE, tesseract_cmd, _POPPLER_DIR = _detect_binaries(verbose=args.verbose)

    if not OCR_IMPORTS_OK:
        print("Note: Python OCR packages (pdf2image/pytesseract) not importable. Proceeding without OCR…", file=sys.stderr)
    if not _OCR_AVAILABLE:
        print("Note: OCR fallback not available (install Tesseract + Poppler, and pip install pdf2image pytesseract pillow). Proceeding without OCR…", file=sys.stderr)
    else:
        if args.verbose:
            print(f"[CHECK] Using tesseract: {tesseract_cmd}", file=sys.stderr)
            print(f"[CHECK] Using poppler dir: {_POPPLER_DIR}", file=sys.stderr)

    total_docs = 0
    total_chunks = 0

    # Stream write to JSONL
    with out_path.open("w", encoding="utf-8") as f:
        for source, text in iter_documents(input_dir, verbose=args.verbose):
            total_docs += 1
            chunks = chunk_text(text, chunk_size=args.size, overlap=args.overlap)
            for i, ch in enumerate(chunks):
                rec = {"source": source, "chunk_id": i, "text": ch}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            total_chunks += len(chunks)

    print(f"Docs: {total_docs} | Chunks: {total_chunks} | Wrote: {out_path}")

    # Optional preview
    if args.preview:
        preview_chunks(
            jsonl_path=str(out_path),
            n=args.preview_n,
            only_source=args.preview_only_source,
            clean=True
        )

if __name__ == "__main__":
    main()
