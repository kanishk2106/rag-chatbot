import sys
import json
import subprocess
from pathlib import Path
import importlib

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

def _import_ingest():
    """Import ingest.py from repo root."""
    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return importlib.import_module("ingest")

LONG_TEXT = "RAG test " * 4000  # long enough to make multiple 500-char chunks

def test_chunk_text_basic():
    ingest = _import_ingest()
    chunks = ingest.chunk_text(LONG_TEXT, chunk_size=500, overlap=50)
    assert isinstance(chunks, list)
    assert len(chunks) > 0, "chunk_text returned 0 chunks"
    if chunks and isinstance(chunks[0], str):
        assert max(len(c) for c in chunks) <= 600

def test_cli_end_to_end_md(tmp_path):
    root = _repo_root()
    input_dir = tmp_path / "data_in"
    input_dir.mkdir(parents=True, exist_ok=True)
    (input_dir / "sample.md").write_text(LONG_TEXT, encoding="utf-8")
    out_path = tmp_path / "chunks.jsonl"

    cmd = [
        sys.executable,
        str(root / "ingest.py"),
        "--input", str(input_dir),
        "--out", str(out_path),
        "--size", "500",
        "--overlap", "50",
    ]
    cp = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
    assert cp.returncode == 0, f"CLI failed: {cp.stderr or cp.stdout}"
    assert out_path.exists(), "Expected output JSONL not found"

    lines = [ln for ln in out_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) > 0, "CLI produced 0 chunks"
    rec = json.loads(lines[0])
    for key in ("source", "chunk_id", "text"):
        assert key in rec
