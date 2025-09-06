
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Progress bar
try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x, **_: x  # fallback

# Vector store & client
import chromadb
from langchain_chroma import Chroma


# ---------- Utilities ----------

def _normalize_record(obj: Dict) -> Tuple[str, Dict]:
    """Normalize one JSONL record into (text, metadata)."""
    text = obj.get("text") or obj.get("content")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Record missing non-empty 'text'/'content'.")

    meta = obj.get("metadata") or obj.get("meta") or {}
    if not isinstance(meta, dict):
        meta = {}

    if "source" in obj and "source" not in meta:
        meta["source"] = obj["source"]
    if "chunk_index" not in meta and "chunk_id" in obj:
        meta["chunk_index"] = obj["chunk_id"]
    if "page" in obj and "page" not in meta:
        meta["page"] = obj["page"]

    return text.strip(), meta


def _read_jsonl(path: Path) -> List[Tuple[str, Dict]]:
    """Load JSONL chunks and return (text, metadata) pairs."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    out: List[Tuple[str, Dict]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {i}: {e}") from e
            text, meta = _normalize_record(obj)
            out.append((text, meta))

    if not out:
        raise ValueError("No lines loaded from JSONL; is the file empty?")
    return out


def _stable_id(text: str, meta: Dict) -> str:
    """Deterministic ID for a chunk using a stable hash."""
    source = str(meta.get("source", ""))
    page = str(meta.get("page", ""))
    idx = str(meta.get("chunk_index", ""))
    basis = "|".join([source, page, idx, text])
    return hashlib.sha1(basis.encode("utf-8")).hexdigest()


def _batched(lst: List, n: int):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


# ---------- Embeddings ----------

def build_embeddings(model: Optional[str]):
    """Construct HuggingFace embeddings object."""
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError as e:
        raise SystemExit(
            "Missing dependency: langchain-huggingface. Add it to requirements.txt."
        ) from e
    model_name = model or "sentence-transformers/all-MiniLM-L6-v2"
    return HuggingFaceEmbeddings(model_name=model_name)


# ---------- Chroma upsert ----------

def upsert_into_chroma(
    pairs: List[Tuple[str, Dict]],
    persist_dir: Path,
    collection: str,
    embedding_fn,
    batch_size: int = 128,
    clear: bool = False,
) -> int:
    """Upsert documents into a persistent Chroma collection."""
    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))

    if clear:
        try:
            client.delete_collection(collection)
        except Exception:
            pass

    vectordb = Chroma(
        client=client,
        collection_name=collection,
        embedding_function=embedding_fn,
        persist_directory=str(persist_dir),
    )

    texts, metadatas, ids = [], [], []
    for text, meta in pairs:
        texts.append(text)
        metadatas.append(meta)
        ids.append(_stable_id(text, meta))

    written = 0
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for b_texts, b_metas, b_ids in tqdm(
        zip(_batched(texts, batch_size), _batched(metadatas, batch_size), _batched(ids, batch_size)),
        total=total_batches,
        desc="Upserting into Chroma",
    ):
        try:
            vectordb.delete(ids=b_ids)
        except Exception:
            pass
        vectordb.add_texts(texts=b_texts, metadatas=b_metas, ids=b_ids)
        written += len(b_texts)

    return written


# ---------- CLI ----------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Embed JSONL chunks into Chroma (HuggingFace only).")
    p.add_argument("--input", required=True, type=Path, help="Path to chunks JSONL file")
    p.add_argument("--persist-dir", required=True, type=Path, help="Directory for Chroma persistence")
    p.add_argument("--collection", required=True, type=str, help="Chroma collection name")
    p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", help="HuggingFace embedding model")
    p.add_argument("--batch-size", type=int, default=128, help="Batch size for upserts")
    p.add_argument("--clear", action="store_true", help="Delete existing collection before writing")
    p.add_argument("--show-stats", action="store_true", help="Print brief stats from the input file")
    args = p.parse_args(argv)

    pairs = _read_jsonl(args.input)

    if args.show_stats:
        sources = {}
        has_chunk_id = False
        for _, m in pairs:
            s = m.get("source", "unknown")
            sources[s] = sources.get(s, 0) + 1
            if "chunk_index" in m:
                has_chunk_id = True
        print(f"Loaded {len(pairs)} chunks from {args.input}")
        print("Top sources:")
        for k, v in sorted(sources.items(), key=lambda kv: kv[1], reverse=True)[:10]:
            print(f"  {k}: {v}")
        print(f"Has chunk_index metadata: {has_chunk_id}")

    embeddings = build_embeddings(args.model)
    written = upsert_into_chroma(
        pairs=pairs,
        persist_dir=args.persist_dir,
        collection=args.collection,
        embedding_fn=embeddings,
        batch_size=args.batch_size,
        clear=args.clear,
    )
    print(f"✅ Wrote {written} chunks to collection '{args.collection}' at {args.persist_dir}")

    try:
        client = chromadb.PersistentClient(path=str(args.persist_dir))
        coll = client.get_collection(args.collection)
        print(f"📦 Collection '{args.collection}' now contains {coll.count()} embeddings.")
    except Exception as e:
        print(f"(Verification skipped) Unable to count docs: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
