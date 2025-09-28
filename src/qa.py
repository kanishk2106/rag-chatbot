#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG query over a local Chroma DB with HF (Qwen) for answering and bracketed citations.

Quick start (terminal or PyCharm Run Configuration -> Parameters):
  python l.py "What is AI?"
  python l.py "Is Leo and Capricorn compatible?" --persist-dir .chroma --collection rag-chatbot \
      --embed-model sentence-transformers/all-MiniLM-L6-v2 \
      --llm-model Qwen/Qwen2.5-1.5B-Instruct --k 4

Requirements (install these):
  pip install "transformers>=4.41" "accelerate>=0.30" torch --extra-index-url https://download.pytorch.org/whl/cu121
  pip install chromadb langchain-chroma langchain-huggingface
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# ---------- Config defaults ----------
DEFAULT_PERSIST = Path(".chroma")
DEFAULT_COLLECTION = "rag-chatbot"
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"  # free, local, better than TinyLlama


# ---------- LLM (HF transformers, Qwen) ----------
def load_hf_llm(model_name: str = DEFAULT_LLM_MODEL):
    """
    Create a text-generation pipeline using Qwen (or any chat HF model).
    Uses model chat template for best results.
    """
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    import torch

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # Pick dtype/device automatically (CUDA/MPS -> fp16, CPU -> fp32)
    has_cuda = torch.cuda.is_available()
    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    dtype = torch.float16 if (has_cuda or has_mps) else torch.float32
    device_map = "auto" if (has_cuda or has_mps) else None

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map=device_map,
    ).eval()

    gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tok,
        max_new_tokens=512,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=1.05,
        pad_token_id=tok.eos_token_id,
    )
    return gen, tok


# ---------- Embeddings ----------
def load_embeddings(model_name: str):
    return HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={"normalize_embeddings": True},  # improves cosine similarity behavior
    )


# ---------- Retriever ----------
def build_retriever(persist_dir: Path, collection: str, embed_model: str, k: int = 4):
    # Use a client pointing at the on-disk DB
    client = chromadb.PersistentClient(path=str(persist_dir))
    embeddings = load_embeddings(embed_model)

    # With a client, don't also pass persist_directory (avoids warnings)
    vectordb = Chroma(
        client=client,
        collection_name=collection,
        embedding_function=embeddings,
    )

    # MMR tends to diversify results a bit
    return vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": max(10, 3 * k)},
    )


# ---------- Prompt ----------
SYSTEM_INSTRUCTIONS = """\
You are a careful assistant. Use ONLY the provided context to answer.
Cite sources with bracketed numbers like [1], [2] that correspond to the
context entries. If the answer is not present in the context, say "I don't know."
Be concise and factual.
"""


def build_context_and_citations(docs: List) -> Tuple[str, List[Dict]]:
    """
    Format retrieved docs into a numbered context block and produce a citations list.
    Each doc gets an index [i] with its source metadata.
    """
    lines = []
    citations = []
    for i, d in enumerate(docs, start=1):
        meta = d.metadata or {}
        source = meta.get("source", "unknown")
        chunk_index = meta.get("chunk_index", meta.get("id", ""))
        page = meta.get("page")
        head = f"[{i}] Source: {source}"
        if page is not None:
            head += f" | page: {page}"
        if chunk_index != "":
            head += f" | chunk_index: {chunk_index}"
        lines.append(f"{head}\n{d.page_content}\n")
        citations.append(
            {"n": i, "source": source, "page": page, "chunk_index": chunk_index}
        )
    return "\n".join(lines).strip(), citations


# ---------- QA ----------
def answer_question(
    question: str,
    persist_dir: Path = DEFAULT_PERSIST,
    collection: str = DEFAULT_COLLECTION,
    embed_model: str = DEFAULT_EMBED_MODEL,
    llm_model: str = DEFAULT_LLM_MODEL,
    k: int = 4,
) -> Dict:
    retriever = build_retriever(persist_dir, collection, embed_model, k=k)

    # Optional sanity check: ensure the collection exists & is non-empty
    client = chromadb.PersistentClient(path=str(persist_dir))
    existing = [c.name for c in client.list_collections()]
    if collection not in existing:
        raise RuntimeError(
            f"Collection '{collection}' not found in {persist_dir}. "
            f"Existing: {existing}"
        )
    col_count = client.get_collection(collection).count()
    if col_count == 0:
        raise RuntimeError(
            f"Collection '{collection}' exists but has 0 documents. "
            f"Please ingest data first."
        )

    docs = retriever.get_relevant_documents(question)
    context_str, citations = build_context_and_citations(docs)

    llm, tok = load_hf_llm(llm_model)

    # Build a chat-formatted prompt that Qwen expects
    user_block = f"""Question:
{question}

Context:
{context_str}

Answer (use citations like [1], [2] where appropriate):"""
    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTIONS},
        {"role": "user", "content": user_block},
    ]
    chat_prompt = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    out = llm(chat_prompt)[0]["generated_text"]

    # Strip the echoed prompt to keep only the model's answer (robustly)
    if out.startswith(chat_prompt):
        ans = out[len(chat_prompt):].strip()
    else:
        # Fallback if pipeline doesn't echo the prompt verbatim
        ans = out.replace(chat_prompt, "", 1).strip()

    return {
        "answer": ans,
        "citations": citations,  # structured data you can render under the answer
        "used_k": len(docs),
    }


# ---------- CLI ----------
def main():
    import argparse

    ap = argparse.ArgumentParser(
        description="RAG query with source citations over Chroma (Qwen local).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("question", type=str, help="Your question (quote it)")
    ap.add_argument("--persist-dir", default=str(DEFAULT_PERSIST), type=str,
                    help="Path to Chroma DB directory")
    ap.add_argument("--collection", default=DEFAULT_COLLECTION,
                    help="Chroma collection name")
    ap.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL,
                    help="HF embedding model name")
    ap.add_argument("--llm-model", default=DEFAULT_LLM_MODEL,
                    help="HF causal LM (chat) model name")
    ap.add_argument("--k", type=int, default=4,
                    help="Top-K documents to retrieve (MMR diversified)")
    args = ap.parse_args()

    persist_path = Path(args.persist_dir)
    print(f"CWD: {Path().resolve()}")
    print(f"Chroma dir exists: {persist_path.exists()} -> {persist_path.resolve()}")

    try:
        res = answer_question(
            question=args.question,
            persist_dir=persist_path,
            collection=args.collection,
            embed_model=args.embed_model,
            llm_model=args.llm_model,
            k=args.k,
        )
    except Exception as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)

    print("\n=== Answer ===")
    print(res["answer"])
    print("\n=== Citations ===")
    for c in res["citations"]:
        bits = [f'[{c["n"]}]', c["source"]]
        if c.get("page") is not None:
            bits.append(f'page={c["page"]}')
        if c.get("chunk_index") not in ("", None):
            bits.append(f'chunk_index={c["chunk_index"]}')
        print(" • " + " | ".join(bits))
    print("\nused_k:", res["used_k"])


if __name__ == "__main__":
    main()
# === Fast eval helpers (reuse retriever + LLM) ===============================
from typing import Dict, Any

def build_once(
    persist_dir: Path = DEFAULT_PERSIST,
    collection: str = DEFAULT_COLLECTION,
    embed_model: str = DEFAULT_EMBED_MODEL,
    llm_model: str = DEFAULT_LLM_MODEL,
    k: int = 4,
) -> Dict[str, Any]:

    retriever = build_retriever(persist_dir, collection, embed_model, k=k)
    llm, tok = load_hf_llm(llm_model)

    # sanity checks (same as answer_question)
    client = chromadb.PersistentClient(path=str(persist_dir))
    names = [c.name for c in client.list_collections()]
    if collection not in names:
        raise RuntimeError(
            f"Collection '{collection}' not found in {persist_dir}. Existing: {names}"
        )
    if client.get_collection(collection).count() == 0:
        raise RuntimeError(
            f"Collection '{collection}' exists but has 0 documents. Please ingest data first."
        )

    return {"retriever": retriever, "llm": llm, "tok": tok, "k": k}


def generate_answer_fast(question: str, ctx: Dict[str, Any]) -> Dict[str, Any]:

    # 1) retrieve
    docs = ctx["retriever"].get_relevant_documents(question)
    context_str, citations = build_context_and_citations(docs)

    # 2) prompt
    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTIONS},
        {
            "role": "user",
            "content": (
                f"Question:\n{question}\n\n"
                f"Context:\n{context_str}\n\n"
                "Answer (use citations like [1], [2] where appropriate):"
            ),
        },
    ]
    chat_prompt = ctx["tok"].apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # 3) generate
    out = ctx["llm"](chat_prompt)[0]["generated_text"]
    if out.startswith(chat_prompt):
        ans = out[len(chat_prompt):].strip()
    else:
        ans = out.replace(chat_prompt, "", 1).strip()

    # Normalize key names to what eval.py expects
    return {"answer": ans, "sources": citations}