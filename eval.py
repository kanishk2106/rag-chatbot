#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, time, argparse, re, difflib
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))  # repo root (src is a package)

from src.qa import build_once, generate_answer_fast  # uses your qa.py

# --- Questions (unchanged content) ---
QUESTIONS = [
    {"q": "What are the three mathematical foundations essential for understanding deep learning?",
     "ref": "Probability theory, linear algebra, and multivariate calculus."},
    {"q": "What is the main purpose of deep learning?",
     "ref": "To use multilayered neural networks trained on large datasets to solve complex information processing tasks."},
    {"q": "What distinguishes epistemic uncertainty from aleatoric uncertainty?",
     "ref": "Epistemic uncertainty arises from limited data and can be reduced with more data, whereas aleatoric uncertainty stems from inherent randomness and cannot be reduced."},
    {"q": "How is the information content of an event defined in information theory?",
     "ref": "It is defined as h(x) = -log2 p(x), where p(x) is the probability of the event."},
    {"q": "What is the key idea behind representation learning in deep neural networks?",
     "ref": "The network learns to transform input data into semantically meaningful representations that simplify the task for subsequent layers."},
    {"q": "What is the simplest model used for regression in a neural network?",
     "ref": "A linear regression model that uses a linear combination of input variables."},
    {"q": "What are residual connections and why are they important in deep networks?",
     "ref": "Residual connections help mitigate vanishing gradients by facilitating the training of networks with hundreds of layers."},
    {"q": "What is automatic differentiation and why is it important?",
     "ref": "It is a technique that automatically generates code to compute derivatives accurately and efficiently, enabling rapid experimentation with neural architectures."},
    {"q": "What are the three key ideas underlying Variational Autoencoders (VAEs)?",
     "ref": "Using the ELBO to approximate the likelihood, amortized inference with an encoder network, and the reparameterization trick for tractable training."},
    {"q": "What is the simplest measure of a classifier’s performance and what is a potential drawback?",
     "ref": "Accuracy is the simplest measure, but it can be misleading in imbalanced datasets."}
]

# --- Quality helpers ---
def substring_score(answer: str, ref: str) -> float:
    if not ref:
        return float("nan")
    return 1.0 if ref.lower() in (answer or "").lower() else 0.0

def _normalize(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s\-./+=]", "", s)  # keep a few benign symbols
    return s.strip()

def fuzzy_score(answer: str, ref: str) -> float:
    if not ref or not answer:
        return float("nan")
    return difflib.SequenceMatcher(None, _normalize(answer), _normalize(ref)).ratio()

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    def semantic_score(answer: str, ref: str) -> float:
        if not ref or not answer:
            return float("nan")
        ea, er = _model.encode([answer, ref])
        ea = ea / (np.linalg.norm(ea) + 1e-9)
        er = er / (np.linalg.norm(er) + 1e-9)
        return float((ea * er).sum())
except ImportError:
    def semantic_score(answer: str, ref: str) -> float:
        return float("nan")

def main():
    # ---- CLI for eval ----
    ap = argparse.ArgumentParser(
        description="Evaluate RAG over a Chroma collection with fixed questions."
    )
    ap.add_argument("--persist-dir", type=Path, required=True,
                    help="Path to Chroma persistence directory (e.g. ./.chroma)")
    ap.add_argument("--collection", type=str, required=True,
                    help="Chroma collection name (e.g. rag_check)")
    ap.add_argument("--embed-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                    help="HuggingFace embedding model")
    ap.add_argument("--llm-model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
                    help="HF causal LM/chat model used for generation")
    ap.add_argument("--k", type=int, default=4,
                    help="Top-K documents to retrieve per question")
    ap.add_argument("--csv", type=Path, default=ROOT / "eval_results.csv",
                    help="Output CSV path")
    ap.add_argument("--max-qs", type=int, default=len(QUESTIONS),
                    help="Evaluate only the first N questions")
    args = ap.parse_args()

    print("Building retriever + LLM once...")
    ctx = build_once(
        persist_dir=args.persist_dir,
        collection=args.collection,
        embed_model=args.embed_model,
        llm_model=args.llm_model,
        k=args.k,
    )

    rows = []
    for i, item in enumerate(QUESTIONS[: args.max_qs], start=1):
        q, ref = item["q"], item["ref"]
        print(f"\n[{i}] {q}")
        t0 = time.perf_counter()
        out = generate_answer_fast(q, ctx)
        dt_ms = (time.perf_counter() - t0) * 1000

        ans = out.get("answer", "")
        srcs = out.get("sources", [])
        scores = {
            "substring": substring_score(ans, ref),
            "semantic": semantic_score(ans, ref),
            "fuzzy": fuzzy_score(ans, ref),
        }

        rows.append({
            "idx": i,
            "question": q,
            "ref": ref,
            "answer": ans,
            "latency_ms": round(dt_ms, 1),
            "score_substring": scores["substring"],
            "score_semantic": scores["semantic"],
            "score_fuzzy": scores["fuzzy"],
            "n_sources": len(srcs),
            "sources": "; ".join(str(s) for s in srcs),
        })

        print(
            f" → latency {dt_ms:.1f} ms, "
            f"substring={scores['substring']}, "
            f"semantic={scores['semantic']}, "
            f"fuzzy={scores['fuzzy']:.3f}"
        )

    df = pd.DataFrame(rows)
    df.to_csv(args.csv, index=False)

    print("\n=== Summary ===")
    print(df[["latency_ms", "score_substring", "score_semantic", "score_fuzzy"]].describe())
    print(f"\nSaved results to {args.csv.resolve()}")

if __name__ == "__main__":
    main()
