"""
experiments/ablation.py
────────────────────────
Ablation study: vary k and measure Recall@k across the evaluation set.

Hypothesis:
  - k too small → relevant chunks missed entirely
  - k too large → irrelevant chunks pollute the context, degrading answer quality
"""

import json
import os
import sys

from dotenv import dotenv_values, load_dotenv

load_dotenv()
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import EvalSample, RAGConfig
from generator import LLMGenerator
from ingest import load_chunks
from pipeline import RAGPipeline
from retriever import EmbeddingIndex

# ── Evaluation dataset ─────────────────────────────────────────────────────────
# Fill in relevant_chunk_ids after building the index.
# Run:  python cli.py --inspect [question]
# to see which chunk IDs are retrieved, then manually mark the correct ones (or the ones you think are relevant)

EVAL_SAMPLES: list[EvalSample] = [
    EvalSample(
        question="What is the minimum GPA required for the Computer Science BS?",
        relevant_chunk_ids=[1133, 1163],  # ← fill in after indexing
        reference_answer="A minimum 2.500 GPA is required",
    ),
    EvalSample(
        question="Can UCF Computer Science students specialize in Cybersecurity?",
        relevant_chunk_ids=[1125, 1128, 1160, 1126, 1129],  # ← fill in after indexing
        reference_answer="Yes.",
    ),
    EvalSample(
        question="How many total GEP credits are required for the Psychology (B.S.), Clinical Psychology Track?",
        relevant_chunk_ids=[1162, 1658, 1669],  # ← fill in after indexing
        reference_answer="",
    ),
    # TODO: Add more samples here
]

K_VALUES = [1, 3, 5, 10, 20]
OUTPUT_FILE = "results/ablation_results.json"


def recall_at_k(retrieved_ids: list[int], relevant_ids: list[int], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    hits = sum(1 for rid in relevant_ids if rid in top_k)
    return hits / len(relevant_ids)


def run_ablation(index: EmbeddingIndex, samples: list[EvalSample]) -> dict:
    results: dict[int, float] = {}

    print("\n── Ablation Study: Varying k ──────────────────────────────")
    for k in K_VALUES:
        scores: list[float] = []
        for sample in samples:
            retrieved = index.search(sample.question, k=k)
            retrieved_ids = [r.chunk.chunk_id for r in retrieved]
            scores.append(recall_at_k(retrieved_ids, sample.relevant_chunk_ids, k))
        avg = float(np.mean(scores)) if scores else 0.0
        results[k] = avg
        bar = "█" * int(avg * 20)
        print(f"  k={k:2d}  {bar:<20}  Recall@{k} = {avg:.4f}")
    print("────────────────────────────────────────────────────────────\n")
    return results


def main() -> None:
    config = RAGConfig()
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise EnvironmentError("Set the GEMINI_API_KEY environment variable.")

    print("Loading index and chunks...")
    chunks = load_chunks(config.chunks_path)
    index = EmbeddingIndex(model_name=config.embedding_model)
    index.load(config.index_path, chunks)

    # Warn if no chunk IDs have been annotated yet
    unannotated = [s for s in EVAL_SAMPLES if not s.relevant_chunk_ids]
    if unannotated:
        print(
            f"[WARN] {len(unannotated)}/{len(EVAL_SAMPLES)} samples have no "
            "relevant_chunk_ids — Recall@k will be 0 for those. "
            "Run cli.py --inspect to find the correct IDs."
        )

    results = run_ablation(index, EVAL_SAMPLES)

    os.makedirs("results", exist_ok=True)
    output = {
        "k_values": K_VALUES,
        "recall_at_k": results,
        "num_samples": len(EVAL_SAMPLES),
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved ablation results → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
