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
EVAL_SAMPLES: list[EvalSample] = [
    EvalSample(
        question="What is the minimum GPA required for the Computer Science BS?",
        relevant_chunk_ids=[1503],
        reference_answer="A minimum 2.500 GPA is required.",
    ),
    EvalSample(
        question="Can UCF Computer Science students specialize in Cybersecurity?",
        relevant_chunk_ids=[1493, 1542],
        reference_answer="Yes, the program includes a cybersecurity-related specialization or track.",
    ),
    EvalSample(
        question="How many total GEP credits are required for the Psychology (B.S.), Clinical Psychology Track?",
        relevant_chunk_ids=[*range(2181, 2196)],
        reference_answer="The program requires the specified total GEP credits listed in the catalog.",
    ),
    EvalSample(
        question="What are the total credit hours required for the Computer Science BS?",
        relevant_chunk_ids=[1514],
        reference_answer="The degree requires the catalog-listed total credit hours.",
    ),
    EvalSample(
        question="What are the required Common Program Prerequisites for the Chemistry (B.S.), Biochemistry Track?",
        relevant_chunk_ids=[*range(2102, 2125)],
        reference_answer="",
    ),
    EvalSample(
        question="What is the UCF creed?",
        relevant_chunk_ids=[12],
        reference_answer="",
    ),
    EvalSample(
        question="What are the prerequisites for Geotechnical Methods and Case Histories?",
        relevant_chunk_ids=[3295],
        reference_answer="",
    ),
    EvalSample(
        question="What do you need to be classified as an upper division student?",
        relevant_chunk_ids=[2628],
        reference_answer="To be classified as an upper division student at the University of Central Florida, a student must complete the following: A minimum of 60 credit hours of academic work; The English and mathematics requirements of the Gordon Rule; One year of college instruction in a single foreign language. (This requirement applies to those students admitted to the University without the required two units of foreign language in high school.)",
    ),
    EvalSample(
        question="What are the requirements to make it on the Dean's List?",
        relevant_chunk_ids=[2592],
        reference_answer="",
    ),
    EvalSample(
        question="What happens if I repeat a course and how does it affect GPA?",
        relevant_chunk_ids=[],
        reference_answer="",
    ),
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
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise EnvironmentError("Set the OPENAI_API_KEY environment variable.")

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
