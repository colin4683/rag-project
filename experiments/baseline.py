"""
experiments/baseline.py
────────────────────────
Zero-shot baseline study.

Sends each evaluation question directly to the LLM with no retrieved context,
then saves the side-by-side comparison with RAG answers to JSON.
"""

import json
import os
import sys

from dotenv import dotenv_values, load_dotenv

load_dotenv()
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import RAGConfig
from generator import LLMGenerator
from ingest import load_chunks
from pipeline import RAGPipeline
from retriever import EmbeddingIndex

# ── Questions to compare ───────────────────────────────────────────────────────
# These should be the same questions used in evaluation dataset.

BASELINE_QUESTIONS = [
    "What is the minimum GPA required for the Computer Science BS program at UCF?",
    "How many total credit hours are required to graduate with a Psychology BS?",
    "What elective tracks are available in the Electrical Engineering BS?",
    "What is the UCF creed?",
    "What are the prerequisites for Geotechnical Methods and Case Histories?",
    # TODO: Add more questions here
]

OUTPUT_FILE = "results/baseline_comparison.json"


def run_baseline(pipeline: RAGPipeline, llm: LLMGenerator) -> list[dict]:
    comparisons: list[dict] = []

    for i, question in enumerate(BASELINE_QUESTIONS):
        print(f"\n[{i + 1}/{len(BASELINE_QUESTIONS)}] {question}")
        try:
            rag_result = pipeline.query(question)
            zero_shot = llm.generate_zero_shot(question)

            print(f"  RAG       : {rag_result.answer[:100]}...")
            print(f"  Zero-shot : {zero_shot[:100]}...")

            comparisons.append(
                {
                    "question": question,
                    "rag_answer": rag_result.answer,
                    "zero_shot_answer": zero_shot,
                    "retrieved_sources": [
                        r.chunk.source_url for r in rag_result.retrieved_docs
                    ],
                }
            )
        except Exception as e:
            print(f"  Error    : {str(e)}")

    return comparisons


def main() -> None:
    config = RAGConfig()
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise EnvironmentError("Set the OPENAI_API_KEY environment variable.")

    print("Loading index and chunks...")
    chunks = load_chunks(config.chunks_path)
    index = EmbeddingIndex(model_name=config.embedding_model)
    index.load(config.index_path, chunks)

    llm = LLMGenerator(api_key=api_key, model=config.llm_model)
    pipeline = RAGPipeline(config, index, llm)

    print("\n── Baseline Comparison ─────────────────────────────────────")
    results = run_baseline(pipeline, llm)

    os.makedirs("results", exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} comparisons → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
