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
        relevant_chunk_ids=[1509, 1542],
        reference_answer="Yes. The Computer Science BS includes a Cybersecurity technical-elective area, and UCF also offers a Cyber Operations Certificate recommended for CS students.",
    ),
    EvalSample(
        question="How many total GEP credits are required for the Psychology (B.S.), Clinical Psychology Track?",
        relevant_chunk_ids=[2192, 2194],
        reference_answer="The program follows the 36-hour GEP structure, with the catalog listing a GEP grand total of 36-41 credits depending on course choices.",
    ),
    EvalSample(
        question="What are the total credit hours required for the Computer Science BS?",
        relevant_chunk_ids=[1505],
        reference_answer="Total Undergraduate Credit Hours Required: 120.",
    ),
    EvalSample(
        question="What are the required Common Program Prerequisites for the Chemistry (B.S.), Biochemistry Track?",
        relevant_chunk_ids=[2108],
        reference_answer="CHM 2045C, CHM 2046, CHM 2046L, CHM 2210, CHM 2211, CHM 2211L, MAC 2312, and PHY 2049C (or PHY 2049 + PHY 2049L), plus MAC 2311C and PHY 2048C identified via GEP.",
    ),
    EvalSample(
        question="What is the UCF creed?",
        relevant_chunk_ids=[12],
        reference_answer="",
    ),
    EvalSample(
        question="What are the prerequisites for Geotechnical Methods and Case Histories?",
        relevant_chunk_ids=[3295],
        reference_answer="CEG4011C (Geotechnical Engineering I) and CGN3405 (Applied Numerical Methods for Civil Engineering), each with a minimum grade of C.",
    ),
    EvalSample(
        question="What do you need to be classified as an upper division student?",
        relevant_chunk_ids=[2628],
        reference_answer="To be classified as an upper division student at the University of Central Florida, a student must complete the following: A minimum of 60 credit hours of academic work; The English and mathematics requirements of the Gordon Rule; One year of college instruction in a single foreign language. (This requirement applies to those students admitted to the University without the required two units of foreign language in high school.)",
    ),
    EvalSample(
        question="What are the requirements to make it on the Dean's List?",
        relevant_chunk_ids=[2592],
        reference_answer="A minimum 3.4 GPA, at least 12 credit hours, no grade below C, and no I, U, N, NC, or WF grades in the term.",
    ),
    EvalSample(
        question="What happens if I repeat a course and how does it affect GPA?",
        relevant_chunk_ids=[2668, 2669, 2677],
        reference_answer="Without grade forgiveness, UCF applies the best grade from the first two attempts; for three or more attempts, the last attempt applies to degree/program requirements. With approved grade forgiveness, the repeated grade is what counts in GPA while all attempts remain on the transcript.",
    ),
    EvalSample(
        question="How many courses can a student use Grade Forgiveness for at UCF?",
        relevant_chunk_ids=[2668],
        reference_answer="Grade Forgiveness is limited to two courses.",
    ),
    EvalSample(
        question="What is the minimum separation period required to apply for Academic Amnesty?",
        relevant_chunk_ids=[2590],
        reference_answer="The student must be separated from UCF for five or more years.",
    ),
    EvalSample(
        question="What are the prerequisites for CDA3103C Computer Logic and Organization?",
        relevant_chunk_ids=[3292],
        reference_answer="COP3223C Introduction to Programming with C with a minimum grade of C.",
    ),
    EvalSample(
        question="What terms is CEG4011C Geotechnical Engineering I offered?",
        relevant_chunk_ids=[3293],
        reference_answer="Every Semester.",
    ),
    EvalSample(
        question="What is the prerequisite for CEG4012 Geotechnical Engineering II?",
        relevant_chunk_ids=[3294],
        reference_answer="CEG4011C Geotechnical Engineering I with a minimum grade of C.",
    ),
    EvalSample(
        question="What terms is CEG4203 Geotechnical Methods and Case Histories offered?",
        relevant_chunk_ids=[3295],
        reference_answer="Spring.",
    ),
    EvalSample(
        question="What are the prerequisites for CEG4801C Geotechnical Engineering Design?",
        relevant_chunk_ids=[3296],
        reference_answer="CEG4012 Geotechnical Engineering II and CGN3090 Engineering Reports & Liability, each with a minimum grade of C.",
    ),
    EvalSample(
        question="What is the prerequisite for CEN4360 Mobile Device Software Development?",
        relevant_chunk_ids=[3297],
        reference_answer="COP3330 Object Oriented Programming (or consent of instructor).",
    ),
    EvalSample(
        question="What are the prerequisites for CES4743C Structural Design?",
        relevant_chunk_ids=[3302],
        reference_answer="CES4605, CES4702, and CGN3090, all with grade C (2.0) or better.",
    ),
    EvalSample(
        question="How many credits is CGN3090 Engineering Reports & Liability?",
        relevant_chunk_ids=[3303],
        reference_answer="2 credits.",
    ),
    EvalSample(
        question="What are the prerequisites for CGN3405 Applied Numerical Methods for Civil Engineering?",
        relevant_chunk_ids=[3304],
        reference_answer="MAC2311C, MAC2312, and MAC2313 (all with minimum C), plus PHY2048C or PHY2048+PHY2048L, and CHS1440C or CHM2045C with minimum C.",
    ),
    EvalSample(
        question="What terms is CGN4808C CECE Capstone Design offered?",
        relevant_chunk_ids=[3309],
        reference_answer="Every Semester.",
    ),
    EvalSample(
        question="Is CGS1060C Introduction to Computers open to Computer Science majors?",
        relevant_chunk_ids=[3310],
        reference_answer="No, it is not open to Computer Science majors.",
    ),
    EvalSample(
        question="What terms is CGS2545C Database Concepts offered?",
        relevant_chunk_ids=[3312],
        reference_answer="Spring and Fall.",
    ),
]

K_VALUES = [1, 3, 5, 10, 20, 30]
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
