"""
cli.py
──────
Interactive CLI for querying the UCF RAG system.

Modes
─────
  Interactive loop (default):
    python cli.py

  Single question:
    python cli.py --question "What are the CS degree requirements?"

  Inspect mode — shows retrieved chunks with scores, no LLM call:
    python cli.py --inspect "What are the CS degree requirements?"

  Build / rebuild the index from scratch:
    python cli.py --build

Flags
─────
  --k INT         Number of chunks to retrieve (default: from RAGConfig)
  --show-sources  Print the source URLs and chunk IDs for each answer
"""

import argparse
import os
import sys
import textwrap

from dotenv import dotenv_values, load_dotenv
from rich.console import Console
from rich.markdown import Markdown

load_dotenv()
from config import RAGConfig
from generator import LLMGenerator
from ingest import build_chunks, fetch_kuali_pages, load_chunks, save_chunks
from pipeline import RAGPipeline
from retriever import EmbeddingIndex

# Rich Console for markdown printing
console = Console()

# ── Kuali content IDs ─────────────────────────────────────────────────────────
KUALI_IDS: list[str] = [
    "66bcc88df93938001c54837f",  # Student Account Services - Tuition & Fees
    "60be66e79ef701001ca5b7c8",  # Mission Creed
    "60be67539ef701001ca5b7ca",  # President's Message
    "66bcc88df93938001c548380",  # Department of English
    "66bcc891f93938001c54839d",  # Department of History
    "66bcc893f93938001c5483b0",  # Department of Modern Languages and Literatures
    "66bcc890f93938001c54839b",  # Department of Philosophy
    "66bcc892f93938001c5483aa",  # Department of Writing and Rhetoric
    "66bcc88ff93938001c548392",  # Program in Women’s and Gender Studies
    "66bcc894f93938001c5483b8",  # School of Performing Arts
    "66bcc892f93938001c5483a8",  # School of Visual Arts and Design
    "66bcc891f93938001c54839e",  # Department of Economics
    "66bcc891f93938001c5483a0",  # Department of Finance
    "66bcc894f93938001c5483bb",  # Department of Integrated Business
    "66bcc890f93938001c548397",  # Department of Management
    "66bcc893f93938001c5483ae",  # Department of Marketing
    "66bcc893f93938001c5483ab",  # Dr. P. Phillips School of Real Estate
    "66bcc893f93938001c5483b3",  # Kenneth G. Dixon School of Accounting
    "66bcc891f93938001c5483a3",  # Department of Counselor Education and School Psychology
    "66bcc894f93938001c5483ba",  # Department of Criminal Justice
    "66bcc891f93938001c54839c",  # Department of Educational Leadership and Higher Education
    "66bcc890f93938001c548398",  # Department of Learning Sciences and Educational Research
    "66bcc890f93938001c548396",  # Department of Legal Studies
    "685454056db9e229ba86c51a",  # School of Interdisciplinary Studies
    "66bcc890f93938001c548395",  # School of Public Administration
    "66bcc895f93938001c5483be",  # School of Teacher Education
    "66bcc88ff93938001c548390",  # Department of Civil, Environmental, and Construction Engineering
    "66bcc894f93938001c5483bd",  # Department of Computer Science
    "66bcc890f93938001c548399",  # Department of Electrical and Computer Engineering
    "66bcc890f93938001c548393",  # Department of Industrial Engineering and Management Systems
    "66bcc88ff93938001c54838f",  # Department of Materials Science and Engineering
    "66bcc893f93938001c5483ac",  # Department of Mechanical and Aerospace Engineering
    "66bcc893f93938001c5483af",  # Reserve Officer Training Corp: Air Force ROTC
    "66bcc890f93938001c548394",  # Reserve Officer Training Corp: Army ROTC
    "66bcc891f93938001c54839f",  # Department of Health Sciences
    "66bcc893f93938001c5483b2",  # School of Communication Sciences and Disorders
    "66bcc88ff93938001c54838e",  # School of Global Health Management and Informatics
    "66bcc893f93938001c5483b1",  # School of Kinesiology and Physical Therapy
    "66bcc88ff93938001c548391",  # School of Social Work
    "66bcc893f93938001c5483b4",  # Burnett School of Biomedical Sciences
    "66bcc892f93938001c5483a6",  # College of Nursing.
    "66bcc894f93938001c5483b6",  # College of Optics and Photonics.
    "66bcc890f93938001c54839a",  # Department of Anthropology
    "66bcc894f93938001c5483b9",  # Department of Biology
    "66bcc895f93938001c5483bf",  # Department of Chemistry
    "66bcc894f93938001c5483bc",  # Department of Mathematics
    "66bcc897f93938001c5483d4",  # Department of Physics
    "66bcc895f93938001c5483c2",  # Department of Psychology
    "66bcc896f93938001c5483cc",  # Department of Sociology
    "66bcc896f93938001c5483c8",  # Department of Statistics and Data Science
    "66bcc895f93938001c5483c3",  # Nicholson School of Communication and Media
    "66bcc896f93938001c5483c9",  # School of Politics, Security, and International Affairs
    "66bcc895f93938001c5483c5",  # Department of Foodservices and Lodging Management
    "66bcc896f93938001c5483d0",  # Department of Hospitality Services
    "66bcc896f93938001c5483ca",  # Department of Tourism, Events and Attractions
]


# ── Index bootstrap ────────────────────────────────────────────────────────────
def load_or_build_pipeline(
    config: RAGConfig,
    api_key: str,
    force_rebuild: bool = False,
    force_embed: bool = False,
) -> RAGPipeline:
    index = EmbeddingIndex(model_name=config.embedding_model)

    if (
        not force_rebuild
        and os.path.exists(config.index_path)
        and os.path.exists(config.chunks_path)
    ):
        print("Loading existing index...")
        chunks = load_chunks(config.chunks_path)
        if force_embed:
            print("Rebuilding embeddings...")
            index.build(chunks, batch_size=16)
        else:
            index.load(config.index_path, chunks)
    else:
        print("Building index from Kuali API...")
        pages = fetch_kuali_pages(KUALI_IDS)
        chunks = build_chunks(pages, config)
        save_chunks(chunks, config.chunks_path)
        index.build(chunks, batch_size=16)
        index.save(config.index_path)

    llm = LLMGenerator(
        api_key=api_key, model=config.llm_model, max_tokens=config.max_tokens
    )
    return RAGPipeline(config, index, llm)


# ── Display helpers ────────────────────────────────────────────────────────────
def print_answer(result, show_sources: bool = False) -> None:
    print(f"\n{'─' * 60}")
    print(f"Answer:\n")
    console.print(Markdown(result.answer))

    if show_sources:
        print(f"\nSources ({len(result.retrieved_docs)} chunks retrieved):")
        for doc in result.retrieved_docs:
            print(
                f"  [{doc.chunk.chunk_id}] {doc.chunk.program_name} "
                f"(score={doc.score:.4f})"
            )
            print(f"       {doc.chunk.source_url}")
    print(f"{'─' * 60}\n")


def inspect_retrieval(index: EmbeddingIndex, question: str, k: int) -> None:
    """Show the raw retrieved chunks without calling the LLM."""

    def _extract_labeled_fields(text: str) -> dict[str, str]:
        fields: dict[str, str] = {}
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or ":" not in line:
                continue
            label, value = line.split(":", 1)
            label = label.strip()
            value = value.strip()
            if label and value and label not in fields:
                fields[label] = value
        return fields

    def _compact_preview(text: str, max_len: int = 260) -> str:
        compact = " ".join(text.split())
        if len(compact) <= max_len:
            return compact
        return compact[: max_len - 3].rstrip() + "..."

    def _print_field(label: str, value: str) -> None:
        print(
            textwrap.fill(
                value,
                width=110,
                initial_indent=f"       {label:<12}: ",
                subsequent_indent=" " * 26,
            )
        )

    results = index.search(question, k=k)
    print(f'\nTop {k} retrieved chunks for: "{question}"\n')
    for i, doc in enumerate(results, start=1):
        fields = _extract_labeled_fields(doc.chunk.text)
        print(f"{'-' * 112}")
        print(f"  [{i:>2}] chunk_id={doc.chunk.chunk_id}  score={doc.score:.4f}")
        _print_field("Program", doc.chunk.program_name)
        _print_field("Source", doc.chunk.source_url)

        # Prefer course-style fields when present in chunk text.
        preferred_fields = [
            "Course Code",
            "Title",
            "Subject",
            "Credits",
            "Terms Offered",
            "Prerequisites",
            "Description",
        ]
        for key in preferred_fields:
            if key in fields:
                _print_field(key, fields[key])

        _print_field("Preview", _compact_preview(doc.chunk.text))
        print()


# ── CLI entry point ────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="UCF Course Catalog RAG — question answering CLI"
    )
    parser.add_argument("--question", type=str, help="Ask a single question and exit")
    parser.add_argument(
        "--inspect", type=str, help="Show retrieved chunks only (no LLM)"
    )
    parser.add_argument("--build", action="store_true", help="Force rebuild the index")
    parser.add_argument(
        "--embed", action="store_true", help="Force rebuild the embeddings"
    )
    parser.add_argument(
        "--k", type=int, default=None, help="Number of chunks to retrieve"
    )
    parser.add_argument(
        "--show-sources", action="store_true", help="Print source URLs with answers"
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        sys.exit("Error: set the OPENAI_API_KEY environment variable.")

    config = RAGConfig()
    if args.k:
        config.k = args.k

    pipeline = load_or_build_pipeline(
        config, api_key, force_rebuild=args.build, force_embed=args.embed
    )

    # ── Inspect mode ──
    if args.inspect:
        inspect_retrieval(pipeline.index, args.inspect, k=config.k)
        return

    # ── Single question mode ──
    if args.question:
        result = pipeline.query(args.question)
        print_answer(result, show_sources=args.show_sources)
        return

    # Clear the screen
    os.system("cls" if os.name == "nt" else "clear")

    # ── Interactive loop ──
    print("\nUCF Undergraduate Course Catalog RAG")
    print("Type your question and press Enter. Type 'quit' or 'exit' to stop.")
    print("Prefix with '!inspect ' to see raw retrieved chunks.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break
        if user_input.lower() in {"clear", "cls"}:
            os.system("cls" if os.name == "nt" else "clear")
            continue
        if user_input.lower().startswith("!inspect "):
            question = user_input[len("!inspect ") :]
            inspect_retrieval(pipeline.index, question, k=config.k)
            continue

        result = pipeline.query(user_input)
        print_answer(result, show_sources=args.show_sources)


if __name__ == "__main__":
    main()
