"""
config.py
─────────
Shared dataclasses used across the entire project.
Import from here rather than from any specific module to avoid circular imports.
"""

from dataclasses import dataclass, field


@dataclass
class RAGConfig:
    # Retrieval
    k: int = 10  # Number of documents to retrieve
    chunk_size: int = 300  # Max words per chunk
    chunk_overlap: int = 50  # Overlap between consecutive chunks

    # Models
    embedding_model: str = "all-MiniLM-L6-v2"
    llm_model: str = "gpt-4o-mini"
    max_tokens: int = 2048

    # Paths
    index_path: str = "generated/faiss_index.bin"
    chunks_path: str = "generated/chunks.json"

    # Evaluation
    recall_k_values: list = field(default_factory=lambda: [1, 3, 5, 10])


@dataclass
class Chunk:
    chunk_id: int
    source_url: str
    program_name: str
    text: str


@dataclass
class RetrievedDoc:
    chunk: "Chunk"
    score: float  # L2 distance — lower means more similar


@dataclass
class RAGResult:
    question: str
    retrieved_docs: list[RetrievedDoc]
    answer: str
    context_used: str


@dataclass
class EvalSample:
    question: str
    relevant_chunk_ids: list[int]  # ground-truth chunk IDs
    reference_answer: str  # human-written reference answer
