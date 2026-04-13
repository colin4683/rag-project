"""
pipeline.py
───────────
Wires together the retriever and generator into a single query interface.
"""

from typing import Optional

from config import RAGConfig, RAGResult
from generator import LLMGenerator
from retriever import EmbeddingIndex


class RAGPipeline:
    def __init__(
        self, config: RAGConfig, embedding_index: EmbeddingIndex, llm: LLMGenerator
    ):
        self.config = config
        self.index = embedding_index
        self.llm = llm

    def query(self, question: str, k: Optional[int] = None) -> RAGResult:
        k = k or self.config.k
        retrieved = self.index.search(question, k=k)
        print(f"Retrieved {len(retrieved)} documents for question: {question}")
        # Build context string from retrieved chunks
        context_parts = []
        for i, doc in enumerate(retrieved):
            context_parts.append(
                f"[{i + 1}] Program: {doc.chunk.program_name}\n"
                f"Source: {doc.chunk.source_url}\n"
                f"{doc.chunk.text}"
            )
        context = "\n\n---\n\n".join(context_parts)

        # answer = self.llm.generate(question, context)
        answer = "TODO"
        return RAGResult(
            question=question,
            retrieved_docs=retrieved,
            answer=answer,
            context_used=context,
        )
