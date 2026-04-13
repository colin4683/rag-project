"""
retriever.py
────────────
Embedding model (Sentence-BERT) + FAISS index.
Handles encoding chunks, building/saving/loading the index,
and searching for the top-k most similar chunks at query time.
"""

from typing import Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config import Chunk, RetrievedDoc


class EmbeddingIndex:
    """
    Wraps Sentence-BERT + FAISS for similarity search.
    Uses a flat L2 index (exact search) — appropriate for ~50 pages of chunks.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.index: Optional[faiss.IndexFlatL2] = None
        self.chunks: list[Chunk] = []

    def build(self, chunks: list[Chunk], batch_size: int = 16):
        """Encode all chunks and build FAISS index."""
        self.chunks = chunks
        texts = [c.text for c in chunks]

        print(f"Encoding {len(texts)} chunks (batch_size={batch_size})...")
        embeddings = self.model.encode(
            texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True
        ).astype("float32")

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(x=np.asarray(embeddings))
        print(f"FAISS index built: {self.index.ntotal} vectors, dim={dim}")

    def save(self, index_path: str):
        faiss.write_index(self.index, index_path)
        print(f"Saved FAISS index → {index_path}")

    def load(self, index_path: str, chunks: list[Chunk]):
        self.index = faiss.read_index(index_path)
        self.chunks = chunks
        print(f"Loaded FAISS index: {self.index.ntotal} vectors")

    def search(self, query: str, k: int) -> list[RetrievedDoc]:
        """Return top-k most similar chunks for a query."""
        if self.index is None:
            raise RuntimeError("Index is not built yet. Call build() or load() first.")
        q_emb = self.model.encode([query], convert_to_numpy=True).astype("float32")
        distances, indices = self.index.search(x=np.asarray(q_emb), k=k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.chunks):
                results.append(RetrievedDoc(chunk=self.chunks[idx], score=float(dist)))
        return results
