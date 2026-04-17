"""
retriever.py
────────────
Embedding model (Sentence-BERT) + FAISS index.
Handles encoding chunks, building/saving/loading the index,
and searching for the top-k most similar chunks at query time.
"""

import logging
import warnings
from typing import Any, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers.utils import logging as transformers_logging
from yaspin import yaspin

from config import Chunk, RetrievedDoc

warnings.filterwarnings("ignore", module="transformers")
warnings.filterwarnings("ignore", module="huggingface_hub")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
transformers_logging.set_verbosity_error()


class EmbeddingIndex:
    """
    Wraps Sentence-BERT + FAISS for similarity search.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        with yaspin(
            text=f"Loading embedding model ({model_name})", color="magenta"
        ) as sp:
            self.model = SentenceTransformer(model_name)
            sp.ok("✓")
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
        self.index.add(x=np.asarray(embeddings))  # pyright: ignore[reportCallIssue]
        print(f"FAISS index built: {self.index.ntotal} vectors, dim={dim}")

    def save(self, index_path: str):
        faiss.write_index(self.index, index_path)
        print(f"Saved FAISS index → {index_path}")

    def load(self, index_path: str, chunks: list[Chunk]):
        self.index = faiss.read_index(index_path)
        self.chunks = chunks
        print(f"Loaded FAISS index: {self.index.ntotal} vectors")  # pyright: ignore[reportCallIssue]

    def search(self, query: str, k: int) -> list[RetrievedDoc]:
        """Return top-k most similar chunks for a query."""
        if self.index is None:
            raise RuntimeError("Index is not built yet. Call build() or load() first.")
        q_emb = self.model.encode([query], convert_to_numpy=True).astype("float32")
        index = self.index
        distances, indices = index.search(x=np.asarray(q_emb), k=k)  # pyright: ignore[reportCallIssue]
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.chunks):
                results.append(RetrievedDoc(chunk=self.chunks[idx], score=float(dist)))
        return results
