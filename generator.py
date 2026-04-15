"""
generator.py
────────────
Wraps the OpenAI API for answer generation.
Provides two modes:
  - generate()           RAG-augmented: answer using retrieved context
  - generate_zero_shot() Baseline: answer using only model training knowledge
"""

import time

import cursor
from openai import OpenAI
from yaspin import yaspin
from yaspin.spinners import Spinners


class LLMGenerator:
    """Generates answers using OpenAI chat completions."""

    def __init__(
        self, api_key: str, model: str = "gpt-4o-mini", max_tokens: int = 1024
    ):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    def _complete(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=0.2,
        )
        return (response.choices[0].message.content or "").strip()

    def generate(self, question: str, context: str) -> str:
        prompt = (
            "You are a helpful academic advisor assistant for UCF (University of Central Florida). "
            "Answer the student's question using ONLY the provided course catalog context. "
            "If the question needs to be more specific in order to find the answer or to provide more context, prompt the student to do so."
            "If the answer is not found in the context, say so clearly. "
            "Be concise, accurate, and cite the program name when relevant.\n\n"
            f"Context from UCF Course Catalog:\n{context}\n\n"
            f"Student Question: {question}\n\n"
            "Answer:"
        )
        with yaspin().magenta as sp:
            with cursor.HiddenCursor():
                sp.side = "right"
                sp.text = "Searching for an answer "
                time.sleep(3)
                return self._complete(prompt)

    def generate_zero_shot(self, question: str) -> str:
        """Baseline: answer with NO retrieved context."""
        prompt = (
            "You are a helpful academic advisor for UCF (University of Central Florida). "
            "Answer the question using only your existing knowledge. "
            "Use 1-2 complete sentences. "
            "If you are not sure, say you are not sure.\\n\\n"
            f"Question: {question}"
        )
        return self._complete(prompt)
