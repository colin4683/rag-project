"""
generator.py
────────────
Wraps the Google Gemini API for answer generation.
Provides two modes:
  - generate()           RAG-augmented: answer using retrieved context
  - generate_zero_shot() Baseline: answer using only model training knowledge
"""

import time

import cursor
import google.generativeai as genai
from yaspin import yaspin
from yaspin.spinners import Spinners


class LLMGenerator:
    """Generates answers using Google Gemini via the google-generativeai SDK."""

    def __init__(
        self, api_key: str, model: str = "gemini-2.5-flash", max_tokens: int = 1024
    ):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name=model,
            generation_config=genai.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=0.2,
            ),
        )

    def generate(self, question: str, context: str) -> str:
        prompt = (
            "You are a helpful academic advisor assistant for UCF (University of Central Florida). "
            "Answer the student's question using ONLY the provided course catalog context. "
            "If the answer is not found in the context, say so clearly. "
            "Be concise, accurate, and cite the program name when relevant.\n\n"
            f"Context from UCF Course Catalog:\n{context}\n\n"
            f"Student Question: {question}\n\n"
            "Answer:"
        )
        with yaspin().magenta as sp:
            with cursor.HiddenCursor():
                sp.side = "right"
                sp.text = "Generating answer "
                time.sleep(3)
                response = self.model.generate_content(prompt)
        return response.text.strip()

    def generate_zero_shot(self, question: str) -> str:
        """Baseline: answer with NO retrieved context."""
        prompt = (
            "You are a helpful academic advisor for UCF. "
            "Answer the question using only your existing knowledge. "
            "Use 1-2 complete sentences. "
            "Do not use bullet points, lists, or preambles. "
            "If you are not sure, say you are not sure.\\n\\n"
            f"Question: {question}"
        )
        response = self.model.generate_content(prompt)
        return response.text.strip()
