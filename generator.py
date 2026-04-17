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
            "You are a helpful academic catalog assistant for UCF (University of Central Florida). "
            "Answer the student's question using ONLY the provided course catalog context. "
            "If the question needs to be more specific in order to find the answer or to provide more context, prompt the student to do so."
            "Do NOT use outside knowledge.\n\n"
            "Rules:\n"
            "1. Prefer a grounded partial answer over asking the student for more clarification.\n"
            "2. If the student asks what courses are available next, treat any coureses listed by the student as already completed or in progress and assume they are passed, unless the question says otherwise.\n"
            "3. Use the retrieved context to identify program requirements, prerequisite relationships, course availability, and track-specific rules.\n"
            "4. If the context supports only a partial answer, say what is supported and then state exactly what is missing.\n"
            "5. If the answer is not supported by the context, clearly say that the catalog context does not provided enough information.\n"
            "6. Do not invent course relationships, semester plans, or requirements.\n"
            "7. When possible, mention the specific program, track, and course codes used in your reasoning.\n\n"
            "For planning-style questions, follow this process:\n"
            "- Identify the student's program and track.\n"
            "- Identify the completed courses mentioned in the question.\n"
            "- From the context, determine which future courses list those completed courses as prerequisites, or which required next-step courses are now available.\n"
            "- Return the likely next available coureses and briefly explain why each appears eligible.\n"
            "- If multiple possibilities exist, present them as likely options rather than certainties.\n\n"
            "Output format:\n"
            "- Start with a direct answer.\n"
            "- Then give 2-6 bullet points with grounded evidence.\n"
            "- End with a short note about missing information only if needed.\n\n"
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
