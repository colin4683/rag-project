"""
Wires together query processing, retrieval, and generation.
"""

from typing import Optional

from config import RAGConfig, RAGResult, RetrievedDoc
from generator import LLMGenerator
from query import ParsedQuery, QueryIntent, parse_query, retrieve_for_parsed_query
from retriever import EmbeddingIndex

# Keywords that indicate a scheduling / planning question
_PLANNING_KEYWORDS = [
    "next semester",
    "schedule",
    "what can i take",
    "what courses",
    "assuming i pass",
    "after i finish",
    "next classes",
]


def _is_planning_query(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in _PLANNING_KEYWORDS)


def _build_planning_context(
    parsed: ParsedQuery,
    index: EmbeddingIndex,
    k: int,
) -> tuple[list[RetrievedDoc], str]:
    """
    Retrieves and formats the context for a planning/scheduling query.
    """
    codes = parsed.course_codes
    program = parsed.program or ""
    track = parsed.track or ""

    # Degree requirements
    req_queries = [
        f"{program} {track} track required core courses".strip(),
        f"{program} {track} track degree requirements".strip(),
        f"{program} degree requirements total credit hours",
    ]
    req_docs: dict[int, RetrievedDoc] = {}
    for q in req_queries:
        for doc in index.search(q, k=k):
            cid = doc.chunk.chunk_id
            score = doc.score
            # Boost chunks from the exact program
            if program.lower() in doc.chunk.program_name.lower():
                score *= 0.70
            if cid not in req_docs or score < req_docs[cid].score:
                req_docs[cid] = RetrievedDoc(chunk=doc.chunk, score=score)

    top_req = sorted(req_docs.values(), key=lambda d: d.score)[:6]

    # Unlocked courses
    # Each course chunk stores its own prerequisites.  Searching for
    unlock_docs: dict[int, RetrievedDoc] = {}

    # Normalise codes: "EEL 3801C" → both "EEL3801C" and "EEL 3801C"
    code_variants: list[str] = []
    for code in codes:
        code_variants.append(code)
        code_variants.append(code.replace(" ", ""))

    for code in code_variants:
        unlock_queries = [
            f"prerequisite {code}",
            f"requires {code}",
            f"{code} required before",
        ]
        for q in unlock_queries:
            for doc in index.search(q, k=k):
                cid = doc.chunk.chunk_id
                score = doc.score
                text = doc.chunk.text.lower()
                code_lower = code.lower().replace(" ", "")

                # Strong boost: the chunk's text actually contains this code
                # in a prerequisites field (confirms it's a real prereq link)
                if "prerequisite" in text and code_lower in text.replace(" ", ""):
                    score *= 0.60

                # Penalise chunks from completely unrelated programs
                pname = doc.chunk.program_name.lower()
                if program.lower() not in pname:
                    unrelated = any(
                        s in pname
                        for s in [
                            "criminal justice",
                            "architecture",
                            "nursing",
                            "psychology",
                            "business",
                            "hospitality",
                            "biology",
                            "chemistry",
                            "music",
                            "theatre",
                            "praxis",
                        ]
                    )
                    if unrelated:
                        score *= 1.5

                if cid not in unlock_docs or score < unlock_docs[cid].score:
                    unlock_docs[cid] = RetrievedDoc(chunk=doc.chunk, score=score)

    top_unlock = sorted(unlock_docs.values(), key=lambda d: d.score)[:10]

    # Merge and build labelled context
    all_docs = list({d.chunk.chunk_id: d for d in (top_req + top_unlock)}.values())

    context_parts: list[str] = []

    context_parts.append(
        "=== SECTION A: DEGREE / TRACK REQUIREMENTS ===\n"
        "Use this section to identify which courses are required for the "
        f"{program} {track} track and whether an unlocked course is on the plan.\n"
    )
    for i, doc in enumerate(top_req):
        context_parts.append(
            f"[A{i + 1}] Program: {doc.chunk.program_name}\n"
            f"Source: {doc.chunk.source_url}\n"
            f"{doc.chunk.text}"
        )

    context_parts.append(
        "\n=== SECTION B: COURSES UNLOCKED BY COMPLETED PREREQUISITES ===\n"
        "Each chunk below is a course whose prerequisite field lists one of "
        f"the student's current courses ({', '.join(codes)}). "
        "These are candidates for next semester IF they also appear in the "
        "degree requirements above.\n"
    )
    for i, doc in enumerate(top_unlock):
        context_parts.append(
            f"[B{i + 1}] Program: {doc.chunk.program_name}\n"
            f"Source: {doc.chunk.source_url}\n"
            f"{doc.chunk.text}"
        )

    context = "\n\n---\n\n".join(context_parts)
    return all_docs, context


class RAGPipeline:
    def __init__(
        self,
        config: RAGConfig,
        embedding_index: EmbeddingIndex,
        llm: LLMGenerator,
    ):
        self.config = config
        self.index = embedding_index
        self.llm = llm

    def query(
        self,
        question: str,
        k: Optional[int] = None,
        verbose: bool = False,
    ) -> RAGResult:
        k = k or self.config.k
        parsed: ParsedQuery = parse_query(question)

        if verbose:
            print(f"\n[QueryProcessor]")
            print(f"  Intent   : {parsed.intent.value}")
            print(f"  Program  : {parsed.program}")
            print(f"  Track    : {parsed.track}")
            print(f"  Courses  : {parsed.course_codes}")
            print(f"  Category : {parsed.requirement_category}")
            print(f"  Sub-queries ({len(parsed.sub_queries)}):")
            for sq in parsed.sub_queries:
                print(f"    • {sq}")

        # Planning questions get the two-pass context strategy
        if _is_planning_query(question) and parsed.course_codes:
            retrieved, context = _build_planning_context(parsed, self.index, k)
        else:
            retrieved = retrieve_for_parsed_query(
                parsed,
                self.index,
                k_per_query=k,
                max_total=min(k * 2, 20),
            )
            context_parts = []
            for i, doc in enumerate(retrieved):
                context_parts.append(
                    f"[{i + 1}] Program: {doc.chunk.program_name}\n"
                    f"Source: {doc.chunk.source_url}\n"
                    f"{doc.chunk.text}"
                )
            context = "\n\n---\n\n".join(context_parts)

        answer = self.llm.generate(question, context)
        return RAGResult(
            question=question,
            retrieved_docs=retrieved,
            answer=answer,
            context_used=context,
        )
