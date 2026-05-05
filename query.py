"""
Improves retrieval for complex questions by:

  1. Classifying query intent (program, policy, course, admissions, degree)
  2. Extracting structured entities (program name, track, course codes, etc.)
  3. Decomposing multi-part questions into focused sub-queries
  4. Reranking retrieved chunks by relevance to each sub-query

This sits between the user's question and the FAISS search in the pipeline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from config import RetrievedDoc

# Query intent types


class QueryIntent(Enum):
    PROGRAM_REQUIREMENTS = "program_requirements"  # degree/track/elective questions
    POLICY = "policy"  # repeat course, GPA, withdrawal
    COURSE = "course"  # prerequisites, description, credits
    ADMISSIONS = "admissions"  # GPA req, how to get into a major
    GENERAL = "general"  # anything else


@dataclass
class ParsedQuery:
    original: str
    intent: QueryIntent
    program: str | None = None  # e.g. "Computer Engineering"
    track: str | None = None  # e.g. "Comprehensive"
    degree_type: str | None = None  # e.g. "BS", "BSBA"
    course_codes: list[str] = field(default_factory=list)  # e.g. ["COP 3502C"]
    requirement_category: str | None = None  # e.g. "upper-level", "electives", "core"
    sub_queries: list[str] = field(default_factory=list)


# Keyword-based intent classifier

_POLICY_KEYWORDS = {
    "repeat",
    "retake",
    "withdrawal",
    "withdraw",
    "drop",
    "late",
    "fee",
    "gpa",
    "grade forgiveness",
    "academic standing",
    "probation",
    "suspension",
    "refund",
    "financial",
    "appeal",
    "hold",
    "transcript",
    "honor",
    "dismissal",
    "forgiveness",
    "academic renewal",
}

_ADMISSIONS_KEYWORDS = {
    "get into",
    "admitted",
    "admission",
    "prerequisite to the major",
    "requirements to enter",
    "apply",
    "acceptance",
    "how do i get",
    "minimum gpa",
    "limited access",
}

_COURSE_KEYWORDS = {
    "prerequisite",
    "credit hours",
    "credits",
    "offered",
    "when is",
    "syllabus",
    "course description",
    "what is",
    "how many credits",
}

_PROGRAM_KEYWORDS = {
    "track",
    "elective",
    "degree requirement",
    "required courses",
    "major",
    "concentration",
    "upper-level",
    "remaining",
    "still need",
    "how many more",
    "program",
    "core requirement",
    "general education",
    "gen ed",
}


def classify_intent(question: str) -> QueryIntent:
    q = question.lower()

    # Check most specific first
    if any(kw in q for kw in _POLICY_KEYWORDS):
        # Make sure it's not also a program question with a GPA mention
        if any(
            kw in q
            for kw in {
                "repeat",
                "retake",
                "withdraw",
                "forgiveness",
                "probation",
                "suspension",
                "appeal",
                "refund",
            }
        ):
            return QueryIntent.POLICY

    if any(kw in q for kw in _ADMISSIONS_KEYWORDS):
        return QueryIntent.ADMISSIONS

    if any(kw in q for kw in _PROGRAM_KEYWORDS):
        return QueryIntent.PROGRAM_REQUIREMENTS

    if any(kw in q for kw in _COURSE_KEYWORDS):
        return QueryIntent.COURSE

    return QueryIntent.GENERAL


# Entity extraction
# Degree suffixes to look for
_DEGREE_PATTERNS = re.compile(
    r"\b(B\.?S\.?|B\.?A\.?|B\.?S\.?B\.?A\.?|M\.?S\.?|Ph\.?D\.?|bachelor|master)\b",
    re.IGNORECASE,
)

# Course code pattern: 2-4 letters, space, 4 digits, optional letter
_COURSE_CODE_RE = re.compile(r"\b([A-Z]{2,4})\s?(\d{4}[A-Z]?)\b")

# Common track names
_TRACK_KEYWORDS = [
    "comprehensive",
    "thesis",
    "non-thesis",
    "accelerated",
    "traditional",
    "systems",
    "networks",
    "cybersecurity",
    "embedded",
    "software",
    "biomedical",
    "power",
    "communications",
    "data science",
    "artificial intelligence",
]

# Requirement categories
_REQUIREMENT_KEYWORDS = {
    "upper-level": ["upper-level", "upper level", "4000-level", "3000-level"],
    "electives": ["elective", "electives", "free elective"],
    "core": ["core", "required core", "core courses"],
    "gen ed": ["gen ed", "general education", "gordon rule"],
    "prerequisites": ["prerequisite", "prereq"],
}


def extract_entities(question: str) -> dict:
    """
    Extract structured entities from a natural-language question.
    Returns a dict with keys: program, track, degree_type, course_codes,
    requirement_category.
    """
    q_lower = question.lower()
    entities: dict = {
        "program": None,
        "track": None,
        "degree_type": None,
        "course_codes": [],
        "requirement_category": None,
    }

    # Degree type
    dm = _DEGREE_PATTERNS.search(question)
    if dm:
        entities["degree_type"] = dm.group(0).upper().replace(".", "")

    # Course codes (e.g. COP 3502C, MAC2311)
    codes = _COURSE_CODE_RE.findall(question.upper())
    entities["course_codes"] = [f"{letters} {digits}" for letters, digits in codes]

    # Track
    for track in _TRACK_KEYWORDS:
        if track in q_lower:
            entities["track"] = track.title()
            break

    # Requirement category
    for category, keywords in _REQUIREMENT_KEYWORDS.items():
        if any(kw in q_lower for kw in keywords):
            entities["requirement_category"] = category
            break

    # Program name: look for known UCF program patterns.
    # Handles both "Computer Engineering BS" and "Computer Engineering (B.S)"
    program_pattern = re.compile(
        r"(?:in|for|the|doing)\s+([A-Z][a-zA-Z\s]+?)"
        r"\s*(?:\(B\.?S\.?[^)]*\)|\(B\.?A\.?[^)]*\)|"
        r"program|major|degree|track|\bbs\b|\bba\b|bsba|bsee|bscs|bsce)",
        re.IGNORECASE,
    )
    pm = program_pattern.search(question)
    if pm:
        entities["program"] = pm.group(1).strip().title()

    return entities


# Sub-query decomposition


def decompose_query(question: str, intent: QueryIntent, entities: dict) -> list[str]:
    """
    Break a complex question into focused sub-queries for FAISS retrieval.
    More focused queries retrieve more precise chunks.

    For simple questions, returns just [question].
    For complex ones, returns 2-4 targeted sub-queries.
    """
    program = entities.get("program")
    track = entities.get("track")
    codes = entities.get("course_codes", [])
    q_lower = question.lower()

    # For simple or short questions include the original as the anchor query.
    # For long planning/scheduling questions the full text makes a poor FAISS
    # query — it averages over too many concepts and drifts away from every
    # specific chunk. In those cases we rely entirely on the decomposed queries.
    is_planning = any(
        kw in q_lower
        for kw in [
            "next semester",
            "schedule",
            "what can i take",
            "what courses",
            "assuming i pass",
            "after i finish",
            "next classes",
        ]
    )
    sub_queries: list[str] = [] if is_planning else [question]

    if intent == QueryIntent.PROGRAM_REQUIREMENTS:
        if program:
            sub_queries.append(f"{program} degree requirements")
        if track and program:
            sub_queries.append(f"{program} {track} track required courses electives")
        if entities.get("requirement_category"):
            cat = entities["requirement_category"]
            base = f"{program} " if program else ""
            sub_queries.append(f"{base}{cat} requirements")

        # For scheduling/planning questions we need to search in the direction
        # the index actually stores prerequisite info — downstream courses list
        # their own prerequisites, so we search for courses that *require* each
        # completed course, not courses that the completed course unlocks.
        if is_planning and codes:
            for code in codes:
                # "requires COP3502C" finds COP3503C's chunk (which says
                # "Prerequisites: COP3502C"), which is what we actually want.
                sub_queries.append(f"requires {code} prerequisite")
                sub_queries.append(f"prerequisite {code}")

            # Also search for what's available now given the program structure
            if program and track:
                sub_queries.append(
                    f"{program} {track} track core courses prerequisites"
                )

    elif intent == QueryIntent.POLICY:
        # Pull in the specific policy topic
        q_lower = question.lower()
        if "repeat" in q_lower or "retake" in q_lower:
            sub_queries += [
                "course repeat policy GPA grade forgiveness",
                "grade forgiveness academic renewal UCF",
            ]
        if "withdraw" in q_lower or "withdrawal" in q_lower:
            sub_queries += [
                "withdrawal policy grades transcript",
                "drop deadline withdraw after drop period",
            ]
        if "probation" in q_lower or "suspension" in q_lower:
            sub_queries += ["academic standing probation suspension dismissal"]

    elif intent == QueryIntent.ADMISSIONS:
        if program:
            sub_queries += [
                f"{program} admission requirements",
                f"{program} limited access program GPA prerequisites",
            ]
        else:
            sub_queries += [
                "limited access program admission requirements GPA",
                "major prerequisites minimum GPA admission",
            ]

    elif intent == QueryIntent.COURSE:
        for code in codes:
            sub_queries.append(f"{code} course description prerequisites credits")

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for q in sub_queries:
        if q not in seen:
            seen.add(q)
            unique.append(q)
    return unique


# Full query parsing
def parse_query(question: str) -> ParsedQuery:
    """
    Top-level entry point. Returns a ParsedQuery with intent,
    extracted entities, and decomposed sub-queries.
    """
    intent = classify_intent(question)
    entities = extract_entities(question)

    return ParsedQuery(
        original=question,
        intent=intent,
        program=entities["program"],
        track=entities["track"],
        degree_type=entities["degree_type"],
        course_codes=entities["course_codes"],
        requirement_category=entities["requirement_category"],
        sub_queries=decompose_query(question, intent, entities),
    )


# Multi-query retrieval & reranking
def retrieve_for_parsed_query(
    parsed: ParsedQuery,
    index,  # EmbeddingIndex
    k_per_query: int = 5,
    max_total: int = 10,
    intent_doc_type_boost: dict[QueryIntent, list[str]] | None = None,
) -> list[RetrievedDoc]:
    """
    Run each sub-query through FAISS, merge results, deduplicate by chunk_id,
    and return the top max_total chunks ordered by best score.

    intent_doc_type_boost: optional map of intent → preferred URL patterns.
    If a chunk's URL matches a preferred pattern, its score is boosted (lowered,
    since L2 distance means lower = better).
    """
    if intent_doc_type_boost is None:
        intent_doc_type_boost = {
            QueryIntent.POLICY: ["/policy/"],
            QueryIntent.PROGRAM_REQUIREMENTS: ["/program/"],
            QueryIntent.COURSE: ["/course/"],
            QueryIntent.ADMISSIONS: ["/program/", "/policy/"],
        }

    boost_patterns = intent_doc_type_boost.get(parsed.intent, [])

    # Build a set of lowercase keyword tokens from the target program name.
    # Used below to detect chunks that clearly belong to a different program.
    program_tokens: set[str] = set()
    if parsed.program:
        program_tokens = {w.lower() for w in parsed.program.split() if len(w) > 2}

    # Penalize keywords that indicate unrelated programs. Any chunk whose
    # program_name contains one of these but does NOT match the target program
    # will have its score inflated (pushed down the ranking).
    _UNRELATED_SIGNALS = [
        "criminal justice",
        "architecture",
        "biomedical sciences",
        "data science",
        "nursing",
        "psychology",
        "business",
        "history",
        "english",
        "chemistry",
        "biology",
        "music",
        "theatre",
        "art ",
        "accounting",
        "finance",
        "marketing",
        "hospitality",
        "education",
        "praxis",
        "interdisciplinary",
    ]

    def _is_off_program(doc) -> bool:
        """Return True if the chunk clearly belongs to a different program."""
        if not program_tokens:
            return False
        pname = doc.chunk.program_name.lower()
        # If the chunk's program name shares tokens with our target, keep it
        if any(tok in pname for tok in program_tokens):
            return False
        # If the chunk's program name contains a known-unrelated signal, penalize
        return any(sig in pname for sig in _UNRELATED_SIGNALS)

    # Collect results from all sub-queries
    seen_ids: dict[int, RetrievedDoc] = {}

    for sub_query in parsed.sub_queries:
        results = index.search(sub_query, k=k_per_query)
        for doc in results:
            cid = doc.chunk.chunk_id
            score = doc.score

            # Boost chunks whose URL matches preferred doc type for this intent
            if any(pat in doc.chunk.source_url for pat in boost_patterns):
                score *= 0.75  # 25% score reduction = ranks higher

            # Penalize chunks that belong to clearly unrelated programs.
            # Multiplying by 1.6 pushes them well below relevant chunks without
            # discarding them entirely in case nothing better exists.
            if _is_off_program(doc):
                score *= 1.6

            if cid not in seen_ids or score < seen_ids[cid].score:
                from config import RetrievedDoc as RD

                seen_ids[cid] = RD(chunk=doc.chunk, score=score)

    # Sort by score ascending (lower L2 = more similar) and return top N
    ranked = sorted(seen_ids.values(), key=lambda d: d.score)
    return ranked[:max_total]
