"""
ingest.py
─────────
Handles all data collection and preprocessing:
  - Fetching UCF catalog pages from the Kuali JSON API
  - Following program sub-links found in page bodies
  - Chunking plain text into overlapping windows
  - Saving / loading chunks to disk
"""

import json
import re
import time

from config import Chunk, RAGConfig

# ── 1. Kuali Catalog API Fetching ─────────────────────────────────────────────
UCF_UNDERGRADUATE_CATALOG_ID = "66bcc88cf93938001c548373"
KUALI_API_BASE = "https://ucf.kuali.co/api/v1/catalog/content/"
KUALI_PROGRAM_BASE = "https://ucf.kuali.co/api/v1/catalog/program/"
KUALI_POLICY_BASE = "https://ucf.kuali.co/api/v1/catalog/policy/"
KUALI_POLICIES_BASE = "https://ucf.kuali.co/api/v1/catalog/policies/"
KUALI_COURSES_BASE = "https://ucf.kuali.co/api/v1/catalog/courses/"
KUALI_COURSE_BASE = "https://ucf.kuali.co/api/v1/catalog/course/"


def _html_to_text(html: str) -> str:
    """Strip HTML tags and return clean plain text."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return soup.get_text(separator=" ", strip=True)


def _extract_program_links(body_html: str) -> list[str]:
    """
    Find all hrefs of the form  #/programs/{slug}?...
    and return just the slug portion for each.
    Example href:
      #/programs/rJleH1ZOsu?bc=true&bcCurrent=Integrated%20Business...
    Returns: ["rJleH1ZOsu", ...]
    """
    import re

    from bs4 import BeautifulSoup

    soup = BeautifulSoup(body_html, "html.parser")
    slugs: list[str] = []
    for a in soup.find_all("a", href=True):
        match = re.match(r"#/programs/([^?]+)", a["href"])
        if match:
            slugs.append(match.group(1))
    return slugs


def _flatten_program_data(data: dict) -> dict:
    """
    Program API sometimes returns shape:
      { newVersion: { title, ... }, oldVersion: { title, ... } }
    Need a flat dict ready for text extraction.
    """
    if "newVersion" in data or "oldVersion" in data:
        return data.get("newVersion") or data.get("oldVersion") or {}
    return data


PROGRAM_TEXT_FIELDS = [
    "title",
    "programDescription",
    "degreeRequirements",
    "requiredCoreCourses",
    "commonProgramPrerequisites",
    "generalEducation",
    "programContactInformation",
]


def _program_to_text(flat: dict) -> str:
    """
    Concatenate the meaningful fields from a program response into one
    plain-text string, labelling each section so the LLM has context.
    Fields may be plain text or HTML — both are handled.
    """

    parts: list[str] = []
    for field in PROGRAM_TEXT_FIELDS:
        value: str = flat.get(field, "") or ""
        value = value.strip()
        if not value:
            continue
        # Strip HTML if the value looks like it contains tags
        if "<" in value:
            value = _html_to_text(value)
        if value:
            label = field[0].upper() + field[1:]  # camelCase → Capitalised
            parts.append(f"{label}: {value}")
    return "\n\n".join(parts)


def _fetch_program_page(
    content_id: str,
    slug: str,
    session,  # requests.Session
    headers: dict,
) -> dict | None:
    """
    URL: /api/v1/catalog/program/{catalog_id}/{slug}
    """
    url = f"{KUALI_PROGRAM_BASE}{content_id}/{slug}"
    try:
        resp = session.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        flat = _flatten_program_data(data)
        title = (flat.get("title") or slug).strip()
        raw_text = _program_to_text(flat)

        if not raw_text:
            print(f"    [WARN] Empty program data for slug={slug}, skipping.")
            return None

        print(f"    ↳ program: {title}")
        return {"url": url, "program_name": title, "raw_text": raw_text}

    except Exception as e:
        print(f"    [WARN] Failed to fetch program slug={slug}: {e}")
        return None


def fetch_kuali_pages(content_ids: list[str]) -> list[dict]:
    """
    Fetch catalog content pages from the Kuali JSON API, and for any
    #/programs/{slug} links found in a page's body, also fetch the full
    program detail from the program API.

    Returns list of {"url": ..., "program_name": ..., "raw_text": ...}
    """
    try:
        import requests
    except ImportError:
        raise ImportError("Install dependencies: pip install requests beautifulsoup4")

    pages: list[dict] = []
    seen_slugs: set[str] = set()  # avoid duplicate slug fetches
    seen_programs: set[str] = set()  # avoid duplicate program fetches
    seen_courses: set[str] = set()  # avoid duplicate course fetches
    seen_policies: set[str] = set()  # avoid duplicate policy fetches
    headers = {"User-Agent": "Mozilla/5.0 (UCF RAG Research Project)"}
    start_time = time.time()

    with requests.Session() as session:
        for content_id in content_ids:
            api_url = f"{KUALI_API_BASE}{content_id}"
            try:
                print(f"  Scraping: {api_url}")
                resp = session.get(api_url, headers=headers, timeout=15)
                resp.raise_for_status()
                data = resp.json()

                title: str = data.get("title", "").strip()
                body_html: str = data.get("body", "")
                catalog_id: str = data.get("catalogId", "")

                if not body_html:
                    print(f"  [WARN] Empty body for id={content_id}, skipping.")
                    continue

                # ── Parent content page ──
                raw_text = _html_to_text(body_html)
                pages.append(
                    {"url": api_url, "program_name": title, "raw_text": raw_text}
                )

                # ── Follow program sub-links ──
                programs = _extract_program_links(body_html)
                for slug in programs:
                    if slug in seen_programs:
                        continue
                    seen_programs.add(slug)
                    time.sleep(0.3)
                    page = _fetch_program_page(catalog_id, slug, session, headers)
                    if page:
                        pages.append(page)

                seen_slugs.add(catalog_id)

                time.sleep(0.3)

            except Exception as e:
                print(f"  [WARN] Failed to scrape content id={content_id}: {e}")
        try:
            print(f" Scraping policies from {KUALI_POLICY_BASE}")
            policy_url = f"{KUALI_POLICIES_BASE}{UCF_UNDERGRADUATE_CATALOG_ID}"
            resp = session.get(policy_url, headers=headers, timeout=15)
            resp.raise_for_status()
            policies = resp.json()

            for policy in policies:
                policy_id: str = policy.get("pid", "")
                if policy_id in seen_policies:
                    continue
                seen_policies.add(policy_id)
                title: str = policy.get("title", "").strip()
                content: str = policy.get("body", "")
                if not title or not content:
                    continue
                raw_text = _html_to_text(content)
                pages.append(
                    {
                        "program_name": title,
                        "raw_text": raw_text,
                        "url": f"{KUALI_POLICY_BASE}{UCF_UNDERGRADUATE_CATALOG_ID}/{policy_id}",
                    }
                )
                print(f"    ↳ policy added: {title}")
                time.sleep(0.3)
        except Exception as e:
            print(f"  [WARN] Failed to scrape policies: {e}")

        try:
            print(f" Scraping courses from {KUALI_COURSES_BASE}")
            course_url = f"{KUALI_COURSES_BASE}{UCF_UNDERGRADUATE_CATALOG_ID}"
            resp = session.get(course_url, headers=headers, timeout=15)
            resp.raise_for_status()
            courses = resp.json()
            for course in courses:
                course_id: str = course.get("pid", "")
                if course_id in seen_courses:
                    continue
                seen_courses.add(course_id)

                resp = session.get(
                    f"{KUALI_COURSE_BASE}{UCF_UNDERGRADUATE_CATALOG_ID}/{course_id}",
                    headers=headers,
                    timeout=15,
                )
                resp.raise_for_status()
                course_data = resp.json()
                title = course_data.get("title", "")
                description = course_data.get("description", "")
                subjectcode = course_data.get("__catalogCourseId", "")
                if not title or not description:
                    print(
                        f"  [WARN] Skipping course: {title} [{subjectcode}] (no title or description)"
                    )
                    continue
                raw_text = _course_to_text(course_data)
                pages.append(
                    {
                        "program_name": title,
                        "raw_text": raw_text,
                        "url": f"{KUALI_COURSE_BASE}/{UCF_UNDERGRADUATE_CATALOG_ID}{course_id}",
                    }
                )
                print(f"    ↳ course added: {title} [{subjectcode}]")
                time.sleep(0.3)

        except Exception as e:
            print(f"  [WARN] Failed to scrape courses: {e}")

    print(f"Scraped {len(pages)} pages total")
    print(f"    ↳ {len(seen_slugs)} programs scraped")
    print(f"    ↳ {len(seen_programs)} program sub-pages scraped")
    print(f"    ↳ {len(seen_policies)} policies scraped")
    print(f"    ↳ {len(seen_courses)} courses scraped")
    end_time = time.time()
    print(f"  [INFO] Scraping took {end_time - start_time:.1f}s")

    return pages


# ── 2. Chunking ────────────────────────────────────────────────────────────────
def _normalize_text(text: str) -> str:
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _split_paragraphs(text: str) -> list[str]:
    text = _normalize_text(text)
    parts = re.split(r"\n\s*\n", text)
    return [p.strip() for p in parts if p.strip()]


def _chunk_by_words(text: str, chunk_size: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []

    chunks = []
    step = max(1, chunk_size - overlap)
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end]).strip()
        if len(chunk) > 20:
            chunks.append(chunk)
        if end == len(words):
            break
        start += step

    return chunks


def _chunk_by_paragraphs(
    text: str, target_words: int = 220, overlap_words: int = 40
) -> list[str]:
    paragraphs = _split_paragraphs(text)
    if not paragraphs:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for para in paragraphs:
        para_words = len(para.split())

        # If one paragraph is too big, split it directly by words
        if para_words > target_words * 1.5:
            if current:
                combined = "\n\n".join(current).strip()
                if combined:
                    chunks.append(combined)
                current = []
                current_words = 0
            chunks.extend(_chunk_by_words(para, target_words, overlap_words))
            continue

        if current_words + para_words <= target_words:
            current.append(para)
            current_words += para_words
        else:
            combined = "\n\n".join(current).strip()
            if combined:
                chunks.append(combined)

            if overlap_words > 0 and chunks:
                prev_words = combined.split()
                overlap_text = " ".join(prev_words[-overlap_words:])
                current = [overlap_text, para]
                current_words = len(overlap_text.split()) + para_words
            else:
                current = [para]
                current_words = para_words

    if current:
        combined = "\n\n".join(current).strip()
        if combined:
            chunks.append(combined)

    return chunks


def _course_to_text(course_data: dict) -> str:
    title = (course_data.get("title") or "").strip()
    course_code = (course_data.get("__catalogCourseId") or "").strip()
    description = _html_to_text(course_data.get("description", "") or "")
    prerequisites = _html_to_text(course_data.get("prerequisites", "") or "")

    subject = course_data.get("subjectCode", {}) or {}
    subject_name = subject.get("name", "")
    subject_desc = subject.get("description", "")

    credits_obj = course_data.get("credits", {}) or {}
    credit_value = credits_obj.get("value")
    terms = course_data.get("termsOffering", []) or []
    terms_text = ", ".join(t.get("name", "") for t in terms if t.get("name"))

    group1 = (course_data.get("groupFilter1") or {}).get("name", "")
    group2 = (course_data.get("groupFilter2") or {}).get("name", "")

    parts = [
        f"Course Code: {course_code}",
        f"Title: {title}",
        f"Subject: {subject_name} {subject_desc}".strip(),
        f"Description: {description}" if description else "",
        f"Prerequisites: {prerequisites}" if prerequisites else "",
        f"Credits: {credit_value}" if credit_value is not None else "",
        f"Terms Offered: {terms_text}" if terms_text else "",
        f"School: {group1}" if group1 else "",
        f"College: {group2}" if group2 else "",
    ]

    return "\n\n".join(p for p in parts if p.strip())


def detect_doc_type(page: dict) -> str:
    url = page.get("url", "")
    text = page.get("raw_text", "")

    if "/catalog/course/" in url:
        return "course"
    if "/catalog/policy/" in url:
        return "policy"
    if "/catalog/program/" in url:
        return "program"
    if "Prerequisites:" in text and "Course Code:" in text:
        return "course"
    return "content"


def chunk_page(page: dict, config: RAGConfig) -> list[str]:
    text = _normalize_text(page["raw_text"])
    doc_type = detect_doc_type(page)

    if doc_type == "course":
        # Keep course facts together whenever possible.
        word_count = len(text.split())
        if word_count <= 220:
            return [text]
        return _chunk_by_words(text, chunk_size=180, overlap=20)

    if doc_type == "policy":
        return _chunk_by_paragraphs(text, target_words=220, overlap_words=50)

    if doc_type == "program":
        return _chunk_by_paragraphs(text, target_words=260, overlap_words=50)

    # fallback for general content pages
    return _chunk_by_paragraphs(
        text,
        target_words=config.chunk_size,
        overlap_words=config.chunk_overlap,
    )


def build_chunks(pages: list[dict], config: RAGConfig) -> list[Chunk]:
    """Convert scraped pages into Chunk objects with document-aware chunking."""
    chunks: list[Chunk] = []
    chunk_id = 0

    for page in pages:
        text_chunks = chunk_page(page, config)

        for tc in text_chunks:
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    source_url=page["url"],
                    program_name=page.get("program_name", ""),
                    text=tc,
                )
            )
            chunk_id += 1

    print(f"Built {len(chunks)} chunks from {len(pages)} pages.")
    return chunks


def save_chunks(chunks: list[Chunk], path: str):
    data = [
        {
            "chunk_id": c.chunk_id,
            "source_url": c.source_url,
            "program_name": c.program_name,
            "text": c.text,
        }
        for c in chunks
    ]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved chunks at’ {path}")


def load_chunks(path: str) -> list[Chunk]:
    with open(path) as f:
        data = json.load(f)
    return [Chunk(**d) for d in data]
