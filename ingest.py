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
KUALI_API_BASE = "https://ucf.kuali.co/api/v1/catalog/content/"
KUALI_PROGRAM_BASE = "https://ucf.kuali.co/api/v1/catalog/program/"
KUALI_POLICY_BASE = "https://ucf.kuali.co/api/v1/catalog/policy/"
KUALI_POLICIES_BASE = "https://ucf.kuali.co/api/v1/catalog/policies/"


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
    from bs4 import BeautifulSoup

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
    seen_slugs: set[str] = set()  # avoid duplicate program fetches
    headers = {"User-Agent": "Mozilla/5.0 (UCF RAG Research Project)"}

    with requests.Session() as session:
        for content_id in content_ids:
            api_url = f"{KUALI_API_BASE}{content_id}"
            try:
                print(f"  Fetching: {api_url}")
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
                slugs = _extract_program_links(body_html)
                for slug in slugs:
                    if slug in seen_slugs:
                        continue
                    seen_slugs.add(slug)
                    time.sleep(0.3)
                    page = _fetch_program_page(catalog_id, slug, session, headers)
                    if page:
                        pages.append(page)

                time.sleep(0.3)

            except Exception as e:
                print(f"  [WARN] Failed to fetch id={content_id}: {e}")
        try:
            print(f" Fetching policies from {KUALI_POLICY_BASE}")
            policy_url = f"{KUALI_POLICIES_BASE}66bcc88cf93938001c548373"
            resp = session.get(policy_url, headers=headers, timeout=15)
            resp.raise_for_status()
            policies = resp.json()

            for policy in policies:
                title: str = policy.get("title", "").strip()
                content: str = policy.get("body", "")
                if not title or not content:
                    continue
                raw_text = _html_to_text(content)
                pages.append(
                    {
                        "program_name": title,
                        "raw_text": raw_text,
                        "url": f"{KUALI_POLICY_BASE}66bcc88cf93938001c548373/{policy.get('pid', '')}",
                    }
                )
                print(f"    ↳ policy added: {title}")
        except Exception as e:
            print(f"  [WARN] Failed to fetch policies: {e}")

    print(f"Fetched {len(pages)} pages total ({len(seen_slugs)} program sub-pages).")
    return pages


# ── 2. Chunking ────────────────────────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """
    Split text into overlapping word-based chunks.
    chunk_size and overlap are measured in words.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        if len(chunk.strip()) > 20:  # skip near-empty chunks
            chunks.append(chunk)
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


def build_chunks(pages: list[dict], config: RAGConfig) -> list[Chunk]:
    """Convert scraped pages into Chunk objects."""
    chunks = []
    chunk_id = 0
    for page in pages:
        text_chunks = chunk_text(
            page["raw_text"], config.chunk_size, config.chunk_overlap
        )
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
    print(f"Saved chunks → {path}")


def load_chunks(path: str) -> list[Chunk]:
    with open(path) as f:
        data = json.load(f)
    return [Chunk(**d) for d in data]
