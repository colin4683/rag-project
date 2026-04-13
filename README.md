# UCF Course Catalog RAG System

A Retrieval-Augmented Generation (RAG) system that answers student questions
about UCF programs by retrieving relevant information from the UCF course
catalog and generating grounded answers with Google Gemini.

Built for the Deep Learning & Neural Networks group project.

---

## Table of Contents

- [Project Overview](#project-overview)
- [File Structure](#file-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Pipeline Architecture](#pipeline-architecture)
- [Experiments](#experiments)
---

## Project Overview

**Problem:** Students need to search through dozens of UCF catalog pages to
find accurate information about programs, requirements, and fees.

**Solution:** A RAG pipeline that retrieves the most relevant catalog chunks
for a question and uses an LLM to generate a grounded, accurate answer.

**Stack:**
| Component | Choice | Reason |
|---|---|---|
| Embedding model | Sentence-BERT (`all-MiniLM-L6-v2`) | Fast, strong semantic similarity |
| Vector database | FAISS `IndexFlatL2` | Exact search, sufficient at this scale |
| Language model | Gemini 1.5 Flash | Free tier, fast, accurate |
| Data source | UCF Catalog JSON API | Structured, no HTML scraping noise |

---

## File Structure

```
rag-project/
│
├── config.py              # Shared dataclasses: RAGConfig, Chunk, EvalSample, etc.
├── ingest.py              # UCF API fetching, HTML parsing, chunking, persistence
├── retriever.py           # Sentence-BERT embeddings + FAISS index
├── generator.py           # Gemini LLM wrapper (RAG + zero-shot modes)
├── pipeline.py            # RAGPipeline: wires retriever → generator together
│
├── cli.py                 # Interactive question-answering tool (main entry point)
│
├── experiments/
│   ├── baseline.py        # Zero-shot LLM baseline comparison
│   └── ablation.py        # Ablation study: vary k, measure Recall@k
│
├── generated/             # Auto-created
│   ├── chunks.json        # Saved chunks from UCF API
│   └── faiss_index.bin    # Generated FAISS index from chunks
│
├── results/               # Auto-created — stores JSON output from experiments
│   ├── baseline_comparison.json
│   └── ablation_results.json
│
├── requirements.txt
└── README.md
```

### What each file is responsible for

**`config.py`** — All shared dataclasses live here (`RAGConfig`, `Chunk`,
`RetrievedDoc`, `RAGResult`, `EvalSample`). Every other module imports from
this file. Changing a field here propagates everywhere.

**`ingest.py`** — Everything related to getting data into the system. Fetches
content pages from the UCF Kuali API, follows `#/programs/{slug}` sub-links to
fetch full program details, parses HTML bodies to plain text, and chunks the
result into overlapping word windows. Also handles saving and loading
`chunks.json`.

**`retriever.py`** — The `EmbeddingIndex` class. Encodes chunks with
Sentence-BERT, stores them in FAISS, and searches at query time. The index
is saved to `faiss_index.bin` after being built so it only needs to run once.

**`generator.py`** — The `LLMGenerator` class. Wraps the Gemini SDK with two
methods: `generate()` for RAG answers (context injected into prompt) and
`generate_zero_shot()` for the baseline (no context).

**`pipeline.py`** — The `RAGPipeline` class. The only class `cli.py` and the
experiment scripts interact with. Takes a question, calls `retriever.search()`,
formats the context block, calls `generator.generate()`, and returns a
`RAGResult`.

**`cli.py`** — The user-facing tool. Supports an interactive loop, single
questions via `--question`, raw chunk inspection via `--inspect`, and index
rebuilding via `--build`.

**`experiments/baseline.py`** — Runs every evaluation question through both
the full RAG pipeline and zero-shot Gemini, saves side-by-side answers to
`results/baseline_comparison.json` for human scoring.

**`experiments/ablation.py`** — Sweeps `k` over `[1, 3, 5, 10, 20]` and
computes Recall@k at each value using the annotated evaluation set. Saves
results to `results/ablation_results.json`.

---

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/colin4683/rag-project.git
cd rag-project
pip install -r requirements.txt
```

### 2. Get a Gemini API key

Go to [aistudio.google.com](https://aistudio.google.com), click **Get API key**,
and copy the key.

### 3. Set the environment variable

```bash
# macOS / Linux
export GEMINI_API_KEY="your-key-here"

# Windows (Command Prompt)
set GEMINI_API_KEY=your-key-here

# Google Colab
import os
os.environ["GEMINI_API_KEY"] = "your-key-here"
```

---

## Usage

### Interactive mode
If the index is not built it will first build
```bash
python cli.py
```

```
UCF Course Catalog RAG
Type your question and press Enter. Type 'quit' or 'exit' to stop.
Prefix with '!inspect ' to see raw retrieved chunks.

You: How many credit hours does the CS degree require?

────────────────────────────────────────────────────────────
Answer:
The Computer Science B.S. requires 120 total credit hours...
────────────────────────────────────────────────────────────
```

### Single question

```bash
python cli.py --question "What are the prerequisites for the CS program?"
```

### Show sources with answers

```bash
python cli.py --question "What is the late payment fee?" --show-sources
```

### Change number of retrieved chunks
```bash
python cli.py --k 10
```
### Inspect raw retrieval (no LLM)
```bash
python cli.py --inspect "What courses are required for the CS degree?"
```

Output:
```
Top 5 retrieved chunks for: "What courses are required for the CS degree?"

  [1] chunk_id=1133  score=0.7271
       Program : Computer Science (B.S.)
       Source  : https://ucf.kuali.co/api/v1/catalog/program/66bcc88cf93938001c548373/SkbvEJ-_iO
       Preview : A minimum 2.500 GPA is required for courses in this section. Technical Electives 18 Total Credits Complete all of the fo...
  ...
```


### Force build the index
### Estimated runtime: ~10min

```bash
python cli.py --build
```

This fetches:
- Catalog Pages (Mission Statement, Creed, Departments, Policies)
- Individual Policy pages (58 policies)
- Individual Course pages (~3667 courses, some have no content in api)

It then chunks the text, encodes everything with Sentence-BERT, and saves `faiss_index.bin` and `chunks.json`.

### Force build the embeds
```bash
python cli.py --embed
```
This encodes the existing chinks from `chunks.json` with Sentence-BERT, and saves `faiss_index.bin`


---

## Pipeline Architecture

```
Question (string)
       │
       ▼
 Sentence-BERT              ← same model used for both chunks and queries
 all-MiniLM-L6-v2
       │
       ▼ 384-dim float32 vector
       │
 FAISS IndexFlatL2          ← exact L2 search over all chunk vectors
       │
       ▼ top-k (chunk_id, L2 distance) pairs
       │
 Chunk lookup               ← map indices → Chunk objects with text + metadata
       │
       ▼ numbered context block
       │
 Gemini 1.5 Flash           ← instructed to answer ONLY from context
       │
       ▼
 Answer (string)
```

---

## Experiments

### Baseline comparison

Answers each question with and without retrieval to establish a performance floor.

```bash
python experiments/baseline.py
```
TODO

### Ablation study

Sweeps `k` over `[1, 3, 5, 10, 20]` to find the optimal retrieval depth.

```bash
python experiments/ablation.py
```
TODO

---
