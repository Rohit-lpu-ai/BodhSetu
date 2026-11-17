# NiyamSetu RAG Retrieval Evaluation

This directory contains the evaluation corpus, corpus loader, dataset schema, benchmark template,
and evaluation methodology for benchmarking Person 3's NiyamSetu retrieval components
(`BM25Retriever` and `DenseRetriever`) against authoritative regulatory text.

---

## 0. Supplied Evaluation Corpus (Stage 3)

The full 501-segment corpus has been exported from Person 1's PostgreSQL database and placed
under `rag_module/evaluation/data/`.

| File | Description |
| :--- | :--- |
| `data/niyamsetu_segments_501.jsonl` | 501 segment records, one JSON object per line |
| `data/niyamsetu_segments_export_manifest.json` | Export metadata, counts, field mapping, and SHA-256 checksum |

**Verified corpus statistics** (produced by `corpus_loader.load_default_corpus()`):

| Metric | Value |
| :--- | :--- |
| SHA-256 of JSONL | `6a870036940b4b808147c3ba01b9ff9de01a17934c61cb9e14a2b74b6a02b15c` |
| Total segments | **501** |
| Clause-aware segments (Experiment A) | **363** |
| Paragraph segments (Experiment B) | **138** |
| Distinct document IDs | **4** (`DOC-0001` – `DOC-0004`) |
| Distinct version IDs | **4** (`VER-0001` – `VER-0004`) |
| Page-provenance rows | **552** |

> **Important:** The corpus file is read-only input for retrieval evaluation.
> Do not alter, truncate, re-sort, or re-normalise segment text.

---

## 0.1 Loading the Corpus

Use `rag_module.evaluation.corpus_loader` — the only authorised entry-point for
retrieval evaluation code.  It performs all integrity checks before returning
`ClauseSegment` objects.

```python
from rag_module.evaluation.corpus_loader import (
    load_default_corpus,
    experiment_a_corpus,
    experiment_b_corpus,
)

# Load and validate the full corpus (SHA-256 is verified automatically)
segments, stats = load_default_corpus()

# Experiment A — clause_aware only (363 segments)
exp_a = experiment_a_corpus(segments)

# Experiment B — paragraph only (138 segments)
exp_b = experiment_b_corpus(segments)
```

To load from non-default paths (e.g. in scripts outside `rag_module/evaluation/`):

```python
from pathlib import Path
from rag_module.evaluation.corpus_loader import load_corpus

segments, stats = load_corpus(
    jsonl_path=Path("path/to/niyamsetu_segments_501.jsonl"),
    manifest_path=Path("path/to/niyamsetu_segments_export_manifest.json"),
)
```

---

## 0.2 Checksum Verification

`load_corpus` and `load_default_corpus` compute the SHA-256 digest of the JSONL
file and compare it against `jsonl_sha256` in the manifest before parsing any records.
If the digests differ the function raises `ValueError` immediately — no partially-loaded
corpus is returned.

The checksum is deterministic: the backend export script orders rows by
`(document, version, method, ordinal)`, so re-running the export against an unchanged
database reproduces the file byte-for-byte.

---

## 0.3 Why the Two Segmentation Methods Must Not Be Mixed

The corpus contains **two independent segmentations of the same source text**:

- **clause_aware** — structured legal clause boundaries (363 segments).
- **paragraph** — paragraph-level splits (138 segments).

Both cover identical regulatory content.  If a retrieval index is built from the
combined corpus, every relevant passage will appear twice, inflating ranking scores and
invalidating comparative evaluation.  The correct protocol is:

1. Build one index with `experiment_a_corpus()` — evaluate BM25 and DenseRetriever under clause-aware segmentation.
2. Build a **separate** index with `experiment_b_corpus()` — evaluate the same retrievers under paragraph segmentation.
3. Report results for each experiment independently.

`experiment_a_corpus()` and `experiment_b_corpus()` return disjoint sets of `ClauseSegment`
objects (verified by test `test_a_and_b_are_disjoint`).

---

## 0.4 Limitation: Page Offsets Reference Normalized Page Text Not Included in This Export

Each `PageProvenance` entry carries `char_start` / `char_end` offsets that index into the
`normalized_text` column stored per page in the backend database (`offset_basis = "normalized_text"`
for every segment in this corpus — see CLAUDE.md D-11).

The `normalized_text` pages are **not included in this export**.  As a result:

- The offsets are preserved verbatim in every `ClauseSegment.pages` entry so that future
  citation verification tools can slice the correct column once they have page access.
- It is **not possible** to reconstruct or verify the exact character span from the JSONL
  alone.
- Do not slice `segment.text` using `char_start` / `char_end` — those offsets index the
  page column, not the segment text.

---

## 1. Corpus Availability & Access Audit

An inspection of the backend integration, fixtures, and source repositories established the following state:

### Backend API Status
- **Base URL:** `http://127.0.0.1:8000` (configurable via `BACKEND_URL` in `rag_module/config/settings.py`)
- **Status:** **Offline** during this audit (`BackendConnectionError: [WinError 10061] No connection could be made because the target machine actively refused it`).
- **Authentication:** None required for read/retrieval endpoints (`/api/health`, `/api/documents`, `/api/documents/{id}/clauses`).
- **Endpoint Contracts:**
  - `GET /api/health`: Health probe returning `{"status": "ok", "version": "..."}`.
  - `GET /api/documents`: Returns list of indexed document metadata objects (`id`, `title`, `version`, `clause_count`, `created_at`).
  - `GET /api/documents/{id}/clauses`: Returns array of segment/clause objects for a given document.

### Real Corpus vs. Unit Test Fixtures
- **Authoritative Corpus (501 segments):**
  - Generated and maintained by Person 1's ingestion and segmentation pipeline (`data/annotations/segmentation/*.jsonl`, indexed into PostgreSQL).
  - Comprises **4 primary regulatory documents** containing **501 total segments**:
    - **363 clause-aware segments** (structured legal clauses with numbered sections/headings).
    - **138 paragraph segments** (narrative, preamble, or annexure text).
  - Segment attributes in real schema:
    - `document_id`: Standardized document code (e.g. `DOC-0001`).
    - `version_id`: Version identifier (e.g. `VER-0001`).
    - `clause_id`: Unique clause identifier (e.g. `CLS-0000001` through `CLS-0000363`).
    - `fingerprint`: Content hash (e.g. `SEG-4f8a9b2c1d0e3f4a`).
    - `text`: Text body of the clause or paragraph.
    - `heading`: Section title or heading if present.
    - `clause_number`: Official numbering (e.g. `4.2`, `12(a)`).
    - `page_start`, `page_end`, `pages`: Page provenance array mapping to physical PDF source pages.
- **Mock Fixture (`rag_module/client/fixtures/mock_segments.json`):**
  - Contains only **2 dummy segments** (`cl_001` in `doc_001` and `cl_002` in `doc_002`).
  - **Purpose:** Exclusively intended for fast, offline, deterministic unit testing of `BackendClient`, `MockBackendClient`, `BM25Retriever`, and `DenseRetriever`.
  - **Notice:** This fixture **must not** be confused with or substituted for the full 501-segment evaluation corpus. Full benchmark evaluation requires either a running backend instance or an exported offline dump of the 501 segments.

---

## 2. Evaluation Dataset Design & Schema

The benchmark dataset format is defined in [`dataset_schema.json`](./dataset_schema.json) following JSON Schema Draft 2020-12. The benchmark dataset itself is stored in JSON Lines format (`.jsonl`), where each line is an independent JSON object representing a single evaluation question and its ground-truth annotations.

### Schema Fields

| Field | Type | Required | Description |
| :--- | :--- | :---: | :--- |
| `question_id` | String | **Yes** | Unique ID matching `^EVAL-[A-Z0-9_-]+$` (e.g. `EVAL-EN-0001`, `EVAL-HI-0005`). |
| `question` | String | **Yes** | Natural language compliance query (minimum 3 characters). |
| `language` | String (`en` \| `hi`) | **Yes** | ISO 639-1 code for the query language. |
| `is_answerable` | Boolean | **Yes** | `true` if answered by corpus; `false` if out-of-scope or ungrounded. |
| `relevant_clause_ids` | Array of Strings | **Yes** | Authoritative clause IDs (e.g. `["CLS-0000142"]`). Empty `[]` if `is_answerable` is `false`. |
| `relevant_document_ids`| Array of Strings | No | Logical document IDs (e.g. `["DOC-0001"]`). Default `[]`. |
| `relevant_version_ids` | Array of Strings | No | Document version IDs (e.g. `["VER-0001"]`). Default `[]`. |
| `category` | Enum | No | Benchmark category (see below). |
| `difficulty` | Enum (`easy` \| `medium` \| `hard`) | No | Subjective complexity of retrieval. |
| `query_type` | Enum | No | `factoid`, `procedural`, `definitional`, `comparative`. |
| `notes` | String | No | Rationale for ground-truth relevance or reason for unanswerability. |
| `annotator_id` | String | No | Identifier of the legal or domain expert authoring the example. |

### Benchmark Categories
1. `single_clause`: Direct answer found in exactly one clause.
2. `multi_clause`: Query requires synthesizing information spanning two or more distinct clauses.
3. `clause_number_heading`: Queries explicitly referring to a clause number (e.g. "Clause 5.3") or heading title.
4. `unanswerable`: Out-of-scope, contradictory, or ungrounded queries testing false-positive rejection.
5. `cross_lingual`: Hindi natural language queries matched against English regulatory clauses.
6. `table_annexure`: Queries targeting tabular data or statutory schedules.

---

## 3. Treatment of Unanswerable & Out-of-Scope Queries

Unanswerable queries represent a critical requirement in legal/regulatory RAG to prevent hallucination.

### Representation
```json
{
  "question_id": "EVAL-EN-UNANS-0001",
  "question": "What is the penalty for violating state vehicle registration in this central labor act?",
  "language": "en",
  "is_answerable": false,
  "relevant_clause_ids": [],
  "relevant_document_ids": [],
  "relevant_version_ids": [],
  "category": "unanswerable",
  "difficulty": "hard",
  "query_type": "factoid",
  "notes": "Query asks about vehicle registration, which is absent from labor regulation corpus.",
  "annotator_id": "LEGAL_REVIEWER_1"
}
```

### Retrieval Evaluation Protocol for Unanswerable Queries
1. **Separation from Ranking Metrics:**
   - Standard retrieval metrics ($\text{Recall}@K$, $\text{MRR}$, $\text{NDCG}@K$) require a non-empty ground-truth set ($|R| > 0$). If applied to queries with $|R| = 0$, the denominator is undefined ($0/0$).
   - Therefore, unanswerable queries **must be excluded** from the computation of Recall@K, MRR, and NDCG.
2. **Rejection & Fallback Threshold Evaluation:**
   - Unanswerable queries are evaluated separately using:
     - **Max Retrieval Score Distribution:** Distribution of top-1 scores $\max_{d} S(q, d)$ under BM25 and DenseRetriever.
     - **Rejection Precision / Precision-at-Empty:** The proportion of unanswerable queries where the top retrieval score falls below a confidence threshold $\tau_{\text{refusal}}$.

---

## 4. Ground-Truth Annotation Protocol

To guarantee consistency, human annotators must adhere to the following rules:

1. **No Content Fabrication:** Questions must only be created against verified clauses present in the corpus. Do not synthesize hypothetical regulations.
2. **Authoritative Clause Mapping:** All `relevant_clause_ids` must correspond to real IDs in the corpus (`CLS-XXXXXXX`).
3. **Multi-Clause Relevance Ordering:** For `multi_clause` queries, list clause IDs in order of importance or dependency.
4. **Natural Query Variation:** Questions must range from exact keyword searches to broad semantic questions and Hindi formulations.
5. **Template Usage:** Annotators should start from [`benchmark_template.jsonl`](./benchmark_template.jsonl) as an unpopulated structural guide.

---

## 5. Metrics & Benchmark Methodology (Stage 3, Step 4)

Retrieval evaluation metrics are implemented in [`rag_module.evaluation.metrics`](file:///e:/Coding/NLP_Project/NiyamSetu-diw_backend/rag_module/evaluation/metrics.py) using pure Python (standard library only).

### Metric Definitions

- **Recall@K** ($K \in \{1, 3, 5, 10\}$):
  $$\text{Recall}@K = \frac{|\text{Retrieved}_K \cap \text{Relevant}|}{|\text{Relevant}|}$$
  Where $\text{Retrieved}_K$ is the deduplicated set of retrieved clause IDs in the top $K$, and $\text{Relevant}$ is the non-empty set of authoritative relevant clause IDs. Returns $0.0$ if no relevant items are retrieved.

- **Reciprocal Rank / Mean Reciprocal Rank (MRR)**:
  $$\text{RR} = \begin{cases} \frac{1}{\text{rank}_1} & \text{if a relevant clause is retrieved} \\ 0.0 & \text{otherwise} \end{cases}$$
  Where $\text{rank}_1$ is the 1-based index of the first relevant retrieved clause in the deduplicated ranked list. MRR is the macro-average of RR across all evaluated queries:
  $$\text{MRR} = \frac{1}{|Q|} \sum_{q \in Q} \text{RR}(q)$$

- **Normalized Discounted Cumulative Gain (NDCG@K)**:
  Uses binary relevance ($\text{rel}_i \in \{0, 1\}$) and standard logarithmic discounting:
  $$\text{DCG}@K = \sum_{i=1}^{\min(K, |T|)} \frac{\text{rel}_i}{\log_2(i + 1)}$$
  $$\text{IDCG}@K = \sum_{i=1}^{\min(K, |\text{Relevant}|)} \frac{1}{\log_2(i + 1)}$$
  $$\text{NDCG}@K = \begin{cases} \frac{\text{DCG}@K}{\text{IDCG}@K} & \text{if } \text{IDCG}@K > 0 \\ 0.0 & \text{otherwise} \end{cases}$$

### Macro-Averaging & Subgroup Breakdowns

The evaluator computes macro-averages across queries, with separate breakdowns for:
- **`dataset_group`**:
  - `core`: Core regulatory compliance questions (15 questions in `annotation_worksheet.jsonl`).
  - `auxiliary`: Diagnostic queries (3 bilingual metadata preambles, 1 segmentation stress test).
- **`language`**:
  - `en`: English natural language queries.
  - `hi`: Hindi queries (Devanagari lexical preambles + cross-lingual).

### Usage Examples

#### 1. Evaluating a Single Query
```python
from rag_module.evaluation.metrics import evaluate_query

retrieved = ["CLS-0000007", "CLS-0000008"]
relevant = ["CLS-0000007"]

scores = evaluate_query(retrieved, relevant, k_values=(1, 3, 5, 10))
# Returns: {'recall@1': 1.0, 'recall@3': 1.0, ..., 'mrr': 1.0, 'ndcg@1': 1.0, ...}
```

#### 2. Evaluating a Full Benchmark Against the Annotation Worksheet
```python
from rag_module.evaluation.metrics import evaluate_benchmark, load_ground_truth

# 1. Load ground truth from the human annotation worksheet
gt = load_ground_truth("rag_module/evaluation/annotation_worksheet.jsonl", use_verified=True)

# 2. Collect retriever predictions: dict of {question_id: [retrieved_clause_ids or RetrievedPassage objects]}
predictions = {
    "CAND-EN-0001": ["CLS-0000007"],
    # ...
}

# 3. Evaluate benchmark with macro-averages and subgroup breakdowns
summary = evaluate_benchmark(predictions, gt, k_values=(1, 3, 5, 10))

print("Overall Recall@1:", summary.overall["recall@1"])
print("Overall MRR:", summary.overall["mrr"])
print("Core MRR:", summary.by_group["core"]["mrr"])
print("Auxiliary MRR:", summary.by_group["auxiliary"]["mrr"])
print("English MRR:", summary.by_language["en"]["mrr"])
print("Hindi MRR:", summary.by_language["hi"]["mrr"])
```

---

## 0.5 Automated Benchmark Runner (BM25 vs DenseRetriever)

The benchmark runner `rag_module/evaluation/benchmark_runner.py` provides an isolated, reproducible execution harness that compares BM25 against DenseRetriever on the authoritative evaluation dataset.

### Key Capabilities
- **Ground Truth Grounding**: Evaluates against `human_verified_relevant_clause_ids` in `annotation_worksheet.jsonl`.
- **Corpus Isolation**: Enforces separate indexing of Experiment A (363 clause-aware segments) and Experiment B (138 paragraph segments); never combines them.
- **Metric Suite**: Automatically computes Recall@1, Recall@3, Recall@5, Recall@10, MRR, NDCG@5, and NDCG@10.
- **Macro-Averaging & Stratification**: Generates overall macro-averages alongside breakdowns by group (`core` vs `auxiliary`) and language (`en` vs `hi`).
- **Telemetry & Provenance**: Records query latency (ms), index build time, retrieval scores, and model metadata.
- **Persistent Output**: Saves structured JSON reports to `rag_module/evaluation/results/` with collision-proof timestamps (`benchmark_<strategy>_<experiment>_<timestamp>.json`).
- **Failure Transparency**: Handles missing PyTorch / SentenceTransformers dependencies cleanly without returning silent mock/fake embeddings.

### Running Benchmarks via CLI

#### 1. Run BM25 Benchmark (Zero external dependencies)
```bash
python -m rag_module.evaluation.benchmark_runner --strategy bm25 --experiment clause_aware
```

#### 2. Run DenseRetriever Benchmark
*Prerequisites*: Requires `torch` and `sentence-transformers` installed in the Python environment:
```bash
pip install sentence-transformers torch
python -m rag_module.evaluation.benchmark_runner --strategy dense --model-name intfloat/multilingual-e5-small --device cpu
```

#### 3. Run All Strategies
```bash
python -m rag_module.evaluation.benchmark_runner --strategy all
```

### Python API Usage

```python
from rag_module.evaluation.benchmark_runner import BenchmarkConfig, run_benchmark

config = BenchmarkConfig(
    experiment="clause_aware",
    top_k=10,
    k_values=(1, 3, 5, 10),
)

# Run BM25 benchmark and persist JSON report
result, saved_file = run_benchmark(config=config, strategy="bm25", save_results=True)
print(f"BM25 Overall MRR: {result['metrics']['overall']['mrr']:.4f}")
print(f"Results saved to: {saved_file}")
```


