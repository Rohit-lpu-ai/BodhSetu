# NiyamSetu RAG Module (`rag_module/`)

**Author:** Person 3 (Ask Feature + RAG NLP); completed and integrated 2026-11-16.
**Status:** **COMPLETE AND INTEGRATED.** Hybrid retrieval, context construction,
answer generation and citation verification are implemented, and the pipeline is
mounted in the FastAPI application as `POST /api/ask`, with the Ask page wired
to it.
**Contract Version:** 1.0.0 (aligned with `backend-integration-contract.md` v1.0)

> **Read this first.** Sections 4 and 5 below describe BM25 and dense retrieval
> as the teammate wrote them, and remain accurate. Section 10 is the current
> status. The directory name in section 2 is `rag_module/`, which is also what
> every import in the package expects — the module was committed as `rag/`,
> under which **nothing in it could import at all**; renaming it back was the
> first fix of the integration.

**What was added to complete it**

| Piece | File |
|---|---|
| Hybrid retrieval (Reciprocal Rank Fusion) | `retrieval/hybrid_retriever.py` |
| Context construction with a character budget | `generation/context.py` |
| Grounded system prompt and refusal sentinel | `generation/prompt.py` |
| LLM providers over plain HTTP (no vendor SDK) | `generation/providers.py` |
| Answer generation, plus the labelled no-model path | `generation/generator.py` |
| Citation extraction and verification | `citation/verifier.py` |
| The end-to-end pipeline and sufficiency gate | `pipeline.py` |
| Hybrid added to the benchmark harness | `evaluation/benchmark_runner.py` |

**Measured on the same 19 questions as BM25 and dense** (`evaluation/results/`):

| Metric | BM25 | Dense | **Hybrid (RRF)** |
|---|---|---|---|
| Recall@1 | 0.632 | **0.737** | 0.684 |
| Recall@3 | 0.789 | **0.895** | **0.895** |
| Recall@10 | 0.895 | 0.895 | **1.000** |
| MRR | 0.717 | **0.807** | 0.787 |
| NDCG@10 | 0.760 | 0.830 | **0.838** |

Fusion buys complete recall at ten — every question's relevant clause is found,
which neither component manages alone — for a small loss of top-1 precision
against dense. **19 questions, single-annotator labels accepted under delegated
review.** They are not independently annotated multi-reviewer ground truth and
must not be reported as such.

The backend adapter lives outside this module, in `backend/app/ask/`, so
`rag_module` still imports nothing from `backend`.

---

## 1. Module Purpose

The `rag_module/` is the dedicated, self-contained **Retrieval-Augmented Generation** subsystem for NiyamSetu. It powers the user-facing **Ask** feature by:
1. Consuming verified, pre-segmented clauses and paragraphs produced by Person 1's backend ingestion pipeline.
2. Indexing segments across both **BM25 lexical search** and **FAISS dense multilingual embeddings**.
3. Fusing retrieval rankings via **Reciprocal Rank Fusion (RRF)**.
4. Gating synthesis with a **multi-factor evidence sufficiency engine** to refuse unanswerable queries gracefully.
5. Generating grounded answers and performing **dual-track citation verification** that accounts for extracted text defects (broken words) without fake claims.

---

## 2. Directory Layout

```
rag_module/
├── README.md                      # Module documentation, usage, and status
├── requirements.txt               # Isolated RAG dependencies
├── config/                        # Independent configuration and environment settings
│   ├── __init__.py
│   └── settings.py                # Model tiers, thresholds, backend URLs
├── core/                          # Domain schemas and abstract interfaces
│   ├── __init__.py
│   ├── models.py                  # PageProvenance, ClauseSegment, RAGQuery, Citation, RAGResponse
│   └── interfaces.py              # BaseRetriever, BaseEmbedder, BaseGenerator, BaseCitationVerifier
├── client/                        # Upstream integration and offline fixtures
│   ├── __init__.py
│   ├── exceptions.py              # Typed client exceptions
│   ├── backend_client.py          # Read-only HTTP client consuming /api/documents/.../clauses
│   ├── mock_client.py             # Offline mock client reading verified JSON fixtures
│   └── fixtures/
│       └── mock_segments.json     # Verified snapshot of RBI circular segments
├── embeddings/                    # Dense embedding subsystem (Milestone 3)
│   └── __init__.py
├── retrieval/                     # Retrieval strategies
│   ├── __init__.py
│   └── bm25_retriever.py          # BM25Tokenizer + BM25Retriever (Milestone 2 ✓)
├── reranking/                     # Optional cross-encoder reranker (future)
│   └── __init__.py
├── generation/                    # Grounded prompt synthesis & sufficiency gating (future)
│   └── __init__.py
├── citation/                      # Dual-track verbatim & span verification (future)
│   └── __init__.py
├── evaluation/                    # Academic evaluation harness: Recall@k, MRR (future)
│   └── __init__.py
└── tests/                         # Independent test suite
    ├── __init__.py
    ├── test_models.py             # Data model validation and provenance tests
    ├── test_config.py             # Settings loading and env override tests
    ├── test_client.py             # HTTP client mocking and error envelope tests
    ├── test_mock_client.py        # Offline fixture client tests
    └── test_bm25_retriever.py     # BM25 tokenizer and retriever tests (Milestone 2 ✓)
```

---

## 3. Communication with NiyamSetu Backend

The RAG module communicates with Person 1's backend strictly over HTTP through the documented API contract:

```
FastAPI Backend (port 8000)
       │
       │ HTTP GET /api/documents/{id}/clauses?method=clause_aware
       ▼
rag_module.client.BackendClient
       │ (Deserialization into ClauseSegment dataclasses)
       ▼
RAG In-Memory Index / Pipeline
```

* **No direct database queries**: Does not import `backend.app.db` or run SQL.
* **No PDF re-extraction**: Consumes the 501 stored segments (363 clause-aware + 138 paragraph) with pre-calculated character offsets into `normalized_text`.
* **Offline Fallback**: When the backend server or PostgreSQL is unavailable, `MockBackendClient` loads verified snapshots from `client/fixtures/mock_segments.json` without modifying code.

---

## 4. BM25 Lexical Retrieval (Milestone 2)

### 4.1 Overview

`rag_module/retrieval/bm25_retriever.py` implements an authoritative, deterministic **BM25Okapi** lexical retrieval engine using **Lucene-style IDF** over `ClauseSegment` objects fetched from the NiyamSetu backend. It is the first of three planned retrieval strategies:

| Strategy | Status | Notes |
|---|---|---|
| BM25 lexical | **✓ Complete** | In-process, authoritative pure-Python implementation (0 external dependencies) |
| Dense (FAISS + embeddings) | *Planned* | BGE-M3 / multilingual-E5 candidates |
| Hybrid RRF | *Planned* | Reciprocal Rank Fusion of BM25 + Dense |

### 4.2 Composite Searchable Representation & Field Weighting

To support queries that refer to clause numbers (e.g., `"clause 3(2)"`, `"3(2)"`) or headings (e.g., `"D-SIB Surcharge"`, `"Penal Provisions"`) without mutating source data, `BM25Retriever.index()` uses `compose_searchable_text()`:

```
[Searchable Representation]
clause {clause_number} {clause_number} . {heading} {heading} . {text}
```

**Key properties:**
- **Zero data mutation:** The original `ClauseSegment` instance, its `.text`, `.pages`, and all provenance offsets remain completely unmodified.
- **Weighting & length trade-off:** Repeating `clause_number` and `heading` twice provides a modest $2\times$ term-frequency (TF) boost for structural metadata, making structural queries match with priority. In single-field BM25, prepending and repeating metadata tokens increases the tokenized document length $|D|$. As a mathematical consequence of BM25 document-length normalization ($b=0.75$), segments with longer metadata face a mild score penalty on queries that match only their body text compared to heading-less segments.
- **Natural query matching:** Including the prefix keyword `"clause "` allows both `"clause 3(2)"` and bare `"3(2)"` to match the clause identifier naturally.
- **Safe handling of missing metadata:** If `clause_number` or `heading` is empty or `None`, only the non-empty fields and body text are included.

### 4.3 Tokenization Design (`BM25Tokenizer`)

The tokenizer is designed for **Indian government notifications and regulatory documents**:
- Preserves clause numbers with brackets: `(2)`, `(iv)`, `(2A)`, `(1-a)`, `(3.1)`.
- Preserves complex alphanumeric references: `RBI/2026-27/300`, `D-SIBs`, `Section 45-IA`.
- Preserves percentages and emits numeric sub-tokens: `3.5%` -> `['3.5%', '3.5']`.
- Normalizes currency: `₹` is replaced with `rs ` before tokenization to align with `"Rs."` queries.
- Preserves Indian comma-formatted numbers: `5,00,000` -> `['5,00,000', '500000']`.
- Preserves Devanagari script words (U+0900–U+0963, U+0966–U+097F) while cleanly excluding sentence danda punctuation (`।`, `॥`).
- Retains legal qualifiers (`shall`, `not`, `except`, `prior`, `must`, `under`) that carry binding legal semantics.
- **Per-token sub-token isolation:** Sub-token expansion deduplicates only within the current raw token `t`, preserving true document-level term frequencies for repeated references (e.g. three occurrences of `(2)` produce three occurrences of `(2)` and `2`).

### 4.4 Scoring: Lucene-Style BM25Okapi

BM25 scores are **unbounded ranking scores** reflecting term frequency (TF), inverse document frequency (IDF), and document length normalization. They are **NOT calibrated confidence probabilities**. A score of 2.1 does not mean "2.1× better" than a score of 1.0.

The implementation standardizes on the **Lucene BM25 IDF** formula:

$$\text{IDF}(t) = \ln\!\left(1.0 + \frac{N - df(t) + 0.5}{df(t) + 0.5}\right)$$

Parameters:
- $k_1 = 1.5$ (term frequency saturation parameter)
- $b = 0.75$ (document length normalization parameter)

**Why Lucene-style IDF?**
- **Strictly non-negative:** For all valid document frequencies $1 \le df(t) \le N$, the quotient $\frac{N - df(t) + 0.5}{df(t) + 0.5} \ge 0$. Adding $1.0$ guarantees the logarithm argument is $> 1$, ensuring $\text{IDF}(t) > 0$ strictly.
- **Robust for ubiquitous terms:** Terms appearing in every corpus document ($df = N$) receive a small, non-negative IDF ($+0.087$ for $N=5$, $+0.046$ for $N=10$) rather than negative values, allowing documents to be ranked by term frequency and length normalization rather than dropped.
- **100% Deterministic & Zero-Dependency:** Implemented directly in Python standard library (`math`, `re`, `typing`). No external libraries (`rank-bm25`) are required, ensuring identical score values across all development and production environments.

### 4.5 Usage

```python
from rag_module.client.mock_client import MockBackendClient
from rag_module.retrieval.bm25_retriever import BM25Retriever

# 1. Load segments (use real BackendClient against live backend)
client = MockBackendClient()
segments = client.fetch_corpus_clauses(method="clause_aware")

# 2. Build BM25 index
retriever = BM25Retriever()
retriever.index(segments)

# 3. Retrieve by clause number, heading, or body text
results = retriever.retrieve("clause 3(2)", top_k=3)
results_heading = retriever.retrieve("D-SIB Surcharge", top_k=3)
results_body = retriever.retrieve("capital adequacy ratio 11.5%", top_k=5)

for r in results:
    print(f"[{r.rank}] {r.segment.clause_id}  score={r.score:.4f}")
    print(f"    heading: {r.segment.heading} | clause_number: {r.segment.clause_number}")
    print(f"    {r.segment.text[:100]}...")

# 4. Filter by document or version
results_doc = retriever.retrieve("penalties", top_k=3, document_id="doc_rbi_02")
results_ver = retriever.retrieve("MSME credit", top_k=3, version_id="v2")
```

### 4.6 Known Limitations

1. **Vocabulary mismatch**: BM25 relies on exact lexical token overlap. Paraphrased queries (e.g., querying `"financial institution capital standards"` instead of `"scheduled commercial banks CAR"`) may fail to rank top passages. Dense vector retrieval (Milestone 2 next phase) is designed to solve this.
2. **In-memory corpus**: The current BM25 index stores tokenized lists in RAM. For extremely large corpora (>100,000 clauses), an on-disk inverted index should be considered.
3. **Cross-lingual gap**: A Hindi query will not match English clauses unless translated. Multilingual dense embeddings (BGE-M3) will bridge English and Hindi queries.

### 4.7 Performance & Evaluation Scope

- **Computational Efficiency & Latency Verification**: Indexing throughput and search latency (<1 ms per query) have been verified in unit performance tests using synthetic collections of 501 segments to confirm algorithmic and computational efficiency.
- **Distinction from Real Corpus Evaluation**: These synthetic micro-benchmarks measure raw processing speed only; they do **not** establish retrieval quality, precision, or relevance on real regulatory corpora. Comprehensive retrieval quality evaluation (Recall@k, MRR, NDCG) on actual backend documents will be conducted during the formal academic evaluation phase (Milestone 4).

The planned evaluation matrix for Milestone 4 on the NiyamSetu benchmark set:

| Metric | BM25 | Dense (FAISS) | Hybrid RRF |
|---|---|---|---|
| Recall@5 | Baseline | Expected higher | Expected highest |
| Recall@10 | Baseline | — | — |
| MRR | Baseline | — | — |
| Hindi query retrieval | Poor (vocabulary mismatch) | Good (cross-lingual) | Good |
| Exact clause reference (`RBI/2026-27/300`) | **Excellent** | May miss | Excellent |
| Latency (p95) | <5 ms | ~50–200 ms | ~50–200 ms |
| RAM (5000 segments) | <50 MB | ~1–2 GB | ~1–2 GB |

---

## 5. Dense Retrieval

### 5.1 Architecture

Dense retrieval encodes user queries and corpus segments into geometric vector embeddings, ranking passages by exact cosine similarity. It addresses vocabulary mismatch inherent in lexical search by matching semantic representations.

```
Query ────────► BaseEmbedder.embed_query() ──────► Query Vector (normalized)
                                                         │
                                                         ▼ exact dot product
Segments ─────► BaseEmbedder.embed_texts() ─────► Passage Vectors (normalized)
```

The initial baseline model configuration standardizes on `intfloat/multilingual-e5-small` (384 dimensions) via `Settings.embedding_model_name`.

### 5.2 Key Properties

1. **Injected Embedder**: `DenseRetriever` accepts an injected `BaseEmbedder` instance (e.g. `SentenceTransformerEmbedder` for production, `DeterministicMockEmbedder` or `ControllableMockEmbedder` for offline tests). It does not download or instantiate external models internally.
2. **Composite Searchable Text Parity**: Uses `compose_searchable_text()` from `bm25_retriever.py` to construct passage text from `clause_number`, `heading`, and body text, ensuring both lexical and dense retrievers index equivalent document representations.
3. **Pure-Python Cosine Similarity**: Computes similarity via pure-Python dot products on L2-normalized vectors. No external vector databases or C-extensions are required for in-process retrieval.
4. **Preservation of Negative Cosine Scores**: Scores falling in `[-1.0, 0.0)` are valid geometric similarities and are retained and ranked rather than discarded.
5. **Deterministic Tie-Breaking**: Equal similarity scores are sorted deterministically by score DESC, `segment.ordinal` ASC, and `segment.clause_id` ASC.
6. **Filtering**: Supports conjunctive filtering by `document_id` and `version_id`.
7. **Atomic Index Lifecycle**: Indexing is atomic. If embedding or vector validation fails at any point, the prior index state is preserved without partial corruption. Deduplication follows first-occurrence-wins by `clause_id`.
8. **Evaluation Distinction**: Synthetic and mock test suites verify mathematical mechanics, lifecycle consistency, and interface contracts. Real-world semantic quality and retrieval relevance metrics (Recall@k, MRR, NDCG) are evaluated separately in Milestone 4.

---

## 6. Setup & Running Tests

### 6.1 Running All Tests (Standard Library Only)

Milestone 1 and Milestone 2 tests run strictly on the Python Standard Library with zero external dependencies:

```bash
python -m unittest discover -s rag_module/tests -v
```

### 6.2 Running with Pytest

```bash
python -m pytest rag_module/tests -v
```

---

## 7. Provenance & Multi-Page Character Offsets

In NiyamSetu, physical PDF pages are extracted independently by Person 1's pipeline. The character offsets `[char_start, char_end)` index into each page's **independent `normalized_text` buffer**, starting at index 0.

### Rules for Segment Offsets:
- **Single-page segments** (`page_start == page_end`): `.char_start` and `.char_end` return the character slice on that single page.
- **Multi-page segments** (`page_start != page_end`): Accessing `.char_start` or `.char_end` directly raises a `ValueError`. This intentional guard prevents consumers from treating offsets across pages as a continuous document-level character range (which would result in an inverted range `char_end < char_start` because each page starts at 0).
- **Authoritative Offset Method**: Always use `segment.char_span_for_page(page_number)`. This method validates that `page_number` is touched by the segment and returns that specific page's `(char_start, char_end)` slice.

### Rules for Citation Offsets:
- If a citation specifies `char_offset=(start, end)`, it **must** also provide `page_number`.
- `page_number` must fall within `[citation.page_start, citation.page_end]`.
- For citations spanning multiple pages or where character-level positions are not yet resolved, `char_offset` and `page_number` remain `None`.

---

## 8. Mock Fixture Provenance & Limitations

The offline fixture at `client/fixtures/mock_segments.json`:
- **Provenance**: Manually assembled from Person 1's backend test mocks (`backend/tests/test_api.py`), integration contract examples (`backend-integration-contract.md` v1.0), and committed segmentation annotations (`data/annotations/segmentation/VER-6507f32084d73257.jsonl`).
- **Limitation**: It is **not** an automated wire dump from a live running PostgreSQL database. It is intended for offline unit testing, CI validation, and client verification without requiring a local database server.

---

## 9. Dependencies

### Milestone 1 & 2 (Foundation, BM25 & Dense Retrieval): Zero External Dependencies
Milestones 1 & 2 run strictly on the Python Standard Library (Python 3.10+):
- `dataclasses`, `typing`: Models and interfaces
- `urllib.request`, `urllib.error`: Read-only HTTP client
- `json`: Payload serialization and mock fixture loading
- `unittest`: Test runner and test cases
- `math`, `re`: Authoritative Lucene-style BM25 engine, tokenizer, and pure-Python cosine similarity
- `os`, `pathlib`, `logging`: Configuration and logging

### Optional Production Dependencies:
- `sentence-transformers>=3.0.0`: Multilingual dense embeddings (intfloat/multilingual-e5-small)
- `torch>=2.1.0`: Underlying PyTorch tensor runtime

### Future Milestones (not yet required):
- `faiss-cpu>=1.8.0`: Optional accelerator for large vector indices
- `httpx>=0.27.0`: High-concurrency async client for backend integration

---

## 10. Current Implementation Status

| Milestone | Subsystem | Status | Details |
|---|---|---|---|
| **Milestone 1** | Scaffolding & Core Models | **COMPLETE** | All directories created, `models.py` with multi-page offset guards and page-linked citations, `interfaces.py` contracts. |
| **Milestone 1** | Config & Client Subsystems | **COMPLETE** | `settings.py`, `backend_client.py` with pagination loops and clamping, `mock_client.py` with inverted page-range validation. |
| **Milestone 1** | Foundation Tests | **COMPLETE** | 44 unit tests passing (models, config, client, mock client, malformed envelopes, pagination). |
| **Milestone 2** | BM25 Lexical Retrieval | **COMPLETE** | `bm25_retriever.py` with `BM25Tokenizer` (legal/Devanagari-aware), `compose_searchable_text`, pure-Python Lucene BM25 engine. 72 BM25 tests passing. |
| **Milestone 2** | Dense Semantic Retrieval | **COMPLETE** | `dense_retriever.py` with `DenseRetriever` implementing `BaseRetriever`, pure-Python cosine similarity, composite indexing, atomic state updates, and tie-breaking. 27 dense tests passing. |
| **Milestone 2** | Hybrid RRF | **COMPLETE** | `retrieval/hybrid_retriever.py`. Union of both arms merged on `clause_id`, fused by RRF with k=60, candidate depth independent of top_k, deterministic tie-breaking, honest degradation to BM25 when no embedder is available. 33 tests. |
| **Milestone 3** | Citation & Generation | **COMPLETE** | `generation/` (context budget, grounded prompt, two real HTTP providers, two generators) and `citation/` (marker resolution, two-level quote verification, per-page offsets). An invented source is stripped and reported; an uncited answer loses its sufficiency claim. 87 tests. |
| **Milestone 4** | Academic Evaluation | **PARTIAL** | Hybrid is benchmarked through the same harness as BM25 and dense, on the same 19 questions — see the table at the top. Still missing: a second annotator, a larger question set, and any measurement of *answer* quality as opposed to retrieval quality. |
| **Integration** | FastAPI + frontend | **COMPLETE** | `backend/app/ask/` (corpus adapter, service, schemas, router) and `frontend/chat.html`. `POST /api/ask`, `GET /api/ask/status`. 31 tests. |

