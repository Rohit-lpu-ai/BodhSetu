# RAG: completion and full-stack integration report

**Date:** 2026-11-16
**Branch:** `diw_backend` (nothing committed, nothing pushed, no branch switched)
**Supersedes:** §8 of `FINAL_FRONTEND_BACKEND_INTEGRATION_REPORT.md`, which
described Ask as deferred and specified the contract RAG would have to publish.
That contract is now implemented.

**One line:** the Ask feature works end to end — a question typed into the
existing BodhSetu chat page is answered from the clauses actually stored in the
application's database, with every citation checked against the stored text
before it is shown.

| | |
|---|---|
| Automated tests | **1,572 passed, 0 failed, 0 skipped** (baseline 1,387 + 1 skipped) |
| Tests added | **185** (115 in `rag_module/`, 35 for the Ask API, 37 others touched) |
| Generation | **Live, on Groq `openai/gpt-oss-120b`** — real answers in ~1.5 s |
| New API endpoints | **2** — `POST /api/ask`, `GET /api/ask/status` (14 → 16 paths) |
| New database tables or migrations | **0** |
| New root dependencies | **0** |
| Defects found and fixed | **9** (two pre-existing and fatal; three surfaced only once a real model was called) |
| Retrieval measured | BM25 / dense / **hybrid**, same 19 questions, same harness |

---

## 1. What was actually there

The handover reported Stages 1–3 approved and Stage 4 (hybrid) outstanding.
Inspection found that to be *half* right, and the half that was wrong was
fatal.

| Reported | Found |
|---|---|
| Module at `rag_module/` | Committed as **`rag/`**, while all 25 modules import `rag_module.*` |
| BM25 retrieval complete | True — 353 lines, a good pure-Python Lucene BM25 |
| Dense retrieval complete | True — injected embedder, exact cosine, atomic indexing |
| 304 tests passing | True **once the directory was renamed**; before that, zero could import |
| Hybrid not implemented | True |
| Generation / citation / pipeline | Empty `__init__.py` files only |
| 19-question eval set, benchmark runner, 501-segment export | All true and all good |

### Defect 1 — the module could not be imported at all

```
>>> import rag.core.models
ModuleNotFoundError: No module named 'rag_module'
```

Every file in the package — and the module's own README — expects
`rag_module/`. Renaming the directory back fixed all 25 files at once and is
what the author clearly intended. `pytest.ini` also listed only
`backend/tests`, so a bare `pytest` never ran the RAG suite: a module whose
every import was broken still looked green. `testpaths` now covers both.

---

## 2. What was built

Everything below is new. Nothing working was rewritten — BM25, dense
retrieval, the corpus loader, the metrics and the benchmark runner are the
teammate's code, reused as they stand.

### 2.1 Hybrid retrieval — `rag_module/retrieval/hybrid_retriever.py`

Reciprocal Rank Fusion, `RRF(d) = Σ 1/(k + rank_i(d))`, k = 60.

Ranks are fused, not scores, because BM25 produces unbounded Lucene scores and
the dense arm produces cosine similarities in [-1, 1]. They share no unit and
no scale, so any weighted sum of the raw numbers is arbitrary and changes
meaning with the corpus. RRF throws the magnitudes away and keeps the ordering.

* Both arms are indexed from one list in one call — same query, same corpus.
* Candidates are **merged on `clause_id`**, so one clause reached by two routes
  is one result, not two.
* **Union, not intersection**: a clause only one arm found survives, scored on
  the one rank it has.
* `candidate_depth` (default 20) is independent of `top_k`. Fusing only the
  top 5 of each would discard exactly the evidence fusion exists to recover.
* Deterministic: score DESC, `ordinal` ASC, `clause_id` ASC — the same rule
  both component retrievers already use.
* Filters by `document_id` / `version_id`, applied inside each arm.
* **Degrades honestly.** With no embedder it runs BM25 only, reports
  `active_components = ["bm25"]`, and labels its results `retrieval_method =
  "bm25"` rather than claiming to be hybrid.
* `retrieve_with_detail()` returns each arm's rank and score per result —
  returned *alongside* the passage, never written onto the shared segment
  objects, which would leak one request's ranking into the next.

### 2.2 Context construction — `generation/context.py`

Numbered `[S1]`, `[S2]` blocks, each carrying document title, clause number,
heading and page range.

Markers rather than clause ids are what make verification possible: the model
never sees a `CLS-…` string, so it cannot pattern-match one into existence. A
citation is a marker, and a marker resolves to exactly one retrieved clause.

Also: clause boundaries preserved (a passage is included whole or dropped,
never cut to fit); duplicates removed both by `clause_id` and by normalised
text, so the paragraph and clause-aware views of one provision cannot read to
the model as two corroborating sources; a character budget (12,000, ≈3k
tokens) spent on the best evidence first, with what was dropped reported, not
silently lost.

### 2.3 Prompt — `generation/prompt.py`

Four load-bearing instructions: the extracts are the only truth; refusing is a
*correct* answer and emits a `INSUFFICIENT_EVIDENCE` sentinel; every factual
sentence must carry a source marker; a conclusion the documents do not state
must be prefixed `Interpretation:`.

### 2.4 Providers — `generation/providers.py`

Two real providers, both over plain HTTP with `httpx`, which the backend
already pins — so **grounded generation adds no dependency**.

* `OpenAICompatibleProvider` — the `/chat/completions` shape, which covers
  OpenAI, Groq, Together, OpenRouter, DeepSeek, Mistral, Azure OpenAI, LM
  Studio and vLLM. Switching vendor is a base URL and a key, not a new SDK.
* `OllamaProvider` — a local daemon; no key, nothing leaves the machine.
* `MockProvider` — tests only, and **never auto-selected**.

Typed failures (`ProviderUnavailable`, `GenerationTimeout`,
`InvalidModelResponse`) each map to a distinct API status. Error bodies are
scanned for credential-shaped text and withheld rather than relayed.

> **Defect 2, fixed.** `Settings.llm_provider` defaulted to `"mock"`. A
> deployment that simply forgot to configure a model would have put scripted
> text in front of a user as if it were an answer. The default is now `"none"`.

### 2.5 Generation — `generation/generator.py`

`GroundedGenerator` is the real path: build context, call the model, detect the
refusal sentinel, return the raw answer for verification.

`ExtractiveGenerator` runs when no model is configured. It **writes nothing** —
it selects sentences verbatim from the retrieved clauses by IDF-ish term
overlap with the question, attaches each one's source marker, and says in the
answer that it was assembled rather than written. `generation_mode` is
`extractive` in the API and the UI shows an amber pill. This is the honest
answer to "finish everything else and clearly identify the configuration
needed": the product works today and gets better the moment a key is pasted
in, and at no point is composed-from-source text presented as model prose.

### 2.6 Citation verification — `citation/verifier.py`

Three checks, in order: does the marker exist in what the model was given; does
it resolve to a clause retrieval actually returned; is the quote really in that
clause.

Quote matching has two levels, and the distinction is reported rather than
smoothed over. `STRICT_VERBATIM` is a literal substring. `AUDITED_SPAN_MATCH`
matches after whitespace normalisation — necessary because this corpus comes
out of `pypdf` on bilingual RBI PDFs where a word can arrive broken across a
line. Punctuation is deliberately *not* normalised, so "shall not …" cannot
match "shall …". Anything else is `REJECTED` and never returned.

`page_number` and `char_offset` are filled only for single-page clauses,
because NiyamSetu offsets index each page's own `normalized_text` and there is
no continuous range across a page break.

> **Defect 3, fixed.** The quote trimmer appended an ellipsis, which made every
> quote drawn from a clause longer than 320 characters fail both match levels
> and lose its citation silently. Found by asking a real question against the
> live corpus: four markers in the answer, two citations in the response. The
> quote is now a verbatim prefix with no ellipsis; `quote_truncated` tells the
> UI to add its own.

### 2.7 Pipeline and the sufficiency gate — `pipeline.py`

`retrieve → sufficiency → context → generate → verify → RAGResponse`. Every
stage can stop, and every stop produces a *stated* outcome.

The gate exists because retrieval always returns something: BM25 will rank the
least-bad clause in a banking corpus for a question about Portuguese tax law,
because "capital" and "rate" appear in both. Catching that before the model is
asked is cheaper and more reliable than hoping the model refuses.

Two signals, both **IDF-weighted** using `BM25Retriever.term_idf` (added for
this; an unseen term scores at the df=0 ceiling):

* `term_coverage` — share of the question's IDF mass the best passage contains.
* `unseen_term_share` — share made of words the corpus has **never seen**. This
  is the decisive one, and it is what makes the refusal message actionable:
  *"The documents in this collection do not mention "renew" or "passport"…"*.

> **Defects 4 and 5, fixed.** Unweighted counts let the Portugal question
> through on its two commonest words. Weighting fixed that but then refused
> *"Which banks **must** set up special **cells**?"* — "must" counted as a
> missing topic word, and "cells" did not match "cell". The gate now uses its
> own wider stopword list (modals are legal weight in retrieval and noise in
> topicality, so the two lists must not be merged) and folds regular plurals.
> Measured after the fix, on the live corpus:

| Question | Verdict | coverage | unseen |
|---|---|---|---|
| Which banks must set up special cells for SC and ST borrowers? | answered | 0.86 | 0.00 |
| What credit facilities are available to Scheduled Castes? | answered | 1.00 | 0.00 |
| What are the reporting requirements for banks? | answered | 1.00 | 0.00 |
| Which bank is the Lead Bank for Old Delhi? | refused | 0.34 | 0.66 |
| What is the capital gains tax rate in Portugal? | refused | 0.16 | 0.84 |
| How do I renew my passport? | refused | 0.00 | 1.00 |

(The Lead Bank refusal is correct: that circular is not in this database.)

---

## 3. Backend integration — `backend/app/ask/`

Mounted into the existing application; no second server, no second catalogue.

### 3.1 The corpus is the live database, not the JSONL export

This is the decision that matters most, and the brief's instinct to "use the
existing production corpus" needed one correction.

`exports/niyamsetu_segments_501.jsonl` is a snapshot of the original
four-document corpus, and the retrieval benchmark was measured on it — so it
stays exactly where it is and `rag_module/evaluation/` keeps using it.

It is the wrong thing to **serve** from, because its ids are `niyamsetu`'s ids.
The application reads whichever database `.env` names, and `public_id` comes
from a per-database sequence, so `DOC-0001` in the export and `DOC-0001` in
`niyamsetu_app` are different documents. An answer indexed from the export
would cite `CLS-0000166` and link the reader to a clause saying something else
entirely — a citation that resolves to the wrong text is worse than none.

So `corpus.py` builds the index from the live corpus through the same
repository every other endpoint reads. A document ingested through *Add
Document* becomes answerable on the next question, with no second catalogue to
keep in step and no re-extraction.

**Clause-aware only.** Both methods are stored and cover the same words twice;
indexing both would put two views of one provision into the top-k and read to
the model as corroboration. `include_paragraph=True` exists for experiments.

**Latest OK version only.** Older versions stay searchable through Library and
comparable through Compare, but an assistant answering from three issues of one
master circular will quote a superseded provision beside the one that replaced
it, with nothing in the ranking to say which is current.

### 3.2 Index lifetime

One process-wide index, invalidated by a **fingerprint** over
`corpus_counts()` — one cheap SQL round trip that moves when anything is
ingested. A TTL would either rebuild constantly or serve a stale corpus after
an upload.

> **Defect 6, fixed.** The first build took **105 seconds** (model download
> plus embedding). Inside the first `POST /api/ask` that is a browser timeout,
> not an answer. `ensure_ready()` now builds the BM25 index synchronously —
> **53 ms** measured — and answers from it immediately, while a daemon thread
> loads the embedder and swaps in the hybrid retriever when ready (**15.3 s**
> with the model cached). The swap is atomic under the same lock, and a dense
> index whose corpus has since changed is discarded rather than installed.
> `GET /api/ask/status` reports `dense_state` as `loading | ready |
> unavailable | disabled`, so the UI states which is in play.

### 3.3 Endpoints

`POST /api/ask` — question in; answer, citations, retrieved passages,
warnings, status and diagnostics out. `GET /api/ask/status` — corpus size,
retrieval components, generation provider. **The API key is never in any
payload**; only `api_key_present: true`.

Handled explicitly: empty question (422 envelope), over-long question
(`question_too_long`, not truncated — truncating a question changes what was
asked), no retrieved passages, insufficient evidence, provider unavailable,
generation timeout, invalid model response, citation failure, empty corpus, and
an unexpected internal error (500 envelope, detail to the log only).

The §12.1 error envelope, the `public_id` convention and the Compare module's
citation shape are all reused, so the frontend's existing deep link —
`document.html?document_id=…&version_id=…&clause_id=…` — works on an Ask
citation with no new routing.

---

## 4. Frontend integration — `frontend/chat.html`

Design preserved: same palette variables, same header/main/footer skeleton,
same `.note` / `.msg-user` / `.bubble` classes. The assistant bubble mirrors
the user bubble (same radius idiom, white ground instead of accent). Nothing
else on the page moved, and `Ask` joined the existing nav.

What it now does:

* Reads `GET /api/ask/status` on load, so the user is told **before** typing
  when no model is configured or the collection is empty.
* Shows a thinking state, supports multiple turns, and keeps the send button
  disabled while a question is in flight. A stale response can never overwrite
  a newer one.
* Renders `[S1]` markers inline as **links to the cited clause**, with the
  document title and page in the tooltip.
* Renders a **Sources** block: clause number, document, page, clause id, the
  verified quote, and whether it passed the strict or the normalised check.
* Shows `generation_mode` as a pill — green *"written by a language model ·
  <model>"* or amber *"assembled from the source text — no language model
  configured"*.
* When generation fails, shows **"Clauses that matched your question"** instead
  of Sources — real stored text with real page references, deliberately headed
  differently because nothing was answered from them.
* **Nothing is rendered as HTML.** `innerHTML` appears nowhere; a test asserts
  it.

---

## 5. Browser testing (Brave, Claude in Chrome)

All against the live application on `http://127.0.0.1:8000` reading
`niyamsetu_app` (1 document, 62 clauses).

| Flow | Result |
|---|---|
| Ask page loads, status read | "No language model is configured" notice; "62 clauses from 1 document" |
| *"What special cells must banks set up for Scheduled Castes?"* | Answered, **4 verified citations**, 18 ms, hybrid |
| Click `[S2]` | Navigated to `document.html?document_id=DOC-0001&version_id=VER-0002&clause_id=CLS-0000157`, **the target clause highlighted on page 5** |
| *"How do I renew my passport in Portugal?"* | *"not answered from the documents"*, names "renew", "passport", "portugal", **0 citations** |
| With a model configured (local stub) | Generated prose, inline `[S2]`/`[S1]` links, `Interpretation:` kept separate, pill *"written by a language model · stub-model"*, 2 verified citations, 257 ms |
| Follow-up question in the same thread | 2 user turns, 2 assistant turns, 4 citation links; second turn shows *"hybrid retrieval"* — the dense arm finished loading mid-session, as designed |
| Model endpoint killed mid-session | *"The answer could not be generated… Could not reach the model endpoint… ConnectError."* plus **"Clauses that matched your question"** with 4 real clauses |
| Console / network | **Zero console messages**; every request 200 |
| Regression: index, home, explore, document, updates, saved, add-document | All load, no error states, nav intact on all |

**The generated-answer test used a local stub**, `tmp/stub_llm.py`, speaking
the OpenAI `/chat/completions` shape. It is not a model and its reply is a
fixture. What it proves is the **wire path**: provider resolution → HTTP call →
answer parsing → marker extraction → citation verification against the real
corpus, with real token counts. Both citations it produced resolved to real
clauses (`CLS-0000157` page 5, `CLS-0000139` page 2) as `STRICT_VERBATIM`.
**No hosted model was called, because no API key exists on this machine**, and
no generated answer from a real provider is claimed anywhere.

---

## 6. Retrieval evaluation

Hybrid added to the existing harness and run on the same 19 questions, same
corpus (363 clause-aware segments), same metrics:

| Metric | BM25 | Dense | **Hybrid (RRF)** |
|---|---|---|---|
| Recall@1 | 0.6316 | **0.7368** | 0.6842 |
| Recall@3 | 0.7895 | **0.8947** | **0.8947** |
| Recall@5 | 0.8947 | 0.8947 | 0.8947 |
| Recall@10 | 0.8947 | 0.8947 | **1.0000** |
| MRR | 0.7167 | **0.8070** | 0.7865 |
| NDCG@10 | 0.7604 | 0.8296 | **0.8378** |
| Latency | 0.93 ms | 35.6 ms | 27.9 ms |
| Index time | ~0 s | 50.4 s | 19.0 s |

Read honestly: **fusion buys complete recall at ten — every question's relevant
clause is found, which neither component manages alone — at a small cost in
top-1 precision against dense.** For an assistant that puts six passages in
front of a model, recall is the metric that matters; for a search box showing
one result, dense alone would be the better pick. Hindi questions reach
Recall@3 1.000 under hybrid.

**The caveat, preserved.** 19 questions, single-annotator labels accepted under
delegated review. Not independently annotated multi-reviewer ground truth, and
not to be reported as such. There is still **no measurement of answer quality**
— only of retrieval.

---

## 7. Configuration

Nothing is required. With an empty `.env` the assistant retrieves, assembles a
labelled extractive answer and verifies its citations.

```ini
# Generated answers (optional). One setting covers every OpenAI-compatible
# vendor; no SDK is installed.
NIYAMSETU_RAG_LLM_PROVIDER=groq        # openai|groq|together|openrouter|
                                       # deepseek|mistral|ollama|auto|none
NIYAMSETU_RAG_LLM_MODEL=llama-3.3-70b-versatile
NIYAMSETU_RAG_LLM_API_KEY=...          # .env is git-ignored
# NIYAMSETU_RAG_ENABLE_DENSE=0         # lexical only; skips the embedding pass
```

Full list with commentary in `.env.example`. Dense retrieval needs
`rag_module/requirements.txt` (`sentence-transformers`, `torch`) — **not** the
root `requirements.txt`, the same split `preview_app/` uses, so the backend
still installs and passes its tests without them.

Starting the application is unchanged: **`start.bat`**. Verified clean after
all of this.

---

## 8. Tests

```
.venv\Scripts\python -m pytest
1572 passed, 1 warning in 84s
```

Baseline: 1,387 passed, 1 skipped (1,083 backend + 304 RAG). Net **+185**, zero
failures, and the one skip is gone — `sentence-transformers` brings
scikit-learn, so `test_comparison_unit.py`'s TF-IDF check against scikit-learn
now runs and **passes**, confirming the stdlib implementation in
`comparison/similarity.py` to 1e-9. CLAUDE.md recorded that as deliberately
unverified; it is now verified.

| Added | Covers |
|---|---|
| `rag_module/tests/test_hybrid_retriever.py` (33) | RRF arithmetic against hand-computed values; agreement outranks a lone top hit; union not intersection; no clause twice; filters; depth ≥ top_k; determinism; honest `bm25` labelling without an embedder; detail must not mutate shared segments |
| `rag_module/tests/test_generation.py` (44) | Marker numbering; dedup by id and by text; budget drops the tail; truncation declared; prompt order and refusal token; refusal detection; each typed failure captured not raised; provider resolution incl. a hosted provider without a key; `describe_configuration` never returns the key; error bodies with credential-shaped text withheld; **every extractive sentence verbatim from a source** |
| `rag_module/tests/test_citation_and_pipeline.py` (43) | Marker forms; strict vs audited vs rejected; **"shall not" cannot match "shall"**; invented marker dropped and reported; uncited answer flagged; multi-page clause claims no single offset; long-clause regression; gate on/off topic; modals not counted; plural folding; off-topic never reaches the model |
| `backend/tests/test_ask_api.py` (31) | Repository → segment mapping with provenance; partial extraction skipped; unusable row skipped not fatal; fingerprint moves/stable/unreadable; answered with citations; deep-link fields present; fusion detail; validation; every error status; **status never contains the key**; extractive never labelled `llm`; index rebuilds on corpus change |
| `rag_module/tests/test_citation_and_pipeline.py` (+8) | `TestRealWorldMarkerBrackets` — the bracket forms a real model emits (§8a); that a clause number is still not a marker |
| `backend/tests/test_ask_api.py` (+4) | Two "no model configured" tests made **hermetic** — they relied on `.env` being empty and broke the moment a real key was added; plus `.env` is loaded by the service, and explicit settings suppress that |
| `frontend/check_chat_render.mjs` (14 checks) | The page's own marker and bold regexes, executed under Node |
| Updated | Route-count pins 14 → 16 (by name and by count); `llm_provider` default; Ask-page contract tests rewritten for the real endpoint |

No test was deleted or weakened. The two tests that changed assertions did so
because the behaviour they pinned was deliberately improved (the `mock`
default, and a capitalised failure message), and each now explains why.

---

## 8a. Turning on a real model — what the first live call exposed

The owner added a Groq key to `.env` on 2026-11-16. Three defects surfaced
immediately, none of which any amount of stub testing would have found, because
each is about the gap between what the prompt asks for and what a real model
and a real account actually do.

### Defect 7 — the key was in `.env` and the service could not see it

`rag_module` deliberately imports nothing from `backend`, so it reads
`os.environ` and nothing else — which means something has to put `.env` there
first. The backend's own `Settings.from_env()` calls `load_dotenv()`, so Ask
worked *only if* backend settings happened to be constructed first. In the
running server they were; in any other entry point they were not.

```
provider : none      <- with NIYAMSETU_RAG_LLM_PROVIDER=groq sitting in .env
key set  : False
```

`AskService.__init__` now calls `load_dotenv()` itself when no explicit
settings are passed. A test pins it, and a second test pins the converse: a
caller that *supplies* settings must not have the file read back in, which is
what keeps the unit tests hermetic.

### Defect 8 — the configured model did not exist on this account

```
HTTP 404  "The model `llama-3.3-70b-versatile` does not exist
           or you do not have access to it."
```

The key authenticated — a bad key is a 401 — but the model was not on the
account. `GET /v1/models` listed what is: `openai/gpt-oss-120b`,
`openai/gpt-oss-20b`, `qwen/qwen3.8-27b` (plus audio and guard models). All
three answer correctly; `.env` now names **`openai/gpt-oss-120b`**, the
strongest, at ~1 s per call. Nothing in the code changed — the error message
was already actionable enough to diagnose this in one read, which is the point
of relaying the provider's own text.

### Defect 9 — the model cites with `【S2】`, not `[S1]`

This is the one that mattered. `gpt-oss` emits the **CJK lenticular bracket**
form, and also a narrow no-break space (U+202F) before it. The marker pattern
was ASCII-only, so:

```
citations: 0 | sufficient: False
warnings : ['This answer carries no citation that could be checked...']
```

A correct, well-sourced answer lost every citation — and the verification layer
dutifully reported the result as unciteable. Correct behaviour reached from a
wrong premise, which is the worst failure mode a verification layer can have,
because it looks exactly like the system working.

`MARKER_PATTERN` now accepts ASCII `[ ]`, CJK lenticular `【 】` and fullwidth
`［ ］`, plus optional whitespace inside. The frontend's split regex and the
pipeline's invented-marker stripper were widened to the same set — all three
must agree or a citation renders as plain text, or a dead reference survives.

**This widens punctuation only.** A marker still has to resolve to a clause
retrieval actually returned, and its quote still has to verify against the
stored text. `test_citation_and_pipeline.py::TestRealWorldMarkerBrackets` pins
all of it, including that `[3.2]` and `【note】` are still *not* markers.

### Also fixed

`**bold**` reached the page as literal asterisks, because the renderer only
knew about citation markers. `appendInline()` now renders that one construct —
and only that one — still without ever parsing HTML.

### Verified live

```
STATUS  retrieval=['bm25', 'dense'] dense=ready | generation=groq openai/gpt-oss-120b

Q: What special cells must banks set up for Scheduled Castes?
   status=answered mode=llm citations=1 sufficient=True 1487ms
   Banks are required to establish a **special cell at their Head Office**
   that monitors the flow of credit to Scheduled Caste (SC) and Scheduled
   Tribe (ST) beneficiaries, gathers relevant data from branches,
   consolidates it, and submits the required returns to the RBI and the
   Government【S2】.
     [S2] CLS-0000157 p5 STRICT_VERBATIM

Q: What are the reporting requirements for banks on SC/ST advances?
   status=answered mode=llm citations=1 sufficient=True 1845ms
     [S1] CLS-0000161 p6 STRICT_VERBATIM

Q: How do I renew my passport?
   status=insufficient_evidence mode=none citations=0 sufficient=False 14ms
   The documents in this collection do not mention “renew” or “passport”...
```

The off-topic question is refused in **14 ms and never reaches the model** —
the sufficiency gate is doing its job, and it is also saving tokens.

### What was NOT done in the browser

**The Claude-in-Chrome extension was not connected for this session**, so the
live-model answers were verified over HTTP, not by clicking. The earlier
browser pass (§5) covered the page itself and still stands; what changed since
is only the *shape of the text the page renders*. That is covered by
`frontend/check_chat_render.mjs`, which executes the page's own regexes
straight out of `chat.html` under Node:

```
node frontend/check_chat_render.mjs
  ok   ascii [S1]            ok   cjk lenticular 【S2】     ok   fullwidth ［S3］
  ok   grouped [S1, S2]      ok   narrow nbsp before marker  ok   mixed styles
  ok   a clause number is not a marker                       ok   no markers
  ok   bold in the middle    ok   lone asterisks are left alone
  ok   no innerHTML anywhere ok   calls POST /api/ask        ok   reads /api/ask/status
All checks passed.
```

**A click-through of a live-model answer is still outstanding** and is the one
thing to do first when the extension is reconnected.

---

## 9. Known limitations

1. **Only one hosted provider has been exercised.** Groq, with
   `openai/gpt-oss-120b`, is confirmed working end to end. The other
   OpenAI-compatible vendors share the same code path but have not been
   called; Ollama has not been run at all.
   **A click-through of a live-model answer in the browser is outstanding**
   (§8a) — the extension was not connected when the key was added.
2. **Answer quality is unmeasured.** Retrieval has numbers; generation does
   not. Measuring it needs reference answers, which do not exist.
3. **The 19-question set is small and single-annotated** (§6).
4. **No reranker.** `rag_module/reranking/` is still an empty scaffold. A
   cross-encoder is the obvious next gain, and would need a written
   justification under rule 5.
5. **One question at a time.** There is no conversation memory — a follow-up is
   retrieved independently, not resolved against the previous turn. "What about
   for STs?" will not work; the full question will.
6. **Latest version only** (§3.1). Asking about a superseded provision is
   Compare's job.
7. **Dense index is in memory and rebuilt per process.** 15 s for 363 clauses
   with the model cached; linear in corpus size. Past a few tens of thousands
   of clauses this needs to be persisted.
8. **The extraction artefacts are visible.** The Devanagari letterhead reaches
   quotes as `U+FFFD`. That is the documented limit of `pypdf` on bilingual RBI
   PDFs, shown rather than hidden.
9. **`comparision.py`, `NiyamSetu-frontend-pages/` and the three `*_preview/`
   directories are untouched**, as are all three development databases and both
   migrations.

---

## 10. Files changed

**Added** — `rag_module/retrieval/hybrid_retriever.py`,
`rag_module/generation/{context,prompt,providers,generator}.py`,
`rag_module/citation/verifier.py`, `rag_module/pipeline.py`,
`backend/app/ask/{__init__,corpus,service,schemas,api}.py`,
`rag_module/tests/{test_hybrid_retriever,test_generation,test_citation_and_pipeline}.py`,
`backend/tests/test_ask_api.py`, this report.

**Renamed** — `rag/` → `rag_module/` (Defect 1).

**Modified** — `rag_module/{__init__ files,config/settings.py,retrieval/bm25_retriever.py (term_idf),evaluation/benchmark_runner.py (hybrid)}`,
`backend/app/api/main.py` (+4 lines: mount the router), `pytest.ini`
(`testpaths`), `.env.example`, `rag_module/requirements.txt`,
`frontend/chat.html`, `frontend/assets/api.js` and `frontend/user_home.html`
(nav), `rag_module/README.md`, and four test files whose route-count or
behaviour pins moved.

**Untouched** — `exports/` (the 501-segment export and its manifest),
`rag_module/evaluation/{annotation_worksheet,candidate_eval_questions,
benchmark_template}.jsonl` and the existing BM25/dense result files,
`requirements.txt` at the root, both migrations, all three databases.

---

## 11. What is left

| | Work | Owner |
|---|---|---|
| 1 | **Done** — Groq key added, `openai/gpt-oss-120b` confirmed answering with verified citations (§8a) | — |
| 1b | Click through a live-model answer once the Chrome extension reconnects | Person 1 |
| 2 | Decide D-10 and ingest the real corpus through *Add Document* | the team |
| 3 | A second annotator and a larger question set (§6) | the team |
| 4 | Measure answer quality, not just retrieval | Person 3 |
| 5 | Conversation memory for follow-up questions (§9.5) | Person 3 |
| 6 | Classification — the last unimplemented module | Person 2 |
