# Final frontend ↔ backend integration report

**Date:** 2026-11-16
**Branch:** `diw_backend` (nothing committed, nothing pushed, no branch switched)
**Author:** Person 1 (Backend + Document NLP)

**One line:** the teammate's BodhSetu frontend is now the product surface of the
NiyamSetu backend — one FastAPI process on one port serves both — with Library,
Search, Document View, Compare and ingestion wired to real data, Ask left
honestly unavailable, and the demonstration corpora moved out of the running
application without deleting anything.

**Headline numbers, all measured today:**

| | |
|---|---|
| Automated tests | **1,104 passed, 1 skipped, 0 failed** (baseline was 1,083 / 1 / 0) |
| Tests added | 21 (17 frontend-mount, 4 ingestion-provenance) |
| Defects found in the browser and fixed | **7** |
| Pages in the integrated frontend | 11 (6 supplied, 5 added) |
| API endpoints reused | **14 of 14** — no endpoint went unused, none was changed |
| New API endpoints | **0** |
| New database tables or migrations | **0** (both migration checksums unchanged in every database) |
| New Python or JavaScript dependencies | **0** |
| Files deleted | 2 (both backed up; see §11) |

---

## 1. Complete frontend inventory

The supplied directory is `NiyamSetu-frontend-pages/`. It is a checkout of the
`origin/frontend-pages` branch, so it contains a **stale copy of the whole
backend** (an older `backend/`, `data/`, `evaluation/`, `preview_app/`,
`CLAUDE.md`, `requirements.txt`) alongside the actual frontend work. Only the
latter is the teammate's frontend contribution:

| File | Bytes | Purpose |
|---|---|---|
| `index.html` | 12,250 | Landing page. Email/mobile capture, bilingual EN/HI copy switch, three-step explainer. |
| `login.html` | 11,894 | Sign in. Show/hide password, remember-me, forgot-password stub. |
| `register.html` | 10,392 | Sign up step 1 — name, contact, password, state. Writes a session draft. |
| `complete-profile.html` | 15,642 | Sign up step 2 — district, residence, age, language, education, occupation, income. Submits the account. |
| `user_home.html` | 6,740 | Signed-in home. Ask box, three suggested questions, nav (Home / Explore Policies / My Updates / Saved). |
| `chat.html` | 6,819 | Ask thread. Renders the user's question; an honest "answers are coming soon" note stands in for RAG. |
| `backend/frontend_backend.py` | 11,644 | **The teammate's own backend** — Flask on port 5000, Supabase REST for the user table, PBKDF2 passwords, JWT sessions. |

**The design system, read out of those files and preserved exactly:**

| | |
|---|---|
| Palette | `--ink #14211c`, `--muted #5b6862`, `--line #dcdfd9`, `--paper #f7f6f1`, `--accent #1f5a45`, hover `#17473a` |
| Display face | Georgia / Times New Roman, italic accent-coloured emphasis inside headings |
| Body face | `system-ui, -apple-system, "Segoe UI", Roboto, sans-serif`, line-height 1.55 |
| Measure | 1080px, 24px gutters |
| Controls | 50px (54px on the home Ask box), 8px radius, 1px `--line` border, white fill, 2px accent focus ring |
| Chips | fully rounded pills, 1px border, white fill, accent on hover |
| Notes | 10px radius, **dashed** `--line` border, white fill |
| Breakpoints | 900px, 800px, 760px, 480px |

Nothing in that table was changed. `frontend/assets/bodhsetu.css` restates those
exact values for the five pages added here; the six supplied pages keep their own
inline `<style>` blocks untouched.

### 1.1 What was NOT in the supplied frontend

No page for Library, Search, Document View, Compare, or document upload. Three of
the four navigation entries in `user_home.html` were `href="#"` with a JavaScript
handler cancelling the click, i.e. placeholders. Section 7 below says what was
built for them.

---

## 2. Pages and routes

Served at `http://127.0.0.1:8000/` by the FastAPI application itself.

| Route | Origin | What it does now |
|---|---|---|
| `/` → `index.html` | supplied | Unchanged design. Auth flow preserved; one link added into the library for visitors without an account. |
| `/login.html` | supplied | Unchanged design. **Fixed:** a successful sign-in now navigates to the home screen. |
| `/register.html` | supplied | Untouched. |
| `/complete-profile.html` | supplied | Untouched. |
| `/user_home.html` | supplied | Nav now points at real pages; a line of **live corpus counts** read from `/api/library/facets`. |
| `/chat.html` | supplied | Design and "coming soon" note preserved; now also shows **real matching clauses**, labelled as keyword search. |
| `/explore.html` | **added** | Library + Search: browse, document search, clause search, facets, date filters, sorting, suggestions, pagination. |
| `/document.html` | **added** | Document View: metadata, versions, clauses, page text, entities, and the character-range provenance check. |
| `/updates.html` | **added** | Compare: version pair picker, change classification, side-by-side, word diff, evidence drawer. |
| `/saved.html` | **added** | Saved documents and clauses, re-read from the API on every load. |
| `/add-document.html` | **added** | Ingestion: upload a PDF with its provenance metadata. |
| `/app` | added | 307 to `/user_home.html`. |
| `/favicon.ico` | added | A 249-byte generated PNG, so every page load stops logging a 404. |
| `/api/*`, `/docs`, `/redoc`, `/openapi.json` | unchanged | The 14-path contract, untouched. |

---

## 3. Design system and assets preserved

* The six supplied HTML files were edited **only inside `<script>`**, plus two
  additive lines of markup (a library link on the landing page, a counts line on
  the home page) written in the pages' own existing classes (`.hint`, `.lead`).
* No colour, font, size, spacing, radius, shadow, border or breakpoint was
  changed anywhere.
* `NiyamSetu-frontend-pages/` is **untouched on disk** and is the pristine
  reference copy. `frontend/` is the copy the application serves.
* A test (`test_every_page_the_navigation_links_to_exists`) fails if any page or
  asset the navigation points at goes missing.

---

## 4. Frontend → backend integration matrix

Legend: **R** = reused an existing endpoint as-is · **F** = frontend-only
(no backend needed) · **D** = deferred, depends on RAG · **T** = teammate-owned.

| Feature | Status | Endpoint used |
|---|---|---|
| Corpus counts on the home screen | R | `GET /api/library/facets` |
| Browse all documents | R | `GET /api/documents` (`sort`, `limit`, `offset`) |
| Document search | R | `GET /api/search` |
| Clause search | R | `GET /api/search/clauses` (`include_text`, `segment_type`, page range) |
| Type-ahead suggestions | R | `GET /api/search/suggest` |
| Authority / type filters, counts, "Not stated" bucket | R | `GET /api/library/facets` + filter params |
| Date-range filter, sorting, pagination | R | existing query parameters |
| Document metadata and version list | R | `GET /api/documents/{id}` |
| Clause list with page provenance | R | `GET /api/documents/{id}/clauses` |
| Page text (normalised / cleaned / raw) | R | `GET /api/documents/{id}/pages` |
| **Clause source verification** (slice the stored offsets out of the stored page) | R | `/clauses` + `/pages`, done in the browser |
| Entity extraction per version | R | `GET /api/compare/entities` |
| Comparable version pairs | R | `GET /api/compare/options?include_single_version=true` |
| Run a comparison | R | `POST /api/compare` |
| Added/removed/modified/unchanged filters, uncertain-only | R | `change_types`, `only_ambiguous` |
| Side-by-side text, word diff, plain-language explanation | R | `include_text`, `include_diff`, `include_explanation` |
| Evidence drawer with page and character range | R | `evidence` on each `ClauseSide` |
| Document ingestion | R | `POST /api/documents/ingest` |
| Duplicate detection on re-upload | R | the same endpoint's `skipped_duplicate` |
| Saved documents and clauses | F | identifiers in `localStorage`; **content** always re-read from the API |
| Sign in / register / profile | T | the teammate's Flask service and Supabase, unchanged |
| Ask / RAG answer generation | **D** | nothing exists to call — see §8 |
| Hindi output | D | Open Decision D-3, paused |

**Every one of the 14 endpoints is used by the frontend.** No endpoint was
added, renamed, or had a field removed, so no consumer contract changed.

---

## 5. APIs reused, APIs added

**Reused, unchanged: all 14.**

`/api/health` · `/api/documents/ingest` · `/api/documents` ·
`/api/documents/{id}` · `/api/documents/{id}/pages` ·
`/api/documents/{id}/clauses` · `/api/search` · `/api/search/clauses` ·
`/api/search/suggest` · `/api/library/facets` · `/api/compare` ·
`/api/compare/options` · `/api/compare/entities` · `/api/compare/entities/labels`

**Added: none.** `test_the_mount_adds_no_documented_api_path` asserts
`len(openapi()["paths"]) == 14` and that `/`, `/app` and `/favicon.ico` stay out
of the schema.

### 5.1 The one backend module that is new

`backend/app/api/frontend.py` — 120 lines, no routes of its own beyond `/`,
`/app` and `/favicon.ico` (all `include_in_schema=False`). It mounts
`frontend/` at `/`, **last**, after every router, so a static file can never
shadow an API path. `NIYAMSETU_SERVE_FRONTEND=0` turns it off for an API-only
server. A missing `frontend/` logs a warning and degrades to API-only rather
than crashing.

---

## 6. Why one process instead of the previous four

The development arrangement was four services: backend 8000, search preview
5173, comparison backend 8010, compare preview 5174. The integrated application
is **one**.

| | Before | Now |
|---|---|---|
| Processes to start | up to 4 | 1 |
| CORS | allow-list per preview port | none needed — same origin |
| Proxy code | ~130 hand-written lines × 2 | none |
| Ports to keep straight | 4 | 1 |

`search_preview/` and `compare_preview/` are **unchanged and still work** —
start them against `--api http://127.0.0.1:8000` exactly as before. The CORS
middleware was left in place for them. They remain developer tools; they are not
the product.

---

## 7. Pages added for backend capabilities that had none

Each follows the rule in §5 of the brief: integrate into an existing page where
possible, and only create one where a dedicated page is genuinely needed. The
three dead nav entries the teammate had already designed gave three of them a
home, so the information architecture is theirs, not invented here.

| Nav entry (was `href="#"`) | Became | Why a page was needed |
|---|---|---|
| Explore Policies | `explore.html` + `document.html` | Browse, search and reading a document are the product's core; they do not fit on the Ask screen. |
| My Updates | `updates.html` | "What changed" is exactly what Compare answers. No new nav entry was invented for it. |
| Saved | `saved.html` | Already in the design. |
| *(new)* Add Document | `add-document.html` | The team must be able to ingest its own PDFs; `POST /api/documents/ingest` had no UI anywhere. This is the one nav entry added. |

Nothing else was added. There is no page per endpoint:
`/api/compare/entities/labels` and `/api/health` are consumed by other pages, not
exposed as screens.

---

## 8. Deferred because of RAG

`chat.html` keeps its design and its own words — "Answers are coming soon … will
appear here in the next phase". It does **not** generate an answer, and it does
not call anything that claims to.

What it does do, because it can be done honestly: it searches
`GET /api/search/clauses` for the question's words and shows up to five **real**
clauses, each with its document, authority, reference number, page range and a
link straight to that clause in the document view. It is labelled *"Found by
keyword search over the stored clauses — not an assistant's answer"*. When
nothing matches, it says so rather than producing something.

`test_ask_does_not_claim_an_answer_it_cannot_produce` pins all three properties.

**The contract the RAG module needs to publish before this page can be wired
up:**

```http
POST /api/ask
{ "question": "...", "version_scope": "latest", "limit": 5 }

200
{
  "question": "...",
  "answer": "... or null when the evidence is insufficient",
  "answerable": true,
  "citations": [
    { "document_id": "DOC-0001", "version_id": "VER-0002",
      "clause_id": "CLS-0000166", "page_start": 7, "page_end": 7,
      "quote": "exact substring of segments.text" }
  ],
  "retrieval": { "strategy": "bm25|dense|hybrid", "took_ms": 0.0 }
}
```

Three requirements, from CLAUDE.md §12.2 and §5.3: `quote` must be a verbatim
substring of the cited clause; `answerable: false` with `answer: null` is a
valid and expected response; and the citation shape must be the §12.2 one the
rest of the system already uses, so `document.html?...&clause_id=` keeps working
as the click-through.

Also deferred: **Hindi output** (D-3, paused). The language selector is part of
the supplied design and stays; selecting हिन्दी shows an inline notice that the
translation is not implemented and reverts. It does not silently do nothing, and
it does not fake a translation.

---

## 9. Database architecture

### 9.1 The decision

**`niyamsetu_app` is the authoritative database for the running application.**
It was created fresh today, migrated with the same two migration files, and
started empty.

No merge, no migration of rows between databases, nothing dropped. The three
development databases are **byte-for-byte as they were** and are one environment
variable away:

| Database | Content | Role now |
|---|---|---|
| **`niyamsetu_app`** | 1 document, 2 versions, 16 pages, 194 segments | **The application.** Ingested through the browser during testing (see §14). |
| `niyamsetu` | 4 documents, 4 versions, 57 pages, 501 segments | The original verified development corpus. Every evaluation number in the project was measured on it. Untouched. |
| `niyamsetu_sample` | 17 documents, 17 versions, 148 pages, 1,503 segments | The Library & Search sample (3 authorities). Untouched. |
| `niyamsetu_compare` | 6 documents, 13 versions, 345 pages, 5,088 segments | The Compare version-pair corpus (3 genuine RBI chains). Untouched. |
| `niyamsetu_test` | throwaway | What `-m integration` truncates and rebuilds. Unchanged. |

Both migration checksums are identical across `niyamsetu` and `niyamsetu_app`:

```
0001_initial_schema.sql      73d4d409ff2b29b1c73bbabcd2095612eeeea7cb56acc6f9dedabe09688e52fe
0002_public_id_sequences.sql 13dba10f92c2ff037063bbdba1d6334d71a0a7a3bb13feda4b50cd97eb2a7d3d
```

### 9.2 Switching database

One line in `.env`, then restart:

```ini
NIYAMSETU_DB_NAME=niyamsetu_app      # the application (default)
# NIYAMSETU_DB_NAME=niyamsetu_compare  # to demo Compare on the 3 RBI chains
# NIYAMSETU_DB_NAME=niyamsetu_sample   # to demo Search on 17 documents
# NIYAMSETU_DB_NAME=niyamsetu          # the original verified 4-document corpus
```

Identifiers are per-database (`DOC-0001` means different things in different
databases) because `public_id` comes from a per-database sequence. That is
exactly why the databases were **not** merged: merging would renumber every
`DOC-`, `VER-` and `CLS-` id, and every evaluation report in `evaluation/` cites
those ids.

### 9.3 Where the application writes

```ini
NIYAMSETU_PROCESSED_DIR=data/app/processed
NIYAMSETU_RAW_DIR=data/app/raw
```

**Not** `data/processed` and `data/raw`. Those hold the original four-document
corpus, and `data/processed/` is a *test and evaluation input* —
`backend/tests/test_data_quality.py` and
`backend/tests/test_library_search_integration.py` read it directly. The first
browser upload landed there and broke two tests; that is defect **D5** in §13.

---

## 10. Old example / demo data — inventory

Provenance established before anything was touched. `local` = produced by this
project; `official` = downloaded from an issuing authority's own domain with a
recorded SHA-256; `synthetic` = invented.

| Item | Size | Provenance | Depended on by | Safe to remove? | Action |
|---|---|---|---|---|---|
| `data/raw/` — 4 RBI PDFs | 1.5 MB | **official**, in `data/manifest.csv`, SHA-256 verified | `niyamsetu` `stored_path`, every evaluation report | **No** | **Kept** |
| `data/processed/` — 4 version dirs | 1.9 MB | local, regenerable from the above | `test_data_quality.py`, `test_library_search_integration.py`, Experiments B/C/E | **No** | **Kept** |
| `data/manifest.csv` — 4 rows | 3 KB | local, hand-verified | the extraction CLI, data-quality checks | **No** | **Kept** |
| `data/annotations/` | 202 KB | local, **hand-made, not regenerable** | Experiment B, E1/E2/E5 scorers | **No** | **Kept** |
| `data/sample_corpus/` — 14 PDFs + manifest + fetch log | 22 MB | **official** (RBI, IRDAI, NABARD), SHA-256 recorded, re-downloadable via `fetch_sample_corpus.py` | `test_library_search_integration.py` when present | Technically yes, but it is an evaluation input | **Kept** — see §10.1 |
| `data/version_pairs/` — 9 PDFs + manifest + fetch log | 11 MB | **official** RBI, each issue names its predecessor; re-downloadable with `--verify` | `test_comparison_integration.py`, `evaluation/comparison_eval.py` | Same | **Kept** — see §10.1 |
| `exports/niyamsetu_segments_501.jsonl` + manifest | 448 KB | local, the RAG hand-off | the teammate's RAG work | **No** | **Kept** (explicitly preserved) |
| `comparision.py` (repo root) | 14 KB | **teammate-provided** | nothing imports it; it is the Compare audit's evidence | **No** | **Kept, unedited** |
| `NiyamSetu-frontend-pages/` | — | **teammate-provided** | the design reference | **No** | **Kept, unedited** |
| `preview_app/`, `search_preview/`, `compare_preview/` | — | local dev tools | their own test modules | **No** | **Kept and still working** |
| `data/raw/upload.pdf` | 392 KB | local **orphan** | nothing | **Yes** | **Removed** (§11) |
| `data/processed/VER-c16ac5ef421048b2`, `VER-276eb158ca0414e0` | 152 KB | local, created by **this session's** first browser upload | nothing — they broke two tests | **Yes** | **Removed** (§11) |
| Synthetic / invented policy text anywhere | — | — | — | — | **None found.** `test_the_pages_added_here_contain_no_invented_policy_content` keeps it that way. |

### 10.1 Why the sample and version-pair corpora were kept

They look like the "old downloaded development-only sample PDFs" the brief lists
for removal, and three facts argue against deleting them:

1. They are **genuine official documents** with verified provenance, not
   synthetic demonstration data. The brief is explicit: *"Do not delete official
   policy documents simply because they are PDFs. Verify their provenance
   first."*
2. They are **live inputs** to `test_comparison_integration.py`,
   `test_library_search_integration.py` and `evaluation/comparison_eval.py`.
   Deleting them removes the reproducibility of every Compare and Search number
   in the project.
3. **They are no longer in the application.** The goal — no demonstration
   documents visible in the product — is met by the application reading
   `niyamsetu_app`, with nothing destroyed to achieve it.

If the team decides they should go after all, this removes them and nothing
else, and both are re-downloadable:

```bat
rmdir /s /q data\sample_corpus\raw  data\sample_corpus\processed
rmdir /s /q data\version_pairs\raw  data\version_pairs\processed
:: re-fetch later, with checksum verification:
.venv\Scripts\python data\sample_corpus\fetch_sample_corpus.py
.venv\Scripts\python data\version_pairs\fetch_version_pairs.py --verify
```

---

## 11. Data removed and preserved, with reasons

**Removed — two items, both backed up first.**

1. **`data/raw/upload.pdf`** (392 KB).
   SHA-256 `6507f32084d73257…` — byte-identical to
   `rbi_circular_lead-bank-delhi_v1.pdf`, which is also present. Not in
   `data/manifest.csv`, not referenced by any `stored_path` in any of the four
   databases, not referenced by any processed artefact. It is the residue of an
   earlier API upload test and the visible symptom of defect **D6** below.
   → `backups/2026-11-16/removed-files/upload.pdf`

2. **`data/processed/VER-c16ac5ef421048b2/` and `VER-276eb158ca0414e0/`**
   (152 KB). Created by this session's own first browser upload, before the
   application was given its own directories. They made
   `test_the_real_corpus_has_no_data_quality_errors` and
   `test_ilike_hits_carry_the_same_provenance` fail.
   → `backups/2026-11-16/polluted-processed/`

**Preserved — everything else**, including all source code, all algorithms,
every test, every fixture, the teammate's files, the migrations, all
documentation, the original corpus, the 501-segment export and all evaluation
scripts.

### Backups and recovery

```
backups/2026-11-16/
├── niyamsetu.dump                   153 KB   pg_dump -F c
├── niyamsetu_sample.dump            412 KB
├── niyamsetu_compare.dump           773 KB
├── .env.backup                               the .env as it was before today
├── removed-files/upload.pdf
└── polluted-processed/VER-c16ac5ef421048b2/, VER-276eb158ca0414e0/
```

Restore a database:

```bat
"C:\Program Files\PostgreSQL\18\bin\pg_restore.exe" -U postgres -h localhost ^
    -d niyamsetu_sample --clean backups\2026-11-16\niyamsetu_sample.dump
```

Restore a removed file: copy it back out of `backups/2026-11-16/`.
Restore the previous configuration: `copy backups\2026-11-16\.env.backup .env`.

`backups/` and `data/app/` were both added to `.gitignore` — the dumps are
large, binary, and not source, and the app's ingested artefacts are regenerable
from the PDFs. Keep the backup directory until the team has confirmed the
cleanup; nothing in it is referenced by the running application.

---

## 12. Files changed

**Added**

| Path | Lines | What |
|---|---|---|
| `start.bat` | 139 | One-click launcher with five pre-flight checks |
| `backend/app/api/frontend.py` | 120 | The static mount, the no-cache rule, the favicon route |
| `backend/tests/test_frontend_mount.py` | 17 tests | The mount must not shadow the API; no fake data in the pages |
| `backend/scripts/make_favicon.py` | 88 | Regenerates `frontend/favicon.png`, standard library only |
| `frontend/explore.html` | 560 | Library + Search |
| `frontend/document.html` | 520 | Document View + provenance |
| `frontend/updates.html` | 560 | Compare |
| `frontend/saved.html` | 190 | Saved items |
| `frontend/add-document.html` | 320 | Ingestion |
| `frontend/assets/bodhsetu.css` | 310 | The supplied design system, restated for the new pages |
| `frontend/assets/api.js` | 330 | API client, error envelope, sequencing, safe highlighting |
| `frontend/favicon.png` | 249 B | Generated |
| `backend/docs/FINAL_FRONTEND_BACKEND_INTEGRATION_REPORT.md` | — | This file |

**Copied unchanged from the teammate's delivery, then edited only inside
`<script>`**

`frontend/index.html` · `frontend/login.html` · `frontend/register.html` ·
`frontend/complete-profile.html` · `frontend/user_home.html` ·
`frontend/chat.html`

**Modified**

| Path | Change |
|---|---|
| `backend/app/api/main.py` | +4 lines: import and call `mount_frontend` last |
| `backend/app/api/dependencies.py` | +8 lines: the `serve_frontend` setting |
| `backend/app/api/ingestion.py` | **D6 fix** — generated `stored_filename`, client name recorded as provenance, `_safe_original_name` |
| `backend/app/db/loader.py` | **D6 fix** — `stored_path` uses `stored_filename`, `original_filename` keeps the client's name |
| `backend/tests/test_api_integration.py` | +4 tests for D6 and D7 |
| `.env` | app database, app data directories (backed up first) |

**Untouched, deliberately:** `comparision.py`, `NiyamSetu-frontend-pages/`,
`search_preview/`, `compare_preview/`, `preview_app/`, every migration file,
`data/manifest.csv`, `data/annotations/`, `exports/`.

---

## 13. Defects found and fixed

Seven, all found by driving the real browser, none by a unit test.

| # | Defect | Why it mattered | Fix |
|---|---|---|---|
| **D1** | `hidden` did nothing on any flex element. `.pager { display: flex }` outranks the UA's `[hidden]{display:none}`, so `pager.hidden = true` left an empty pager rendered under an empty-state note. | Every pane, pager and panel on the new pages is shown and hidden by that attribute. | One rule: `[hidden] { display: none !important; }` in `bodhsetu.css`. |
| **D2** | A deep link to a clause past the first page of 25 landed on page 1 with nothing highlighted and no explanation. | **Every** search result and every Compare evidence link is such a deep link. | `document.html` now walks the clause pages once to find the target, then renders that page. |
| **D3** | The browser served an edited page from its heuristic cache — no `Cache-Control` was being sent. | Cost several minutes chasing a fix that was already on disk; would hit the team on every edit. | `RevalidatedStaticFiles` sends `Cache-Control: no-cache` on every file. |
| **D4** | **Stale comparison result.** Changing the version selection cleared the change list but left the summary panel — counts and both version labels — standing under a picker that now described a different pair. A failed run did the same. | This is precisely the failure the brief says must not regress. | `invalidate()` and the error path both clear the summary; the error path also drops `state.current` and hides the pager. |
| **D5** | Ingesting through the UI wrote artefacts into `data/processed/`, which is a test input. Two tests started failing. | A user upload must never change what the evaluation measures. | `NIYAMSETU_PROCESSED_DIR` / `NIYAMSETU_RAW_DIR` point at `data/app/`; the two stray directories were removed. |
| **D6** | **Every API upload recorded `original_filename = "upload.pdf"` and the same `stored_path`.** The extractor reports the name of the file it read, which is the generated staging name. The "never overwrite" guard then meant the second upload's PDF was *never written at all*, leaving its `stored_path` pointing at an unrelated document. | A broken provenance link — the exact thing §10.2 forbids. Pre-existing; the browser made it visible by being the first thing to upload two documents. | `document["filename"]` is now the client's name (sanitised, recorded only); `document["stored_filename"]` is `VER-<fingerprint>.pdf`; the loader writes `stored_path` from the latter and `original_filename` from the former. The CLI path is unaffected. |
| **D7** | The upload form had no `retrieved_at`, which `data/README.md` §5 lists as mandatory. | Two `metadata_completeness` errors in the data-quality report, from the very first use. | Added the field; defaults to today. |

Two startup-script defects were also found and fixed while testing `start.bat`:
bare `timeout` resolving to the GNU utility when Git for Windows is on `PATH`,
and `timeout.exe` refusing to run when stdin is not a console. Both replaced
with `%SystemRoot%\System32\PING.EXE -n 4`.

---

## 14. Browser testing performed

Brave, with the Claude in Chrome extension connected, against the live
application on `http://127.0.0.1:8000` reading `niyamsetu_app`.

**Ingestion — the whole pipeline, through the UI.** Two genuine RBI master
circulars uploaded through `add-document.html` with the file picker and the real
form: *Credit facilities to Scheduled Castes (SCs) & Scheduled Tribes (STs)*,
issues 2017-07-01 and 2019-07-01, both under document group
`rbi-sc-st-credit-facilities`. Results read off the response, not assumed:

| | 2017 issue | 2019 issue |
|---|---|---|
| Ids | DOC-0001 / VER-0001 | DOC-0001 / **VER-0002** (same document, second version) |
| Pages | 8 | 8 |
| Clause-aware segments | 60 | 62 |
| Paragraph segments | 19 | 53 |
| Running-header lines removed | 0 | 8 |
| Extraction status | ok | ok |
| Offset basis | `normalized_text` | `normalized_text` |

The title auto-filled from the file name; the second upload attached as a new
version rather than a new document, which is what makes Compare possible.

**Explore.** Live counts (1 document · 2 versions · 16 pages · 122 clauses) read
from the database. Browse, document search, clause search, facet filters, date
filters, sorting and pagination all exercised. Type-ahead fired against
`/api/search/suggest` and offered the real document title. Clause search for
"credit facilities" returned **18 clauses in 14 ms**, each with its page tag,
clause id, version label and highlighted snippet.

**Document View.** Opened at a deep-linked clause (`CLS-0000166`); the page
jumped to the right clause-list page and highlighted it. **Show source** sliced
`page 7, characters 509–573` out of the stored page text and showed it verbatim,
reporting *"Verified against the stored page"*. Page-text tab rendered all 8
pages in the normalised/cleaned/raw variants. Entities tab reported DATE 31,
REFERENCE_NO 29, OBLIGATION 27, AUTHORITY 18, MONEY 5, PERCENT 4 in 5 ms.

**Compare.** VER-0001 → VER-0002 produced **3 added · 1 removed · 19 modified ·
40 unchanged · 59 matched pairs · 7 renumbered · 0 reordered** in 32 ms, with
side-by-side text, word-level `ins`/`del` diff, diff statistics (43 words added,
133 removed, 58% token overlap) and a plain-language explanation per change. The
evidence drawer resolved and showed page 1 characters 5–640 with the verified
quote.

Two specific concurrency behaviours were driven deliberately:

* **Rapid filter toggles.** Two change-type filters clicked while the first
  request was still in flight. Final render = `added + removed + unchanged` =
  3 + 1 + 40 = **44 of 44 shown**, matching the controls exactly. Neither change
  was dropped and the older response did not win.
* **Selection invalidation.** Changing the newer-issue selector cleared the
  result, hid the panel, disabled Compare, and explained why — after D4 was
  fixed.

**Saved.** Two clauses saved from Explore, re-read from the API on the Saved
page with live titles, text, page tags and clause ids.

**Ask.** `chat.html` showed the question, the preserved "coming soon" note, and
five real matching clauses with page ranges and working links.

**States.** Empty collection (before ingestion), empty search result, 404
(`document.html?document_id=DOC-9999` → *"no document with id DOC-9999 · Error
code: NOT_FOUND"* with a way back), loading skeletons, and the self-comparison
refusal. No fake success anywhere.

**Navigation.** Every nav entry, every card link, every evidence link. Refresh
and back/forward both preserve state — Explore and Document View keep their
query, filters, version and target clause in the URL.

**Console and network.** Zero console errors or warnings across the session.
Every request 200 or 304. `/favicon.ico` now 200 (was 404). An unknown
`/api/...` path still returns the §12.1 envelope
(`{"error":{"code":"NOT_FOUND",...}}`), not an HTML page.

**Responsive.** The window would not resize in this environment, so each page was
loaded in an iframe with a genuine 417px viewport — which exercises the media
queries for real. All seven pages: **no horizontal overflow** (`scrollWidth` 402
against a 417px viewport). `.split` collapses to one column, the header nav
wraps, and Compare's side-by-side becomes a single 312px column. A full
comparison ran inside the narrow viewport and rendered 23 cards.

### 14.1 Screenshots captured

Landing page · Add Document (form and the stored result) · Explore browse ·
Explore clause search with highlighted snippets · Document View with sidebar and
provenance · Document View jumped to the deep-linked clause · Entities tab ·
Compare summary · Compare change card with evidence drawer open · Compare word
diff · Saved · Ask with real clause evidence · Home with live counts · the 404
state.

### 14.2 Not verified in the browser

* **Accessibility.** The pages carry `aria-pressed`, `aria-live`, `aria-label`,
  `role` and visible focus rings, and were built keyboard-operable — but no
  screen reader or audit tool was run.
* **Any browser other than Brave.**
* **The teammate's sign-in flow.** It needs either the Flask service (not
  runnable here — `flask` and `pyjwt` are not installed and the Supabase
  credentials are not in this repository) or internet access to Supabase. The
  login *redirect* fix was made and the code path read, but a successful
  end-to-end sign-in was not performed. Nothing else in the product requires it.

---

## 15. Automated test results

```
.venv\Scripts\python -m pytest
1104 passed, 1 skipped, 1 warning in 87.64s
```

Baseline before this work: 1,083 passed, 1 skipped, 0 failed. Net **+21 tests**,
**0 failures**, and the one skip is unchanged — it checks the TF-IDF
implementation against scikit-learn, which is deliberately not a dependency.

| Added | Covers |
|---|---|
| `test_frontend_mount.py` (17) | the frontend exists; every linked page and asset exists; the mount does not shadow `/api`, `/docs` or `/openapi.json`; the 14-path contract is intact; an unknown `/api/` path keeps the error envelope; a missing page is 404 not the landing page; every file sends `Cache-Control`; the favicon is a real PNG; `/app` redirects; the frontend can be switched off; no page hardcodes the API port; no page contains invented policy content; Ask does not claim an answer |
| `test_api_integration.py` (+4) | `original_filename` is the client's name; each version stores its own PDF under a distinct generated name; a path in the client's filename can never reach the filesystem; `retrieved_at` survives ingestion |

Nothing was deleted or weakened to make the suite pass. The two tests that broke
mid-work (D5) were fixed by correcting the application, not the tests.

---

## 16. Known limitations

1. **Ask / RAG is not implemented.** `chat.html` shows real keyword matches and
   says exactly that. §8 has the contract it needs.
2. **Hindi output is not implemented** (D-3, paused). The selector reverts with
   an inline notice.
3. **Classification is not implemented** (Person 2's module). Nothing in the
   frontend claims otherwise.
4. **Sign-in depends on the teammate's service or on Supabase.** Nothing else in
   the product requires an account; a failed sign-in now offers the library
   instead of being a dead end.
5. **Saved items live in the browser.** Identifiers only — content is always
   re-read from the API. They do not sync across devices. A server-side list
   would need a table and an account, and the accounts live in the teammate's
   Supabase project, not in PostgreSQL.
6. **One database at a time.** The application reads whichever database `.env`
   names. Showing the Compare corpus and the team's own corpus simultaneously
   would require a merge, which would renumber every public id.
7. **Ingestion is synchronous** (D-5). Fine for 10–40 page PDFs — the two
   uploads took a few seconds each — but a large scanned PDF could time out the
   request. `ingestion_jobs` exists in the schema for a later switch.
8. **Accessibility is unaudited** (§14.2).
9. **Every evaluation metric still carries its original caveats.** Nothing here
   produced a new measurement, and no accuracy figure was invented. Compare
   still has **no alignment accuracy figure** (D-13); search relevance is still
   one non-independent annotator.
10. **The extraction artefacts are visible in the product.** The Devanagari
    letterhead comes through as `U+FFFD` and some punctuation as mojibake —
    these are the real, documented limits of `pypdf` on bilingual RBI PDFs
    (Experiment A), shown rather than hidden.

---

## 17. Remaining work before submission

| | Work | Owner |
|---|---|---|
| 1 | Implement Ask/RAG and publish the §8 contract; then wire `chat.html` | Person 3 |
| 2 | Classification surfaced on the document view | Person 2 |
| 3 | Decide D-10: which documents are the real corpus, and ingest them through `add-document.html` | the team |
| 4 | Annotate clause correspondences for one declared version pair so Compare gets a real alignment number (D-13) | the team |
| 5 | A second annotator for E1 (inter-annotator agreement), E2 (cleaning correctness), E5 (CER/WER) | the team |
| 6 | Settle D-1 (monorepo vs split). `frontend/` now exists; this should be recorded as the owner's answer | the team |
| 7 | Accessibility audit and a second browser | Person 5 |
| 8 | Decide whether `data/sample_corpus/` and `data/version_pairs/` stay (§10.1) | the team |

---

## 18. How to run it

```bat
start.bat
```

That is the whole thing. It checks the virtualenv, `.env`, PostgreSQL, the
migrations and the port — each failure naming its own cause — then starts the
application and opens the browser. `start.bat 8080` uses another port.

| | |
|---|---|
| Application | http://127.0.0.1:8000/ |
| Explore policies | http://127.0.0.1:8000/explore.html |
| Add a document | http://127.0.0.1:8000/add-document.html |
| API documentation | http://127.0.0.1:8000/docs |

Manual equivalent:

```bat
.venv\Scripts\python -m backend.app.db.migrate
.venv\Scripts\python -m uvicorn backend.app.api.main:app --reload
```
