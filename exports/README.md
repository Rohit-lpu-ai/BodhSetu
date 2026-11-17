# `exports/` — the segment corpus, for Person 3 (Ask / RAG)

A read-only snapshot of the **original 4-document development corpus** as
JSONL: one record per segment, with page and character provenance attached, so
every passage you retrieve can still be turned into a citation.

| | |
|---|---|
| Source database | **`niyamsetu`** — the verified 4-document corpus |
| Records | **501** segments — 138 `paragraph`, 363 `clause_aware` |
| Documents / versions / pages | 4 / 4 / 57 |
| PageProvenance objects | 552 |
| Format | `jsonl/niyamsetu-segments/1.0.0`, UTF-8, LF line endings |
| Produced by | `backend/scripts/export_segments.py` |

Full field-by-field provenance, the document list and the validation results
are in **`niyamsetu_segments_export_manifest.json`**. Read that before using
the file; this page is the summary.

---

## Four things to know before you index this

**1. Both segmentations are in the file, over the same text.** 138 `paragraph`
and 363 `clause_aware` records cover the *same* 57 pages, segmented two ways.
**Filter on `segmentation_method` and index one of them**, or the same passage
will be retrieved twice and your results will look better than they are.
`clause_aware` is the one with clause numbers and headings; `paragraph` is the
baseline condition (`CLAUDE.md` §13.2).

**2. `clause_id` is the citation key.** It is `segments.public_id`
(`CLS-0000123`) — the same identifier `GET /api/documents/{id}/clauses`
returns, so anything you store against it lines up with the API. There is no
separate `clause_id` column in the schema; nothing here was invented.

Use **`fingerprint`** (`SEG-<16 hex>`) if you are persisting an index: it is
derived from `(version, method, ordinal)` rather than from a sequence, so it
survives a re-ingestion that would renumber `public_id`.

**3. `char_start` / `char_end` index `pages.normalized_text`, not `raw_text`.**
Every record carries `offset_basis` saying so. Slice the wrong column and the
span will be wrong — this is D-11 in `CLAUDE.md`, answered and verified. All
501 clauses were confirmed to rebuild from their stored offsets as part of this
export.

**4. A clause that spans a page break has two entries in `pages`.** It is never
collapsed onto one page. Use `pages` for precise highlighting;
`page_start`/`page_end` are the cheap summary for a citation line.

---

## One record

```json
{
  "clause_id": "CLS-0000126",
  "fingerprint": "SEG-...",
  "document_id": "DOC-0003",
  "version_id": "VER-0003",
  "segmentation_method": "clause_aware",
  "segment_type": "clause",
  "ordinal": 1,
  "clause_number": "1",
  "heading": null,
  "text": "1. Planning Process",
  "page_start": 2,
  "page_end": 2,
  "offset_basis": "normalized_text",
  "pages": [
    {"page_number": 2, "char_start": 0, "char_end": 19,
     "line_index_start": 0, "line_index_end": 0}
  ]
}
```

`clause_number` and `heading` are **nullable** and genuinely null a lot:
`clause_number` is null on all 138 paragraph records (a schema `CHECK` forbids
it there) and on 5 clause-aware ones; `heading` is non-null on only 2 of 501.
Nulls are preserved, not replaced with empty strings.

---

## Reproduce it

```
.venv\Scripts\python -m backend.scripts.export_segments --database niyamsetu --expect 501
```

The export is **deterministic** — rows are ordered by
`(document, version, method, ordinal)` — so re-running against an unchanged
database reproduces the file byte for byte, and therefore the same SHA-256.
That was verified: two consecutive runs produced identical files.

`--expect 501` makes exporting the wrong database fail loudly rather than
quietly: `niyamsetu_sample` holds 1,503 segments and `niyamsetu_compare` holds
5,088.

The script opens its connection with `default_transaction_read_only=on`, so
**PostgreSQL itself refuses** INSERT, UPDATE, DELETE, CREATE and DROP on it.
That is a per-connection option; it changes nothing about the database's
configuration.

---

## Not committed

`*.jsonl` here is git-ignored, the same policy `data/processed/` follows: it is
regenerable from the database plus the script, and bulk extracted document text
does not belong in git (`CLAUDE.md` §16, §20). The **manifest and this README
are committed** — they are the provenance record, like `data/manifest.csv`.

Send the `.jsonl` and the manifest together. The manifest's `jsonl_sha256` is
how the recipient checks the file arrived intact.
