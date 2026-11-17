"""Export the stored segments of one database as JSONL, for a teammate.

Run from the repository root::

    .venv\\Scripts\\python -m backend.scripts.export_segments
    .venv\\Scripts\\python -m backend.scripts.export_segments --database niyamsetu
    .venv\\Scripts\\python -m backend.scripts.export_segments --validate-only

Built for Person 3 (Ask / RAG), who needs the segmented corpus as a file rather
than through the API: one JSONL record per segment, with the page and character
provenance attached, so every passage retrieved downstream can still be turned
into a citation.

**This script only reads.** The connection is opened with
``default_transaction_read_only=on``, which is a *server-side* guard for this
connection only -- PostgreSQL itself refuses INSERT, UPDATE, DELETE, CREATE and
DROP on it, so a mistake in this file cannot modify the corpus. It is a
connection parameter, not a server setting: nothing about the database's
configuration is changed, and the next connection is unaffected. There is no
``begin()``, no commit, and no DDL anywhere in this module.

Which database
--------------
``niyamsetu`` by default -- the original 4-document development corpus every
evaluation number in ``backend/docs/final-backend-evaluation.md`` was measured
on. **Not** ``niyamsetu_sample`` (17 documents, 1,503 segments) and **not**
``niyamsetu_compare`` (13 versions, 5,088 segments). The manifest records which
database was used, and ``--expect`` fails the run if the counts do not match
what the caller said they expected, so the wrong database cannot be exported
silently.

What ``clause_id`` is
---------------------
**There is no ``clause_id`` column.** The schema's identifier for a segment is
``segments.public_id`` (``CLS-0000123``), and ``CLAUDE.md`` §12.3 names that
field ``clause_id`` in every API payload -- ``db/repository.py`` maps it the
same way. This export therefore uses ``segments.public_id`` and nothing is
fabricated. ``fingerprint`` (``SEG-<16 hex>``) is carried alongside it: it is
the *content-independent* stable id derived from
``(version, method, ordinal)``, which survives a re-ingestion that renumbers
``public_id``.

Likewise ``document_id`` and ``version_id`` are the ``public_id`` strings
(``DOC-0001``, ``VER-0001``), never the database integers --
``CLAUDE.md`` §12 fixes that for every payload the team exchanges.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection, Engine

from backend.app.db.config import REPO_ROOT, DatabaseConfig, load_dotenv

__all__ = [
    "DEFAULT_DATABASE",
    "DEFAULT_OUT_DIR",
    "read_only_engine",
    "fetch_segments",
    "fetch_corpus_summary",
    "write_jsonl",
    "sha256_of",
    "validate",
    "main",
]

#: The original development corpus. Overridable, but the default is the one
#: this export exists for.
DEFAULT_DATABASE = "niyamsetu"
DEFAULT_OUT_DIR = REPO_ROOT / "exports"
JSONL_NAME = "niyamsetu_segments_501.jsonl"
MANIFEST_NAME = "niyamsetu_segments_export_manifest.json"

EXPORT_FORMAT = "jsonl/niyamsetu-segments/1.0.0"

#: The field each exported key was read from, published in the manifest so a
#: consumer never has to guess and can re-derive the export from the schema.
FIELD_MAPPING: Dict[str, str] = {
    "document_id": "documents.public_id — e.g. 'DOC-0001'. NOT the integer key.",
    "version_id": "document_versions.public_id — e.g. 'VER-0001'. NOT the integer key.",
    "clause_id": (
        "segments.public_id — e.g. 'CLS-0000123'. There is NO clause_id column "
        "in the schema; public_id is the identifier, and CLAUDE.md §12.3 plus "
        "backend/app/db/repository.py already expose it under the name "
        "clause_id in every API payload. Nothing is fabricated."
    ),
    "fingerprint": (
        "segments.fingerprint — 'SEG-<16 hex>', derived from "
        "(version_id, segmentation_method, ordinal) and NOT from the text. It "
        "is stable across a re-ingestion that would renumber public_id, so it "
        "is the safer join key for anything stored outside this database."
    ),
    "text": (
        "segments.text — verbatim, exactly as stored. Not normalised, "
        "re-wrapped, truncated or summarised by this export."
    ),
    "clause_number": (
        "segments.clause_number — as printed ('3.2', '(iv)'). NULL on every "
        "paragraph segment by a schema CHECK, and on clause_aware segments "
        "where no marker was detected."
    ),
    "heading": "segments.heading — NULL unless a heading was detected.",
    "segment_type": (
        "segments.segment_type — one of clause | paragraph | heading | "
        "annexure | preamble | unclassified."
    ),
    "segmentation_method": (
        "segments.segmentation_method — 'paragraph' or 'clause_aware'. NOT in "
        "the requested field list, but added deliberately: without it the two "
        "segmentations are indistinguishable in the file, the 138/363 split "
        "cannot be checked, and a retrieval index would silently contain the "
        "same passage twice."
    ),
    "ordinal": "segments.ordinal — document order within (version, method).",
    "page_start": "segments.page_start — 1-based, inclusive.",
    "page_end": "segments.page_end — 1-based, inclusive. Equals page_start unless the segment spans a page break.",
    "pages": (
        "One PageProvenance object per segment_pages row for this segment, "
        "ordered by page_number. This is the precise provenance; page_start / "
        "page_end are the cheap summary. A segment spanning two pages has TWO "
        "entries and is never collapsed onto one."
    ),
    "offset_basis": (
        "document_versions.offset_basis — which page column the character "
        "offsets index into. NOT in the requested field list, but added "
        "deliberately: the offsets are unusable without it. It is "
        "'normalized_text' for every version in this corpus (D-11, verified), "
        "so char_start/char_end slice pages.normalized_text — NOT raw_text."
    ),
}

PAGE_PROVENANCE_MAPPING: Dict[str, str] = {
    "page_number": "segment_pages.page_number — 1-based.",
    "char_start": (
        "segment_pages.char_start — inclusive offset into that page's text on "
        "the record's offset_basis column."
    ),
    "char_end": "segment_pages.char_end — exclusive offset, same basis.",
    "line_index_start": "segment_pages.line_index_start — 0-based line index on that page.",
    "line_index_end": "segment_pages.line_index_end — 0-based, inclusive.",
}

#: One row per segment, with its pages aggregated in the database rather than
#: in Python: 501 segments would otherwise mean 501 follow-up queries.
#: ``json_agg`` over an ordered subquery keeps the page order deterministic.
SEGMENTS_SQL = """
    SELECT d.public_id            AS document_id,
           v.public_id            AS version_id,
           s.public_id            AS clause_id,
           s.fingerprint          AS fingerprint,
           s.text                 AS text,
           s.clause_number        AS clause_number,
           s.heading              AS heading,
           s.segment_type         AS segment_type,
           s.segmentation_method  AS segmentation_method,
           s.ordinal              AS ordinal,
           s.page_start           AS page_start,
           s.page_end             AS page_end,
           v.offset_basis         AS offset_basis,
           COALESCE(
               (SELECT json_agg(page ORDER BY page_number)
                  FROM (
                       SELECT sp.page_number,
                              sp.char_start,
                              sp.char_end,
                              sp.line_index_start,
                              sp.line_index_end
                         FROM segment_pages sp
                         JOIN pages p ON p.id = sp.page_id
                        WHERE sp.segment_id = s.id
                  ) AS page),
               '[]'::json
           )                      AS pages
      FROM segments s
      JOIN document_versions v ON v.id = s.version_id
      JOIN documents d         ON d.id = v.document_id
     ORDER BY d.public_id, v.public_id, s.segmentation_method, s.ordinal
"""

#: Corpus shape, for the manifest and for the --expect guard.
SUMMARY_SQL = """
    SELECT (SELECT count(*) FROM documents)         AS documents,
           (SELECT count(*) FROM document_versions) AS versions,
           (SELECT count(*) FROM pages)             AS pages,
           (SELECT count(*) FROM segments)          AS segments,
           (SELECT count(*) FROM segments
             WHERE segmentation_method = 'paragraph')    AS paragraph_segments,
           (SELECT count(*) FROM segments
             WHERE segmentation_method = 'clause_aware') AS clause_aware_segments,
           (SELECT count(*) FROM segment_pages)     AS segment_pages
"""

DOCUMENTS_SQL = """
    SELECT d.public_id      AS document_id,
           d.title          AS title,
           d.department     AS department,
           d.document_type  AS document_type,
           d.reference_no   AS reference_no,
           v.public_id      AS version_id,
           v.version_label  AS version_label,
           v.publication_date,
           v.page_count,
           v.extraction_status,
           v.offset_basis,
           v.source_url,
           v.original_filename
      FROM documents d
      JOIN document_versions v ON v.document_id = d.id
     ORDER BY d.public_id, v.public_id
"""


def read_only_engine(database: str = DEFAULT_DATABASE) -> Engine:
    """An engine whose every connection is read-only, enforced by PostgreSQL.

    ``default_transaction_read_only=on`` is passed as a *connection* option, so
    it applies to this connection and nothing else — the database's own
    configuration is untouched and the next client is unaffected. Any write
    statement on it fails with ``ReadOnlySqlTransaction`` before reaching a
    table, which is a stronger guarantee than "this script only contains
    SELECTs".

    Host, port, user and password come from ``.env`` exactly as the rest of the
    project reads them; only the database name is substituted. The password is
    never printed — :meth:`DatabaseConfig.safe_repr` is what the CLI shows.
    """
    load_dotenv()
    configured = DatabaseConfig.from_env()
    if configured.url:
        raise SystemExit(
            "NIYAMSETU_DATABASE_URL is set to a complete URL, so the database "
            "name cannot be substituted safely. Unset it, or point it at the "
            "database you want to export."
        )
    target = DatabaseConfig(
        host=configured.host,
        port=configured.port,
        database=database,
        user=configured.user,
        password=configured.password,
    )
    return create_engine(
        target.to_url(),
        connect_args={"options": "-c default_transaction_read_only=on"},
    )


def fetch_segments(conn: Connection) -> List[Dict[str, Any]]:
    """Every segment, in a deterministic order, with its page provenance.

    Order is ``(document, version, method, ordinal)`` so two runs against an
    unchanged database produce a byte-identical file — which is what makes the
    manifest's checksum a reproducibility claim rather than a timestamp.
    """
    records: List[Dict[str, Any]] = []
    for row in conn.execute(text(SEGMENTS_SQL)):
        records.append(
            {
                "document_id": row.document_id,
                "version_id": row.version_id,
                "clause_id": row.clause_id,
                "fingerprint": row.fingerprint,
                "text": row.text,
                "clause_number": row.clause_number,
                "heading": row.heading,
                "segment_type": row.segment_type,
                "segmentation_method": row.segmentation_method,
                "ordinal": int(row.ordinal),
                "page_start": int(row.page_start),
                "page_end": int(row.page_end),
                "offset_basis": row.offset_basis,
                "pages": [
                    {
                        "page_number": int(page["page_number"]),
                        "char_start": int(page["char_start"]),
                        "char_end": int(page["char_end"]),
                        "line_index_start": int(page["line_index_start"]),
                        "line_index_end": int(page["line_index_end"]),
                    }
                    for page in (row.pages or [])
                ],
            }
        )
    return records


def fetch_corpus_summary(conn: Connection) -> Dict[str, int]:
    row = conn.execute(text(SUMMARY_SQL)).one()
    return {
        "documents": int(row.documents),
        "versions": int(row.versions),
        "pages": int(row.pages),
        "segments": int(row.segments),
        "paragraph_segments": int(row.paragraph_segments),
        "clause_aware_segments": int(row.clause_aware_segments),
        "segment_pages": int(row.segment_pages),
    }


def fetch_documents(conn: Connection) -> List[Dict[str, Any]]:
    """Document and version metadata, for the manifest.

    Kept out of the JSONL on purpose: repeating a title on 501 records is 501
    chances for them to disagree. The manifest carries it once, and
    ``document_id`` / ``version_id`` on each record join to it.
    """
    out: List[Dict[str, Any]] = []
    for row in conn.execute(text(DOCUMENTS_SQL)):
        out.append(
            {
                "document_id": row.document_id,
                "version_id": row.version_id,
                "title": row.title,
                "department": row.department,
                "document_type": row.document_type,
                "reference_no": row.reference_no,
                "version_label": row.version_label,
                "publication_date": (
                    row.publication_date.isoformat() if row.publication_date else None
                ),
                "page_count": int(row.page_count),
                "extraction_status": row.extraction_status,
                "offset_basis": row.offset_basis,
                "source_url": row.source_url,
                "original_filename": row.original_filename,
            }
        )
    return out


def write_jsonl(records: Sequence[Dict[str, Any]], path: Path) -> int:
    """One compact JSON object per line, UTF-8, LF endings.

    ``ensure_ascii=False`` so the Devanagari that survives in some preamble
    segments stays readable rather than becoming escape sequences; the file is
    UTF-8 and says so in the manifest. ``newline="\\n"`` is explicit because
    this runs on Windows, where the default would write CRLF and change the
    checksum for a reason that has nothing to do with the data.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    return len(records)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(
    records: Sequence[Dict[str, Any]],
    summary: Dict[str, int],
    conn: Connection,
) -> List[Dict[str, Any]]:
    """Check the export against the database it came from.

    Every check returns a row rather than raising, so one failure does not hide
    the others — the caller decides what to do with a failing list.
    """
    checks: List[Dict[str, Any]] = []

    def check(name: str, passed: bool, detail: str = "") -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    by_method: Dict[str, int] = {}
    for record in records:
        by_method[record["segmentation_method"]] = (
            by_method.get(record["segmentation_method"], 0) + 1
        )

    check(
        "record count equals segments in the database",
        len(records) == summary["segments"],
        f"{len(records)} exported, {summary['segments']} in the database",
    )
    check(
        "paragraph segment count matches",
        by_method.get("paragraph", 0) == summary["paragraph_segments"],
        f"{by_method.get('paragraph', 0)} exported, {summary['paragraph_segments']} in the database",
    )
    check(
        "clause_aware segment count matches",
        by_method.get("clause_aware", 0) == summary["clause_aware_segments"],
        f"{by_method.get('clause_aware', 0)} exported, {summary['clause_aware_segments']} in the database",
    )

    documents = {r["document_id"] for r in records}
    versions = {r["version_id"] for r in records}
    check(
        "every document is represented",
        len(documents) == summary["documents"],
        f"{len(documents)} distinct document_id, {summary['documents']} in the database",
    )
    check(
        "every version is represented",
        len(versions) == summary["versions"],
        f"{len(versions)} distinct version_id, {summary['versions']} in the database",
    )

    clause_ids = [r["clause_id"] for r in records]
    check(
        "no duplicate clause_id",
        len(set(clause_ids)) == len(clause_ids),
        f"{len(clause_ids) - len(set(clause_ids))} duplicate(s)",
    )
    fingerprints = [r["fingerprint"] for r in records]
    check(
        "no duplicate fingerprint",
        len(set(fingerprints)) == len(fingerprints),
        f"{len(fingerprints) - len(set(fingerprints))} duplicate(s)",
    )

    # Every (document_id, version_id) pair the export claims must be a real
    # pair in the database, not merely two ids that both exist.
    real_pairs = {
        (row.document_id, row.version_id)
        for row in conn.execute(
            text(
                "SELECT d.public_id AS document_id, v.public_id AS version_id "
                "  FROM document_versions v JOIN documents d ON d.id = v.document_id"
            )
        )
    }
    bad_pairs = {
        (r["document_id"], r["version_id"]) for r in records
    } - real_pairs
    check(
        "every record's document/version pairing is real",
        not bad_pairs,
        f"{len(bad_pairs)} invalid pairing(s): {sorted(bad_pairs)[:3]}",
    )

    # Page provenance: present, in range, and pointing at pages that exist.
    stored_pages = {
        (row.version_id, int(row.page_number))
        for row in conn.execute(
            text(
                "SELECT v.public_id AS version_id, p.page_number "
                "  FROM pages p JOIN document_versions v ON v.id = p.version_id"
            )
        )
    }
    missing_pages: List[str] = []
    empty_provenance: List[str] = []
    bad_range: List[str] = []
    span_mismatch: List[str] = []
    for record in records:
        if not record["pages"]:
            empty_provenance.append(record["clause_id"])
            continue
        numbers = [page["page_number"] for page in record["pages"]]
        for page in record["pages"]:
            if (record["version_id"], page["page_number"]) not in stored_pages:
                missing_pages.append(f"{record['clause_id']} p{page['page_number']}")
            if page["char_end"] < page["char_start"] or page["char_start"] < 0:
                bad_range.append(record["clause_id"])
        if min(numbers) != record["page_start"] or max(numbers) != record["page_end"]:
            span_mismatch.append(record["clause_id"])

    check(
        "every record has page provenance",
        not empty_provenance,
        f"{len(empty_provenance)} record(s) with an empty pages array",
    )
    check(
        "every page reference resolves to a stored page",
        not missing_pages,
        f"{len(missing_pages)} unresolved: {missing_pages[:3]}",
    )
    check(
        "character offsets are ordered and non-negative",
        not bad_range,
        f"{len(bad_range)} record(s) with a bad range",
    )
    check(
        "pages array agrees with page_start/page_end",
        not span_mismatch,
        f"{len(span_mismatch)} mismatch(es): {span_mismatch[:3]}",
    )

    # The total number of PageProvenance objects must equal segment_pages.
    total_pages = sum(len(r["pages"]) for r in records)
    check(
        "PageProvenance count equals segment_pages rows",
        total_pages == summary["segment_pages"],
        f"{total_pages} exported, {summary['segment_pages']} in the database",
    )

    # The text must be byte-identical to what is stored, for every record.
    stored_text = {
        row.public_id: row.text
        for row in conn.execute(text("SELECT public_id, text FROM segments"))
    }
    altered = [
        r["clause_id"]
        for r in records
        if stored_text.get(r["clause_id"]) != r["text"]
    ]
    check(
        "text is byte-identical to the stored text",
        not altered,
        f"{len(altered)} altered: {altered[:3]}",
    )

    return checks


def _format(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    add = lines.append
    add("NiyamSetu — segment export")
    add("=" * 70)
    add(f"  database            {report['database']}")
    add(f"  exported_at         {report['exported_at']}")
    add(f"  records             {report['total_records']}")
    add(f"    paragraph         {report['paragraph_segments']}")
    add(f"    clause_aware      {report['clause_aware_segments']}")
    add(f"  documents           {report['document_count']}")
    add(f"  versions            {report['version_count']}")
    add(f"  PageProvenance      {report['page_provenance_records']}")
    add(f"  format              {report['export_format']}")
    add(f"  sha256              {report['jsonl_sha256']}")
    add(f"  bytes               {report['jsonl_bytes']}")
    add("")
    add("VALIDATION")
    add("-" * 70)
    for row in report["validation"]:
        mark = "PASS" if row["passed"] else "FAIL"
        detail = f"  ({row['detail']})" if row["detail"] and not row["passed"] else ""
        add(f"  {mark}  {row['check']}{detail}")
    failed = [r for r in report["validation"] if not r["passed"]]
    add("")
    add(f"  {len(report['validation'])} checks, {len(failed)} failed")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m backend.scripts.export_segments",
        description=(
            "Read-only JSONL export of the stored segments, with page and "
            "character provenance. Writes nothing to the database."
        ),
    )
    parser.add_argument("--database", default=DEFAULT_DATABASE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--jsonl-name", default=JSONL_NAME)
    parser.add_argument("--manifest-name", default=MANIFEST_NAME)
    parser.add_argument(
        "--expect",
        type=int,
        default=None,
        help=(
            "Fail unless the database holds exactly this many segments. Use it "
            "to make exporting the wrong database impossible rather than "
            "merely unlikely."
        ),
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Query and validate, but write no files.",
    )
    args = parser.parse_args(argv)

    engine = read_only_engine(args.database)
    try:
        with engine.connect() as conn:
            # Proof, recorded in the manifest, that the session really is
            # read-only -- not merely that this file contains only SELECTs.
            read_only = conn.execute(text("SHOW transaction_read_only")).scalar_one()
            actual_db = conn.execute(text("SELECT current_database()")).scalar_one()
            if actual_db != args.database:
                raise SystemExit(
                    f"connected to {actual_db!r} but asked for {args.database!r}"
                )

            summary = fetch_corpus_summary(conn)
            if args.expect is not None and summary["segments"] != args.expect:
                raise SystemExit(
                    f"refusing to export {actual_db!r}: it holds "
                    f"{summary['segments']} segments, and --expect said "
                    f"{args.expect}. Check the --database argument."
                )

            records = fetch_segments(conn)
            documents = fetch_documents(conn)
            checks = validate(records, summary, conn)
    finally:
        engine.dispose()

    by_method: Dict[str, int] = {}
    for record in records:
        by_method[record["segmentation_method"]] = (
            by_method.get(record["segmentation_method"], 0) + 1
        )

    jsonl_path = args.out_dir / args.jsonl_name
    manifest_path = args.out_dir / args.manifest_name

    report: Dict[str, Any] = {
        "export_format": EXPORT_FORMAT,
        "database": args.database,
        "exported_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "source": {
            "host": "localhost (from .env; credentials are never recorded here)",
            "session_transaction_read_only": read_only,
            "written_by": "backend/scripts/export_segments.py",
            "reproducible": (
                "Deterministic: rows are ordered by (document, version, method, "
                "ordinal), so re-running against an unchanged database "
                "reproduces this file byte for byte, and therefore this "
                "checksum."
            ),
        },
        "total_records": len(records),
        "paragraph_segments": by_method.get("paragraph", 0),
        "clause_aware_segments": by_method.get("clause_aware", 0),
        "document_count": len({r["document_id"] for r in records}),
        "version_count": len({r["version_id"] for r in records}),
        "page_provenance_records": sum(len(r["pages"]) for r in records),
        "database_counts": summary,
        "field_mapping": FIELD_MAPPING,
        "page_provenance_mapping": PAGE_PROVENANCE_MAPPING,
        "nullable_fields": {
            "clause_number": (
                "NULL on every paragraph segment (enforced by a schema CHECK) "
                "and on any clause_aware segment with no detected marker."
            ),
            "heading": "NULL unless a heading was detected. Rare.",
            "department / document_type / reference_no": (
                "Document-level and nullable; they live in this manifest's "
                "`documents` block, not on each record."
            ),
        },
        "notes": [
            "text is verbatim from segments.text. Nothing was normalised, "
            "re-wrapped, truncated or summarised.",
            "Both segmentations are present and cover the SAME source text. "
            "Index ONE of them — filter on segmentation_method — or the same "
            "passage will be retrieved twice.",
            "char_start/char_end index pages.normalized_text (offset_basis), "
            "NOT raw_text. Slicing the wrong column gives the wrong span.",
            "A segment spanning a page break has two entries in `pages`; it is "
            "never collapsed onto one page.",
        ],
        "documents": documents,
        "validation": checks,
    }

    if args.validate_only:
        report["jsonl_sha256"] = None
        report["jsonl_bytes"] = None
        print(_format(report))
        return 0 if all(c["passed"] for c in checks) else 1

    write_jsonl(records, jsonl_path)
    report["jsonl_sha256"] = sha256_of(jsonl_path)
    report["jsonl_bytes"] = jsonl_path.stat().st_size
    report["files"] = {
        "jsonl": str(jsonl_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "manifest": str(manifest_path.relative_to(REPO_ROOT)).replace("\\", "/"),
    }

    manifest_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(_format(report))
    print()
    print(f"written: {jsonl_path}")
    print(f"written: {manifest_path}")
    return 0 if all(c["passed"] for c in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
