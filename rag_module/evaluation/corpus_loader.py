"""Evaluation corpus loader for the NiyamSetu RAG retrieval benchmark.

Loads and validates the 501-segment export produced by Person 1's backend
pipeline (``backend/scripts/export_segments.py``).  Segments are converted
into the canonical ``ClauseSegment`` / ``PageProvenance`` objects defined in
``rag_module.core.models`` and then separated into the two independent
experiment corpora.

**Experiment A** – clause_aware segmentation (363 segments)
**Experiment B** – paragraph segmentation (138 segments)

The two corpora must never be combined into one retrieval index, because both
methods cover the same source text.  Mixing them would cause every passage to
be retrieved twice.

Offset note
-----------
Character offsets (``char_start`` / ``char_end``) in ``pages`` provenance
index each page's ``normalized_text`` column in the backend database.  That
page text is NOT included in this export.  The offsets are preserved verbatim
so that downstream tools can slice the correct column; they do not make it
possible to reconstruct the full page text here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rag_module.core.models import ClauseSegment, PageProvenance

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

VALID_SEGMENTATION_METHODS: Tuple[str, ...] = ("clause_aware", "paragraph")

REQUIRED_RECORD_FIELDS: Tuple[str, ...] = (
    "clause_id",
    "document_id",
    "version_id",
    "fingerprint",
    "ordinal",
    "segment_type",
    "segmentation_method",
    "text",
    "page_start",
    "page_end",
    "pages",
    "offset_basis",
)


# ---------------------------------------------------------------------------
# Validation result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorpusStats:
    """Summary statistics produced after a successful corpus load."""

    total: int
    clause_aware: int
    paragraph: int
    document_ids: Tuple[str, ...]
    version_ids: Tuple[str, ...]
    page_provenance_records: int
    jsonl_sha256: str
    jsonl_bytes: int


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    """Return the hex SHA-256 digest of a file, reading in 64 KiB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(65536):
            h.update(chunk)
    return h.hexdigest()


def _validate_record(record: Dict, line_no: int, valid_doc_version_pairs: Optional[set]) -> None:
    """Raise ``ValueError`` with a descriptive message if *record* is invalid.

    Args:
        record: Parsed JSON object from the JSONL file.
        line_no: 1-based line number in the JSONL file (for error messages).
        valid_doc_version_pairs: Set of (document_id, version_id) pairs that are
            valid according to the manifest.  ``None`` disables this check.

    Raises:
        ValueError: On any validation failure, identifying the affected record.
    """
    cid = record.get("clause_id", f"<unknown, line {line_no}>")
    ctx = f"clause_id={cid!r} (line {line_no})"

    # 1. Required fields present
    for field in REQUIRED_RECORD_FIELDS:
        if field not in record:
            raise ValueError(f"Missing required field {field!r} in record {ctx}")

    # 2. Non-empty strings for key identifiers
    for field in ("clause_id", "document_id", "version_id", "fingerprint", "text"):
        val = record[field]
        if not isinstance(val, str) or not val.strip():
            raise ValueError(
                f"Field {field!r} must be a non-empty string in record {ctx}; got {val!r}"
            )

    # 3. segmentation_method
    method = record["segmentation_method"]
    if method not in VALID_SEGMENTATION_METHODS:
        raise ValueError(
            f"Invalid segmentation_method {method!r} in record {ctx}. "
            f"Must be one of {VALID_SEGMENTATION_METHODS}"
        )

    # 4. ordinal is a non-negative integer
    ordinal = record["ordinal"]
    if not isinstance(ordinal, int) or ordinal < 0:
        raise ValueError(
            f"Field 'ordinal' must be a non-negative integer in record {ctx}; got {ordinal!r}"
        )

    # 5. page_start and page_end
    ps = record["page_start"]
    pe = record["page_end"]
    if not isinstance(ps, int) or ps < 1:
        raise ValueError(f"page_start must be an integer >= 1 in record {ctx}; got {ps!r}")
    if not isinstance(pe, int) or pe < ps:
        raise ValueError(
            f"page_end ({pe}) must be >= page_start ({ps}) in record {ctx}"
        )

    # 6. pages provenance: non-empty list, each entry has required keys
    pages = record["pages"]
    if not isinstance(pages, list) or len(pages) == 0:
        raise ValueError(f"'pages' must be a non-empty list in record {ctx}")

    prov_page_numbers = []
    for idx, prov in enumerate(pages):
        p_ctx = f"record {ctx}, pages[{idx}]"
        for pfield in ("page_number", "char_start", "char_end"):
            if pfield not in prov:
                raise ValueError(f"Missing field {pfield!r} in page provenance of {p_ctx}")
            if not isinstance(prov[pfield], int):
                raise ValueError(
                    f"Field {pfield!r} must be an integer in page provenance of {p_ctx}; "
                    f"got {prov[pfield]!r}"
                )

        pnum = prov["page_number"]
        cs = prov["char_start"]
        ce = prov["char_end"]

        if pnum < 1:
            raise ValueError(
                f"page_number must be >= 1 in page provenance of {p_ctx}; got {pnum}"
            )
        if cs < 0:
            raise ValueError(
                f"char_start must be >= 0 in page provenance of {p_ctx}; got {cs}"
            )
        if ce < cs:
            raise ValueError(
                f"char_end ({ce}) must be >= char_start ({cs}) in page provenance of {p_ctx}"
            )
        prov_page_numbers.append(pnum)

    # 7. page_start / page_end agree with pages array
    if min(prov_page_numbers) != ps:
        raise ValueError(
            f"page_start ({ps}) does not match the minimum page_number in 'pages' "
            f"({min(prov_page_numbers)}) in record {ctx}"
        )
    if max(prov_page_numbers) != pe:
        raise ValueError(
            f"page_end ({pe}) does not match the maximum page_number in 'pages' "
            f"({max(prov_page_numbers)}) in record {ctx}"
        )

    # 8. Valid document/version pairing (if manifest pairs provided)
    if valid_doc_version_pairs is not None:
        pair = (record["document_id"], record["version_id"])
        if pair not in valid_doc_version_pairs:
            raise ValueError(
                f"Invalid document/version pairing {pair} in record {ctx}. "
                f"Valid pairings from manifest: {sorted(valid_doc_version_pairs)}"
            )

    # 9. nullable fields: clause_number and heading may be None or str
    for nullable in ("clause_number", "heading"):
        if nullable in record:
            val = record[nullable]
            if val is not None and not isinstance(val, str):
                raise ValueError(
                    f"Field {nullable!r} must be a string or null in record {ctx}; got {val!r}"
                )


# ---------------------------------------------------------------------------
# Core loader
# ---------------------------------------------------------------------------


def load_corpus(
    jsonl_path: Path,
    manifest_path: Path,
    *,
    skip_sha256: bool = False,
) -> Tuple[List[ClauseSegment], CorpusStats]:
    """Load and validate the full 501-segment evaluation corpus.

    Performs all integrity checks (SHA-256, counts, uniqueness, field types,
    page provenance) before returning.  Raises informative errors on any
    failure without silently repairing records.

    Args:
        jsonl_path: Path to ``niyamsetu_segments_501.jsonl``.
        manifest_path: Path to ``niyamsetu_segments_export_manifest.json``.
        skip_sha256: If ``True``, skip the file-checksum comparison.  Intended
            only for unit tests that deliberately provide truncated fixtures.

    Returns:
        Tuple of (list of ClauseSegment objects, CorpusStats).

    Raises:
        FileNotFoundError: If either supplied file does not exist.
        ValueError: On any integrity violation, identifying the affected record.
    """
    # --- Existence checks ---------------------------------------------------
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL corpus file not found: {jsonl_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    # --- Load manifest ------------------------------------------------------
    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    expected_sha = manifest.get("jsonl_sha256", "")
    expected_bytes = int(manifest.get("jsonl_bytes", -1))
    expected_total = int(manifest["total_records"])
    expected_paragraph = int(manifest["paragraph_segments"])
    expected_clause_aware = int(manifest["clause_aware_segments"])
    expected_doc_count = int(manifest["document_count"])
    expected_ver_count = int(manifest["version_count"])
    expected_provenance = int(manifest["page_provenance_records"])

    # Build valid (doc_id, ver_id) pairings from manifest documents list
    manifest_docs: List[Dict] = manifest.get("documents", [])
    valid_pairs: set = {
        (d["document_id"], d["version_id"]) for d in manifest_docs
    }

    # --- SHA-256 and byte size check ----------------------------------------
    actual_bytes = jsonl_path.stat().st_size
    if expected_bytes >= 0 and actual_bytes != expected_bytes:
        raise ValueError(
            f"JSONL file size mismatch: expected {expected_bytes} bytes, "
            f"got {actual_bytes} bytes at {jsonl_path}"
        )

    if not skip_sha256:
        actual_sha = _sha256_file(jsonl_path)
        if actual_sha != expected_sha:
            raise ValueError(
                f"SHA-256 mismatch for {jsonl_path}:\n"
                f"  expected: {expected_sha}\n"
                f"  actual:   {actual_sha}"
            )
    else:
        actual_sha = _sha256_file(jsonl_path) if expected_sha else ""

    # --- Parse JSONL records ------------------------------------------------
    records: List[Dict] = []
    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                rec = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSON parse error on line {line_no} of {jsonl_path}: {exc}"
                ) from exc
            records.append((line_no, rec))

    # --- Total count --------------------------------------------------------
    actual_total = len(records)
    if actual_total != expected_total:
        raise ValueError(
            f"Record count mismatch: expected {expected_total}, got {actual_total}"
        )

    # --- Per-record validation, duplicate detection -------------------------
    seen_clause_ids: Dict[str, int] = {}   # clause_id  -> first line_no
    seen_fingerprints: Dict[str, int] = {} # fingerprint -> first line_no

    method_counts: Dict[str, int] = {}
    doc_ids_seen: set = set()
    ver_ids_seen: set = set()
    total_provenance = 0

    for line_no, rec in records:
        # Field + type + provenance validation (raises on failure)
        _validate_record(rec, line_no, valid_pairs)

        # Duplicate clause_id
        cid = rec["clause_id"]
        if cid in seen_clause_ids:
            raise ValueError(
                f"Duplicate clause_id {cid!r}: first seen on line {seen_clause_ids[cid]}, "
                f"repeated on line {line_no}"
            )
        seen_clause_ids[cid] = line_no

        # Duplicate fingerprint
        fp = rec["fingerprint"]
        if fp in seen_fingerprints:
            raise ValueError(
                f"Duplicate fingerprint {fp!r}: first seen on line {seen_fingerprints[fp]}, "
                f"repeated on line {line_no}"
            )
        seen_fingerprints[fp] = line_no

        method = rec["segmentation_method"]
        method_counts[method] = method_counts.get(method, 0) + 1
        doc_ids_seen.add(rec["document_id"])
        ver_ids_seen.add(rec["version_id"])
        total_provenance += len(rec["pages"])

    # --- Method counts ------------------------------------------------------
    actual_clause_aware = method_counts.get("clause_aware", 0)
    actual_paragraph = method_counts.get("paragraph", 0)

    if actual_clause_aware != expected_clause_aware:
        raise ValueError(
            f"clause_aware segment count mismatch: "
            f"expected {expected_clause_aware}, got {actual_clause_aware}"
        )
    if actual_paragraph != expected_paragraph:
        raise ValueError(
            f"paragraph segment count mismatch: "
            f"expected {expected_paragraph}, got {actual_paragraph}"
        )

    # --- Document and version counts ----------------------------------------
    if len(doc_ids_seen) != expected_doc_count:
        raise ValueError(
            f"Document count mismatch: expected {expected_doc_count} distinct document IDs, "
            f"got {len(doc_ids_seen)}: {sorted(doc_ids_seen)}"
        )
    if len(ver_ids_seen) != expected_ver_count:
        raise ValueError(
            f"Version count mismatch: expected {expected_ver_count} distinct version IDs, "
            f"got {len(ver_ids_seen)}: {sorted(ver_ids_seen)}"
        )

    # --- Page provenance total ----------------------------------------------
    if total_provenance != expected_provenance:
        raise ValueError(
            f"Page provenance record count mismatch: "
            f"expected {expected_provenance}, got {total_provenance}"
        )

    # --- Convert to ClauseSegment objects -----------------------------------
    segments: List[ClauseSegment] = []
    for _line_no, rec in records:
        pages = [PageProvenance.from_dict(p) for p in rec["pages"]]
        seg = ClauseSegment(
            document_id=rec["document_id"],
            version_id=rec["version_id"],
            clause_id=rec["clause_id"],
            fingerprint=rec["fingerprint"],
            ordinal=int(rec["ordinal"]),
            segment_type=str(rec["segment_type"]),
            text=str(rec["text"]),
            page_start=int(rec["page_start"]),
            page_end=int(rec["page_end"]),
            pages=pages,
            clause_number=rec.get("clause_number"),
            clause_number_ambiguous=bool(rec.get("clause_number_ambiguous", False)),
            heading=rec.get("heading"),
            segmentation_method=str(rec["segmentation_method"]),
            offset_basis=str(rec.get("offset_basis", "normalized_text")),
            level=int(rec.get("level", 0)),
            marker_style=rec.get("marker_style"),
            parent_clause_id=rec.get("parent_clause_id"),
            metadata=dict(rec.get("metadata") or {}),
        )
        segments.append(seg)

    stats = CorpusStats(
        total=actual_total,
        clause_aware=actual_clause_aware,
        paragraph=actual_paragraph,
        document_ids=tuple(sorted(doc_ids_seen)),
        version_ids=tuple(sorted(ver_ids_seen)),
        page_provenance_records=total_provenance,
        jsonl_sha256=actual_sha,
        jsonl_bytes=actual_bytes,
    )
    return segments, stats


# ---------------------------------------------------------------------------
# Experiment corpus selectors
# ---------------------------------------------------------------------------


def experiment_a_corpus(segments: List[ClauseSegment]) -> List[ClauseSegment]:
    """Return only the clause_aware segments (Experiment A).

    Args:
        segments: Full validated corpus from :func:`load_corpus`.

    Returns:
        Filtered list containing exactly the ``clause_aware`` segments,
        in their original ordinal order.

    Raises:
        ValueError: If the filtered list is empty (indicates a corrupt corpus).
    """
    result = [s for s in segments if s.segmentation_method == "clause_aware"]
    if not result:
        raise ValueError(
            "experiment_a_corpus: no clause_aware segments found in the provided corpus."
        )
    return result


def experiment_b_corpus(segments: List[ClauseSegment]) -> List[ClauseSegment]:
    """Return only the paragraph segments (Experiment B).

    Args:
        segments: Full validated corpus from :func:`load_corpus`.

    Returns:
        Filtered list containing exactly the ``paragraph`` segments,
        in their original ordinal order.

    Raises:
        ValueError: If the filtered list is empty (indicates a corrupt corpus).
    """
    result = [s for s in segments if s.segmentation_method == "paragraph"]
    if not result:
        raise ValueError(
            "experiment_b_corpus: no paragraph segments found in the provided corpus."
        )
    return result


# ---------------------------------------------------------------------------
# Convenience: load from default paths relative to this file
# ---------------------------------------------------------------------------

_EVAL_DATA_DIR = Path(__file__).parent / "data"
_DEFAULT_JSONL = _EVAL_DATA_DIR / "niyamsetu_segments_501.jsonl"
_DEFAULT_MANIFEST = _EVAL_DATA_DIR / "niyamsetu_segments_export_manifest.json"


def load_default_corpus(
    *,
    skip_sha256: bool = False,
) -> Tuple[List[ClauseSegment], CorpusStats]:
    """Load the corpus from the standard evaluation/data/ paths.

    This is the entry point for all evaluation scripts.  It calls
    :func:`load_corpus` with the two files that Person 1 exported from the
    PostgreSQL database.

    Returns:
        Same as :func:`load_corpus`.
    """
    return load_corpus(_DEFAULT_JSONL, _DEFAULT_MANIFEST, skip_sha256=skip_sha256)
