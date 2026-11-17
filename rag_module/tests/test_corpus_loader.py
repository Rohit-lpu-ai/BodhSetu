"""Tests for rag_module.evaluation.corpus_loader.

All tests use small synthetic fixtures written to temporary files.  The full
501-segment production corpus is never duplicated or fabricated here.

Test categories
---------------
- SHA-256 validation (correct digest accepted, wrong digest rejected)
- Record count validation
- Per-method count validation (clause_aware, paragraph)
- Duplicate clause_id rejection
- Duplicate fingerprint rejection
- Missing required field rejection
- Empty text rejection
- Invalid document/version pairing rejection
- Invalid page provenance rejection (empty list, bad offsets, page mismatch)
- Offset ordering validation
- experiment_a_corpus and experiment_b_corpus filters
- No accidental mixing between experiments
- ClauseSegment and PageProvenance conversion correctness
- Original text and nullable field preservation
"""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from rag_module.core.models import ClauseSegment, PageProvenance
from rag_module.evaluation.corpus_loader import (
    CorpusStats,
    experiment_a_corpus,
    experiment_b_corpus,
    load_corpus,
)

# ---------------------------------------------------------------------------
# Shared fixture builders
# ---------------------------------------------------------------------------


def _make_provenance(page_number=1, char_start=0, char_end=100,
                     line_index_start=0, line_index_end=5):
    return {
        "page_number": page_number,
        "char_start": char_start,
        "char_end": char_end,
        "line_index_start": line_index_start,
        "line_index_end": line_index_end,
    }


def _make_record(
    clause_id="CLS-0000001",
    document_id="DOC-0001",
    version_id="VER-0001",
    fingerprint="SEG-aaaaaaaaaaaaaaaa",
    ordinal=0,
    segment_type="clause",
    segmentation_method="clause_aware",
    text="Sample regulatory text.",
    page_start=1,
    page_end=1,
    pages=None,
    offset_basis="normalized_text",
    clause_number=None,
    heading=None,
):
    if pages is None:
        pages = [_make_provenance()]
    return {
        "clause_id": clause_id,
        "document_id": document_id,
        "version_id": version_id,
        "fingerprint": fingerprint,
        "ordinal": ordinal,
        "segment_type": segment_type,
        "segmentation_method": segmentation_method,
        "text": text,
        "page_start": page_start,
        "page_end": page_end,
        "pages": pages,
        "offset_basis": offset_basis,
        "clause_number": clause_number,
        "heading": heading,
    }


def _write_jsonl(path: Path, records: list) -> str:
    """Write records as JSONL; return the resulting SHA-256 hex digest."""
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(65536):
            h.update(chunk)
    return h.hexdigest()


def _write_manifest(path: Path, records: list, sha256: str,
                    clause_aware_count: int = None, paragraph_count: int = None,
                    doc_count: int = 1, ver_count: int = 1,
                    provenance_count: int = None,
                    documents: list = None) -> None:
    """Write a minimal manifest that matches the given records."""
    total = len(records)
    if clause_aware_count is None:
        clause_aware_count = sum(1 for r in records if r.get("segmentation_method") == "clause_aware")
    if paragraph_count is None:
        paragraph_count = sum(1 for r in records if r.get("segmentation_method") == "paragraph")
    if provenance_count is None:
        provenance_count = sum(len(r.get("pages", [])) for r in records)
    if documents is None:
        documents = [{"document_id": "DOC-0001", "version_id": "VER-0001"}]
    manifest = {
        "total_records": total,
        "paragraph_segments": paragraph_count,
        "clause_aware_segments": clause_aware_count,
        "document_count": doc_count,
        "version_count": ver_count,
        "page_provenance_records": provenance_count,
        "jsonl_sha256": sha256,
        "jsonl_bytes": path.parent.joinpath(path.name.replace("manifest", "corpus")).stat().st_size
        if False else -1,  # skip byte check in mini-fixtures
        "documents": documents,
    }
    # Fix the byte count to match what was actually written
    jsonl_path = path.parent / "corpus.jsonl"
    if jsonl_path.exists():
        manifest["jsonl_bytes"] = jsonl_path.stat().st_size
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh)


class FixtureHelper(unittest.TestCase):
    """Base class providing a shared temporary directory."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmpdir.name)
        self.jsonl = self.tmp / "corpus.jsonl"
        self.manifest = self.tmp / "manifest.json"

    def tearDown(self):
        self._tmpdir.cleanup()

    def write(self, records: list, *, clause_aware_count=None, paragraph_count=None,
              doc_count=1, ver_count=1, provenance_count=None, documents=None):
        """Write records + a consistent manifest; return computed sha256."""
        sha = _write_jsonl(self.jsonl, records)
        manifest_docs = documents or [{"document_id": "DOC-0001", "version_id": "VER-0001"}]
        total = len(records)
        ca = clause_aware_count if clause_aware_count is not None else sum(
            1 for r in records if r.get("segmentation_method") == "clause_aware")
        pg = paragraph_count if paragraph_count is not None else sum(
            1 for r in records if r.get("segmentation_method") == "paragraph")
        prov = provenance_count if provenance_count is not None else sum(
            len(r.get("pages", [])) for r in records)
        manifest = {
            "total_records": total,
            "paragraph_segments": pg,
            "clause_aware_segments": ca,
            "document_count": doc_count,
            "version_count": ver_count,
            "page_provenance_records": prov,
            "jsonl_sha256": sha,
            "jsonl_bytes": self.jsonl.stat().st_size,
            "documents": manifest_docs,
        }
        with open(self.manifest, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh)
        return sha


# ---------------------------------------------------------------------------
# 1. SHA-256 validation
# ---------------------------------------------------------------------------


class TestChecksumValidation(FixtureHelper):

    def test_correct_checksum_accepted(self):
        recs = [_make_record()]
        self.write(recs)
        segs, stats = load_corpus(self.jsonl, self.manifest)
        self.assertEqual(len(segs), 1)
        self.assertIsInstance(stats, CorpusStats)

    def test_wrong_checksum_rejected(self):
        recs = [_make_record()]
        sha = _write_jsonl(self.jsonl, recs)
        # write manifest with a deliberately wrong hash
        manifest = {
            "total_records": 1,
            "paragraph_segments": 0,
            "clause_aware_segments": 1,
            "document_count": 1,
            "version_count": 1,
            "page_provenance_records": 1,
            "jsonl_sha256": "0" * 64,
            "jsonl_bytes": self.jsonl.stat().st_size,
            "documents": [{"document_id": "DOC-0001", "version_id": "VER-0001"}],
        }
        with open(self.manifest, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh)
        with self.assertRaises(ValueError) as ctx:
            load_corpus(self.jsonl, self.manifest)
        self.assertIn("SHA-256 mismatch", str(ctx.exception))

    def test_skip_sha256_flag_bypasses_check(self):
        recs = [_make_record()]
        _write_jsonl(self.jsonl, recs)
        # Incorrect hash in manifest — but skip_sha256 must bypass it
        manifest = {
            "total_records": 1,
            "paragraph_segments": 0,
            "clause_aware_segments": 1,
            "document_count": 1,
            "version_count": 1,
            "page_provenance_records": 1,
            "jsonl_sha256": "0" * 64,
            "jsonl_bytes": -1,
            "documents": [{"document_id": "DOC-0001", "version_id": "VER-0001"}],
        }
        with open(self.manifest, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh)
        segs, stats = load_corpus(self.jsonl, self.manifest, skip_sha256=True)
        self.assertEqual(len(segs), 1)

    def test_missing_jsonl_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_corpus(self.tmp / "nonexistent.jsonl", self.manifest)

    def test_missing_manifest_raises_file_not_found(self):
        recs = [_make_record()]
        _write_jsonl(self.jsonl, recs)
        with self.assertRaises(FileNotFoundError):
            load_corpus(self.jsonl, self.tmp / "nonexistent.json")


# ---------------------------------------------------------------------------
# 2. Record count validation
# ---------------------------------------------------------------------------


class TestRecordCounts(FixtureHelper):

    def test_correct_total_accepted(self):
        recs = [_make_record(clause_id=f"CLS-{i:07d}", fingerprint=f"SEG-{i:016x}")
                for i in range(1, 4)]
        self.write(recs, doc_count=1, ver_count=1)
        segs, stats = load_corpus(self.jsonl, self.manifest, skip_sha256=True)
        self.assertEqual(stats.total, 3)

    def test_total_mismatch_rejected(self):
        recs = [_make_record(clause_id="CLS-0000001", fingerprint="SEG-aaaaaaaaaaaaaaaa"),
                _make_record(clause_id="CLS-0000002", fingerprint="SEG-bbbbbbbbbbbbbbbb")]
        sha = _write_jsonl(self.jsonl, recs)
        # Manifest says total=5 but file has only 2
        manifest = {
            "total_records": 5,
            "paragraph_segments": 0,
            "clause_aware_segments": 2,
            "document_count": 1,
            "version_count": 1,
            "page_provenance_records": 2,
            "jsonl_sha256": sha,
            "jsonl_bytes": -1,
            "documents": [{"document_id": "DOC-0001", "version_id": "VER-0001"}],
        }
        with open(self.manifest, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh)
        with self.assertRaises(ValueError) as ctx:
            load_corpus(self.jsonl, self.manifest, skip_sha256=True)
        self.assertIn("count mismatch", str(ctx.exception).lower())

    def test_correct_method_counts_accepted(self):
        recs = [
            _make_record(clause_id="CLS-0000001", fingerprint="SEG-aaaaaaaaaaaaaaaa",
                         segmentation_method="clause_aware"),
            _make_record(clause_id="CLS-0000002", fingerprint="SEG-bbbbbbbbbbbbbbbb",
                         segmentation_method="paragraph"),
        ]
        self.write(recs, doc_count=1, ver_count=1)
        segs, stats = load_corpus(self.jsonl, self.manifest, skip_sha256=True)
        self.assertEqual(stats.clause_aware, 1)
        self.assertEqual(stats.paragraph, 1)

    def test_clause_aware_count_mismatch_rejected(self):
        recs = [
            _make_record(clause_id="CLS-0000001", fingerprint="SEG-aaaaaaaaaaaaaaaa",
                         segmentation_method="clause_aware"),
        ]
        sha = _write_jsonl(self.jsonl, recs)
        manifest = {
            "total_records": 1,
            "paragraph_segments": 0,
            "clause_aware_segments": 99,   # wrong
            "document_count": 1,
            "version_count": 1,
            "page_provenance_records": 1,
            "jsonl_sha256": sha,
            "jsonl_bytes": -1,
            "documents": [{"document_id": "DOC-0001", "version_id": "VER-0001"}],
        }
        with open(self.manifest, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh)
        with self.assertRaises(ValueError) as ctx:
            load_corpus(self.jsonl, self.manifest, skip_sha256=True)
        self.assertIn("clause_aware segment count", str(ctx.exception))

    def test_paragraph_count_mismatch_rejected(self):
        recs = [
            _make_record(clause_id="CLS-0000001", fingerprint="SEG-aaaaaaaaaaaaaaaa",
                         segmentation_method="paragraph"),
        ]
        sha = _write_jsonl(self.jsonl, recs)
        manifest = {
            "total_records": 1,
            "paragraph_segments": 99,   # wrong
            "clause_aware_segments": 0,
            "document_count": 1,
            "version_count": 1,
            "page_provenance_records": 1,
            "jsonl_sha256": sha,
            "jsonl_bytes": -1,
            "documents": [{"document_id": "DOC-0001", "version_id": "VER-0001"}],
        }
        with open(self.manifest, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh)
        with self.assertRaises(ValueError) as ctx:
            load_corpus(self.jsonl, self.manifest, skip_sha256=True)
        self.assertIn("paragraph segment count", str(ctx.exception))


# ---------------------------------------------------------------------------
# 3. Duplicate rejection
# ---------------------------------------------------------------------------


class TestDuplicateRejection(FixtureHelper):

    def test_duplicate_clause_id_rejected(self):
        recs = [
            _make_record(clause_id="CLS-0000001", fingerprint="SEG-aaaaaaaaaaaaaaaa"),
            _make_record(clause_id="CLS-0000001", fingerprint="SEG-bbbbbbbbbbbbbbbb"),  # dup cid
        ]
        self.write(recs, doc_count=1, ver_count=1)
        with self.assertRaises(ValueError) as ctx:
            load_corpus(self.jsonl, self.manifest, skip_sha256=True)
        self.assertIn("Duplicate clause_id", str(ctx.exception))
        self.assertIn("CLS-0000001", str(ctx.exception))

    def test_duplicate_fingerprint_rejected(self):
        recs = [
            _make_record(clause_id="CLS-0000001", fingerprint="SEG-aaaaaaaaaaaaaaaa"),
            _make_record(clause_id="CLS-0000002", fingerprint="SEG-aaaaaaaaaaaaaaaa"),  # dup fp
        ]
        self.write(recs, doc_count=1, ver_count=1)
        with self.assertRaises(ValueError) as ctx:
            load_corpus(self.jsonl, self.manifest, skip_sha256=True)
        self.assertIn("Duplicate fingerprint", str(ctx.exception))

    def test_unique_ids_accepted(self):
        recs = [
            _make_record(clause_id="CLS-0000001", fingerprint="SEG-aaaaaaaaaaaaaaaa"),
            _make_record(clause_id="CLS-0000002", fingerprint="SEG-bbbbbbbbbbbbbbbb"),
        ]
        self.write(recs, doc_count=1, ver_count=1)
        segs, _ = load_corpus(self.jsonl, self.manifest, skip_sha256=True)
        self.assertEqual(len(segs), 2)


# ---------------------------------------------------------------------------
# 4. Required field validation
# ---------------------------------------------------------------------------


class TestRequiredFields(FixtureHelper):

    def _load_one_record(self, record: dict):
        sha = _write_jsonl(self.jsonl, [record])
        manifest = {
            "total_records": 1,
            "paragraph_segments": 1 if record.get("segmentation_method") == "paragraph" else 0,
            "clause_aware_segments": 1 if record.get("segmentation_method", "clause_aware") == "clause_aware" else 0,
            "document_count": 1,
            "version_count": 1,
            "page_provenance_records": len(record.get("pages", [])),
            "jsonl_sha256": sha,
            "jsonl_bytes": -1,
            "documents": [{"document_id": record.get("document_id", "DOC-0001"),
                           "version_id": record.get("version_id", "VER-0001")}],
        }
        with open(self.manifest, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh)
        return load_corpus(self.jsonl, self.manifest, skip_sha256=True)

    def test_missing_clause_id_rejected(self):
        rec = _make_record()
        del rec["clause_id"]
        with self.assertRaises(ValueError) as ctx:
            self._load_one_record(rec)
        self.assertIn("clause_id", str(ctx.exception))

    def test_missing_text_rejected(self):
        rec = _make_record()
        del rec["text"]
        with self.assertRaises(ValueError) as ctx:
            self._load_one_record(rec)
        self.assertIn("text", str(ctx.exception))

    def test_missing_document_id_rejected(self):
        rec = _make_record()
        del rec["document_id"]
        with self.assertRaises(ValueError) as ctx:
            self._load_one_record(rec)
        self.assertIn("document_id", str(ctx.exception))

    def test_missing_version_id_rejected(self):
        rec = _make_record()
        del rec["version_id"]
        with self.assertRaises(ValueError) as ctx:
            self._load_one_record(rec)
        self.assertIn("version_id", str(ctx.exception))

    def test_missing_fingerprint_rejected(self):
        rec = _make_record()
        del rec["fingerprint"]
        with self.assertRaises(ValueError) as ctx:
            self._load_one_record(rec)
        self.assertIn("fingerprint", str(ctx.exception))

    def test_missing_segmentation_method_rejected(self):
        rec = _make_record()
        del rec["segmentation_method"]
        with self.assertRaises(ValueError) as ctx:
            self._load_one_record(rec)
        self.assertIn("segmentation_method", str(ctx.exception))

    def test_missing_pages_rejected(self):
        rec = _make_record()
        del rec["pages"]
        with self.assertRaises(ValueError) as ctx:
            self._load_one_record(rec)
        self.assertIn("pages", str(ctx.exception))


# ---------------------------------------------------------------------------
# 5. Empty text rejection
# ---------------------------------------------------------------------------


class TestEmptyText(FixtureHelper):

    def test_empty_string_text_rejected(self):
        recs = [_make_record(text="")]
        self.write(recs, doc_count=1, ver_count=1)
        with self.assertRaises(ValueError) as ctx:
            load_corpus(self.jsonl, self.manifest, skip_sha256=True)
        self.assertIn("text", str(ctx.exception))

    def test_whitespace_only_text_rejected(self):
        recs = [_make_record(text="   \t\n")]
        self.write(recs, doc_count=1, ver_count=1)
        with self.assertRaises(ValueError) as ctx:
            load_corpus(self.jsonl, self.manifest, skip_sha256=True)
        self.assertIn("text", str(ctx.exception))


# ---------------------------------------------------------------------------
# 6. Invalid document/version pairing
# ---------------------------------------------------------------------------


class TestDocVersionPairing(FixtureHelper):

    def test_valid_pairing_accepted(self):
        recs = [_make_record(document_id="DOC-0001", version_id="VER-0001")]
        self.write(recs,
                   documents=[{"document_id": "DOC-0001", "version_id": "VER-0001"}])
        segs, _ = load_corpus(self.jsonl, self.manifest, skip_sha256=True)
        self.assertEqual(len(segs), 1)

    def test_invalid_pairing_rejected(self):
        recs = [_make_record(document_id="DOC-0001", version_id="VER-0099")]
        self.write(recs,
                   documents=[{"document_id": "DOC-0001", "version_id": "VER-0001"}])
        with self.assertRaises(ValueError) as ctx:
            load_corpus(self.jsonl, self.manifest, skip_sha256=True)
        self.assertIn("Invalid document/version pairing", str(ctx.exception))


# ---------------------------------------------------------------------------
# 7. Invalid page provenance
# ---------------------------------------------------------------------------


class TestPageProvenance(FixtureHelper):

    def _one_record_load(self, record):
        sha = _write_jsonl(self.jsonl, [record])
        manifest = {
            "total_records": 1,
            "paragraph_segments": 0,
            "clause_aware_segments": 1,
            "document_count": 1,
            "version_count": 1,
            "page_provenance_records": len(record.get("pages", [])),
            "jsonl_sha256": sha,
            "jsonl_bytes": -1,
            "documents": [{"document_id": "DOC-0001", "version_id": "VER-0001"}],
        }
        with open(self.manifest, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh)
        return load_corpus(self.jsonl, self.manifest, skip_sha256=True)

    def test_empty_pages_array_rejected(self):
        rec = _make_record(pages=[])
        with self.assertRaises(ValueError) as ctx:
            self._one_record_load(rec)
        self.assertIn("pages", str(ctx.exception))

    def test_negative_char_start_rejected(self):
        rec = _make_record(pages=[_make_provenance(char_start=-1, char_end=10)])
        with self.assertRaises(ValueError) as ctx:
            self._one_record_load(rec)
        self.assertIn("char_start", str(ctx.exception))

    def test_char_end_less_than_char_start_rejected(self):
        rec = _make_record(pages=[_make_provenance(char_start=50, char_end=10)])
        with self.assertRaises(ValueError) as ctx:
            self._one_record_load(rec)
        self.assertIn("char_end", str(ctx.exception))

    def test_page_number_zero_rejected(self):
        rec = _make_record(
            pages=[_make_provenance(page_number=0)],
            page_start=0, page_end=0,
        )
        with self.assertRaises(ValueError) as ctx:
            self._one_record_load(rec)
        # Either page_start or page_number error
        err = str(ctx.exception)
        self.assertTrue("page_number" in err or "page_start" in err)

    def test_page_start_mismatch_rejected(self):
        # pages array has page 2, but page_start says 1
        rec = _make_record(
            pages=[_make_provenance(page_number=2)],
            page_start=1,
            page_end=2,
        )
        with self.assertRaises(ValueError) as ctx:
            self._one_record_load(rec)
        self.assertIn("page_start", str(ctx.exception))

    def test_page_end_mismatch_rejected(self):
        # pages has max page 1 but page_end says 2
        rec = _make_record(
            pages=[_make_provenance(page_number=1)],
            page_start=1,
            page_end=2,
        )
        with self.assertRaises(ValueError) as ctx:
            self._one_record_load(rec)
        self.assertIn("page_end", str(ctx.exception))

    def test_valid_multi_page_provenance_accepted(self):
        rec = _make_record(
            pages=[
                _make_provenance(page_number=1, char_start=0, char_end=200),
                _make_provenance(page_number=2, char_start=0, char_end=50),
            ],
            page_start=1,
            page_end=2,
        )
        segs, _ = self._one_record_load(rec)
        self.assertEqual(len(segs[0].pages), 2)


# ---------------------------------------------------------------------------
# 8. Experiment filtering
# ---------------------------------------------------------------------------


class TestExperimentFilters(FixtureHelper):

    def setUp(self):
        super().setUp()
        # Build a small mixed corpus: 2 clause_aware + 1 paragraph
        self.recs = [
            _make_record(clause_id="CLS-0000001", fingerprint="SEG-aaaaaaaaaaaaaaaa",
                         segmentation_method="clause_aware"),
            _make_record(clause_id="CLS-0000002", fingerprint="SEG-bbbbbbbbbbbbbbbb",
                         segmentation_method="clause_aware"),
            _make_record(clause_id="CLS-0000003", fingerprint="SEG-cccccccccccccccc",
                         segmentation_method="paragraph"),
        ]
        self.write(self.recs, doc_count=1, ver_count=1)
        self.segs, _ = load_corpus(self.jsonl, self.manifest, skip_sha256=True)

    def test_experiment_a_returns_only_clause_aware(self):
        a = experiment_a_corpus(self.segs)
        self.assertEqual(len(a), 2)
        for seg in a:
            self.assertEqual(seg.segmentation_method, "clause_aware")

    def test_experiment_b_returns_only_paragraph(self):
        b = experiment_b_corpus(self.segs)
        self.assertEqual(len(b), 1)
        self.assertEqual(b[0].segmentation_method, "paragraph")

    def test_a_and_b_are_disjoint(self):
        a_ids = {s.clause_id for s in experiment_a_corpus(self.segs)}
        b_ids = {s.clause_id for s in experiment_b_corpus(self.segs)}
        self.assertFalse(a_ids & b_ids, "Experiment A and B share clause IDs — mixing detected!")

    def test_a_plus_b_equals_total(self):
        a = experiment_a_corpus(self.segs)
        b = experiment_b_corpus(self.segs)
        self.assertEqual(len(a) + len(b), len(self.segs))

    def test_experiment_a_empty_corpus_raises(self):
        only_para = [s for s in self.segs if s.segmentation_method == "paragraph"]
        with self.assertRaises(ValueError) as ctx:
            experiment_a_corpus(only_para)
        self.assertIn("clause_aware", str(ctx.exception))

    def test_experiment_b_empty_corpus_raises(self):
        only_ca = [s for s in self.segs if s.segmentation_method == "clause_aware"]
        with self.assertRaises(ValueError) as ctx:
            experiment_b_corpus(only_ca)
        self.assertIn("paragraph", str(ctx.exception))


# ---------------------------------------------------------------------------
# 9. ClauseSegment and PageProvenance conversion correctness
# ---------------------------------------------------------------------------


class TestConversion(FixtureHelper):

    def _load_single(self, record):
        sha = _write_jsonl(self.jsonl, [record])
        manifest = {
            "total_records": 1,
            "paragraph_segments": 1 if record.get("segmentation_method") == "paragraph" else 0,
            "clause_aware_segments": 1 if record.get("segmentation_method", "clause_aware") == "clause_aware" else 0,
            "document_count": 1,
            "version_count": 1,
            "page_provenance_records": len(record.get("pages", [])),
            "jsonl_sha256": sha,
            "jsonl_bytes": -1,
            "documents": [{"document_id": record.get("document_id", "DOC-0001"),
                           "version_id": record.get("version_id", "VER-0001")}],
        }
        with open(self.manifest, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh)
        segs, stats = load_corpus(self.jsonl, self.manifest, skip_sha256=True)
        return segs[0], stats

    def test_clause_segment_type(self):
        seg, _ = self._load_single(_make_record())
        self.assertIsInstance(seg, ClauseSegment)

    def test_page_provenance_type(self):
        seg, _ = self._load_single(_make_record())
        self.assertEqual(len(seg.pages), 1)
        self.assertIsInstance(seg.pages[0], PageProvenance)

    def test_original_text_preserved(self):
        original_text = "Every employer shall maintain a register of workers."
        seg, _ = self._load_single(_make_record(text=original_text))
        self.assertEqual(seg.text, original_text)

    def test_unicode_text_preserved(self):
        hindi_text = "प्रत्येक नियोक्ता श्रमिकों का रजिस्टर बनाए रखेगा।"
        seg, _ = self._load_single(_make_record(text=hindi_text))
        self.assertEqual(seg.text, hindi_text)

    def test_nullable_heading_none_preserved(self):
        seg, _ = self._load_single(_make_record(heading=None))
        self.assertIsNone(seg.heading)

    def test_nullable_heading_string_preserved(self):
        seg, _ = self._load_single(_make_record(heading="Penal Provisions"))
        self.assertEqual(seg.heading, "Penal Provisions")

    def test_nullable_clause_number_none_preserved(self):
        seg, _ = self._load_single(_make_record(clause_number=None,
                                                  segmentation_method="paragraph"))
        self.assertIsNone(seg.clause_number)

    def test_nullable_clause_number_string_preserved(self):
        seg, _ = self._load_single(_make_record(clause_number="3(2)"))
        self.assertEqual(seg.clause_number, "3(2)")

    def test_all_ids_preserved(self):
        rec = _make_record(
            clause_id="CLS-0000042",
            document_id="DOC-0001",
            version_id="VER-0001",
            fingerprint="SEG-deadbeefdeadbeef",
            ordinal=7,
        )
        seg, _ = self._load_single(rec)
        self.assertEqual(seg.clause_id, "CLS-0000042")
        self.assertEqual(seg.document_id, "DOC-0001")
        self.assertEqual(seg.version_id, "VER-0001")
        self.assertEqual(seg.fingerprint, "SEG-deadbeefdeadbeef")
        self.assertEqual(seg.ordinal, 7)

    def test_offset_basis_preserved(self):
        seg, _ = self._load_single(_make_record(offset_basis="normalized_text"))
        self.assertEqual(seg.offset_basis, "normalized_text")

    def test_segmentation_method_preserved(self):
        seg, _ = self._load_single(_make_record(segmentation_method="clause_aware"))
        self.assertEqual(seg.segmentation_method, "clause_aware")

    def test_page_provenance_fields_preserved(self):
        prov = _make_provenance(page_number=3, char_start=10, char_end=200,
                                line_index_start=1, line_index_end=8)
        rec = _make_record(pages=[prov], page_start=3, page_end=3)
        seg, _ = self._load_single(rec)
        p = seg.pages[0]
        self.assertEqual(p.page_number, 3)
        self.assertEqual(p.char_start, 10)
        self.assertEqual(p.char_end, 200)
        self.assertEqual(p.line_index_start, 1)
        self.assertEqual(p.line_index_end, 8)

    def test_corpus_stats_returned(self):
        recs = [
            _make_record(clause_id="CLS-0000001", fingerprint="SEG-aaaaaaaaaaaaaaaa",
                         segmentation_method="clause_aware"),
            _make_record(clause_id="CLS-0000002", fingerprint="SEG-bbbbbbbbbbbbbbbb",
                         segmentation_method="paragraph"),
        ]
        self.write(recs, doc_count=1, ver_count=1)
        _, stats = load_corpus(self.jsonl, self.manifest, skip_sha256=True)
        self.assertIsInstance(stats, CorpusStats)
        self.assertEqual(stats.total, 2)
        self.assertEqual(stats.clause_aware, 1)
        self.assertEqual(stats.paragraph, 1)
        self.assertIn("DOC-0001", stats.document_ids)
        self.assertIn("VER-0001", stats.version_ids)


# ---------------------------------------------------------------------------
# 10. Invalid segmentation method
# ---------------------------------------------------------------------------


class TestInvalidSegmentationMethod(FixtureHelper):

    def test_unknown_method_rejected(self):
        recs = [_make_record(segmentation_method="unknown_method")]
        self.write(recs, doc_count=1, ver_count=1)
        with self.assertRaises(ValueError) as ctx:
            load_corpus(self.jsonl, self.manifest, skip_sha256=True)
        self.assertIn("segmentation_method", str(ctx.exception))


# ---------------------------------------------------------------------------
# 11. Error messages identify the offending record
# ---------------------------------------------------------------------------


class TestErrorMessages(FixtureHelper):

    def test_error_includes_clause_id(self):
        recs = [
            _make_record(clause_id="CLS-0000001", fingerprint="SEG-aaaaaaaaaaaaaaaa"),
            _make_record(clause_id="CLS-BAD-EMPTY", fingerprint="SEG-bbbbbbbbbbbbbbbb",
                         text=""),
        ]
        self.write(recs, doc_count=1, ver_count=1)
        with self.assertRaises(ValueError) as ctx:
            load_corpus(self.jsonl, self.manifest, skip_sha256=True)
        # Error should identify the bad record
        self.assertIn("CLS-BAD-EMPTY", str(ctx.exception))

    def test_duplicate_error_shows_both_lines(self):
        recs = [
            _make_record(clause_id="CLS-0000001", fingerprint="SEG-aaaaaaaaaaaaaaaa"),
            _make_record(clause_id="CLS-0000001", fingerprint="SEG-bbbbbbbbbbbbbbbb"),
        ]
        self.write(recs, doc_count=1, ver_count=1)
        with self.assertRaises(ValueError) as ctx:
            load_corpus(self.jsonl, self.manifest, skip_sha256=True)
        err = str(ctx.exception)
        self.assertIn("CLS-0000001", err)
        self.assertIn("line", err.lower())


if __name__ == "__main__":
    unittest.main()
