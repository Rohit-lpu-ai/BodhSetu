"""Unit tests for RAG data models."""

import unittest

from rag_module.core.models import (
    Citation,
    ClauseListResponse,
    ClauseSegment,
    DocumentSummary,
    PageProvenance,
    RAGQuery,
    RAGResponse,
    RetrievedPassage,
)


class TestPageProvenance(unittest.TestCase):
    def test_valid_page_provenance(self) -> None:
        p = PageProvenance(
            page_number=1,
            char_start=100,
            char_end=250,
            line_index_start=2,
            line_index_end=5,
        )
        self.assertEqual(p.page_number, 1)
        self.assertEqual(p.char_start, 100)
        self.assertEqual(p.char_end, 250)
        d = p.to_dict()
        self.assertEqual(d["page_number"], 1)
        p_rebuilt = PageProvenance.from_dict(d)
        self.assertEqual(p_rebuilt.char_end, 250)

    def test_invalid_page_number_raises(self) -> None:
        with self.assertRaises(ValueError):
            PageProvenance(page_number=0, char_start=0, char_end=10)

    def test_invalid_char_offsets_raises(self) -> None:
        with self.assertRaises(ValueError):
            PageProvenance(page_number=1, char_start=50, char_end=20)


class TestClauseSegment(unittest.TestCase):
    def setUp(self) -> None:
        self.provenance = [
            PageProvenance(page_number=1, char_start=1719, char_end=1955),
            PageProvenance(page_number=2, char_start=119, char_end=318),
        ]
        self.segment = ClauseSegment(
            document_id="DOC-0004",
            version_id="VER-0004",
            clause_id="CLS-0000253",
            fingerprint="SEG-2146f692b738fe91",
            ordinal=5,
            segment_type="clause",
            clause_number="(2)",
            clause_number_ambiguous=False,
            heading=None,
            text="(2) The minimum leverage ratio shall be 3.5 per cent.",
            page_start=1,
            page_end=2,
            pages=self.provenance,
            segmentation_method="clause_aware",
            offset_basis="normalized_text",
            level=2,
            marker_style="paren_number",
            parent_clause_id="CLS-0000250",
            metadata={"marker_style": "paren_number"},
        )

    def test_multi_page_char_offsets_raise(self) -> None:
        """Multi-page segments must raise ValueError on flat char_start/char_end."""
        with self.assertRaises(ValueError) as ctx:
            _ = self.segment.char_start
        self.assertIn("multi-page segment", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            _ = self.segment.char_end
        self.assertIn("multi-page segment", str(ctx.exception))

    def test_single_page_char_offsets_succeed(self) -> None:
        single_page_seg = ClauseSegment(
            document_id="DOC-0001",
            version_id="VER-0001",
            clause_id="CLS-0000001",
            fingerprint="SEG-single",
            ordinal=0,
            segment_type="clause",
            text="Single page clause text.",
            page_start=3,
            page_end=3,
            pages=[PageProvenance(page_number=3, char_start=50, char_end=74)],
        )
        self.assertEqual(single_page_seg.char_start, 50)
        self.assertEqual(single_page_seg.char_end, 74)

    def test_char_span_for_page(self) -> None:
        self.assertEqual(self.segment.char_span_for_page(1), (1719, 1955))
        self.assertEqual(self.segment.char_span_for_page(2), (119, 318))
        with self.assertRaises(ValueError) as ctx:
            self.segment.char_span_for_page(3)
        self.assertIn("not part of clause", str(ctx.exception))

    def test_serialization_round_trip(self) -> None:
        data = self.segment.to_dict()
        self.assertEqual(data["clause_id"], "CLS-0000253")
        self.assertIsNone(data["char_start"])  # Multi-page segment has None for flat offsets
        self.assertIsNone(data["char_end"])
        self.assertEqual(len(data["pages"]), 2)

        rebuilt = ClauseSegment.from_dict(data)
        self.assertEqual(rebuilt.clause_id, self.segment.clause_id)
        self.assertEqual(rebuilt.text, self.segment.text)
        self.assertEqual(rebuilt.level, 2)
        self.assertEqual(rebuilt.pages[1].page_number, 2)

    def test_missing_metadata_handled_gracefully(self) -> None:
        minimal_data = {
            "clause_id": "CLS-0001",
            "document_id": "DOC-0001",
            "version_id": "VER-0001",
            "text": "Some text",
            "pages": [{"page_number": 1, "char_start": 0, "char_end": 9}],
        }
        seg = ClauseSegment.from_dict(minimal_data)
        self.assertEqual(seg.clause_number, None)
        self.assertFalse(seg.clause_number_ambiguous)
        self.assertEqual(seg.metadata, {})
        self.assertEqual(seg.segmentation_method, "clause_aware")

    def test_empty_text_raises_validation_error(self) -> None:
        with self.assertRaises(ValueError):
            ClauseSegment(
                document_id="DOC-0001",
                version_id="VER-0001",
                clause_id="CLS-0001",
                fingerprint="SEG-1",
                ordinal=0,
                segment_type="clause",
                text="",
                page_start=1,
                page_end=1,
                pages=[PageProvenance(1, 0, 10)],
            )


class TestRAGQueryAndResponse(unittest.TestCase):
    def test_rag_query_defaults_and_validation(self) -> None:
        q = RAGQuery(question="What is the leverage ratio?")
        self.assertEqual(q.segmentation_method, "clause_aware")
        self.assertEqual(q.retrieval_strategy, "hybrid")
        self.assertEqual(q.top_k, 5)

        with self.assertRaises(ValueError):
            RAGQuery(question="")

        with self.assertRaises(ValueError):
            RAGQuery(question="Valid?", top_k=0)

        with self.assertRaises(ValueError):
            RAGQuery(question="Valid?", retrieval_strategy="unknown")

    def test_citation_validation_and_roundtrip(self) -> None:
        c = Citation(
            document_id="DOC-0004",
            version_id="VER-0004",
            clause_id="CLS-0000253",
            page_start=1,
            page_end=2,
            quote="3.5 per cent",
            page_number=1,
            verification_status="STRICT_VERBATIM",
            extracted_raw_quote=None,
            char_offset=(10, 22),
        )
        d = c.to_dict()
        self.assertEqual(d["verification_status"], "STRICT_VERBATIM")
        self.assertEqual(d["char_offset"], [10, 22])
        self.assertEqual(d["page_number"], 1)

        c2 = Citation.from_dict(d)
        self.assertEqual(c2.char_offset, (10, 22))
        self.assertEqual(c2.page_number, 1)

    def test_citation_missing_page_number_when_char_offset_present(self) -> None:
        """char_offset requires page_number because offsets are per-page."""
        with self.assertRaises(ValueError) as ctx:
            Citation(
                document_id="DOC-0004",
                version_id="VER-0004",
                clause_id="CLS-0000253",
                page_start=1,
                page_end=2,
                quote="3.5 per cent",
                page_number=None,
                char_offset=(10, 22),
            )
        self.assertIn("page_number must be provided", str(ctx.exception))

    def test_citation_invalid_page_number_range(self) -> None:
        """page_number must fall within [page_start, page_end]."""
        with self.assertRaises(ValueError) as ctx:
            Citation(
                document_id="DOC-0004",
                version_id="VER-0004",
                clause_id="CLS-0000253",
                page_start=1,
                page_end=2,
                quote="3.5 per cent",
                page_number=3,
                char_offset=(10, 22),
            )
        self.assertIn("must be between page_start", str(ctx.exception))

    def test_citation_invalid_char_offset_range(self) -> None:
        """char_offset end must be >= start and non-negative."""
        with self.assertRaises(ValueError):
            Citation(
                document_id="DOC-0004",
                version_id="VER-0004",
                clause_id="CLS-0000253",
                page_start=1,
                page_end=2,
                quote="3.5 per cent",
                page_number=1,
                char_offset=(50, 20),
            )

    def test_citation_without_char_offset(self) -> None:
        """Citation without char_offset is valid with page_number=None or valid page_number."""
        c = Citation(
            document_id="DOC-0004",
            version_id="VER-0004",
            clause_id="CLS-0000253",
            page_start=1,
            page_end=2,
            quote="General quote",
            page_number=None,
            char_offset=None,
        )
        self.assertIsNone(c.page_number)
        self.assertIsNone(c.char_offset)

    def test_citation_invalid_status_raises(self) -> None:
        with self.assertRaises(ValueError):
            Citation(
                document_id="D",
                version_id="V",
                clause_id="C",
                page_start=1,
                page_end=1,
                quote="q",
                verification_status="INVALID_STATUS",
            )

    def test_rag_response_roundtrip(self) -> None:
        c = Citation(
            document_id="DOC-0004",
            version_id="VER-0004",
            clause_id="CLS-0000253",
            page_start=1,
            page_end=2,
            quote="3.5 per cent",
            verification_status="STRICT_VERBATIM",
        )
        resp = RAGResponse(
            query="What is the minimum ratio?",
            answer="The minimum ratio is 3.5 per cent.",
            sufficient_evidence=True,
            citations=[c],
            segmentation_method="clause_aware",
            retrieval_strategy="hybrid",
        )
        d = resp.to_dict()
        self.assertTrue(d["sufficient_evidence"])
        self.assertEqual(len(d["citations"]), 1)

        resp2 = RAGResponse.from_dict(d)
        self.assertEqual(resp2.answer, resp.answer)
        self.assertEqual(resp2.citations[0].clause_id, "CLS-0000253")


if __name__ == "__main__":
    unittest.main()

