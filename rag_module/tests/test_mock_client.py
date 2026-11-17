"""Unit tests for MockBackendClient offline fixture loading."""

import unittest

from rag_module.client.exceptions import (
    BackendClientError,
    BackendConnectionError,
    BackendValidationError,
    DocumentNotFoundError,
)
from rag_module.client.mock_client import MockBackendClient


class TestMockBackendClient(unittest.TestCase):
    def setUp(self) -> None:
        self.client = MockBackendClient()

    def test_mock_health(self) -> None:
        health = self.client.health()
        self.assertEqual(health["status"], "ok")
        self.assertEqual(health["mode"], "offline-fixture")

    def test_list_documents(self) -> None:
        docs = self.client.list_documents()
        self.assertGreaterEqual(len(docs), 2)
        doc_ids = [d.document_id for d in docs]
        self.assertIn("DOC-0001", doc_ids)
        self.assertIn("DOC-0004", doc_ids)

    def test_list_documents_filtering(self) -> None:
        filtered = self.client.list_documents(q="leverage")
        self.assertEqual(len(filtered), 0)  # "leverage" is in clause text, not in title

        delhi_docs = self.client.list_documents(q="Delhi")
        self.assertEqual(len(delhi_docs), 1)
        self.assertEqual(delhi_docs[0].document_id, "DOC-0001")

    def test_get_clause_aware_segments(self) -> None:
        res = self.client.get_document_clauses("DOC-0001", method="clause_aware")
        self.assertEqual(res.document_id, "DOC-0001")
        self.assertEqual(res.segmentation_method, "clause_aware")
        self.assertGreaterEqual(len(res.items), 3)

        # Check provenance preservation
        first = res.items[0]
        self.assertEqual(first.clause_id, "CLS-0000010")
        self.assertEqual(first.page_start, 1)
        self.assertEqual(len(first.pages), 1)
        self.assertEqual(first.pages[0].char_start, 0)

    def test_get_paragraph_segments(self) -> None:
        res = self.client.get_document_clauses("DOC-0001", method="paragraph")
        self.assertEqual(res.segmentation_method, "paragraph")
        self.assertEqual(len(res.items), 1)
        self.assertEqual(res.items[0].segment_type, "paragraph")

    def test_multi_page_clause_provenance(self) -> None:
        # DOC-0004 has clause CLS-0000253 touching pages 1 and 2
        res = self.client.get_document_clauses("DOC-0004", method="clause_aware")
        multi_page = [c for c in res.items if c.clause_id == "CLS-0000253"]
        self.assertEqual(len(multi_page), 1)
        clause = multi_page[0]
        self.assertEqual(clause.page_start, 1)
        self.assertEqual(clause.page_end, 2)
        self.assertEqual(len(clause.pages), 2)
        self.assertEqual(clause.char_span_for_page(1), (1719, 1955))
        self.assertEqual(clause.char_span_for_page(2), (119, 318))

    def test_page_range_filtering(self) -> None:
        # page_from=2 should exclude segments strictly on page 1
        res = self.client.get_document_clauses("DOC-0001", page_from=2)
        for item in res.items:
            self.assertGreaterEqual(item.page_end, 2)

    def test_inverted_page_range_raises_validation_error(self) -> None:
        """Inverted page ranges (page_to < page_from) must raise HTTP 422 BackendValidationError."""
        with self.assertRaises(BackendValidationError) as ctx:
            self.client.get_document_clauses("DOC-0001", page_from=5, page_to=2)
        self.assertEqual(ctx.exception.status_code, 422)
        self.assertIn("page_to must be greater than or equal to page_from", str(ctx.exception))

    def test_malformed_fixture_raises_client_error(self) -> None:
        """Loading a corrupted or non-JSON fixture file raises BackendClientError."""
        import tempfile
        from pathlib import Path
        from rag_module.client.exceptions import BackendClientError

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            f.write("NOT_VALID_JSON{:::}")
            bad_path = Path(f.name)

        try:
            with self.assertRaises(BackendClientError):
                MockBackendClient(fixture_path=bad_path)
        finally:
            if bad_path.exists():
                bad_path.unlink()

    def test_not_found_raises(self) -> None:
        with self.assertRaises(DocumentNotFoundError):
            self.client.get_document_clauses("DOC-9999")

    def test_fetch_corpus_clauses(self) -> None:
        corpus = self.client.fetch_corpus_clauses()
        self.assertGreater(len(corpus), 0)
        doc_ids = {c.document_id for c in corpus}
        self.assertIn("DOC-0001", doc_ids)
        self.assertIn("DOC-0004", doc_ids)

    def test_error_simulation(self) -> None:
        failing_client = MockBackendClient(
            fail_with=BackendConnectionError("Simulated connection failure")
        )
        with self.assertRaises(BackendConnectionError):
            failing_client.health()


if __name__ == "__main__":
    unittest.main()

