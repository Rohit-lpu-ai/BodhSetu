"""Unit tests for BackendClient HTTP operations and error mapping."""

import io
import json
import socket
import unittest
import urllib.error
from typing import Any, Dict
from unittest.mock import MagicMock, patch

from rag_module.client.backend_client import BackendClient
from rag_module.client.exceptions import (
    BackendConnectionError,
    BackendResponseError,
    BackendTimeoutError,
    BackendValidationError,
    DocumentNotFoundError,
)


class TestBackendClient(unittest.TestCase):
    def setUp(self) -> None:
        self.client = BackendClient(base_url="http://mock-backend:8000/api", timeout=2.0)

    def _mock_http_response(self, status: int, data: Dict[str, Any]) -> MagicMock:
        body = json.dumps(data).encode("utf-8")
        resp = MagicMock()
        resp.getcode.return_value = status
        resp.read.return_value = body
        resp.__enter__.return_value = resp
        resp.__exit__.return_value = None
        return resp

    @patch("urllib.request.urlopen")
    def test_health_check_success(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = self._mock_http_response(
            200, {"status": "ok", "database": "ok", "version": "0.1.0"}
        )
        health = self.client.health()
        self.assertEqual(health["status"], "ok")
        self.assertEqual(health["database"], "ok")

    @patch("urllib.request.urlopen")
    def test_list_documents_parsing(self, mock_urlopen: MagicMock) -> None:
        payload = {
            "total": 1,
            "limit": 20,
            "offset": 0,
            "items": [
                {
                    "document_id": "DOC-0001",
                    "title": "Lead Bank Directions",
                    "department": "FIDD",
                    "document_type": "circular",
                    "reference_no": "RBI/2026/1",
                    "version_count": 1,
                    "latest_publication_date": "2026-10-26",
                }
            ],
        }
        mock_urlopen.return_value = self._mock_http_response(200, payload)
        docs = self.client.list_documents(department="FIDD")
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].document_id, "DOC-0001")
        self.assertEqual(docs[0].department, "FIDD")

    @patch("urllib.request.urlopen")
    def test_get_document_clauses_parsing(self, mock_urlopen: MagicMock) -> None:
        payload = {
            "document_id": "DOC-0001",
            "version_id": "VER-0001",
            "segmentation_method": "clause_aware",
            "offset_basis": "normalized_text",
            "total": 1,
            "limit": 50,
            "offset": 0,
            "items": [
                {
                    "clause_id": "CLS-0001",
                    "fingerprint": "SEG-0001",
                    "ordinal": 0,
                    "segment_type": "clause",
                    "clause_number": "1",
                    "clause_number_ambiguous": False,
                    "heading": None,
                    "text": "1. Test clause text.",
                    "page_start": 1,
                    "page_end": 1,
                    "pages": [{"page_number": 1, "char_start": 0, "char_end": 20}],
                    "level": 1,
                    "marker_style": "numeric_dotted",
                    "parent_clause_id": None,
                    "metadata": {},
                }
            ],
        }
        mock_urlopen.return_value = self._mock_http_response(200, payload)
        response = self.client.get_document_clauses("DOC-0001")
        self.assertEqual(response.document_id, "DOC-0001")
        self.assertEqual(len(response.items), 1)
        self.assertEqual(response.items[0].clause_id, "CLS-0001")
        self.assertEqual(response.items[0].char_start, 0)
        self.assertEqual(response.items[0].char_end, 20)

    @patch("urllib.request.urlopen")
    def test_fetch_all_clauses_pagination(self, mock_urlopen: MagicMock) -> None:
        def side_effect(req, timeout=None):
            url = req.full_url
            if "offset=0" in url:
                payload = {
                    "document_id": "DOC-0001",
                    "version_id": "VER-0001",
                    "segmentation_method": "clause_aware",
                    "offset_basis": "normalized_text",
                    "total": 2,
                    "limit": 1,
                    "offset": 0,
                    "items": [
                        {
                            "clause_id": "CLS-0001",
                            "text": "Clause 1",
                            "pages": [{"page_number": 1, "char_start": 0, "char_end": 8}],
                        }
                    ],
                }
            else:
                payload = {
                    "document_id": "DOC-0001",
                    "version_id": "VER-0001",
                    "segmentation_method": "clause_aware",
                    "offset_basis": "normalized_text",
                    "total": 2,
                    "limit": 1,
                    "offset": 1,
                    "items": [
                        {
                            "clause_id": "CLS-0002",
                            "text": "Clause 2",
                            "pages": [{"page_number": 1, "char_start": 9, "char_end": 17}],
                        }
                    ],
                }
            return self._mock_http_response(200, payload)

        mock_urlopen.side_effect = side_effect
        all_clauses = self.client.fetch_all_clauses("DOC-0001", page_size=1)
        self.assertEqual(len(all_clauses), 2)
        self.assertEqual(all_clauses[0].clause_id, "CLS-0001")
        self.assertEqual(all_clauses[1].clause_id, "CLS-0002")

    @patch("urllib.request.urlopen")
    def test_404_maps_to_document_not_found(self, mock_urlopen: MagicMock) -> None:
        err_body = json.dumps(
            {"error": {"code": "NOT_FOUND", "message": "no document DOC-9999"}}
        ).encode("utf-8")
        err = urllib.error.HTTPError(
            url="http://mock-backend:8000/api/documents/DOC-9999/clauses",
            code=404,
            msg="Not Found",
            hdrs={},  # type: ignore[arg-type]
            fp=io.BytesIO(err_body),
        )
        mock_urlopen.side_effect = err

        with self.assertRaises(DocumentNotFoundError) as ctx:
            self.client.get_document_clauses("DOC-9999")
        self.assertIn("no document DOC-9999", str(ctx.exception))
        self.assertEqual(ctx.exception.status_code, 404)

    @patch("urllib.request.urlopen")
    def test_422_maps_to_backend_validation_error(self, mock_urlopen: MagicMock) -> None:
        err_body = json.dumps(
            {"error": {"code": "VALIDATION_ERROR", "message": "bad method"}}
        ).encode("utf-8")
        err = urllib.error.HTTPError(
            url="http://mock-backend:8000/api/documents/DOC-0001/clauses",
            code=422,
            msg="Unprocessable Entity",
            hdrs={},  # type: ignore[arg-type]
            fp=io.BytesIO(err_body),
        )
        mock_urlopen.side_effect = err

        with self.assertRaises(BackendValidationError) as ctx:
            self.client.get_document_clauses("DOC-0001", method="invalid_method")
        self.assertEqual(ctx.exception.status_code, 422)

    @patch("urllib.request.urlopen")
    def test_500_maps_to_backend_response_error(self, mock_urlopen: MagicMock) -> None:
        err_body = json.dumps(
            {"error": {"code": "INTERNAL_ERROR", "message": "An internal error occurred."}}
        ).encode("utf-8")
        err = urllib.error.HTTPError(
            url="http://mock-backend:8000/api/documents",
            code=500,
            msg="Internal Server Error",
            hdrs={},  # type: ignore[arg-type]
            fp=io.BytesIO(err_body),
        )
        mock_urlopen.side_effect = err

        with self.assertRaises(BackendResponseError) as ctx:
            self.client.list_documents()
        self.assertEqual(ctx.exception.status_code, 500)

    @patch("urllib.request.urlopen")
    def test_connection_error_maps_to_backend_connection_error(
        self, mock_urlopen: MagicMock
    ) -> None:
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")
        with self.assertRaises(BackendConnectionError):
            self.client.health()

    @patch("urllib.request.urlopen")
    def test_timeout_maps_to_backend_timeout_error(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = socket.timeout("timed out")
        with self.assertRaises(BackendTimeoutError):
            self.client.health()

    @patch("urllib.request.urlopen")
    def test_fetch_all_documents_pagination_over_100(self, mock_urlopen: MagicMock) -> None:
        """fetch_all_documents must correctly paginate when catalog exceeds 100 documents."""
        docs_page1 = [
            {"document_id": f"DOC-{i:04d}", "title": f"Doc {i}"}
            for i in range(1, 101)
        ]
        docs_page2 = [
            {"document_id": f"DOC-{i:04d}", "title": f"Doc {i}"}
            for i in range(101, 151)
        ]

        def side_effect(req, timeout=None):
            url = req.full_url
            if "offset=0" in url:
                payload = {"total": 150, "limit": 100, "offset": 0, "items": docs_page1}
            elif "offset=100" in url:
                payload = {"total": 150, "limit": 100, "offset": 100, "items": docs_page2}
            else:
                payload = {"total": 150, "limit": 100, "offset": 200, "items": []}
            return self._mock_http_response(200, payload)

        mock_urlopen.side_effect = side_effect
        all_docs = self.client.fetch_all_documents(page_size=100)

        self.assertEqual(len(all_docs), 150)
        # Verify no duplicate documents
        doc_ids = [d.document_id for d in all_docs]
        self.assertEqual(len(doc_ids), len(set(doc_ids)))
        self.assertEqual(doc_ids[0], "DOC-0001")
        self.assertEqual(doc_ids[99], "DOC-0100")
        self.assertEqual(doc_ids[100], "DOC-0101")
        self.assertEqual(doc_ids[149], "DOC-0150")

    @patch("urllib.request.urlopen")
    def test_fetch_all_clauses_clamps_page_size(self, mock_urlopen: MagicMock) -> None:
        """page_size > 100 must be clamped to 100 to respect backend le=100 validator."""
        mock_urlopen.return_value = self._mock_http_response(
            200,
            {
                "document_id": "DOC-0001",
                "version_id": "VER-0001",
                "segmentation_method": "clause_aware",
                "offset_basis": "normalized_text",
                "total": 1,
                "limit": 100,
                "offset": 0,
                "items": [
                    {
                        "clause_id": "CLS-0001",
                        "text": "Text",
                        "pages": [{"page_number": 1, "char_start": 0, "char_end": 4}],
                    }
                ],
            },
        )
        self.client.fetch_all_clauses("DOC-0001", page_size=500)
        req = mock_urlopen.call_args[0][0]
        self.assertIn("limit=100", req.full_url)
        self.assertNotIn("limit=500", req.full_url)

    @patch("urllib.request.urlopen")
    def test_malformed_json_response_raises_response_error(
        self, mock_urlopen: MagicMock
    ) -> None:
        """Malformed JSON from a 200 OK response raises BackendResponseError."""
        resp = MagicMock()
        resp.getcode.return_value = 200
        resp.read.return_value = b"<html>Not JSON</html>"
        resp.__enter__.return_value = resp
        resp.__exit__.return_value = None
        mock_urlopen.return_value = resp

        with self.assertRaises(BackendResponseError) as ctx:
            self.client.health()
        self.assertIn("Failed to parse JSON response", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

