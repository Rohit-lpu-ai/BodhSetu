"""Read-only HTTP client consuming Person 1's NiyamSetu Backend API contract.

This client follows the specifications in backend-integration-contract.md (v1.0):
- Interacts exclusively via HTTP GET requests under /api.
- Does not import database drivers, SQLAlchemy, or FastAPI backend modules.
- Maps backend errors (using the standard {"error": {"code": ...}} envelope)
  to typed Python exceptions.
- Handles pagination automatically to assemble full clause collections.
"""

from __future__ import annotations

import json
import logging
import socket
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

from rag_module.client.exceptions import (
    BackendClientError,
    BackendConnectionError,
    BackendResponseError,
    BackendTimeoutError,
    BackendValidationError,
    DocumentNotFoundError,
)
from rag_module.config.settings import Settings, get_settings
from rag_module.core.models import (
    ClauseListResponse,
    ClauseSegment,
    DocumentSummary,
)

logger = logging.getLogger(__name__)


class BackendClient:
    """Read-only client for the NiyamSetu REST API."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        cfg = settings or get_settings()
        self.base_url = (base_url or cfg.backend_base_url).rstrip("/")
        self.timeout = timeout if timeout is not None else cfg.http_timeout_seconds

    def _request(
        self, endpoint: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute a GET request against the backend API and parse JSON response."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        if params:
            clean_params = {
                k: str(v) for k, v in params.items() if v is not None
            }
            if clean_params:
                url = f"{url}?{urllib.parse.urlencode(clean_params)}"

        req = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "NiyamSetu-RAG-Client/0.1.0",
            },
            method="GET",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                status_code = response.getcode()
                raw_bytes = response.read()
                try:
                    payload = json.loads(raw_bytes.decode("utf-8"))
                except json.JSONDecodeError as exc:
                    raise BackendResponseError(
                        f"Failed to parse JSON response from {url}: {exc}",
                        status_code=status_code,
                    ) from exc
                return payload

        except urllib.error.HTTPError as exc:
            status_code = exc.code
            error_body: Dict[str, Any] = {}
            try:
                raw_err = exc.read().decode("utf-8")
                parsed = json.loads(raw_err)
                if isinstance(parsed, dict) and "error" in parsed:
                    error_body = parsed["error"]
                else:
                    error_body = parsed
            except Exception:
                error_body = {"raw": str(exc)}

            code = error_body.get("code", "UNKNOWN_ERROR")
            message = error_body.get("message", exc.reason or f"HTTP {status_code}")
            details = error_body.get("details", {})

            if status_code == 404:
                raise DocumentNotFoundError(
                    message=message, status_code=status_code, details=details
                ) from exc
            if status_code == 422:
                raise BackendValidationError(
                    message=message, status_code=status_code, details=details
                ) from exc
            raise BackendResponseError(
                message=f"[{code}] {message}",
                status_code=status_code,
                details=details,
            ) from exc

        except urllib.error.URLError as exc:
            reason = exc.reason
            if isinstance(reason, (socket.timeout, TimeoutError)):
                raise BackendTimeoutError(
                    f"Request to {url} timed out after {self.timeout}s"
                ) from exc
            raise BackendConnectionError(
                f"Failed to connect to backend at {url}: {reason}"
            ) from exc

        except (socket.timeout, TimeoutError) as exc:
            raise BackendTimeoutError(
                f"Request to {url} timed out after {self.timeout}s"
            ) from exc

        except BackendClientError:
            raise

        except Exception as exc:
            raise BackendClientError(
                f"Unexpected error communicating with backend at {url}: {exc}"
            ) from exc

    def health(self) -> Dict[str, Any]:
        """Check the health and database reachability of the backend."""
        return self._request("health")

    def list_documents(
        self,
        q: Optional[str] = None,
        department: Optional[str] = None,
        document_type: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> List[DocumentSummary]:
        """Fetch a single page of the document catalog.

        The backend clamps limit to le=100.
        """
        clamped_limit = min(max(1, limit), 100)
        params = {
            "q": q,
            "department": department,
            "document_type": document_type,
            "limit": clamped_limit,
            "offset": max(0, offset),
        }
        res = self._request("documents", params=params)
        items = res.get("items", [])
        return [DocumentSummary.from_dict(item) for item in items]

    def fetch_all_documents(
        self,
        q: Optional[str] = None,
        department: Optional[str] = None,
        document_type: Optional[str] = None,
        page_size: int = 100,
    ) -> List[DocumentSummary]:
        """Exhaustively fetch all documents in the catalog using backend pagination."""
        clamped_size = min(max(1, page_size), 100)
        offset = 0
        all_docs: List[DocumentSummary] = []

        while True:
            params = {
                "q": q,
                "department": department,
                "document_type": document_type,
                "limit": clamped_size,
                "offset": offset,
            }
            res = self._request("documents", params=params)
            raw_items = res.get("items", [])
            total = int(res.get("total", 0))

            items = [DocumentSummary.from_dict(item) for item in raw_items]
            all_docs.extend(items)
            offset += len(items)

            if offset >= total or not items:
                break

        return all_docs

    def get_document_clauses(
        self,
        document_id: str,
        version_id: Optional[str] = None,
        method: str = "clause_aware",
        segment_type: Optional[str] = None,
        page_from: Optional[int] = None,
        page_to: Optional[int] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> ClauseListResponse:
        """Fetch a single page of clauses/segments for a document version."""
        if not document_id:
            raise ValueError("document_id must not be empty")

        clamped_limit = min(max(1, limit), 100)
        params = {
            "version_id": version_id,
            "method": method,
            "segment_type": segment_type,
            "page_from": page_from,
            "page_to": page_to,
            "limit": clamped_limit,
            "offset": max(0, offset),
        }
        endpoint = f"documents/{document_id}/clauses"
        res = self._request(endpoint, params=params)
        return ClauseListResponse.from_dict(res)

    def fetch_all_clauses(
        self,
        document_id: str,
        version_id: Optional[str] = None,
        method: str = "clause_aware",
        page_size: int = 100,
    ) -> List[ClauseSegment]:
        """Exhaustively fetch all clauses for a given document version via pagination.

        Clamps page_size to the backend's allowed [1, 100] range.
        """
        clamped_size = min(max(1, page_size), 100)
        offset = 0
        all_clauses: List[ClauseSegment] = []

        while True:
            response = self.get_document_clauses(
                document_id=document_id,
                version_id=version_id,
                method=method,
                limit=clamped_size,
                offset=offset,
            )
            all_clauses.extend(response.items)
            offset += len(response.items)

            if offset >= response.total or not response.items:
                break

        return all_clauses

    def fetch_corpus_clauses(
        self,
        document_ids: Optional[List[str]] = None,
        method: str = "clause_aware",
    ) -> List[ClauseSegment]:
        """Fetch all clauses across the specified documents or the entire catalog."""
        if document_ids is None:
            docs = self.fetch_all_documents()
            target_ids = [d.document_id for d in docs]
        else:
            target_ids = document_ids

        corpus: List[ClauseSegment] = []
        for doc_id in target_ids:
            clauses = self.fetch_all_clauses(doc_id, method=method)
            corpus.extend(clauses)

        return corpus

