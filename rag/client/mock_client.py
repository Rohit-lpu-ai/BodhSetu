"""Offline mock client for NiyamSetu RAG module.

Loads verified backend API responses from a local JSON fixture without requiring
a running PostgreSQL instance or FastAPI server. Ideal for offline development,
CI runs, and unit tests.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from rag_module.client.exceptions import (
    BackendClientError,
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


class MockBackendClient:
    """Mock implementation of BackendClient reading from verified fixture files."""

    def __init__(
        self,
        fixture_path: Optional[Path] = None,
        fail_with: Optional[Exception] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        cfg = settings or get_settings()
        self.fixture_path = (
            fixture_path or cfg.fixture_dir / "mock_segments.json"
        )
        self.fail_with = fail_with
        self._data: Dict[str, Any] = self._load_fixture()

    def _guard(self) -> None:
        """Simulate configured failure states."""
        if self.fail_with is not None:
            raise self.fail_with

    def _load_fixture(self) -> Dict[str, Any]:
        """Load fixture JSON from disk, or return empty structures if missing."""
        if not self.fixture_path.exists():
            logger.warning(
                "Fixture file not found at %s. Mock client initialized with empty data.",
                self.fixture_path,
            )
            return {"documents": [], "clauses": {}}

        try:
            with open(self.fixture_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.error("Failed to parse fixture at %s: %s", self.fixture_path, exc)
            raise BackendClientError(
                f"Failed to load mock fixture from {self.fixture_path}: {exc}"
            ) from exc

    def health(self) -> Dict[str, Any]:
        """Simulate health endpoint."""
        self._guard()
        return {
            "status": "ok",
            "database": "ok",
            "version": "0.1.0-mock",
            "mode": "offline-fixture",
        }

    def list_documents(
        self,
        q: Optional[str] = None,
        department: Optional[str] = None,
        document_type: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> List[DocumentSummary]:
        """List documents from the fixture with optional filtering and pagination."""
        self._guard()
        raw_docs = self._data.get("documents", [])
        filtered: List[Dict[str, Any]] = []

        for d in raw_docs:
            if q and q.lower() not in d.get("title", "").lower() and q.lower() not in d.get("reference_no", "").lower():
                continue
            if department and department.lower() != (d.get("department") or "").lower():
                continue
            if document_type and document_type.lower() != (d.get("document_type") or "").lower():
                continue
            filtered.append(d)

        paginated = filtered[offset : offset + limit]
        return [DocumentSummary.from_dict(item) for item in paginated]

    def fetch_all_documents(
        self,
        q: Optional[str] = None,
        department: Optional[str] = None,
        document_type: Optional[str] = None,
        page_size: int = 100,
    ) -> List[DocumentSummary]:
        """Fetch all documents matching filters from fixture data."""
        return self.list_documents(
            q=q,
            department=department,
            document_type=document_type,
            limit=10000,
            offset=0,
        )

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
        """Fetch segments for a document from fixture data."""
        self._guard()
        if not document_id:
            raise BackendValidationError("document_id must not be empty", status_code=422)

        if page_from is not None and page_to is not None and page_to < page_from:
            raise BackendValidationError(
                "page_to must be greater than or equal to page_from.",
                status_code=422,
                details={"page_from": page_from, "page_to": page_to},
            )

        doc_clauses = self._data.get("clauses", {}).get(document_id)
        if doc_clauses is None:
            raise DocumentNotFoundError(
                f"no document with id {document_id}", status_code=404
            )

        if method not in doc_clauses:
            raise BackendValidationError(
                f"unsupported method '{method}'. Available: {list(doc_clauses.keys())}",
                status_code=422,
            )

        raw_items: List[Dict[str, Any]] = doc_clauses[method]

        # Apply filtering
        filtered: List[Dict[str, Any]] = []
        for item in raw_items:
            if segment_type and item.get("segment_type") != segment_type:
                continue
            item_start = item.get("page_start", 1)
            item_end = item.get("page_end", 1)
            if page_from is not None and item_end < page_from:
                continue
            if page_to is not None and item_start > page_to:
                continue
            filtered.append(item)

        total = len(filtered)
        paginated_raw = filtered[offset : offset + limit]

        items = [
            ClauseSegment.from_dict(
                item,
                default_document_id=document_id,
                default_version_id=version_id or "VER-0001",
                default_method=method,
            )
            for item in paginated_raw
        ]

        return ClauseListResponse(
            document_id=document_id,
            version_id=version_id or "VER-0001",
            segmentation_method=method,
            offset_basis="normalized_text",
            total=total,
            limit=limit,
            offset=offset,
            items=items,
        )

    def fetch_all_clauses(
        self,
        document_id: str,
        version_id: Optional[str] = None,
        method: str = "clause_aware",
        page_size: int = 100,
    ) -> List[ClauseSegment]:
        """Fetch all clauses for a document under the specified method."""
        res = self.get_document_clauses(
            document_id=document_id,
            version_id=version_id,
            method=method,
            limit=page_size,
            offset=0,
        )
        return res.items

    def fetch_corpus_clauses(
        self,
        document_ids: Optional[List[str]] = None,
        method: str = "clause_aware",
    ) -> List[ClauseSegment]:
        """Fetch all clauses across all documents in the fixture."""
        target_ids = (
            document_ids
            if document_ids is not None
            else list(self._data.get("clauses", {}).keys())
        )
        corpus: List[ClauseSegment] = []
        for doc_id in target_ids:
            try:
                clauses = self.fetch_all_clauses(doc_id, method=method)
                corpus.extend(clauses)
            except DocumentNotFoundError:
                continue
        return corpus

