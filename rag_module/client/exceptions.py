"""Typed exceptions for NiyamSetu RAG client."""

from __future__ import annotations

from typing import Any, Dict, Optional


class BackendClientError(Exception):
    """Base exception for all backend client errors."""

    def __init__(self, message: str, status_code: Optional[int] = None, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details or {}

    def __str__(self) -> str:
        if self.status_code:
            return f"[{self.status_code}] {self.message}"
        return self.message


class BackendConnectionError(BackendClientError):
    """Raised when the backend server cannot be reached."""


class BackendTimeoutError(BackendClientError):
    """Raised when a request to the backend times out."""


class DocumentNotFoundError(BackendClientError):
    """Raised when a requested document or clause is not found (404)."""


class BackendValidationError(BackendClientError):
    """Raised when the backend rejects a query or parameter (422)."""


class BackendResponseError(BackendClientError):
    """Raised when the backend returns an unexpected payload or 500 error."""

