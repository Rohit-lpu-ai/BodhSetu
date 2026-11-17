"""Client subsystem for fetching data from NiyamSetu Backend or offline fixtures."""

from rag_module.client.backend_client import BackendClient
from rag_module.client.exceptions import (
    BackendClientError,
    BackendConnectionError,
    BackendResponseError,
    BackendTimeoutError,
    BackendValidationError,
    DocumentNotFoundError,
)
from rag_module.client.mock_client import MockBackendClient

__all__ = [
    "BackendClient",
    "MockBackendClient",
    "BackendClientError",
    "BackendConnectionError",
    "BackendTimeoutError",
    "DocumentNotFoundError",
    "BackendValidationError",
    "BackendResponseError",
]

