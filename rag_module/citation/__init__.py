"""Citation extraction and verification against the retrieved source text."""

from rag_module.citation.verifier import (
    DEFAULT_QUOTE_CHARS,
    MARKER_PATTERN,
    CitationReport,
    CitationVerifier,
    extract_markers,
)

__all__ = [
    "CitationVerifier",
    "CitationReport",
    "extract_markers",
    "MARKER_PATTERN",
    "DEFAULT_QUOTE_CHARS",
]
