"""Embeddings subsystem for NiyamSetu RAG module (Milestone 2 dense retrieval)."""

from rag_module.embeddings.mock import (
    ControllableMockEmbedder,
    DeterministicMockEmbedder,
    MockEmbedder,
)
from rag_module.embeddings.sentence_transformer import (
    DEFAULT_E5_DIMENSION,
    DEFAULT_E5_MODEL,
    SentenceTransformerEmbedder,
)
from rag_module.embeddings.validation import (
    PASSAGE_PREFIX,
    QUERY_PREFIX,
    format_passage_prefix,
    format_query_prefix,
    is_finite_number,
    normalize_l2,
    validate_vector,
    validate_vectors,
    vector_norm,
)

__all__ = [
    "SentenceTransformerEmbedder",
    "MockEmbedder",
    "DeterministicMockEmbedder",
    "ControllableMockEmbedder",
    "DEFAULT_E5_MODEL",
    "DEFAULT_E5_DIMENSION",
    "QUERY_PREFIX",
    "PASSAGE_PREFIX",
    "format_query_prefix",
    "format_passage_prefix",
    "is_finite_number",
    "normalize_l2",
    "vector_norm",
    "validate_vector",
    "validate_vectors",
]
