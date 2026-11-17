"""Abstract base interfaces for NiyamSetu RAG subsystems.

These interfaces establish clean contracts between subsystems so that
implementations (e.g., BM25 vs FAISS, Mock vs Ollama vs API generators) can be
swapped and tested without coupling.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from rag_module.core.models import (
    Citation,
    ClauseSegment,
    RetrievedPassage,
)


class BaseRetriever(ABC):
    """Abstract interface for passage retrieval."""

    @abstractmethod
    def index(self, segments: List[ClauseSegment]) -> None:
        """Build or update the retrieval index with the provided segments."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedPassage]:
        """Retrieve the top-k most relevant segments for a query."""


class BaseEmbedder(ABC):
    """Abstract interface for dense vector embeddings."""

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate dense embeddings for a batch of text passages."""

    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """Generate a dense embedding for a single user query."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensionality of the embedding vectors."""


class BaseGenerator(ABC):
    """Abstract interface for grounded answer synthesis."""

    @abstractmethod
    def generate(
        self, query: str, passages: List[RetrievedPassage]
    ) -> Tuple[str, List[Citation], bool]:
        """Generate an answer grounded in the provided passages.

        Returns:
            Tuple of (answer_text, extracted_citations, is_sufficient_evidence).
        """


class BaseCitationVerifier(ABC):
    """Abstract interface for citation and provenance verification."""

    @abstractmethod
    def verify(self, citation: Citation, segment: ClauseSegment) -> Citation:
        """Verify that a quotation accurately references the given segment.

        Updates verification_status, extracted_raw_quote, and char_offset.
        """

