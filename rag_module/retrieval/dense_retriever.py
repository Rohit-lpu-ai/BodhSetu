"""Dense semantic retrieval for NiyamSetu RAG module.

Implements DenseRetriever: an in-process dense vector retriever compatible with
BaseRetriever that embeds corpus segments and ranks them by cosine similarity.

Design principles:
- Pure-Python exact cosine via dot product of L2-normalized vectors (no external vector DB).
- Atomic indexing: a failed embed_texts() call never corrupts the previous index state.
- Feature parity with BM25Retriever: uses compose_searchable_text() for composite indexing.
- Optional filtering by document_id and/or version_id.
- Deterministic tie-breaking: score DESC, ordinal ASC, clause_id ASC.
- Negative cosine scores are preserved and ranked (not filtered out).
- Deduplication by clause_id: first-occurrence-wins when indexing.
- Avoids redundant re-embedding if the corpus has not changed.

Scoring note:
Cosine similarity via dot product of L2-normalized vectors returns values in [-1.0, 1.0].
These scores reflect geometric vector orientation and are NOT calibrated probabilities.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence, Set, Tuple

from rag_module.core.interfaces import BaseEmbedder, BaseRetriever
from rag_module.core.models import ClauseSegment, RetrievedPassage
from rag_module.embeddings.validation import validate_vector, validate_vectors
from rag_module.retrieval.bm25_retriever import compose_searchable_text

logger = logging.getLogger(__name__)


def _dot_product(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """Compute the dot product of two equal-length float sequences.

    For unit L2-normalized vectors, this equals exact cosine similarity.
    """
    return sum(float(a) * float(b) for a, b in zip(vec_a, vec_b))


class DenseRetriever(BaseRetriever):
    """Dense vector retriever using an injected BaseEmbedder.

    Uses the same composite searchable text representation as BM25Retriever
    (via compose_searchable_text) so both lexical and dense retrievers index
    equivalent document content.

    Args:
        embedder: A BaseEmbedder instance. If None, instantiates a lazy
            SentenceTransformerEmbedder via SentenceTransformerEmbedder.from_settings()
            without downloading or loading weights immediately.
    """

    def __init__(self, embedder: Optional[BaseEmbedder] = None) -> None:
        if embedder is None:
            from rag_module.embeddings.sentence_transformer import (
                SentenceTransformerEmbedder,
            )

            self.embedder: BaseEmbedder = SentenceTransformerEmbedder.from_settings()
        elif isinstance(embedder, BaseEmbedder):
            self.embedder = embedder
        else:
            raise TypeError(
                f"embedder must implement BaseEmbedder, got {type(embedder).__name__}"
            )

        self._segments: List[ClauseSegment] = []
        self._vectors: List[List[float]] = []
        self._clause_ids: Set[str] = set()
        self._corpus_fingerprint: Optional[Tuple[Any, ...]] = None

    @property
    def segment_count(self) -> int:
        """Number of unique segments currently indexed."""
        return len(self._segments)

    def index(self, segments: List[ClauseSegment]) -> None:
        """Build or update the dense retrieval index with the provided segments.

        Lifecycle:
        1. If segments is empty, clears the index state completely.
        2. Deduplicates segments by clause_id (first-occurrence-wins).
        3. Checks if deduplicated corpus matches the cached corpus fingerprint;
           if identical, avoids redundant re-embedding.
        4. Builds composite searchable text for each segment via compose_searchable_text().
        5. Embeds all texts via self.embedder.embed_texts().
        6. Validates vectors (dimension, finite values, unit L2 norm).
        7. Commits state atomically only after successful embedding and validation.

        Args:
            segments: List of ClauseSegment objects to index.

        Raises:
            ValueError: If embedder returns vector count/dimension mismatch or invalid floats.
            Any exception raised by embed_texts() propagates without corrupting previous state.
        """
        if not segments:
            self._segments = []
            self._vectors = []
            self._clause_ids = set()
            self._corpus_fingerprint = None
            logger.debug("DenseRetriever: index cleared (empty segment list).")
            return

        # 1. Deduplicate by clause_id (first-occurrence-wins)
        seen_ids: Set[str] = set()
        new_segments: List[ClauseSegment] = []
        for seg in segments:
            if seg.clause_id not in seen_ids:
                seen_ids.add(seg.clause_id)
                new_segments.append(seg)

        # 2. Check corpus fingerprint to avoid redundant re-embedding
        new_fingerprint = tuple(
            (s.clause_id, s.fingerprint, s.text) for s in new_segments
        )
        if (
            self._corpus_fingerprint is not None
            and self._corpus_fingerprint == new_fingerprint
        ):
            logger.debug("DenseRetriever: corpus unchanged; skipping re-indexing.")
            return

        # 3. Build searchable texts using BM25's compose_searchable_text()
        texts = [compose_searchable_text(seg) for seg in new_segments]

        # 4. Embed passages (staged in local variable for atomicity)
        new_vectors = self.embedder.embed_texts(texts)

        # 5. Validate output vectors
        if len(new_vectors) != len(new_segments):
            raise ValueError(
                f"Embedder returned {len(new_vectors)} vectors for {len(new_segments)} segments."
            )

        validate_vectors(
            new_vectors,
            expected_dim=self.embedder.dimension,
            check_normalized=True,
        )

        # 6. Atomic commit
        self._segments = new_segments
        self._vectors = new_vectors
        self._clause_ids = seen_ids
        self._corpus_fingerprint = new_fingerprint
        logger.debug(
            "DenseRetriever: successfully indexed %d segments (deduplicated from %d).",
            len(new_segments),
            len(segments),
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        document_id: Optional[str] = None,
        version_id: Optional[str] = None,
    ) -> List[RetrievedPassage]:
        """Retrieve top-k segments by dense cosine similarity.

        Args:
            query: User search query.
            top_k: Maximum number of passages to return. If <= 0, returns [].
            document_id: Optional filter for a specific document_id.
            version_id: Optional filter for a specific version_id.

        Returns:
            List of RetrievedPassage objects ranked by score (descending).
            Negative cosine scores are preserved and ranked.
            Ties are broken deterministically: score DESC, ordinal ASC, clause_id ASC.
        """
        if top_k <= 0:
            return []

        if not query or not query.strip():
            return []

        if not self._segments:
            return []

        # Generate query vector and validate
        query_vec = self.embedder.embed_query(query)
        validate_vector(
            query_vec,
            expected_dim=self.embedder.dimension,
            check_normalized=True,
        )

        # Score segments and apply optional filters
        scored_candidates: List[Tuple[float, int, str, ClauseSegment]] = []
        for seg, vec in zip(self._segments, self._vectors):
            if document_id is not None and seg.document_id != document_id:
                continue
            if version_id is not None and seg.version_id != version_id:
                continue

            score = _dot_product(query_vec, vec)
            # Retain all scores including negative numbers
            scored_candidates.append((score, seg.ordinal, seg.clause_id, seg))

        # Sort: score DESC (-score ASC), ordinal ASC, clause_id ASC
        scored_candidates.sort(key=lambda x: (-x[0], x[1], x[2]))

        results: List[RetrievedPassage] = []
        for rank_idx, (score, _, _, seg) in enumerate(
            scored_candidates[:top_k], start=1
        ):
            results.append(
                RetrievedPassage(
                    segment=seg,
                    score=float(score),
                    rank=rank_idx,
                    retrieval_method="dense",
                )
            )

        return results
