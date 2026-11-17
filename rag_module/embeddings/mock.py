"""Mock and controllable embedding implementations for testing and offline development.

IMPORTANT DESIGN NOTE:
Hash-based synthetic vectors verify mathematical pipeline mechanics, dimension
consistency, deterministic sorting, and tie-breaking. They DO NOT model linguistic
semantics or semantic understanding.

For tests that require verifying semantic similarity and ranking mathematically, use
ControllableMockEmbedder with predefined vectors (e.g. dot product = 0.85 vs 0.20).
"""

from __future__ import annotations

import hashlib
import struct
from typing import Dict, List, Optional, Sequence

from rag_module.core.interfaces import BaseEmbedder
from rag_module.embeddings.validation import (
    format_passage_prefix,
    format_query_prefix,
    normalize_l2,
    validate_vector,
    validate_vectors,
)


def _generate_deterministic_vector(text: str, dim: int) -> List[float]:
    """Generate a deterministic pseudo-random unit vector from text using SHA-256.

    Uses SHA-256 counter blocks to produce reproducible, uniformly distributed
    floats in [-1.0, 1.0], then normalizes the resulting vector to unit L2 norm.
    """
    raw_bytes = text.encode("utf-8")
    vec: List[float] = []
    block_index = 0

    while len(vec) < dim:
        digest = hashlib.sha256(raw_bytes + struct.pack(">I", block_index)).digest()
        # 32 bytes -> 8 signed 32-bit big-endian integers
        for i in range(0, 32, 4):
            val = struct.unpack(">i", digest[i : i + 4])[0]
            # Map signed 32-bit int to float in (-1.0, 1.0)
            vec.append(val / 2147483648.0)
            if len(vec) == dim:
                break
        block_index += 1

    return normalize_l2(vec)


class DeterministicMockEmbedder(BaseEmbedder):
    """Deterministic, offline mock embedder generating reproducible synthetic vectors.

    Produces unit L2-normalized vectors of the specified dimension without external
    machine learning libraries or downloads.

    Note:
        These embeddings DO NOT reflect semantic understanding. They test dimension
        consistency, batching, caching, sorting, and interface compliance.
    """

    def __init__(self, dimension: int = 384) -> None:
        if dimension < 1:
            raise ValueError(f"dimension must be positive, got {dimension}")
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_query(self, query: str) -> List[float]:
        """Generate a deterministic unit vector for a user query.

        Applies 'query: ' prefix automatically.
        """
        if not isinstance(query, str):
            raise TypeError(
                f"embed_query() expects a str, got {type(query).__name__}"
            )
        if not query or not query.strip():
            raise ValueError("Query text cannot be empty or whitespace-only")

        prefixed = format_query_prefix(query)
        vec = _generate_deterministic_vector(prefixed, self._dimension)
        validate_vector(vec, expected_dim=self._dimension, check_normalized=True)
        return vec

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate deterministic unit vectors for a batch of text passages.

        Applies 'passage: ' prefix automatically to each passage.
        """
        if not texts:
            return []

        for i, text in enumerate(texts):
            if not isinstance(text, str):
                raise TypeError(
                    f"embed_texts() expects all items to be str, got {type(text).__name__} at index {i}"
                )

        vecs: List[List[float]] = []
        for text in texts:
            prefixed = format_passage_prefix(text)
            vecs.append(_generate_deterministic_vector(prefixed, self._dimension))

        validate_vectors(vecs, expected_dim=self._dimension, check_normalized=True)
        return vecs


class ControllableMockEmbedder(BaseEmbedder):
    """Controllable test embedder that returns predefined vectors for specified texts.

    Enables deterministic mathematical tests for ranking and similarity, e.g.:
      embedder.set_vector("query: capital", [1.0, 0.0, 0.0])
      embedder.set_vector("passage: seg_high", [0.9, 0.4359, 0.0])  # dot = 0.90
      embedder.set_vector("passage: seg_low",  [0.2, 0.9798, 0.0])  # dot = 0.20

    If a text is not found in the predefined mapping:
    - If fallback_to_deterministic is True (default), generates a deterministic vector.
    - If fallback_to_deterministic is False, raises KeyError.
    """

    def __init__(
        self,
        dimension: int = 384,
        predefined_vectors: Optional[Dict[str, Sequence[float]]] = None,
        fallback_to_deterministic: bool = True,
        auto_normalize: bool = True,
    ) -> None:
        if dimension < 1:
            raise ValueError(f"dimension must be positive, got {dimension}")
        self._dimension = dimension
        self.fallback_to_deterministic = fallback_to_deterministic
        self.auto_normalize = auto_normalize
        self._vectors: Dict[str, List[float]] = {}

        if predefined_vectors:
            for text, vec in predefined_vectors.items():
                self.set_vector(text, vec)

    @property
    def dimension(self) -> int:
        return self._dimension

    def set_vector(self, text: str, vector: Sequence[float]) -> None:
        """Register a predefined vector for a given text or prefixed text."""
        vec = list(vector)
        if self.auto_normalize:
            vec = normalize_l2(vec)
        validate_vector(
            vec,
            expected_dim=self._dimension,
            check_normalized=self.auto_normalize,
        )
        # Store under stripped text and prefixed variations for flexible matching
        self._vectors[text.strip()] = vec

    def _lookup_vector(self, text: str, prefixed_text: str) -> List[float]:
        """Look up vector by raw text or prefixed text."""
        # 1. Exact match on prefixed text (e.g. 'query: CAR' or 'passage: seg_01')
        if prefixed_text in self._vectors:
            return self._vectors[prefixed_text]

        # 2. Match on stripped raw text (e.g. 'CAR' or 'seg_01')
        raw = text.strip()
        if raw in self._vectors:
            return self._vectors[raw]

        # 3. Match with prefix removed if text was passed with prefix
        for prefix in ("query: ", "passage: "):
            if raw.startswith(prefix):
                unprefixed = raw[len(prefix) :].strip()
                if unprefixed in self._vectors:
                    return self._vectors[unprefixed]

        if self.fallback_to_deterministic:
            return _generate_deterministic_vector(prefixed_text, self._dimension)

        raise KeyError(
            f"No predefined vector registered for text '{text}' "
            f"(checked '{prefixed_text}' and '{raw}')"
        )

    def embed_query(self, query: str) -> List[float]:
        """Embed a single query using predefined mapping or deterministic fallback."""
        if not isinstance(query, str):
            raise TypeError(
                f"embed_query() expects a str, got {type(query).__name__}"
            )
        if not query or not query.strip():
            raise ValueError("Query text cannot be empty or whitespace-only")

        prefixed = format_query_prefix(query)
        vec = self._lookup_vector(query, prefixed)
        validate_vector(vec, expected_dim=self._dimension, check_normalized=True)
        return vec

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of passages using predefined mapping or deterministic fallback."""
        if not texts:
            return []

        for i, text in enumerate(texts):
            if not isinstance(text, str):
                raise TypeError(
                    f"embed_texts() expects all items to be str, got {type(text).__name__} at index {i}"
                )

        vecs: List[List[float]] = []
        for text in texts:
            prefixed = format_passage_prefix(text)
            vecs.append(self._lookup_vector(text, prefixed))

        validate_vectors(vecs, expected_dim=self._dimension, check_normalized=True)
        return vecs


class MockEmbedder(ControllableMockEmbedder):
    """General mock embedder combining deterministic synthesis and controllable overrides.

    Default usage generates deterministic unit vectors:
        embedder = MockEmbedder()
        q_vec = embedder.embed_query("capital requirements")

    Test usage can inject exact vectors:
        embedder = MockEmbedder(predefined_vectors={"seg_a": [1.0, 0.0, ...]})
    """
