"""Production dense embedder using Sentence Transformers for NiyamSetu RAG.

Standardizes on 'intfloat/multilingual-e5-small' (384 dimensions).
Applies asymmetric task-specific prefixes:
- 'query: ' for user search queries
- 'passage: ' for indexed document text segments

Includes lazy loading, dependency validation, batching, and L2 normalization.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence

from rag_module.core.interfaces import BaseEmbedder
from rag_module.embeddings.validation import (
    format_passage_prefix,
    format_query_prefix,
    validate_vector,
    validate_vectors,
)

logger = logging.getLogger(__name__)

DEFAULT_E5_MODEL = "intfloat/multilingual-e5-small"
DEFAULT_E5_DIMENSION = 384


class SentenceTransformerEmbedder(BaseEmbedder):
    """Dense vector embedder wrapping Sentence Transformers.

    Features:
    1. Lazy loading: sentence_transformers is imported and weights are loaded
       only on the first call to embed_* or explicit load(). Module import is
       instantaneous and zero-dependency.
    2. Asymmetric prefixing: transparently prepends 'query: ' or 'passage: '
       without duplicate prefixing.
    3. Unit L2 normalization: produces unit vectors for exact cosine similarity
       via standard dot product.
    4. Safe validation: validates dimension consistency and rejects NaNs/Infs.
    5. Clean error handling: raises clear, actionable ImportError or RuntimeError
       if dependencies or model weights are unavailable; never returns silent fake data.
    """

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_E5_MODEL,
        device: str = "cpu",
        batch_size: int = 32,
        dimension: int = DEFAULT_E5_DIMENSION,
        normalize_embeddings: bool = True,
        model: Optional[Any] = None,
    ) -> None:
        """Initialize SentenceTransformerEmbedder.

        Args:
            model_name_or_path: HuggingFace model ID or local path.
            device: Computation device ('cpu', 'cuda', etc.).
            batch_size: Batch size for passage encoding.
            dimension: Expected embedding dimensionality (384 for E5-small).
            normalize_embeddings: Whether to L2-normalize vectors to unit length.
            model: Optional pre-loaded SentenceTransformer instance (for testing).
        """
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.batch_size = max(1, batch_size)
        self._dimension = dimension
        self.normalize_embeddings = normalize_embeddings
        self._model: Optional[Any] = model

    @property
    def dimension(self) -> int:
        """Dimensionality of the embedding vectors."""
        return self._dimension

    @property
    def is_loaded(self) -> bool:
        """Return True if the underlying model has been loaded into memory."""
        return self._model is not None

    def load(self) -> None:
        """Load the SentenceTransformer model if not already loaded.

        Raises:
            ImportError: If sentence-transformers or torch is not installed.
            RuntimeError: If model weights cannot be loaded.
        """
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as err:
            raise ImportError(
                "SentenceTransformerEmbedder requires 'sentence-transformers' and 'torch'. "
                "Install them in your project virtual environment via: "
                "pip install sentence-transformers torch\n"
                f"Underlying error: {err}"
            ) from err

        try:
            logger.info(
                "Loading SentenceTransformer '%s' on %s...",
                self.model_name_or_path,
                self.device,
            )
            model = SentenceTransformer(self.model_name_or_path, device=self.device)
        except Exception as err:
            raise RuntimeError(
                f"Failed to load embedding model '{self.model_name_or_path}' "
                f"on device '{self.device}': {err}"
            ) from err

        actual_dim = getattr(model, "get_sentence_embedding_dimension", lambda: None)()
        if actual_dim is not None and actual_dim != self._dimension:
            logger.warning(
                "Loaded model '%s' reports dimension %d, differing from expected %d. Updating.",
                self.model_name_or_path,
                actual_dim,
                self._dimension,
            )
            self._dimension = actual_dim

        self._model = model

    @classmethod
    def from_settings(cls, settings: Optional[Any] = None) -> "SentenceTransformerEmbedder":
        """Construct a SentenceTransformerEmbedder from a Settings instance.

        Args:
            settings: A Settings instance. If None, calls get_settings() to retrieve
                the global singleton.

        Returns:
            SentenceTransformerEmbedder configured from settings attributes:
            - settings.embedding_model_name
            - settings.embedding_device
            - settings.embedding_batch_size
        """
        if settings is None:
            from rag_module.config.settings import get_settings
            settings = get_settings()
        return cls(
            model_name_or_path=settings.embedding_model_name,
            device=settings.embedding_device,
            batch_size=settings.embedding_batch_size,
        )

    def embed_query(self, query: str) -> List[float]:
        """Generate a dense embedding vector for a single user query.

        Applies 'query: ' prefix automatically.

        Args:
            query: Query string.

        Returns:
            List of floats representing the L2-normalized query vector.

        Raises:
            TypeError: If query is not a string.
            ValueError: If query is empty or whitespace-only.
            ImportError: If dependencies are not installed.
            RuntimeError: If model encoding fails.
        """
        if not isinstance(query, str):
            raise TypeError(
                f"embed_query() expects a str, got {type(query).__name__}"
            )
        if not query or not query.strip():
            raise ValueError("Query text cannot be empty or whitespace-only")

        self.load()
        assert self._model is not None  # for type checker

        prefixed_query = format_query_prefix(query)

        try:
            raw_vec = self._model.encode(
                prefixed_query,
                normalize_embeddings=self.normalize_embeddings,
                show_progress_bar=False,
            )
        except Exception as err:
            raise RuntimeError(f"Error encoding query: {err}") from err

        vec: List[float]
        if hasattr(raw_vec, "tolist"):
            vec = [float(x) for x in raw_vec.tolist()]
        else:
            vec = [float(x) for x in raw_vec]

        validate_vector(
            vec,
            expected_dim=self._dimension,
            check_normalized=self.normalize_embeddings,
        )
        return vec

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate dense embeddings for a batch of text passages.

        Applies 'passage: ' prefix automatically to each passage.

        Args:
            texts: List of passage strings to encode.

        Returns:
            List of L2-normalized float vectors. Returns [] if texts is empty.

        Raises:
            TypeError: If any element of texts is not a string.
            ImportError: If dependencies are not installed.
            RuntimeError: If model encoding fails.
            ValueError: If output vectors fail dimension or finite-value checks.
        """
        if not texts:
            return []

        for i, t in enumerate(texts):
            if not isinstance(t, str):
                raise TypeError(
                    f"embed_texts() expects all items to be str, "
                    f"got {type(t).__name__} at index {i}"
                )

        self.load()
        assert self._model is not None  # for type checker

        prefixed_texts = [format_passage_prefix(t) for t in texts]

        try:
            raw_vecs = self._model.encode(
                prefixed_texts,
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize_embeddings,
                show_progress_bar=False,
            )
        except Exception as err:
            raise RuntimeError(f"Error encoding passages: {err}") from err

        vecs: List[List[float]] = []
        if hasattr(raw_vecs, "tolist"):
            vecs = [[float(x) for x in v] for v in raw_vecs.tolist()]
        else:
            vecs = [[float(x) for x in v] for v in raw_vecs]

        validate_vectors(
            vecs,
            expected_dim=self._dimension,
            check_normalized=self.normalize_embeddings,
        )
        return vecs
