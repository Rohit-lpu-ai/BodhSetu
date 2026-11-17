"""Unit tests for dense embedding layer (Milestone 2 Stage 1).

Validates:
1. Prefix handling:
   - 'query: ' and 'passage: ' asymmetric prefix injection
   - Duplicate prefix prevention (idempotence)
   - Whitespace stripping before prefixing

2. Vector validation utilities:
   - Vector norm and L2 normalization
   - Finite numeric checks (rejects NaN, Inf)
   - Dimension mismatch detection
   - Zero-norm vector error handling
   - Empty batch handling

3. DeterministicMockEmbedder:
   - Interface compliance with BaseEmbedder
   - Dimension consistency
   - Deterministic vector generation across calls
   - L2 unit normalization
   - Batch vs single embedding consistency
   - Empty input and invalid query handling

4. ControllableMockEmbedder / MockEmbedder:
   - Predefined vector injection
   - Exact mathematical similarity ranking verification
   - Fallback to deterministic generation
   - KeyError on missing key when fallback is disabled
   - Dynamic vector registration via set_vector

5. SentenceTransformerEmbedder:
   - Lazy loading (does not import sentence-transformers at initialization)
   - Missing dependency raises clear, actionable ImportError
   - Dependency injection of mock model for offline execution
   - Model encoding failure raises RuntimeError (no silent fake output)
   - Batch encoding and batch_size passthrough
   - Unit L2 normalization verification
"""

from __future__ import annotations

import math
import sys
import unittest
from typing import Any, List, Optional, Sequence
from unittest.mock import patch

from rag_module.core.interfaces import BaseEmbedder
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


class FakeSentenceTransformerModel:
    """Mock SentenceTransformer model for offline dependency-injected unit tests."""

    def __init__(self, dimension: int = DEFAULT_E5_DIMENSION) -> None:
        self._dim = dimension
        self.recorded_calls: List[dict[str, Any]] = []

    def get_sentence_embedding_dimension(self) -> int:
        return self._dim

    def encode(
        self,
        sentences: Any,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ) -> Any:
        self.recorded_calls.append(
            {
                "sentences": sentences,
                "batch_size": batch_size,
                "normalize_embeddings": normalize_embeddings,
            }
        )
        if isinstance(sentences, str):
            # Return single 1D unit vector
            vec = [1.0] + [0.0] * (self._dim - 1)
            return vec

        # Return 2D list of unit vectors
        return [[1.0] + [0.0] * (self._dim - 1) for _ in sentences]


# ---------------------------------------------------------------------------
# 1. Prefix Handling Tests
# ---------------------------------------------------------------------------


class TestPrefixHandling(unittest.TestCase):
    """Test E5 asymmetric query and passage prefix formatting."""

    def test_query_prefix_added(self) -> None:
        self.assertEqual(
            format_query_prefix("capital adequacy ratio"),
            "query: capital adequacy ratio",
        )

    def test_query_prefix_idempotent(self) -> None:
        self.assertEqual(
            format_query_prefix("query: capital adequacy ratio"),
            "query: capital adequacy ratio",
        )

    def test_query_prefix_strips_whitespace(self) -> None:
        self.assertEqual(
            format_query_prefix("   capital adequacy   "),
            "query: capital adequacy",
        )

    def test_passage_prefix_added(self) -> None:
        self.assertEqual(
            format_passage_prefix("clause 3(1) capital buffer"),
            "passage: clause 3(1) capital buffer",
        )

    def test_passage_prefix_idempotent(self) -> None:
        self.assertEqual(
            format_passage_prefix("passage: clause 3(1) capital buffer"),
            "passage: clause 3(1) capital buffer",
        )

    def test_passage_prefix_strips_whitespace(self) -> None:
        self.assertEqual(
            format_passage_prefix("   clause 3(1)   "),
            "passage: clause 3(1)",
        )

    def test_query_prefix_case_insensitive(self) -> None:
        """Query prefix should be detected regardless of case and normalized to lowercase."""
        self.assertEqual(
            format_query_prefix("QUERY: capital adequacy ratio"),
            "query: capital adequacy ratio",
        )
        self.assertEqual(
            format_query_prefix("Query: capital adequacy ratio"),
            "query: capital adequacy ratio",
        )

    def test_passage_prefix_case_insensitive(self) -> None:
        """Passage prefix should be detected regardless of case and normalized to lowercase."""
        self.assertEqual(
            format_passage_prefix("PASSAGE: clause 3(1) buffer"),
            "passage: clause 3(1) buffer",
        )
        self.assertEqual(
            format_passage_prefix("Passage: clause 3(1) buffer"),
            "passage: clause 3(1) buffer",
        )

    def test_duplicate_prefix_prevention(self) -> None:
        """Applying prefix function multiple times must not accumulate duplicate prefixes."""
        once = format_query_prefix("capital requirements")
        twice = format_query_prefix(once)
        self.assertEqual(twice, "query: capital requirements")

        p_once = format_passage_prefix("reserve requirements")
        p_twice = format_passage_prefix(p_once)
        self.assertEqual(p_twice, "passage: reserve requirements")

    def test_query_prefix_non_string_raises_type_error(self) -> None:
        """Non-string inputs to format_query_prefix must raise TypeError."""
        with self.assertRaises(TypeError):
            format_query_prefix(None)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            format_query_prefix(123)  # type: ignore[arg-type]

    def test_passage_prefix_non_string_raises_type_error(self) -> None:
        """Non-string inputs to format_passage_prefix must raise TypeError."""
        with self.assertRaises(TypeError):
            format_passage_prefix(None)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            format_passage_prefix(123)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 2. Vector Validation Utilities Tests
# ---------------------------------------------------------------------------


class TestValidationUtilities(unittest.TestCase):
    """Test vector validation, norm calculation, and normalization."""

    def test_is_finite_number(self) -> None:
        self.assertTrue(is_finite_number(0.0))
        self.assertTrue(is_finite_number(42))
        self.assertTrue(is_finite_number(-3.14159))
        self.assertFalse(is_finite_number(float("nan")))
        self.assertFalse(is_finite_number(float("inf")))
        self.assertFalse(is_finite_number(float("-inf")))
        self.assertFalse(is_finite_number("string"))  # type: ignore

    def test_vector_norm_calculation(self) -> None:
        self.assertAlmostEqual(vector_norm([3.0, 4.0]), 5.0, places=6)
        self.assertAlmostEqual(vector_norm([1.0, 0.0, 0.0]), 1.0, places=6)

    def test_normalize_l2_unit_length(self) -> None:
        vec = [3.0, 4.0]
        normed = normalize_l2(vec)
        self.assertAlmostEqual(normed[0], 0.6, places=6)
        self.assertAlmostEqual(normed[1], 0.8, places=6)
        self.assertAlmostEqual(vector_norm(normed), 1.0, places=6)

    def test_normalize_l2_zero_norm_raises_value_error(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            normalize_l2([0.0, 0.0, 0.0])
        self.assertIn("Cannot normalize zero-norm vector", str(ctx.exception))

    def test_validate_vector_valid(self) -> None:
        validate_vector([1.0, 0.0, 0.0], expected_dim=3, check_normalized=True)

    def test_validate_vector_type_error(self) -> None:
        with self.assertRaises(TypeError):
            validate_vector("not a vector")  # type: ignore

    def test_validate_vector_dimension_mismatch(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            validate_vector([1.0, 2.0], expected_dim=384)
        self.assertIn("dimension mismatch", str(ctx.exception))

    def test_validate_vector_with_nan_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            validate_vector([1.0, float("nan"), 0.0])
        self.assertIn("Non-finite value", str(ctx.exception))

    def test_validate_vector_with_inf_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            validate_vector([1.0, float("inf"), 0.0])
        self.assertIn("Non-finite value", str(ctx.exception))

    def test_validate_vector_unnormalized_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            validate_vector([2.0, 0.0], check_normalized=True)
        self.assertIn("not L2-normalized", str(ctx.exception))

    def test_validate_vectors_empty_batch(self) -> None:
        # Empty batch is valid no-op
        validate_vectors([])

    def test_validate_vectors_inconsistent_dimensions(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            validate_vectors([[1.0, 2.0], [1.0, 2.0, 3.0]])
        self.assertIn("dimension mismatch", str(ctx.exception))


# ---------------------------------------------------------------------------
# 3. Deterministic Mock Embedder Tests
# ---------------------------------------------------------------------------


class TestDeterministicMockEmbedder(unittest.TestCase):
    """Test DeterministicMockEmbedder generation, determinism, and normalization."""

    def setUp(self) -> None:
        self.embedder = DeterministicMockEmbedder(dimension=384)

    def test_implements_base_embedder(self) -> None:
        self.assertIsInstance(self.embedder, BaseEmbedder)

    def test_dimension_property(self) -> None:
        self.assertEqual(self.embedder.dimension, 384)
        custom = DeterministicMockEmbedder(dimension=64)
        self.assertEqual(custom.dimension, 64)

    def test_invalid_dimension_raises(self) -> None:
        with self.assertRaises(ValueError):
            DeterministicMockEmbedder(dimension=0)

    def test_query_embedding_dimension_and_normalization(self) -> None:
        vec = self.embedder.embed_query("Capital Adequacy Ratio")
        self.assertEqual(len(vec), 384)
        self.assertAlmostEqual(vector_norm(vec), 1.0, places=5)

    def test_passage_embedding_dimension_and_normalization(self) -> None:
        vecs = self.embedder.embed_texts(["Section 45-IA of RBI Act."])
        self.assertEqual(len(vecs), 1)
        self.assertEqual(len(vecs[0]), 384)
        self.assertAlmostEqual(vector_norm(vecs[0]), 1.0, places=5)

    def test_determinism_across_calls(self) -> None:
        """Embedding the exact same text produces bit-identical vectors across calls."""
        q1 = self.embedder.embed_query("test query string")
        q2 = self.embedder.embed_query("test query string")
        self.assertEqual(q1, q2)

        p1 = self.embedder.embed_texts(["paragraph one", "paragraph two"])
        p2 = self.embedder.embed_texts(["paragraph one", "paragraph two"])
        self.assertEqual(p1, p2)

    def test_different_texts_produce_different_vectors(self) -> None:
        v1 = self.embedder.embed_query("Topic A")
        v2 = self.embedder.embed_query("Topic B")
        self.assertNotEqual(v1, v2)

    def test_batch_matches_single_embedding(self) -> None:
        batch_vecs = self.embedder.embed_texts(["text one", "text two"])
        single_1 = self.embedder.embed_texts(["text one"])[0]
        single_2 = self.embedder.embed_texts(["text two"])[0]
        self.assertEqual(batch_vecs[0], single_1)
        self.assertEqual(batch_vecs[1], single_2)

    def test_empty_batch_returns_empty_list(self) -> None:
        self.assertEqual(self.embedder.embed_texts([]), [])

    def test_empty_query_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            self.embedder.embed_query("")
        with self.assertRaises(ValueError):
            self.embedder.embed_query("   ")

    def test_mock_embed_query_non_string_raises_type_error(self) -> None:
        """DeterministicMockEmbedder.embed_query must raise TypeError on non-string."""
        with self.assertRaises(TypeError):
            self.embedder.embed_query(None)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            self.embedder.embed_query(123)  # type: ignore[arg-type]

    def test_mock_embed_texts_non_string_raises_type_error(self) -> None:
        """DeterministicMockEmbedder.embed_texts must raise TypeError on non-string item."""
        with self.assertRaises(TypeError):
            self.embedder.embed_texts(["valid text", None])  # type: ignore[list-item]
        with self.assertRaises(TypeError):
            self.embedder.embed_texts(["valid text", 42])  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# 4. Controllable Mock Embedder Tests
# ---------------------------------------------------------------------------


class TestControllableMockEmbedder(unittest.TestCase):
    """Test ControllableMockEmbedder and predefined vector mathematical ranking."""

    def test_predefined_vector_retrieval(self) -> None:
        v_known = [1.0] + [0.0] * 383
        embedder = ControllableMockEmbedder(
            dimension=384,
            predefined_vectors={"known text": v_known},
            fallback_to_deterministic=False,
        )
        # Check that query and passage retrieve the exact known vector
        self.assertEqual(embedder.embed_query("known text"), v_known)
        self.assertEqual(embedder.embed_texts(["known text"])[0], v_known)

    def test_predefined_vector_similarity_ranking_math(self) -> None:
        """Verify that predefined vectors allow exact mathematical cosine ranking tests."""
        dim = 4
        # Query aligned with axis 0
        q_vec = [1.0, 0.0, 0.0, 0.0]
        # Passage A: dot product with q is 0.90
        p_a = [0.9, math.sqrt(1 - 0.9**2), 0.0, 0.0]
        # Passage B: dot product with q is 0.20
        p_b = [0.2, math.sqrt(1 - 0.2**2), 0.0, 0.0]

        embedder = ControllableMockEmbedder(
            dimension=dim,
            predefined_vectors={
                "target query": q_vec,
                "relevant passage": p_a,
                "irrelevant passage": p_b,
            },
        )

        q = embedder.embed_query("target query")
        passages = embedder.embed_texts(["relevant passage", "irrelevant passage"])

        # Dot product with normalized vectors equals cosine similarity
        score_a = sum(x * y for x, y in zip(q, passages[0]))
        score_b = sum(x * y for x, y in zip(q, passages[1]))

        self.assertAlmostEqual(score_a, 0.90, places=4)
        self.assertAlmostEqual(score_b, 0.20, places=4)
        self.assertGreater(score_a, score_b)

    def test_fallback_to_deterministic_when_enabled(self) -> None:
        embedder = ControllableMockEmbedder(
            dimension=384,
            predefined_vectors={"fixed": [1.0] + [0.0] * 383},
            fallback_to_deterministic=True,
        )
        # "other" is not in predefined_vectors, should fall back deterministically
        vec = embedder.embed_query("other")
        self.assertEqual(len(vec), 384)
        self.assertAlmostEqual(vector_norm(vec), 1.0, places=5)

    def test_missing_key_raises_when_fallback_disabled(self) -> None:
        embedder = ControllableMockEmbedder(
            dimension=384,
            predefined_vectors={"fixed": [1.0] + [0.0] * 383},
            fallback_to_deterministic=False,
        )
        with self.assertRaises(KeyError):
            embedder.embed_query("unregistered text")

    def test_set_vector_dynamically(self) -> None:
        embedder = ControllableMockEmbedder(dimension=3)
        embedder.set_vector("dynamic", [0.0, 1.0, 0.0])
        self.assertEqual(embedder.embed_query("dynamic"), [0.0, 1.0, 0.0])

    def test_mock_embedder_alias_supports_both_modes(self) -> None:
        """MockEmbedder provides both deterministic and controllable modes."""
        embedder = MockEmbedder(dimension=384)
        v_det = embedder.embed_query("some query")
        self.assertEqual(len(v_det), 384)

        embedder.set_vector("overridden", [1.0] + [0.0] * 383)
        self.assertEqual(embedder.embed_query("overridden"), [1.0] + [0.0] * 383)

    def test_controllable_embed_query_non_string_raises_type_error(self) -> None:
        """ControllableMockEmbedder.embed_query must raise TypeError on non-string."""
        embedder = ControllableMockEmbedder(dimension=384)
        with self.assertRaises(TypeError):
            embedder.embed_query(None)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            embedder.embed_query(123)  # type: ignore[arg-type]

    def test_controllable_embed_texts_non_string_raises_type_error(self) -> None:
        """ControllableMockEmbedder.embed_texts must raise TypeError on non-string item."""
        embedder = ControllableMockEmbedder(dimension=384)
        with self.assertRaises(TypeError):
            embedder.embed_texts(["valid text", None])  # type: ignore[list-item]
        with self.assertRaises(TypeError):
            embedder.embed_texts(["valid text", 42])  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# 5. SentenceTransformerEmbedder Tests (Offline via Mock Injection)
# ---------------------------------------------------------------------------


class TestSentenceTransformerEmbedder(unittest.TestCase):
    """Test SentenceTransformerEmbedder lazy loading, prefixing, and error handling."""

    def test_implements_base_embedder(self) -> None:
        embedder = SentenceTransformerEmbedder()
        self.assertIsInstance(embedder, BaseEmbedder)

    def test_lazy_loading_not_loaded_on_init(self) -> None:
        """Instantiating SentenceTransformerEmbedder must NOT import sentence_transformers."""
        embedder = SentenceTransformerEmbedder()
        self.assertFalse(embedder.is_loaded)
        self.assertEqual(embedder.dimension, DEFAULT_E5_DIMENSION)
        self.assertEqual(embedder.model_name_or_path, DEFAULT_E5_MODEL)

    def test_missing_dependency_raises_informative_import_error(self) -> None:
        """Calling load() when sentence_transformers is absent raises actionable ImportError."""
        embedder = SentenceTransformerEmbedder()

        # Simulate missing sentence_transformers module
        with patch.dict(sys.modules, {"sentence_transformers": None}):
            with self.assertRaises(ImportError) as ctx:
                embedder.load()

            error_msg = str(ctx.exception)
            self.assertIn("sentence-transformers", error_msg)
            self.assertIn("torch", error_msg)
            self.assertIn("pip install", error_msg)

    def test_model_load_failure_raises_runtime_error_no_silent_fake(self) -> None:
        """If model loading fails, raise RuntimeError rather than returning silent fake vectors."""
        embedder = SentenceTransformerEmbedder(model_name_or_path="nonexistent-model")

        fake_st_module = unittest.mock.MagicMock()
        fake_st_module.SentenceTransformer.side_effect = Exception("Network connection error")

        with patch.dict(sys.modules, {"sentence_transformers": fake_st_module}):
            with self.assertRaises(RuntimeError) as ctx:
                embedder.load()

            self.assertIn("Failed to load embedding model", str(ctx.exception))
            self.assertFalse(embedder.is_loaded)

    def test_dependency_injection_embed_query_applies_query_prefix(self) -> None:
        """embed_query prepends 'query: ' and calls model.encode()."""
        fake_model = FakeSentenceTransformerModel(dimension=384)
        embedder = SentenceTransformerEmbedder(model=fake_model)

        self.assertTrue(embedder.is_loaded)
        vec = embedder.embed_query("commercial bank CAR")

        self.assertEqual(len(vec), 384)
        self.assertAlmostEqual(vector_norm(vec), 1.0, places=5)

        # Inspect recorded call on fake model
        self.assertEqual(len(fake_model.recorded_calls), 1)
        call = fake_model.recorded_calls[0]
        self.assertEqual(call["sentences"], "query: commercial bank CAR")
        self.assertTrue(call["normalize_embeddings"])

    def test_dependency_injection_embed_texts_applies_passage_prefix(self) -> None:
        """embed_texts prepends 'passage: ' to each passage and respects batching."""
        fake_model = FakeSentenceTransformerModel(dimension=384)
        embedder = SentenceTransformerEmbedder(batch_size=16, model=fake_model)

        vecs = embedder.embed_texts(["Section 45-IA", "D-SIB CET1 buffer"])
        self.assertEqual(len(vecs), 2)
        self.assertEqual(len(vecs[0]), 384)

        self.assertEqual(len(fake_model.recorded_calls), 1)
        call = fake_model.recorded_calls[0]
        self.assertEqual(
            call["sentences"],
            ["passage: Section 45-IA", "passage: D-SIB CET1 buffer"],
        )
        self.assertEqual(call["batch_size"], 16)
        self.assertTrue(call["normalize_embeddings"])

    def test_prefix_idempotence_in_sentence_transformer(self) -> None:
        """Pre-prefixed strings must not receive duplicate prefixes."""
        fake_model = FakeSentenceTransformerModel(dimension=384)
        embedder = SentenceTransformerEmbedder(model=fake_model)

        embedder.embed_query("query: already prefixed query")
        self.assertEqual(
            fake_model.recorded_calls[0]["sentences"],
            "query: already prefixed query",
        )

        embedder.embed_texts(["passage: already prefixed passage"])
        self.assertEqual(
            fake_model.recorded_calls[1]["sentences"],
            ["passage: already prefixed passage"],
        )

    def test_embed_texts_empty_list_returns_empty_without_loading(self) -> None:
        embedder = SentenceTransformerEmbedder()
        self.assertEqual(embedder.embed_texts([]), [])
        # Model should still not be loaded
        self.assertFalse(embedder.is_loaded)

    def test_embed_query_empty_raises_value_error(self) -> None:
        fake_model = FakeSentenceTransformerModel(dimension=384)
        embedder = SentenceTransformerEmbedder(model=fake_model)
        with self.assertRaises(ValueError):
            embedder.embed_query("")
        with self.assertRaises(ValueError):
            embedder.embed_query("   ")

    def test_from_settings_default(self) -> None:
        """from_settings() with no argument uses global get_settings()."""
        from rag_module.config.settings import Settings, get_settings
        import rag_module.config.settings as _settings_mod

        custom = Settings(
            embedding_model_name="intfloat/multilingual-e5-small",
            embedding_device="cpu",
            embedding_batch_size=32,
        )
        original = _settings_mod._DEFAULT_SETTINGS
        _settings_mod._DEFAULT_SETTINGS = custom
        try:
            embedder = SentenceTransformerEmbedder.from_settings()
            self.assertIsInstance(embedder, SentenceTransformerEmbedder)
            self.assertEqual(embedder.model_name_or_path, "intfloat/multilingual-e5-small")
            self.assertEqual(embedder.device, "cpu")
            self.assertEqual(embedder.batch_size, 32)
            self.assertFalse(embedder.is_loaded)
        finally:
            _settings_mod._DEFAULT_SETTINGS = original

    def test_from_settings_custom(self) -> None:
        """from_settings(settings) uses provided Settings object directly."""
        from rag_module.config.settings import Settings

        custom = Settings(
            embedding_model_name="BAAI/bge-m3",
            embedding_device="cuda",
            embedding_batch_size=64,
        )
        embedder = SentenceTransformerEmbedder.from_settings(custom)
        self.assertEqual(embedder.model_name_or_path, "BAAI/bge-m3")
        self.assertEqual(embedder.device, "cuda")
        self.assertEqual(embedder.batch_size, 64)
        self.assertFalse(embedder.is_loaded)

    def test_embed_query_non_string_raises_type_error(self) -> None:
        """embed_query raises TypeError for non-string queries (None, int)."""
        embedder = SentenceTransformerEmbedder()
        with self.assertRaises(TypeError):
            embedder.embed_query(None)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            embedder.embed_query(123)  # type: ignore[arg-type]

    def test_embed_texts_non_string_item_raises_type_error(self) -> None:
        """embed_texts raises TypeError if any item is not a string (None, int)."""
        embedder = SentenceTransformerEmbedder()
        with self.assertRaises(TypeError):
            embedder.embed_texts(["valid", None])  # type: ignore[list-item]
        with self.assertRaises(TypeError):
            embedder.embed_texts(["valid", 42])  # type: ignore[list-item]

    def test_validation_occurring_before_model_loading(self) -> None:
        """Input validation must reject invalid inputs before attempting model loading."""
        embedder = SentenceTransformerEmbedder()
        self.assertFalse(embedder.is_loaded)

        with self.assertRaises(TypeError):
            embedder.embed_query(None)  # type: ignore[arg-type]
        self.assertFalse(embedder.is_loaded)

        with self.assertRaises(TypeError):
            embedder.embed_query(999)  # type: ignore[arg-type]
        self.assertFalse(embedder.is_loaded)

        with self.assertRaises(TypeError):
            embedder.embed_texts(["ok passage", None])  # type: ignore[list-item]
        self.assertFalse(embedder.is_loaded)

        with self.assertRaises(TypeError):
            embedder.embed_texts(["ok passage", 999])  # type: ignore[list-item]
        self.assertFalse(embedder.is_loaded)


if __name__ == "__main__":
    unittest.main()
