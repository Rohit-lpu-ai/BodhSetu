"""Tests for DenseRetriever (dense semantic retrieval).

Coverage:
1. Interface conformance (BaseRetriever subclass, injected embedder, default embedder).
2. Cosine similarity mathematics and score ordering with exact dot-product verification.
3. Composite text indexing (feature parity with BM25) and segment immutability.
4. Deterministic tie-breaking (score DESC, ordinal ASC, clause_id ASC).
5. Filtering by document_id and version_id.
6. Index lifecycle (empty corpus, empty query, top_k <= 0, deduplication, re-index, caching, atomicity).
7. Vector validation (dimension mismatch, NaN, infinity, zero-norm).

IMPORTANT NOTE ON MOCK EMBEDDINGS:
ControllableMockEmbedder and DeterministicMockEmbedder use synthetic vectors.
These tests verify mathematical pipeline mechanics (dot-product ranking, tie-breaking,
filtering, lifecycle atomicity) and NOT linguistic semantic understanding.
Relevance and semantic evaluation is conducted in separate evaluation milestones.
"""

from __future__ import annotations

import math
import unittest
from typing import List, Optional

from rag_module.core.interfaces import BaseRetriever
from rag_module.core.models import ClauseSegment, PageProvenance, RetrievedPassage
from rag_module.embeddings.mock import (
    ControllableMockEmbedder,
    DeterministicMockEmbedder,
)
from rag_module.embeddings.sentence_transformer import SentenceTransformerEmbedder
from rag_module.retrieval.bm25_retriever import compose_searchable_text
from rag_module.retrieval.dense_retriever import DenseRetriever, _dot_product


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_segment(
    clause_id: str = "seg_001",
    text: str = "Capital adequacy requirements for banks.",
    document_id: str = "doc_rbi_001",
    version_id: str = "v1",
    ordinal: int = 1,
    clause_number: Optional[str] = None,
    heading: Optional[str] = None,
    page_start: int = 1,
    page_end: int = 1,
) -> ClauseSegment:
    """Construct a minimal ClauseSegment for testing."""
    return ClauseSegment(
        document_id=document_id,
        version_id=version_id,
        clause_id=clause_id,
        fingerprint=f"fp_{clause_id}",
        ordinal=ordinal,
        segment_type="clause",
        text=text,
        page_start=page_start,
        page_end=page_end,
        pages=[
            PageProvenance(
                page_number=page_start,
                char_start=0,
                char_end=len(text),
            )
        ],
        clause_number=clause_number,
        heading=heading,
    )


def _unit_vec(components: List[float]) -> List[float]:
    """Return the unit L2-normalized version of a vector."""
    norm = math.sqrt(sum(x * x for x in components))
    return [x / norm for x in components]


# ---------------------------------------------------------------------------
# 1. Interface Conformance Tests
# ---------------------------------------------------------------------------


class TestDenseRetrieverInterface(unittest.TestCase):
    """DenseRetriever must satisfy the BaseRetriever contract."""

    def test_implements_base_retriever(self) -> None:
        retriever = DenseRetriever(embedder=DeterministicMockEmbedder(dimension=16))
        self.assertIsInstance(retriever, BaseRetriever)

    def test_default_embedder_created_from_settings(self) -> None:
        """DenseRetriever() without embedder kwarg uses SentenceTransformerEmbedder (lazy)."""
        retriever = DenseRetriever()
        self.assertIsInstance(retriever.embedder, SentenceTransformerEmbedder)
        # Model must NOT be loaded on initialization
        self.assertFalse(retriever.embedder.is_loaded)

    def test_injected_embedder_used(self) -> None:
        mock = DeterministicMockEmbedder(dimension=32)
        retriever = DenseRetriever(embedder=mock)
        self.assertIs(retriever.embedder, mock)
        self.assertEqual(retriever.segment_count, 0)

    def test_invalid_embedder_type_raises(self) -> None:
        with self.assertRaises(TypeError):
            DenseRetriever(embedder="not_an_embedder")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 2. Cosine Similarity Mathematics and Score Ordering
# ---------------------------------------------------------------------------


class TestDenseRetrieverCosineMathAndRanking(unittest.TestCase):
    """Verify exact cosine similarity values, ranking, and negative-score retention.

    Predefined vectors verify mathematical ranking only, NOT semantic understanding.
    """

    def _build_embedder_and_segments(self):
        dim = 4

        seg_a = _make_segment(
            clause_id="seg_A", text="High relevant text", ordinal=1
        )
        seg_b = _make_segment(
            clause_id="seg_B", text="Moderate relevant text", ordinal=2
        )
        seg_c = _make_segment(
            clause_id="seg_C", text="Slightly irrelevant text", ordinal=3
        )
        seg_d = _make_segment(
            clause_id="seg_D", text="Very irrelevant text", ordinal=4
        )

        text_a = compose_searchable_text(seg_a)
        text_b = compose_searchable_text(seg_b)
        text_c = compose_searchable_text(seg_c)
        text_d = compose_searchable_text(seg_d)

        def vec_with_score(s: float) -> List[float]:
            other = math.sqrt(max(0.0, 1.0 - s * s))
            return [s, other, 0.0, 0.0] if other > 0 else [s, 0.0, 0.0, 0.0]

        embedder = ControllableMockEmbedder(
            dimension=dim,
            fallback_to_deterministic=False,
        )
        # Unit query vector along axis 0
        embedder.set_vector("query: search capital", [1.0, 0.0, 0.0, 0.0])
        embedder.set_vector(f"passage: {text_a}", vec_with_score(0.90))
        embedder.set_vector(f"passage: {text_b}", vec_with_score(0.40))
        embedder.set_vector(f"passage: {text_c}", vec_with_score(-0.30))
        embedder.set_vector(f"passage: {text_d}", vec_with_score(-0.85))

        return embedder, [seg_a, seg_b, seg_c, seg_d]

    def test_exact_cosine_scores_and_rank_order(self) -> None:
        """Scores equal exact dot products; ranking is score DESC."""
        embedder, segments = self._build_embedder_and_segments()
        retriever = DenseRetriever(embedder=embedder)
        retriever.index(segments)

        results = retriever.retrieve("search capital", top_k=4)

        self.assertEqual(len(results), 4)
        self.assertEqual(results[0].clause_id, "seg_A")
        self.assertEqual(results[1].clause_id, "seg_B")
        self.assertEqual(results[2].clause_id, "seg_C")
        self.assertEqual(results[3].clause_id, "seg_D")

        self.assertAlmostEqual(results[0].score, 0.90, places=5)
        self.assertAlmostEqual(results[1].score, 0.40, places=5)
        self.assertAlmostEqual(results[2].score, -0.30, places=5)
        self.assertAlmostEqual(results[3].score, -0.85, places=5)

        for i, r in enumerate(results, start=1):
            self.assertEqual(r.rank, i)
            self.assertEqual(r.retrieval_method, "dense")

    def test_negative_scores_not_filtered(self) -> None:
        """Negative cosine scores must be retained and ranked, not silently dropped."""
        embedder, segments = self._build_embedder_and_segments()
        retriever = DenseRetriever(embedder=embedder)
        retriever.index(segments)

        results = retriever.retrieve("search capital", top_k=10)
        self.assertEqual(len(results), 4)
        self.assertTrue(any(r.score < 0 for r in results))

    def test_top_k_limits_results(self) -> None:
        """top_k caps results to at most top_k items."""
        embedder, segments = self._build_embedder_and_segments()
        retriever = DenseRetriever(embedder=embedder)
        retriever.index(segments)

        results = retriever.retrieve("search capital", top_k=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].clause_id, "seg_A")
        self.assertEqual(results[1].clause_id, "seg_B")

    def test_top_k_zero_or_negative_returns_empty(self) -> None:
        """top_k <= 0 returns an empty list immediately."""
        embedder, segments = self._build_embedder_and_segments()
        retriever = DenseRetriever(embedder=embedder)
        retriever.index(segments)

        self.assertEqual(retriever.retrieve("search capital", top_k=0), [])
        self.assertEqual(retriever.retrieve("search capital", top_k=-5), [])


# ---------------------------------------------------------------------------
# 3. Composite Text Indexing and Segment Immutability
# ---------------------------------------------------------------------------


class TestDenseRetrieverCompositeTextAndImmutability(unittest.TestCase):
    """DenseRetriever must use compose_searchable_text and preserve segment objects."""

    def test_composite_text_matches_bm25_format(self) -> None:
        dim = 4
        seg = _make_segment(
            clause_id="cl_3_2",
            clause_number="3(2)",
            heading="D-SIB Surcharge",
            text="Buffer shall be 0.80%.",
            ordinal=1,
        )
        expected_composite = compose_searchable_text(seg)

        embedder = ControllableMockEmbedder(
            dimension=dim, fallback_to_deterministic=False
        )
        embedder.set_vector("query: d-sib surcharge buffer", [1.0, 0.0, 0.0, 0.0])
        embedder.set_vector(f"passage: {expected_composite}", [1.0, 0.0, 0.0, 0.0])

        retriever = DenseRetriever(embedder=embedder)
        retriever.index([seg])

        self.assertEqual(retriever.segment_count, 1)
        results = retriever.retrieve("d-sib surcharge buffer", top_k=1)
        self.assertEqual(len(results), 1)
        self.assertAlmostEqual(results[0].score, 1.0, places=5)

    def test_segment_object_identity_preserved(self) -> None:
        """Retrieved passage.segment must be the exact same object as was indexed."""
        dim = 8
        seg = _make_segment(clause_id="seg_identity", text="Provenance test.")
        embedder = DeterministicMockEmbedder(dimension=dim)
        retriever = DenseRetriever(embedder=embedder)
        retriever.index([seg])
        results = retriever.retrieve("provenance test", top_k=1)
        self.assertEqual(len(results), 1)
        self.assertIs(results[0].segment, seg)

    def test_multi_page_segment_char_start_raises(self) -> None:
        """Multi-page segment char_start raises ValueError (model contract preserved)."""
        seg = ClauseSegment(
            document_id="doc1",
            version_id="v1",
            clause_id="multi_page",
            fingerprint="fp_mp",
            ordinal=1,
            segment_type="clause",
            text="Spans two pages of regulatory content.",
            page_start=3,
            page_end=4,
            pages=[
                PageProvenance(page_number=3, char_start=100, char_end=500),
                PageProvenance(page_number=4, char_start=0, char_end=200),
            ],
        )
        embedder = DeterministicMockEmbedder(dimension=16)
        retriever = DenseRetriever(embedder=embedder)
        retriever.index([seg])
        results = retriever.retrieve("regulatory content", top_k=1)
        self.assertEqual(len(results), 1)
        with self.assertRaises(ValueError):
            _ = results[0].segment.char_start


# ---------------------------------------------------------------------------
# 4. Tie-Breaking
# ---------------------------------------------------------------------------


class TestDenseRetrieverTieBreaking(unittest.TestCase):
    """When scores are equal, sort by ordinal ASC then clause_id ASC."""

    def test_tie_broken_by_ordinal_then_clause_id(self) -> None:
        dim = 4
        seg_z = _make_segment(clause_id="seg_z", text="Same score z.", ordinal=3)
        seg_a = _make_segment(clause_id="seg_a", text="Same score a.", ordinal=1)
        seg_m = _make_segment(clause_id="seg_m", text="Same score m.", ordinal=2)

        identical_vec = _unit_vec([0.5, 0.5, 0.5, 0.5])
        query_vec = _unit_vec([0.5, 0.5, 0.5, 0.5])

        embedder = ControllableMockEmbedder(
            dimension=dim, fallback_to_deterministic=False
        )
        embedder.set_vector("query: tie query", query_vec)
        for seg in [seg_z, seg_a, seg_m]:
            ct = compose_searchable_text(seg)
            embedder.set_vector(f"passage: {ct}", identical_vec)

        retriever = DenseRetriever(embedder=embedder)
        retriever.index([seg_z, seg_a, seg_m])
        results = retriever.retrieve("tie query", top_k=3)

        self.assertEqual(len(results), 3)
        # Expected order: ordinal 1 (seg_a), ordinal 2 (seg_m), ordinal 3 (seg_z)
        self.assertEqual(results[0].clause_id, "seg_a")
        self.assertEqual(results[1].clause_id, "seg_m")
        self.assertEqual(results[2].clause_id, "seg_z")


# ---------------------------------------------------------------------------
# 5. Filtering
# ---------------------------------------------------------------------------


class TestDenseRetrieverFiltering(unittest.TestCase):
    """document_id and version_id filters are applied during retrieve()."""

    def _build_multi_doc_retriever(self):
        dim = 16
        segs = [
            _make_segment(
                "s1",
                "Capital adequacy.",
                document_id="doc_A",
                version_id="v1",
                ordinal=1,
            ),
            _make_segment(
                "s2",
                "Liquidity norms.",
                document_id="doc_A",
                version_id="v2",
                ordinal=2,
            ),
            _make_segment(
                "s3",
                "Reserve requirements.",
                document_id="doc_B",
                version_id="v1",
                ordinal=3,
            ),
            _make_segment(
                "s4",
                "Exposure limits.",
                document_id="doc_B",
                version_id="v2",
                ordinal=4,
            ),
        ]
        embedder = DeterministicMockEmbedder(dimension=dim)
        retriever = DenseRetriever(embedder=embedder)
        retriever.index(segs)
        return retriever

    def test_filter_by_document_id(self) -> None:
        retriever = self._build_multi_doc_retriever()
        results = retriever.retrieve("capital", top_k=10, document_id="doc_A")
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r.document_id == "doc_A" for r in results))

    def test_filter_by_version_id(self) -> None:
        retriever = self._build_multi_doc_retriever()
        results = retriever.retrieve("capital", top_k=10, version_id="v1")
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r.version_id == "v1" for r in results))

    def test_filter_by_document_and_version(self) -> None:
        retriever = self._build_multi_doc_retriever()
        results = retriever.retrieve(
            "capital", top_k=10, document_id="doc_A", version_id="v2"
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].clause_id, "s2")

    def test_filter_non_matching_returns_empty(self) -> None:
        retriever = self._build_multi_doc_retriever()
        results = retriever.retrieve("capital", top_k=10, document_id="doc_X")
        self.assertEqual(results, [])


# ---------------------------------------------------------------------------
# 6. Index Lifecycle
# ---------------------------------------------------------------------------


class TestDenseRetrieverIndexLifecycle(unittest.TestCase):
    """Index lifecycle: empty corpus, empty query, deduplication, re-index, caching, atomicity."""

    def test_empty_corpus_retrieve_returns_empty(self) -> None:
        retriever = DenseRetriever(embedder=DeterministicMockEmbedder(dimension=16))
        self.assertEqual(retriever.retrieve("capital"), [])
        self.assertEqual(retriever.segment_count, 0)

    def test_index_empty_list_clears_state(self) -> None:
        dim = 16
        retriever = DenseRetriever(embedder=DeterministicMockEmbedder(dimension=dim))
        seg = _make_segment("seg_1", "Capital adequacy.")
        retriever.index([seg])
        self.assertEqual(retriever.segment_count, 1)

        retriever.index([])
        self.assertEqual(retriever.segment_count, 0)
        self.assertEqual(retriever.retrieve("capital"), [])

    def test_empty_or_whitespace_query_returns_empty(self) -> None:
        retriever = DenseRetriever(embedder=DeterministicMockEmbedder(dimension=16))
        retriever.index([_make_segment("seg_1", "Capital adequacy.")])
        self.assertEqual(retriever.retrieve(""), [])
        self.assertEqual(retriever.retrieve("   "), [])

    def test_deduplication_first_occurrence_wins(self) -> None:
        dim = 16
        seg_first = _make_segment("dup_id", "First occurrence text.", ordinal=1)
        seg_second = _make_segment("dup_id", "Second occurrence text.", ordinal=2)
        embedder = DeterministicMockEmbedder(dimension=dim)
        retriever = DenseRetriever(embedder=embedder)
        retriever.index([seg_first, seg_second])

        self.assertEqual(retriever.segment_count, 1)
        results = retriever.retrieve("first occurrence", top_k=5)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].segment.text, "First occurrence text.")

    def test_reindex_replaces_previous_index(self) -> None:
        dim = 16
        embedder = DeterministicMockEmbedder(dimension=dim)
        retriever = DenseRetriever(embedder=embedder)
        seg_a = _make_segment("seg_A", "First corpus segment.")
        seg_b = _make_segment("seg_B", "Second corpus segment.")

        retriever.index([seg_a])
        self.assertEqual(retriever.segment_count, 1)

        retriever.index([seg_b])
        self.assertEqual(retriever.segment_count, 1)
        results = retriever.retrieve("second corpus", top_k=5)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].clause_id, "seg_B")

    def test_repeated_index_same_corpus_skips_reembedding(self) -> None:
        """Calling index() with the exact same segments reuses cached state."""
        dim = 16
        embedder = DeterministicMockEmbedder(dimension=dim)
        call_count = 0
        original_embed_texts = embedder.embed_texts

        def _counting_embed_texts(texts):
            nonlocal call_count
            call_count += 1
            return original_embed_texts(texts)

        embedder.embed_texts = _counting_embed_texts  # type: ignore[method-assign]
        retriever = DenseRetriever(embedder=embedder)
        seg = _make_segment("seg_cache", "Cached corpus test.")

        retriever.index([seg])
        self.assertEqual(call_count, 1)

        # Second index call with identical segment
        retriever.index([seg])
        self.assertEqual(call_count, 1)  # embed_texts was not called again

    def test_failed_index_preserves_prior_state(self) -> None:
        """If embed_texts raises an exception, the prior successful index is preserved."""
        dim = 16
        embedder = DeterministicMockEmbedder(dimension=dim)
        retriever = DenseRetriever(embedder=embedder)

        seg_good = _make_segment("good_seg", "Capital adequacy requirements.")
        retriever.index([seg_good])
        self.assertEqual(retriever.segment_count, 1)

        def _failing_embed_texts(texts):
            raise RuntimeError("Simulated embedding failure")

        embedder.embed_texts = _failing_embed_texts  # type: ignore[method-assign]

        bad_seg = _make_segment("bad_seg", "This will fail to embed.")
        with self.assertRaises(RuntimeError):
            retriever.index([bad_seg])

        # Prior state must remain intact
        self.assertEqual(retriever.segment_count, 1)
        results = retriever.retrieve("capital adequacy", top_k=5)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].clause_id, "good_seg")


# ---------------------------------------------------------------------------
# 7. Vector Validation
# ---------------------------------------------------------------------------


class TestDenseRetrieverValidation(unittest.TestCase):
    """DenseRetriever validates embedder output; malformed vectors raise ValueError."""

    def test_dimension_mismatch_raises_value_error(self) -> None:
        dim = 4

        class WrongDimEmbedder(ControllableMockEmbedder):
            def embed_texts(self, texts):
                return [[0.5] * (dim * 2)] * len(texts)

        retriever = DenseRetriever(embedder=WrongDimEmbedder(dimension=dim))
        seg = _make_segment("seg_dim", "Dimension mismatch test.")
        with self.assertRaises((ValueError, TypeError)):
            retriever.index([seg])

    def test_nan_in_vector_raises_value_error(self) -> None:
        dim = 4

        class NaNEmbedder(DeterministicMockEmbedder):
            def embed_texts(self, texts):
                return [[float("nan"), 0.0, 0.0, 0.0]] * len(texts)

        retriever = DenseRetriever(embedder=NaNEmbedder(dimension=dim))
        seg = _make_segment("seg_nan", "NaN vector test.")
        with self.assertRaises(ValueError) as ctx:
            retriever.index([seg])
        self.assertIn("Non-finite", str(ctx.exception))

    def test_infinity_in_vector_raises_value_error(self) -> None:
        dim = 4

        class InfEmbedder(DeterministicMockEmbedder):
            def embed_texts(self, texts):
                return [[float("inf"), 0.0, 0.0, 0.0]] * len(texts)

        retriever = DenseRetriever(embedder=InfEmbedder(dimension=dim))
        seg = _make_segment("seg_inf", "Inf vector test.")
        with self.assertRaises(ValueError) as ctx:
            retriever.index([seg])
        self.assertIn("Non-finite", str(ctx.exception))

    def test_zero_norm_vector_raises_value_error(self) -> None:
        dim = 4

        class ZeroNormEmbedder(DeterministicMockEmbedder):
            def embed_texts(self, texts):
                return [[0.0, 0.0, 0.0, 0.0]] * len(texts)

        retriever = DenseRetriever(embedder=ZeroNormEmbedder(dimension=dim))
        seg = _make_segment("seg_zero", "Zero-norm vector test.")
        with self.assertRaises(ValueError) as ctx:
            retriever.index([seg])
        self.assertIn("zero", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
