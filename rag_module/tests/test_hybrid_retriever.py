"""Hybrid retrieval: RRF arithmetic, fusion behaviour, filters, degradation.

These tests use the mock embedders, so they run with no model and no network.
The arithmetic is checked against hand-computed values rather than against the
implementation, because a fusion formula that is merely self-consistent is
worthless.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rag_module.core.models import ClauseSegment, PageProvenance  # noqa: E402
from rag_module.embeddings.mock import ControllableMockEmbedder  # noqa: E402
from rag_module.retrieval.bm25_retriever import BM25Retriever  # noqa: E402
from rag_module.retrieval.dense_retriever import DenseRetriever  # noqa: E402
from rag_module.retrieval.hybrid_retriever import (  # noqa: E402
    DEFAULT_RRF_K,
    HybridRetriever,
    rrf_score,
)


def seg(
    clause_id: str,
    text: str,
    *,
    ordinal: int = 0,
    document_id: str = "DOC-0001",
    version_id: str = "VER-0001",
    clause_number: str | None = None,
) -> ClauseSegment:
    return ClauseSegment(
        document_id=document_id,
        version_id=version_id,
        clause_id=clause_id,
        fingerprint=f"SEG-{clause_id}",
        ordinal=ordinal,
        segment_type="clause",
        text=text,
        page_start=1,
        page_end=1,
        pages=[PageProvenance(page_number=1, char_start=0, char_end=len(text))],
        clause_number=clause_number,
    )


CORPUS = [
    seg("CLS-1", "Lead Bank responsibilities for the district of Old Delhi.", ordinal=1),
    seg("CLS-2", "Capital adequacy ratio for scheduled commercial banks.", ordinal=2),
    seg("CLS-3", "Penal provisions for non compliance with reporting.", ordinal=3),
    seg("CLS-4", "Reporting requirements and quarterly returns.", ordinal=4),
    seg("CLS-5", "Canara Bank is assigned to the Old Delhi district.", ordinal=5),
]


class TestRrfScore(unittest.TestCase):
    def test_single_rank(self) -> None:
        self.assertAlmostEqual(rrf_score([1]), 1.0 / 61.0, places=12)

    def test_two_ranks_sum(self) -> None:
        self.assertAlmostEqual(
            rrf_score([1, 2]), 1.0 / 61.0 + 1.0 / 62.0, places=12
        )

    def test_rank_one_in_both_beats_rank_one_in_one(self) -> None:
        """The property RRF exists for: agreement outranks a lone top hit."""
        self.assertGreater(rrf_score([1, 1]), rrf_score([1]))

    def test_two_mid_ranks_can_beat_one_top_rank(self) -> None:
        """k damps rank 1 so corroboration can win. This is the whole point."""
        self.assertGreater(rrf_score([2, 3]), rrf_score([1]))

    def test_k_zero_is_permitted_and_sharpens_rank_one(self) -> None:
        self.assertAlmostEqual(rrf_score([1], k=0), 1.0, places=12)

    def test_custom_k_flattens(self) -> None:
        self.assertLess(rrf_score([1], k=1000), rrf_score([1], k=10))

    def test_zero_rank_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            rrf_score([0])

    def test_negative_rank_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            rrf_score([-1])

    def test_negative_k_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            rrf_score([1], k=-1)

    def test_empty_ranks_is_zero(self) -> None:
        self.assertEqual(rrf_score([]), 0.0)


class TestHybridConstruction(unittest.TestCase):
    def test_without_an_embedder_it_is_not_hybrid(self) -> None:
        retriever = HybridRetriever()
        self.assertFalse(retriever.is_hybrid)
        self.assertEqual(retriever.active_components, ["bm25"])

    def test_with_an_embedder_both_arms_are_active(self) -> None:
        retriever = HybridRetriever(embedder=ControllableMockEmbedder(dimension=8))
        self.assertTrue(retriever.is_hybrid)
        self.assertEqual(retriever.active_components, ["bm25", "dense"])

    def test_negative_rrf_k_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            HybridRetriever(rrf_k=-1)

    def test_zero_candidate_depth_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            HybridRetriever(candidate_depth=0)

    def test_existing_retrievers_can_be_reused(self) -> None:
        bm25 = BM25Retriever()
        dense = DenseRetriever(embedder=ControllableMockEmbedder(dimension=8))
        retriever = HybridRetriever(bm25=bm25, dense=dense)
        self.assertIs(retriever.bm25, bm25)
        self.assertIs(retriever.dense, dense)


class TestHybridIndexing(unittest.TestCase):
    def test_both_arms_receive_the_same_corpus(self) -> None:
        retriever = HybridRetriever(embedder=ControllableMockEmbedder(dimension=8))
        retriever.index(CORPUS)
        self.assertEqual(retriever.segment_count, len(CORPUS))
        self.assertEqual(retriever.bm25.corpus_size, len(CORPUS))
        self.assertEqual(retriever.dense.segment_count, len(CORPUS))

    def test_empty_corpus_retrieves_nothing(self) -> None:
        retriever = HybridRetriever()
        retriever.index([])
        self.assertEqual(retriever.retrieve("anything"), [])

    def test_reindexing_replaces_the_corpus(self) -> None:
        retriever = HybridRetriever()
        retriever.index(CORPUS)
        retriever.index(CORPUS[:2])
        self.assertEqual(retriever.segment_count, 2)


class TestHybridRetrieval(unittest.TestCase):
    def setUp(self) -> None:
        self.retriever = HybridRetriever()
        self.retriever.index(CORPUS)

    def test_empty_query_returns_nothing(self) -> None:
        self.assertEqual(self.retriever.retrieve(""), [])
        self.assertEqual(self.retriever.retrieve("   "), [])

    def test_zero_top_k_returns_nothing(self) -> None:
        self.assertEqual(self.retriever.retrieve("bank", top_k=0), [])

    def test_results_are_ranked_from_one(self) -> None:
        results = self.retriever.retrieve("Old Delhi district", top_k=3)
        self.assertTrue(results)
        self.assertEqual([r.rank for r in results], list(range(1, len(results) + 1)))

    def test_scores_are_non_increasing(self) -> None:
        results = self.retriever.retrieve("bank reporting", top_k=5)
        scores = [r.score for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_retrieval_method_reflects_what_actually_ran(self) -> None:
        """BM25-only must not label its output "hybrid"."""
        results = self.retriever.retrieve("Old Delhi", top_k=1)
        self.assertEqual(results[0].retrieval_method, "bm25")

        hybrid = HybridRetriever(embedder=ControllableMockEmbedder(dimension=8))
        hybrid.index(CORPUS)
        results = hybrid.retrieve("Old Delhi", top_k=1)
        self.assertEqual(results[0].retrieval_method, "hybrid")

    def test_document_filter_is_applied(self) -> None:
        corpus = CORPUS + [seg("CLS-9", "Old Delhi again", document_id="DOC-0002")]
        retriever = HybridRetriever()
        retriever.index(corpus)
        results = retriever.retrieve("Old Delhi", top_k=10, document_id="DOC-0002")
        self.assertTrue(results)
        self.assertTrue(all(r.document_id == "DOC-0002" for r in results))

    def test_version_filter_is_applied(self) -> None:
        corpus = CORPUS + [seg("CLS-9", "Old Delhi again", version_id="VER-0002")]
        retriever = HybridRetriever()
        retriever.index(corpus)
        results = retriever.retrieve("Old Delhi", top_k=10, version_id="VER-0002")
        self.assertTrue(results)
        self.assertTrue(all(r.version_id == "VER-0002" for r in results))

    def test_a_clause_is_never_returned_twice(self) -> None:
        """Merging on clause_id is what stops two arms double-counting one clause."""
        hybrid = HybridRetriever(embedder=ControllableMockEmbedder(dimension=8))
        hybrid.index(CORPUS)
        results = hybrid.retrieve("bank Old Delhi reporting", top_k=10)
        clause_ids = [r.clause_id for r in results]
        self.assertEqual(len(clause_ids), len(set(clause_ids)))

    def test_candidate_depth_is_at_least_top_k(self) -> None:
        """Fusing a window narrower than the output would truncate the answer."""
        retriever = HybridRetriever(candidate_depth=1)
        retriever.index(CORPUS)
        results = retriever.retrieve("bank", top_k=4)
        self.assertGreater(len(results), 1)

    def test_union_not_intersection(self) -> None:
        """A clause only one arm found must survive fusion."""
        embedder = ControllableMockEmbedder(dimension=8)
        hybrid = HybridRetriever(embedder=embedder)
        hybrid.index(CORPUS)
        lexical = {r.clause_id for r in hybrid.bm25.retrieve("Canara", top_k=10)}
        fused = {r.clause_id for r in hybrid.retrieve("Canara", top_k=10)}
        self.assertTrue(lexical.issubset(fused))


class TestFusionDetail(unittest.TestCase):
    def test_detail_reports_each_arms_rank(self) -> None:
        hybrid = HybridRetriever(embedder=ControllableMockEmbedder(dimension=8))
        hybrid.index(CORPUS)
        detailed = hybrid.retrieve_with_detail("Old Delhi district", top_k=3)
        self.assertTrue(detailed)
        for _passage, detail in detailed:
            self.assertIn("component_ranks", detail)
            self.assertTrue(set(detail["component_ranks"]) <= {"bm25", "dense"})
            self.assertEqual(detail["rrf_k"], DEFAULT_RRF_K)

    def test_detail_score_matches_the_rrf_of_its_ranks(self) -> None:
        """The reported score must be the formula applied to the reported ranks."""
        hybrid = HybridRetriever(embedder=ControllableMockEmbedder(dimension=8))
        hybrid.index(CORPUS)
        for passage, detail in hybrid.retrieve_with_detail("bank reporting", top_k=5):
            expected = rrf_score(list(detail["component_ranks"].values()),
                                 k=detail["rrf_k"])
            self.assertAlmostEqual(passage.score, expected, places=12)

    def test_detail_does_not_mutate_the_indexed_segments(self) -> None:
        """Per-query detail must never be written onto shared segment objects."""
        hybrid = HybridRetriever()
        hybrid.index(CORPUS)
        before = [dict(s.metadata) for s in CORPUS]
        hybrid.retrieve_with_detail("bank", top_k=5)
        after = [dict(s.metadata) for s in CORPUS]
        self.assertEqual(before, after)


class TestDeterminism(unittest.TestCase):
    def test_same_query_gives_the_same_order(self) -> None:
        retriever = HybridRetriever()
        retriever.index(CORPUS)
        first = [r.clause_id for r in retriever.retrieve("bank reporting", top_k=5)]
        second = [r.clause_id for r in retriever.retrieve("bank reporting", top_k=5)]
        self.assertEqual(first, second)

    def test_ties_break_on_ordinal_then_clause_id(self) -> None:
        """Two clauses each found only by BM25 at adjacent ranks differ in score;
        construct a true tie by giving both the same single rank in one arm."""
        tied = [
            seg("CLS-B", "identical wording here", ordinal=2),
            seg("CLS-A", "identical wording here", ordinal=1),
        ]
        retriever = HybridRetriever()
        retriever.index(tied)
        results = retriever.retrieve("identical wording here", top_k=2)
        # Both match identically, so BM25 itself tie-breaks on ordinal first.
        self.assertEqual([r.clause_id for r in results], ["CLS-A", "CLS-B"])


if __name__ == "__main__":
    unittest.main()
