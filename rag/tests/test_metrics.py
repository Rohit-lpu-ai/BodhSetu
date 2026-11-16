"""Tests for rag_module.evaluation.metrics.

Verifies:
- Recall@K (perfect, zero, partial, cutoff effects, duplicates)
- Reciprocal Rank / MRR (rank 1, 2, 3, not found, cutoff K)
- DCG@K and NDCG@K (perfect, zero, ranking-order differences, logarithmic discounting)
- Duplicate retrieved clause ID handling
- RetrievedPassage and ClauseSegment object compatibility
- Input validation (invalid K, empty relevance, malformed types, duplicate question IDs)
- Macro-averaging and subgroup breakdown (core vs auxiliary, en vs hi)
- Integration with real annotation_worksheet.jsonl
"""

import math
import tempfile
import unittest
from pathlib import Path

from rag_module.core.models import ClauseSegment, PageProvenance, RetrievedPassage
from rag_module.evaluation.metrics import (
    EvaluationSummary,
    QueryResult,
    dcg_at_k,
    evaluate_benchmark,
    evaluate_query,
    load_ground_truth,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)


def _make_dummy_passage(clause_id: str, score: float = 1.0, rank: int = 1) -> RetrievedPassage:
    """Helper to construct a RetrievedPassage object for testing."""
    seg = ClauseSegment(
        document_id="DOC-0001",
        version_id="VER-0001",
        clause_id=clause_id,
        fingerprint="SEG-dummy",
        ordinal=0,
        segment_type="clause",
        text="Dummy text",
        page_start=1,
        page_end=1,
        pages=[PageProvenance(page_number=1, char_start=0, char_end=10)],
    )
    return RetrievedPassage(
        segment=seg,
        score=score,
        rank=rank,
        retrieval_method="bm25",
    )


class TestRecallAtK(unittest.TestCase):
    """Unit tests for recall_at_k."""

    def test_perfect_retrieval(self):
        retrieved = ["CLS-001", "CLS-002", "CLS-003"]
        relevant = ["CLS-001", "CLS-002"]
        # Both relevant items in top 2
        self.assertAlmostEqual(recall_at_k(retrieved, relevant, k=2), 1.0)
        self.assertAlmostEqual(recall_at_k(retrieved, relevant, k=5), 1.0)

    def test_zero_retrieval(self):
        retrieved = ["CLS-003", "CLS-004"]
        relevant = ["CLS-001", "CLS-002"]
        self.assertEqual(recall_at_k(retrieved, relevant, k=2), 0.0)
        self.assertEqual(recall_at_k([], relevant, k=5), 0.0)

    def test_partial_retrieval(self):
        retrieved = ["CLS-001", "CLS-999"]
        relevant = ["CLS-001", "CLS-002"]
        self.assertAlmostEqual(recall_at_k(retrieved, relevant, k=2), 0.5)

    def test_k_cutoff_boundary(self):
        retrieved = ["CLS-999", "CLS-001"]
        relevant = ["CLS-001"]
        # At K=1, CLS-001 is outside top 1
        self.assertEqual(recall_at_k(retrieved, relevant, k=1), 0.0)
        # At K=2, CLS-001 is included
        self.assertAlmostEqual(recall_at_k(retrieved, relevant, k=2), 1.0)

    def test_duplicate_retrieved_ids_does_not_inflate(self):
        # CLS-001 repeated in retrieved
        retrieved = ["CLS-001", "CLS-001", "CLS-001"]
        relevant = ["CLS-001", "CLS-002"]
        # Unique match is 1, total relevant is 2 -> 0.5
        self.assertAlmostEqual(recall_at_k(retrieved, relevant, k=3), 0.5)

    def test_retrieved_passage_objects_accepted(self):
        retrieved = [
            _make_dummy_passage("CLS-001", 0.9, 1),
            _make_dummy_passage("CLS-002", 0.8, 2),
        ]
        relevant = ["CLS-001"]
        self.assertAlmostEqual(recall_at_k(retrieved, relevant, k=1), 1.0)

    def test_invalid_k_raises(self):
        with self.assertRaises(ValueError):
            recall_at_k(["CLS-001"], ["CLS-001"], k=0)
        with self.assertRaises(ValueError):
            recall_at_k(["CLS-001"], ["CLS-001"], k=-5)
        with self.assertRaises(TypeError):
            recall_at_k(["CLS-001"], ["CLS-001"], k="1")  # type: ignore

    def test_empty_relevant_raises(self):
        with self.assertRaises(ValueError):
            recall_at_k(["CLS-001"], [], k=5)
        with self.assertRaises(ValueError):
            recall_at_k(["CLS-001"], set(), k=5)


class TestReciprocalRank(unittest.TestCase):
    """Unit tests for reciprocal_rank."""

    def test_first_rank(self):
        retrieved = ["CLS-001", "CLS-002"]
        relevant = ["CLS-001"]
        self.assertAlmostEqual(reciprocal_rank(retrieved, relevant), 1.0)

    def test_second_rank(self):
        retrieved = ["CLS-999", "CLS-001"]
        relevant = ["CLS-001"]
        self.assertAlmostEqual(reciprocal_rank(retrieved, relevant), 0.5)

    def test_third_rank(self):
        retrieved = ["CLS-999", "CLS-888", "CLS-001"]
        relevant = ["CLS-001"]
        self.assertAlmostEqual(reciprocal_rank(retrieved, relevant), 1.0 / 3.0)

    def test_not_retrieved_returns_zero(self):
        retrieved = ["CLS-999", "CLS-888"]
        relevant = ["CLS-001"]
        self.assertEqual(reciprocal_rank(retrieved, relevant), 0.0)
        self.assertEqual(reciprocal_rank([], relevant), 0.0)

    def test_first_matching_relevant_used_when_multiple(self):
        # Multiple relevant items in ground truth; RR uses the highest rank
        retrieved = ["CLS-999", "CLS-002", "CLS-001"]
        relevant = ["CLS-001", "CLS-002"]
        # CLS-002 found at rank 2
        self.assertAlmostEqual(reciprocal_rank(retrieved, relevant), 0.5)

    def test_k_cutoff_honored(self):
        retrieved = ["CLS-999", "CLS-001"]
        relevant = ["CLS-001"]
        # Cutoff k=1 does not reach rank 2
        self.assertEqual(reciprocal_rank(retrieved, relevant, k=1), 0.0)
        self.assertAlmostEqual(reciprocal_rank(retrieved, relevant, k=2), 0.5)

    def test_duplicate_retrieved_ids_preserves_true_rank(self):
        # Duplicate of non-relevant item before relevant item
        retrieved = ["CLS-999", "CLS-999", "CLS-001"]
        relevant = ["CLS-001"]
        # After deduplication: ["CLS-999", "CLS-001"] -> rank 2 -> 0.5
        self.assertAlmostEqual(reciprocal_rank(retrieved, relevant), 0.5)


class TestNDCG(unittest.TestCase):
    """Unit tests for dcg_at_k and ndcg_at_k."""

    def test_perfect_retrieval_single_relevant(self):
        retrieved = ["CLS-001", "CLS-999"]
        relevant = ["CLS-001"]
        # DCG@1 = 1 / log2(2) = 1.0, IDCG@1 = 1.0 -> NDCG@1 = 1.0
        self.assertAlmostEqual(ndcg_at_k(retrieved, relevant, k=1), 1.0)
        self.assertAlmostEqual(ndcg_at_k(retrieved, relevant, k=5), 1.0)

    def test_perfect_retrieval_multiple_relevant(self):
        retrieved = ["CLS-001", "CLS-002", "CLS-999"]
        relevant = ["CLS-001", "CLS-002"]
        self.assertAlmostEqual(ndcg_at_k(retrieved, relevant, k=2), 1.0)
        self.assertAlmostEqual(ndcg_at_k(retrieved, relevant, k=5), 1.0)

    def test_zero_retrieval(self):
        retrieved = ["CLS-999", "CLS-888"]
        relevant = ["CLS-001"]
        self.assertEqual(ndcg_at_k(retrieved, relevant, k=5), 0.0)
        self.assertEqual(ndcg_at_k([], relevant, k=5), 0.0)

    def test_ranking_order_sensitivity(self):
        # Rank 1 vs Rank 2:
        # If relevant item is at rank 1:
        ndcg_rank1 = ndcg_at_k(["CLS-001", "CLS-999"], ["CLS-001"], k=2)
        # If relevant item is at rank 2:
        ndcg_rank2 = ndcg_at_k(["CLS-999", "CLS-001"], ["CLS-001"], k=2)

        self.assertAlmostEqual(ndcg_rank1, 1.0)
        expected_rank2 = (1.0 / math.log2(3)) / (1.0 / math.log2(2))
        self.assertAlmostEqual(ndcg_rank2, expected_rank2)
        self.assertGreater(ndcg_rank1, ndcg_rank2)

    def test_partial_multi_clause_ndcg(self):
        # 2 relevant clauses, 1 retrieved at rank 1, second missing
        retrieved = ["CLS-001", "CLS-999"]
        relevant = ["CLS-001", "CLS-002"]
        dcg = 1.0 / math.log2(2)
        idcg = (1.0 / math.log2(2)) + (1.0 / math.log2(3))
        expected = dcg / idcg
        self.assertAlmostEqual(ndcg_at_k(retrieved, relevant, k=2), expected)

    def test_duplicate_retrieved_does_not_gain_discount(self):
        retrieved = ["CLS-001", "CLS-001", "CLS-002"]
        relevant = ["CLS-001", "CLS-002"]
        # After deduplication: ["CLS-001", "CLS-002"] -> ideal rank -> 1.0
        self.assertAlmostEqual(ndcg_at_k(retrieved, relevant, k=2), 1.0)


class TestEvaluateQuery(unittest.TestCase):
    """Unit tests for evaluate_query."""

    def test_returns_all_requested_metrics(self):
        retrieved = ["CLS-001", "CLS-002"]
        relevant = ["CLS-001"]
        res = evaluate_query(retrieved, relevant, k_values=(1, 3, 5))
        expected_keys = {"recall@1", "recall@3", "recall@5", "mrr", "ndcg@1", "ndcg@3", "ndcg@5"}
        self.assertEqual(set(res.keys()), expected_keys)
        self.assertAlmostEqual(res["recall@1"], 1.0)
        self.assertAlmostEqual(res["mrr"], 1.0)
        self.assertAlmostEqual(res["ndcg@1"], 1.0)


class TestLoadGroundTruth(unittest.TestCase):
    """Unit tests for load_ground_truth."""

    def test_load_from_worksheet(self):
        path = Path("rag_module/evaluation/annotation_worksheet.jsonl")
        if not path.exists():
            self.skipTest("annotation_worksheet.jsonl not found")
        gt = load_ground_truth(path, use_verified=True)
        self.assertEqual(len(gt), 19)
        # Verify CAND-EN-0001
        self.assertIn("CAND-EN-0001", gt)
        self.assertEqual(gt["CAND-EN-0001"]["relevant_clause_ids"], ["CLS-0000007"])
        self.assertEqual(gt["CAND-EN-0001"]["dataset_group"], "core")
        # Verify CAND-EN-0002 has sole verified ID CLS-0000013
        self.assertEqual(gt["CAND-EN-0002"]["relevant_clause_ids"], ["CLS-0000013"])

    def test_duplicate_question_id_raises(self):
        records = [
            {"question_id": "Q-001", "relevant_clause_ids": ["CLS-01"]},
            {"question_id": "Q-001", "relevant_clause_ids": ["CLS-02"]},
        ]
        with self.assertRaises(ValueError) as ctx:
            load_ground_truth(records)
        self.assertIn("Duplicate question_id", str(ctx.exception))

    def test_empty_relevance_raises(self):
        records = [{"question_id": "Q-001", "relevant_clause_ids": []}]
        with self.assertRaises(ValueError) as ctx:
            load_ground_truth(records)
        self.assertIn("empty relevance", str(ctx.exception).lower())

    def test_missing_file_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_ground_truth("nonexistent_gt_file.jsonl")


class TestEvaluateBenchmark(unittest.TestCase):
    """Unit tests for evaluate_benchmark macro-averages and breakdowns."""

    def setUp(self):
        self.ground_truth = {
            "Q1": {
                "question_id": "Q1",
                "dataset_group": "core",
                "language": "en",
                "relevant_clause_ids": ["CLS-01"],
            },
            "Q2": {
                "question_id": "Q2",
                "dataset_group": "core",
                "language": "hi",
                "relevant_clause_ids": ["CLS-02"],
            },
            "Q3": {
                "question_id": "Q3",
                "dataset_group": "auxiliary",
                "language": "hi",
                "relevant_clause_ids": ["CLS-03"],
            },
        }

    def test_perfect_benchmark_evaluation(self):
        # All queries achieve rank 1
        predictions = {
            "Q1": ["CLS-01"],
            "Q2": ["CLS-02"],
            "Q3": ["CLS-03"],
        }
        summary = evaluate_benchmark(predictions, self.ground_truth, k_values=(1, 3))
        self.assertEqual(summary.question_count, 3)
        self.assertAlmostEqual(summary.overall["recall@1"], 1.0)
        self.assertAlmostEqual(summary.overall["mrr"], 1.0)
        self.assertAlmostEqual(summary.overall["ndcg@1"], 1.0)
        self.assertEqual(summary.group_counts, {"auxiliary": 1, "core": 2})
        self.assertEqual(summary.language_counts, {"en": 1, "hi": 2})

    def test_zero_retrieval_benchmark(self):
        predictions = {
            "Q1": ["CLS-WRONG"],
            "Q2": [],
            "Q3": ["CLS-IRRELEVANT"],
        }
        summary = evaluate_benchmark(predictions, self.ground_truth, k_values=(1, 3))
        self.assertAlmostEqual(summary.overall["recall@1"], 0.0)
        self.assertAlmostEqual(summary.overall["mrr"], 0.0)
        self.assertAlmostEqual(summary.overall["ndcg@1"], 0.0)

    def test_subgroup_breakdown_correctness(self):
        # Q1 (core, en): perfect (mrr = 1.0)
        # Q2 (core, hi): rank 2 (mrr = 0.5)
        # Q3 (auxiliary, hi): not retrieved (mrr = 0.0)
        predictions = {
            "Q1": ["CLS-01"],
            "Q2": ["CLS-WRONG", "CLS-02"],
            "Q3": ["CLS-WRONG"],
        }
        summary = evaluate_benchmark(predictions, self.ground_truth, k_values=(1, 3))

        # Overall MRR: (1.0 + 0.5 + 0.0) / 3 = 0.5
        self.assertAlmostEqual(summary.overall["mrr"], 0.5)

        # Core MRR: Q1(1.0) + Q2(0.5) / 2 = 0.75
        self.assertAlmostEqual(summary.by_group["core"]["mrr"], 0.75)
        # Auxiliary MRR: Q3(0.0) = 0.0
        self.assertAlmostEqual(summary.by_group["auxiliary"]["mrr"], 0.0)

        # English MRR: Q1(1.0) = 1.0
        self.assertAlmostEqual(summary.by_language["en"]["mrr"], 1.0)
        # Hindi MRR: Q2(0.5) + Q3(0.0) / 2 = 0.25
        self.assertAlmostEqual(summary.by_language["hi"]["mrr"], 0.25)

    def test_prediction_as_list_of_dicts(self):
        predictions = [
            {"question_id": "Q1", "retrieved": ["CLS-01"]},
            {"question_id": "Q2", "retrieved": ["CLS-02"]},
            {"question_id": "Q3", "retrieved": ["CLS-03"]},
        ]
        summary = evaluate_benchmark(predictions, self.ground_truth, k_values=(1,))
        self.assertAlmostEqual(summary.overall["recall@1"], 1.0)

    def test_duplicate_prediction_question_id_raises(self):
        predictions = [
            {"question_id": "Q1", "retrieved": ["CLS-01"]},
            {"question_id": "Q1", "retrieved": ["CLS-02"]},  # duplicate
        ]
        with self.assertRaises(ValueError) as ctx:
            evaluate_benchmark(predictions, self.ground_truth)
        self.assertIn("Duplicate question_id", str(ctx.exception))

    def test_unknown_prediction_question_id_raises(self):
        predictions = {"Q_UNKNOWN": ["CLS-01"]}
        with self.assertRaises(ValueError) as ctx:
            evaluate_benchmark(predictions, self.ground_truth)
        self.assertIn("not found in ground truth", str(ctx.exception))

    def test_integration_with_actual_annotation_worksheet(self):
        ws_path = Path("rag_module/evaluation/annotation_worksheet.jsonl")
        if not ws_path.exists():
            self.skipTest("annotation_worksheet.jsonl not found")

        gt = load_ground_truth(ws_path, use_verified=True)
        # Simulate perfect predictions using the ground truth itself
        perfect_predictions = {qid: list(item["relevant_clause_ids"]) for qid, item in gt.items()}

        summary = evaluate_benchmark(perfect_predictions, gt, k_values=(1, 3, 5, 10))
        self.assertEqual(summary.question_count, 19)
        self.assertEqual(summary.group_counts["core"], 15)
        self.assertEqual(summary.group_counts["auxiliary"], 4)
        self.assertAlmostEqual(summary.overall["recall@1"], 1.0)
        self.assertAlmostEqual(summary.overall["mrr"], 1.0)
        self.assertAlmostEqual(summary.overall["ndcg@1"], 1.0)
        self.assertAlmostEqual(summary.by_group["core"]["recall@1"], 1.0)
        self.assertAlmostEqual(summary.by_group["auxiliary"]["recall@1"], 1.0)


if __name__ == "__main__":
    unittest.main()
