"""Unit tests for Benchmark Runner (BM25 vs DenseRetriever).

Tests benchmark configuration, corpus loading and isolation, dependency injection,
result persistence, metric calculation, error handling, and CLI functionality.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
import tempfile
from typing import Any, Dict, List
import unittest
from unittest.mock import MagicMock, patch

from rag_module.core.interfaces import BaseRetriever
from rag_module.core.models import ClauseSegment, PageProvenance, RetrievedPassage
from rag_module.embeddings.mock import MockEmbedder
from rag_module.evaluation.benchmark_runner import (
    DEFAULT_RESULTS_DIR,
    DEFAULT_WORKSHEET_PATH,
    BenchmarkConfig,
    create_retriever,
    format_summary_table,
    generate_result_filepath,
    load_evaluation_corpus,
    main,
    run_benchmark,
    run_single_retriever_benchmark,
)
from rag_module.retrieval.bm25_retriever import BM25Retriever
from rag_module.retrieval.dense_retriever import DenseRetriever


class MockDummyRetriever(BaseRetriever):
    """Simple mock retriever for isolated runner testing."""

    def __init__(self, fixed_passages: List[RetrievedPassage]) -> None:
        self.fixed_passages = fixed_passages
        self.indexed_count = 0

    def index(self, segments: List[ClauseSegment]) -> None:
        self.indexed_count = len(segments)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        document_id: str | None = None,
        version_id: str | None = None,
    ) -> List[RetrievedPassage]:
        return self.fixed_passages[:top_k]


def _make_dummy_segment(clause_id: str, text: str = "Test regulatory passage.") -> ClauseSegment:
    return ClauseSegment(
        document_id="DOC-0001",
        version_id="VER-0001",
        clause_id=clause_id,
        fingerprint=f"fp_{clause_id}",
        ordinal=1,
        segment_type="clause",
        text=text,
        page_start=1,
        page_end=1,
        pages=[PageProvenance(page_number=1, char_start=0, char_end=len(text))],
        clause_number="1",
        heading="Definitions",
    )


class TestBenchmarkConfig(unittest.TestCase):
    """Test BenchmarkConfig validation and defaults."""

    def test_default_config(self) -> None:
        cfg = BenchmarkConfig()
        self.assertEqual(cfg.experiment, "clause_aware")
        self.assertEqual(cfg.k_values, (1, 3, 5, 10))
        self.assertEqual(cfg.top_k, 10)
        self.assertEqual(cfg.worksheet_path, DEFAULT_WORKSHEET_PATH)
        self.assertEqual(cfg.results_dir, DEFAULT_RESULTS_DIR)

    def test_invalid_experiment_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            BenchmarkConfig(experiment="invalid_experiment")
        self.assertIn("Invalid experiment", str(ctx.exception))

    def test_top_k_clamps_k_values(self) -> None:
        cfg = BenchmarkConfig(k_values=(1, 5, 10), top_k=5)
        self.assertEqual(cfg.k_values, (1, 5))


class TestCorpusLoading(unittest.TestCase):
    """Test experiment corpus loading and separation."""

    def test_load_experiment_a_corpus(self) -> None:
        cfg = BenchmarkConfig(experiment="clause_aware")
        corpus, total = load_evaluation_corpus(cfg)
        self.assertEqual(len(corpus), 363)
        self.assertEqual(total, 501)
        for seg in corpus:
            self.assertNotEqual(seg.segment_type, "paragraph")
            self.assertEqual(seg.segmentation_method, "clause_aware")

    def test_load_experiment_b_corpus(self) -> None:
        cfg = BenchmarkConfig(experiment="paragraph")
        corpus, total = load_evaluation_corpus(cfg)
        self.assertEqual(len(corpus), 138)
        self.assertEqual(total, 501)
        for seg in corpus:
            self.assertEqual(seg.segment_type, "paragraph")


class TestCreateRetriever(unittest.TestCase):
    """Test retriever creation and dependency injection."""

    def test_create_bm25_retriever(self) -> None:
        cfg = BenchmarkConfig()
        retriever = create_retriever("bm25", cfg)
        self.assertIsInstance(retriever, BM25Retriever)

    def test_create_dense_with_injected_embedder(self) -> None:
        cfg = BenchmarkConfig()
        mock_emb = MockEmbedder(dimension=64)
        retriever = create_retriever("dense", cfg, embedder=mock_emb)
        self.assertIsInstance(retriever, DenseRetriever)
        self.assertIs(retriever.embedder, mock_emb)

    def test_create_with_preconstructed_retriever(self) -> None:
        cfg = BenchmarkConfig()
        custom = MockDummyRetriever([])
        retriever = create_retriever("bm25", cfg, retriever_instance=custom)
        self.assertIs(retriever, custom)

    def test_unknown_strategy_raises(self) -> None:
        cfg = BenchmarkConfig()
        with self.assertRaises(ValueError) as ctx:
            create_retriever("unsupported", cfg)
        self.assertIn("Unknown retrieval strategy", str(ctx.exception))


class TestRunSingleRetrieverBenchmark(unittest.TestCase):
    """Test execution of single retriever benchmark with synthetic data."""

    def test_synthetic_run(self) -> None:
        seg1 = _make_dummy_segment("CLS-001", "Lead bank Canara Bank Old Delhi.")
        seg2 = _make_dummy_segment("CLS-002", "Other text.")
        corpus = [seg1, seg2]

        passages = [
            RetrievedPassage(segment=seg1, score=0.95, rank=1, retrieval_method="test"),
            RetrievedPassage(segment=seg2, score=0.40, rank=2, retrieval_method="test"),
        ]
        retriever = MockDummyRetriever(passages)

        gt = {
            "Q1": {
                "question_id": "Q1",
                "question": "Which bank is lead bank?",
                "language": "en",
                "dataset_group": "core",
                "relevant_clause_ids": ["CLS-001"],
            }
        }
        cfg = BenchmarkConfig(k_values=(1, 3), top_k=2)

        res = run_single_retriever_benchmark(
            strategy="mock_strat",
            retriever=retriever,
            corpus=corpus,
            ground_truth=gt,
            config=cfg,
        )

        self.assertEqual(retriever.indexed_count, 2)
        meta = res["benchmark_metadata"]
        self.assertEqual(meta["strategy"], "mock_strat")
        self.assertEqual(meta["corpus_size"], 2)
        self.assertEqual(meta["total_questions_evaluated"], 1)

        metrics = res["metrics"]["overall"]
        self.assertEqual(metrics["recall@1"], 1.0)
        self.assertEqual(metrics["mrr"], 1.0)

        query_res = res["per_query_results"][0]
        self.assertEqual(query_res["question_id"], "Q1")
        self.assertIn("latency_ms", query_res)
        self.assertEqual(len(query_res["retrieved"]), 2)
        self.assertEqual(query_res["retrieved"][0]["clause_id"], "CLS-001")


class TestDenseDependencyHandling(unittest.TestCase):
    """Test that missing dependencies raise explicit ImportError and never return fake results."""

    def test_missing_sentence_transformers_raises_import_error(self) -> None:
        cfg = BenchmarkConfig()
        # When attempting to create dense retriever without injected embedder or dependencies:
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            with self.assertRaises(ImportError) as ctx:
                retriever = create_retriever("dense", cfg)
                # Trigger lazy load
                retriever.index([_make_dummy_segment("CLS-001")])
            self.assertIn("requires 'sentence-transformers' and 'torch'", str(ctx.exception))


class TestResultNonOverwriting(unittest.TestCase):
    """Test result filepath generation never collides or overwrites."""

    def test_non_overwriting_filenames(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            p1 = generate_result_filepath(out_dir, "bm25", "clause_aware", "20261116T000000Z")
            p1.touch()

            p2 = generate_result_filepath(out_dir, "bm25", "clause_aware", "20261116T000000Z")
            self.assertNotEqual(p1, p2)
            self.assertTrue(str(p2).endswith("_1.json"))


class TestEndToEndBenchmarkRuns(unittest.TestCase):
    """Test end-to-end benchmark runs with BM25 and injected Dense embedders."""

    def test_bm25_end_to_end_on_worksheet(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = BenchmarkConfig(
                experiment="clause_aware",
                worksheet_path=DEFAULT_WORKSHEET_PATH,
                results_dir=Path(tmpdir),
                k_values=(1, 3, 5, 10),
                top_k=10,
            )

            result, saved_file = run_benchmark(config=cfg, strategy="bm25", save_results=True)

            self.assertIsNotNone(saved_file)
            self.assertTrue(saved_file.exists())

            # Validate metadata
            meta = result["benchmark_metadata"]
            self.assertEqual(meta["strategy"], "bm25")
            self.assertEqual(meta["experiment"], "clause_aware")
            self.assertEqual(meta["corpus_size"], 363)
            self.assertEqual(meta["total_questions_evaluated"], 19)

            # Validate metrics structure
            metrics = result["metrics"]
            overall = metrics["overall"]
            for m in ("recall@1", "recall@3", "recall@5", "recall@10", "mrr", "ndcg@5", "ndcg@10"):
                self.assertIn(m, overall)
                self.assertGreaterEqual(overall[m], 0.0)
                self.assertLessEqual(overall[m], 1.0)

            # Validate group breakdowns
            by_group = metrics["by_group"]
            self.assertIn("core", by_group)
            self.assertIn("auxiliary", by_group)
            counts = metrics["counts"]
            self.assertEqual(counts["by_group"]["core"], 15)
            self.assertEqual(counts["by_group"]["auxiliary"], 4)

            # Validate language breakdowns
            by_lang = metrics["by_language"]
            self.assertIn("en", by_lang)
            self.assertIn("hi", by_lang)
            self.assertEqual(counts["by_language"]["en"], 14)
            self.assertEqual(counts["by_language"]["hi"], 5)

            # Validate per-query results
            per_query = result["per_query_results"]
            self.assertEqual(len(per_query), 19)
            for q in per_query:
                self.assertIn("question_id", q)
                self.assertIn("latency_ms", q)
                self.assertIn("retrieved", q)
                self.assertIn("metrics", q)
                self.assertLessEqual(len(q["retrieved"]), 10)
                self.assertGreaterEqual(len(q["retrieved"]), 0)

            # Validate table formatting
            table_str = format_summary_table(result)
            self.assertIn("NIYAMSETU RETRIEVAL BENCHMARK REPORT: BM25", table_str)
            self.assertIn("Recall@1", table_str)
            self.assertIn("NDCG@10", table_str)

    def test_dense_end_to_end_with_injected_mock_embedder(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = BenchmarkConfig(
                experiment="clause_aware",
                worksheet_path=DEFAULT_WORKSHEET_PATH,
                results_dir=Path(tmpdir),
                k_values=(1, 3, 5, 10),
                top_k=10,
            )
            mock_emb = MockEmbedder(dimension=32)

            result, saved_file = run_benchmark(
                config=cfg,
                strategy="dense",
                embedder_instance=mock_emb,
                save_results=True,
            )

            self.assertIsNotNone(saved_file)
            self.assertTrue(saved_file.exists())
            meta = result["benchmark_metadata"]
            self.assertEqual(meta["strategy"], "dense")
            self.assertEqual(meta["corpus_size"], 363)
            self.assertEqual(meta["total_questions_evaluated"], 19)

    def test_paragraph_experiment_raises_when_no_annotated_questions(self) -> None:
        cfg = BenchmarkConfig(
            experiment="paragraph",
            worksheet_path=DEFAULT_WORKSHEET_PATH,
        )
        with self.assertRaises(ValueError) as ctx:
            run_benchmark(config=cfg, strategy="bm25")
        self.assertIn("No ground-truth questions found matching experiment 'paragraph'", str(ctx.exception))


class TestCLI(unittest.TestCase):
    """Test benchmark runner CLI options."""

    def test_cli_help(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            main(["--help"])
        self.assertEqual(ctx.exception.code, 0)

    def test_cli_bm25_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            code = main([
                "--strategy", "bm25",
                "--results-dir", tmpdir,
                "--top-k", "5",
            ])
            self.assertEqual(code, 0)
            json_files = list(Path(tmpdir).glob("*.json"))
            self.assertEqual(len(json_files), 1)

    def test_cli_dense_missing_dependencies_reports_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Running dense CLI without torch/sentence_transformers should cleanly catch ImportError
            # and exit with code 1 without unhandled stacktrace
            with patch.dict("sys.modules", {"sentence_transformers": None}):
                code = main([
                    "--strategy", "dense",
                    "--results-dir", tmpdir,
                ])
                self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
