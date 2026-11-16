"""Evaluation harness and retrieval benchmark scorers for NiyamSetu RAG."""

from typing import TYPE_CHECKING

from rag_module.evaluation.corpus_loader import (
    CorpusStats,
    experiment_a_corpus,
    experiment_b_corpus,
    load_corpus,
    load_default_corpus,
)
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

if TYPE_CHECKING:
    from rag_module.evaluation.benchmark_runner import (
        BenchmarkConfig,
        format_summary_table,
        run_benchmark,
        run_single_retriever_benchmark,
    )


def __getattr__(name: str):
    if name in {
        "BenchmarkConfig",
        "run_benchmark",
        "run_single_retriever_benchmark",
        "format_summary_table",
    }:
        import rag_module.evaluation.benchmark_runner as runner

        return getattr(runner, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "load_corpus",
    "load_default_corpus",
    "experiment_a_corpus",
    "experiment_b_corpus",
    "CorpusStats",
    "recall_at_k",
    "reciprocal_rank",
    "dcg_at_k",
    "ndcg_at_k",
    "evaluate_query",
    "evaluate_benchmark",
    "load_ground_truth",
    "QueryResult",
    "EvaluationSummary",
    "BenchmarkConfig",
    "run_benchmark",
    "run_single_retriever_benchmark",
    "format_summary_table",
]
