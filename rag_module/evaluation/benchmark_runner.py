"""Benchmark runner comparing BM25 against DenseRetriever for NiyamSetu RAG.

This module provides a reproducible, isolated benchmark harness that:
1. Loads human-verified evaluation questions from annotation_worksheet.jsonl.
2. Preserves core and auxiliary dataset groups and language splits (en/hi).
3. Evaluates Experiment A (363 clause-aware segments) and Experiment B (138 paragraph
   segments) strictly separately, never combining their corpora.
4. Executes BM25Retriever and DenseRetriever independently against the selected corpus.
5. Computes standard IR metrics: Recall@1, @3, @5, @10, MRR, NDCG@5, and NDCG@10.
6. Reports macro-averaged results overall and broken down by group and language.
7. Records per-query rankings, scores, latency (ms), and metric values.
8. Records benchmark metadata: model, device, corpus size, configuration, and ISO timestamp.
9. Persists JSON benchmark reports to rag_module/evaluation/results/ without overwriting.
10. Handles missing dependencies (PyTorch/sentence-transformers) cleanly and explicitly;
    never returns silent mock/fake embeddings as real dense evaluation results.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from rag_module.core.interfaces import BaseEmbedder, BaseRetriever
from rag_module.core.models import ClauseSegment, RetrievedPassage
from rag_module.embeddings.sentence_transformer import (
    DEFAULT_E5_MODEL,
    SentenceTransformerEmbedder,
)
from rag_module.evaluation.corpus_loader import (
    experiment_a_corpus,
    experiment_b_corpus,
    load_corpus,
    load_default_corpus,
)
from rag_module.evaluation.metrics import (
    EvaluationSummary,
    QueryResult,
    evaluate_benchmark,
    load_ground_truth,
)
from rag_module.retrieval.bm25_retriever import BM25Retriever
from rag_module.retrieval.dense_retriever import DenseRetriever
from rag_module.retrieval.hybrid_retriever import (
    DEFAULT_CANDIDATE_DEPTH,
    DEFAULT_RRF_K,
    HybridRetriever,
)

logger = logging.getLogger(__name__)

# Default file paths relative to evaluation package
_EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_WORKSHEET_PATH = _EVAL_DIR / "annotation_worksheet.jsonl"
DEFAULT_DATA_DIR = _EVAL_DIR / "data"
DEFAULT_RESULTS_DIR = _EVAL_DIR / "results"
DEFAULT_CORPUS_JSONL = DEFAULT_DATA_DIR / "niyamsetu_segments_501.jsonl"
DEFAULT_MANIFEST_JSON = DEFAULT_DATA_DIR / "niyamsetu_segments_export_manifest.json"


@dataclass
class BenchmarkConfig:
    """Configuration for retrieval benchmark execution."""

    experiment: str = "clause_aware"
    worksheet_path: Path = DEFAULT_WORKSHEET_PATH
    corpus_jsonl_path: Optional[Path] = None
    manifest_path: Optional[Path] = None
    results_dir: Path = DEFAULT_RESULTS_DIR
    k_values: Tuple[int, ...] = (1, 3, 5, 10)
    top_k: int = 10
    model_name: str = DEFAULT_E5_MODEL
    device: str = "cpu"
    batch_size: int = 32
    use_verified: bool = True
    #: Reciprocal Rank Fusion constant, used only by the hybrid strategy.
    rrf_k: int = DEFAULT_RRF_K
    #: How deep each arm retrieves before fusion. Independent of top_k, so
    #: fusing cannot be starved of the candidates it exists to recover.
    candidate_depth: int = DEFAULT_CANDIDATE_DEPTH

    def __post_init__(self) -> None:
        self.worksheet_path = Path(self.worksheet_path)
        self.results_dir = Path(self.results_dir)
        if self.corpus_jsonl_path is not None:
            self.corpus_jsonl_path = Path(self.corpus_jsonl_path)
        if self.manifest_path is not None:
            self.manifest_path = Path(self.manifest_path)

        valid_experiments = {"clause_aware", "paragraph"}
        if self.experiment not in valid_experiments:
            raise ValueError(
                f"Invalid experiment {self.experiment!r}. Must be one of {valid_experiments}."
            )
        if self.top_k < max(self.k_values, default=1):
            self.k_values = tuple(k for k in self.k_values if k <= self.top_k)
            if not self.k_values:
                self.k_values = (self.top_k,)


def load_evaluation_corpus(
    config: BenchmarkConfig,
) -> Tuple[List[ClauseSegment], int]:
    """Load and isolate the corpus segments for the configured experiment.

    Enforces strict separation between Experiment A (clause-aware) and
    Experiment B (paragraph). Never combines corpora.

    Returns:
        Tuple of (isolated_segments, total_exported_records).
    """
    if config.corpus_jsonl_path and config.manifest_path:
        raw_segments, stats = load_corpus(
            jsonl_path=config.corpus_jsonl_path,
            manifest_path=config.manifest_path,
            skip_sha256=False,
        )
    else:
        raw_segments, stats = load_default_corpus(skip_sha256=False)

    if config.experiment == "clause_aware":
        corpus = experiment_a_corpus(raw_segments)
    elif config.experiment == "paragraph":
        corpus = experiment_b_corpus(raw_segments)
    else:
        raise ValueError(f"Unsupported experiment: {config.experiment}")

    return corpus, stats.total


def generate_result_filepath(
    results_dir: Path,
    strategy: str,
    experiment: str,
    timestamp_str: str,
) -> Path:
    """Generate a non-colliding filepath for benchmark results.

    Ensures that existing result files are never overwritten.
    """
    base_name = f"benchmark_{strategy}_{experiment}_{timestamp_str}"
    target = results_dir / f"{base_name}.json"
    counter = 1
    while target.exists():
        target = results_dir / f"{base_name}_{counter}.json"
        counter += 1
    return target


def create_retriever(
    strategy: str,
    config: BenchmarkConfig,
    embedder: Optional[BaseEmbedder] = None,
    retriever_instance: Optional[BaseRetriever] = None,
) -> BaseRetriever:
    """Instantiate a retriever according to strategy and configuration.

    Supports dependency injection via `retriever_instance` or `embedder`
    for unit testing without downloading models or requiring PyTorch.
    """
    if retriever_instance is not None:
        return retriever_instance

    normalized = strategy.lower().strip()
    if normalized == "bm25":
        return BM25Retriever()
    elif normalized == "dense":
        if embedder is not None:
            return DenseRetriever(embedder=embedder)
        # Attempt to instantiate production SentenceTransformerEmbedder
        # If PyTorch or sentence-transformers is missing, this will fail cleanly
        # when indexing or loading weights.
        dense_embedder = SentenceTransformerEmbedder(
            model_name_or_path=config.model_name,
            device=config.device,
            batch_size=config.batch_size,
        )
        return DenseRetriever(embedder=dense_embedder)
    elif normalized == "hybrid":
        # Hybrid is BM25 and dense fused by Reciprocal Rank Fusion. It is
        # benchmarked through exactly the same harness as its two components,
        # on the same corpus and the same 19 questions, because a fusion that
        # is only ever measured on its own terms proves nothing about whether
        # fusing helped.
        if embedder is None:
            embedder = SentenceTransformerEmbedder(
                model_name_or_path=config.model_name,
                device=config.device,
                batch_size=config.batch_size,
            )
        return HybridRetriever(
            embedder=embedder,
            rrf_k=getattr(config, "rrf_k", DEFAULT_RRF_K),
            candidate_depth=getattr(config, "candidate_depth", DEFAULT_CANDIDATE_DEPTH),
        )
    else:
        raise ValueError(
            f"Unknown retrieval strategy: {strategy!r}. "
            "Supported: 'bm25', 'dense', 'hybrid'."
        )


def run_single_retriever_benchmark(
    strategy: str,
    retriever: BaseRetriever,
    corpus: List[ClauseSegment],
    ground_truth: Dict[str, Dict[str, Any]],
    config: BenchmarkConfig,
    metadata_extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute evaluation queries for a single retriever and compute IR metrics.

    Args:
        strategy: Name of retrieval strategy ('bm25' or 'dense').
        retriever: Indexed or unindexed BaseRetriever instance.
        corpus: Experiment-isolated segment corpus.
        ground_truth: Ground-truth annotations mapping question_id -> record.
        config: Benchmark configuration.
        metadata_extra: Optional extra metadata dictionary.

    Returns:
        Structured benchmark result dictionary.
    """
    start_index = time.perf_counter()
    logger.info("Indexing %d segments with %s retriever...", len(corpus), strategy)
    retriever.index(corpus)
    index_latency_ms = (time.perf_counter() - start_index) * 1000.0

    predictions: Dict[str, List[RetrievedPassage]] = {}
    query_details: List[Dict[str, Any]] = []

    for qid, gt_entry in ground_truth.items():
        query_text = gt_entry["question"]
        t0 = time.perf_counter()
        passages = retriever.retrieve(query=query_text, top_k=config.top_k)
        t_elapsed_ms = (time.perf_counter() - t0) * 1000.0

        predictions[qid] = passages

        ranked_passages = [
            {
                "rank": idx,
                "clause_id": p.clause_id,
                "score": float(p.score),
                "document_id": p.document_id,
            }
            for idx, p in enumerate(passages, start=1)
        ]

        query_details.append(
            {
                "question_id": qid,
                "question": query_text,
                "language": gt_entry.get("language", "en"),
                "dataset_group": gt_entry.get("dataset_group", "core"),
                "latency_ms": round(t_elapsed_ms, 3),
                "relevant_clause_ids": gt_entry.get("relevant_clause_ids", []),
                "retrieved": ranked_passages,
            }
        )

    # Compute IR metrics using authoritative pure-Python metrics module
    summary = evaluate_benchmark(
        predictions=predictions,
        ground_truth=ground_truth,
        k_values=config.k_values,
        use_verified=config.use_verified,
    )

    # Merge per-query metric scores into query details
    per_query_map = {res.question_id: res.metrics for res in summary.per_query}
    for q_detail in query_details:
        qid = q_detail["question_id"]
        q_detail["metrics"] = per_query_map.get(qid, {})

    # Model and configuration metadata
    model_name = "None (Lexical BM25Okapi Lucene-IDF)"
    device = "cpu"
    if strategy.lower() == "hybrid":
        dense_arm = getattr(retriever, "dense", None)
        embedder_obj = getattr(dense_arm, "embedder", None) if dense_arm else None
        model_name = (
            getattr(embedder_obj, "model_name_or_path", str(embedder_obj))
            if embedder_obj is not None
            else "None (lexical only - no embedder available)"
        )
        device = getattr(embedder_obj, "device", config.device) if embedder_obj else "cpu"
    elif strategy.lower() == "dense":
        model_name = config.model_name
        device = config.device
        if hasattr(retriever, "embedder") and retriever.embedder is not None:
            embedder = retriever.embedder
            model_name = getattr(embedder, "model_name_or_path", str(embedder))
            device = getattr(embedder, "device", config.device)

    benchmark_ts = datetime.now(timezone.utc).isoformat()

    retrieval_config: Dict[str, Any] = {
        "k_values": list(config.k_values),
        "top_k": config.top_k,
    }
    if strategy.lower() == "bm25":
        retrieval_config["k1"] = getattr(retriever, "k1", 1.5)
        retrieval_config["b"] = getattr(retriever, "b", 0.75)
        retrieval_config["tokenizer"] = "BM25Tokenizer (legal regex, Devanagari, clause-aware)"
    elif strategy.lower() == "dense":
        retrieval_config["batch_size"] = config.batch_size
        retrieval_config["similarity"] = "cosine"
        retrieval_config["normalize_embeddings"] = True
    elif strategy.lower() == "hybrid":
        retrieval_config["fusion"] = "reciprocal_rank_fusion"
        retrieval_config["rrf_k"] = getattr(retriever, "rrf_k", DEFAULT_RRF_K)
        retrieval_config["candidate_depth"] = getattr(
            retriever, "candidate_depth", DEFAULT_CANDIDATE_DEPTH
        )
        retrieval_config["components"] = list(
            getattr(retriever, "active_components", ["bm25"])
        )
        retrieval_config["batch_size"] = config.batch_size

    if metadata_extra:
        retrieval_config.update(metadata_extra)

    # Average query latency
    total_query_lat = sum(q["latency_ms"] for q in query_details)
    avg_latency_ms = round(total_query_lat / len(query_details), 3) if query_details else 0.0

    result_payload: Dict[str, Any] = {
        "benchmark_metadata": {
            "benchmark_timestamp": benchmark_ts,
            "strategy": strategy.lower(),
            "experiment": config.experiment,
            "corpus_size": len(corpus),
            "model_name": model_name,
            "device": device,
            "retrieval_configuration": retrieval_config,
            "index_latency_ms": round(index_latency_ms, 3),
            "avg_query_latency_ms": avg_latency_ms,
            "ground_truth_file": str(config.worksheet_path.name),
            "total_questions_evaluated": summary.question_count,
        },
        "metrics": {
            "overall": summary.overall,
            "by_group": summary.by_group,
            "by_language": summary.by_language,
            "counts": {
                "total": summary.question_count,
                "by_group": summary.group_counts,
                "by_language": summary.language_counts,
            },
        },
        "per_query_results": query_details,
    }

    return result_payload


def run_benchmark(
    config: BenchmarkConfig,
    strategy: str = "bm25",
    retriever_instance: Optional[BaseRetriever] = None,
    embedder_instance: Optional[BaseEmbedder] = None,
    save_results: bool = True,
) -> Tuple[Dict[str, Any], Optional[Path]]:
    """Run a complete retrieval benchmark according to strategy and configuration.

    Args:
        config: Benchmark configuration.
        strategy: 'bm25', 'dense' or 'hybrid'.
        retriever_instance: Optional pre-constructed BaseRetriever.
        embedder_instance: Optional pre-constructed BaseEmbedder (for dense testing).
        save_results: Whether to persist result JSON to config.results_dir.

    Returns:
        Tuple of (result_payload, saved_filepath_or_none).

    Raises:
        FileNotFoundError: If corpus or worksheet files are missing.
        ValueError: If experiment has no matching ground-truth questions.
        ImportError: If dense retrieval dependencies are missing.
        RuntimeError: If model loading or dense indexing fails.
    """
    strategy_norm = strategy.lower().strip()
    if strategy_norm not in {"bm25", "dense", "hybrid"}:
        raise ValueError(
            f"Invalid strategy: {strategy!r}. Must be 'bm25', 'dense' or 'hybrid'."
        )

    # 1. Load experiment-isolated corpus
    corpus, _ = load_evaluation_corpus(config)
    if not corpus:
        raise ValueError(
            f"Corpus for experiment {config.experiment!r} is empty. Cannot evaluate."
        )

    # 2. Load ground-truth annotations
    all_gt = load_ground_truth(config.worksheet_path, use_verified=config.use_verified)
    gt_for_experiment = {
        qid: rec
        for qid, rec in all_gt.items()
        if rec.get("experiment", "clause_aware") == config.experiment
    }

    if not gt_for_experiment:
        raise ValueError(
            f"No ground-truth questions found matching experiment {config.experiment!r} "
            f"in {config.worksheet_path}."
        )

    # 3. Instantiate retriever
    retriever = create_retriever(
        strategy=strategy_norm,
        config=config,
        embedder=embedder_instance,
        retriever_instance=retriever_instance,
    )

    # 4. Execute benchmark
    result = run_single_retriever_benchmark(
        strategy=strategy_norm,
        retriever=retriever,
        corpus=corpus,
        ground_truth=gt_for_experiment,
        config=config,
    )

    # 5. Persist results
    saved_path: Optional[Path] = None
    if save_results:
        config.results_dir.mkdir(parents=True, exist_ok=True)
        ts_slug = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        saved_path = generate_result_filepath(
            results_dir=config.results_dir,
            strategy=strategy_norm,
            experiment=config.experiment,
            timestamp_str=ts_slug,
        )
        with open(saved_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, ensure_ascii=False)
        logger.info("Saved benchmark results to: %s", saved_path)

    return result, saved_path


def format_summary_table(result: Dict[str, Any]) -> str:
    """Format benchmark evaluation summary into a human-readable table."""
    meta = result.get("benchmark_metadata", {})
    metrics = result.get("metrics", {})
    overall = metrics.get("overall", {})
    by_group = metrics.get("by_group", {})
    by_lang = metrics.get("by_language", {})
    counts = metrics.get("counts", {})

    lines: List[str] = [
        "================================================================================",
        f" NIYAMSETU RETRIEVAL BENCHMARK REPORT: {meta.get('strategy', '').upper()}",
        "================================================================================",
        f" Timestamp:    {meta.get('benchmark_timestamp')}",
        f" Strategy:     {meta.get('strategy')}",
        f" Experiment:   {meta.get('experiment')} (Corpus size: {meta.get('corpus_size')} segments)",
        f" Model:        {meta.get('model_name')}",
        f" Device:       {meta.get('device')}",
        f" Avg Latency:  {meta.get('avg_query_latency_ms')} ms/query",
        f" Index Time:   {meta.get('index_latency_ms')} ms",
        f" Questions:    {meta.get('total_questions_evaluated')} total "
        f"({counts.get('by_group', {}).get('core', 0)} core, {counts.get('by_group', {}).get('auxiliary', 0)} aux; "
        f"{counts.get('by_language', {}).get('en', 0)} EN, {counts.get('by_language', {}).get('hi', 0)} HI)",
        "--------------------------------------------------------------------------------",
        f"{'Metric':<14} | {'Overall':<10} | {'Core':<10} | {'Auxiliary':<10} | {'English':<10} | {'Hindi':<10}",
        "--------------------------------------------------------------------------------",
    ]

    metric_keys = [
        ("Recall@1", "recall@1"),
        ("Recall@3", "recall@3"),
        ("Recall@5", "recall@5"),
        ("Recall@10", "recall@10"),
        ("MRR", "mrr"),
        ("NDCG@5", "ndcg@5"),
        ("NDCG@10", "ndcg@10"),
    ]

    core_m = by_group.get("core", {})
    aux_m = by_group.get("auxiliary", {})
    en_m = by_lang.get("en", {})
    hi_m = by_lang.get("hi", {})

    for label, key in metric_keys:
        ov = f"{overall.get(key, 0.0):.4f}" if key in overall else "N/A"
        co = f"{core_m.get(key, 0.0):.4f}" if key in core_m else "N/A"
        au = f"{aux_m.get(key, 0.0):.4f}" if key in aux_m else "N/A"
        en = f"{en_m.get(key, 0.0):.4f}" if key in en_m else "N/A"
        hi = f"{hi_m.get(key, 0.0):.4f}" if key in hi_m else "N/A"
        lines.append(f"{label:<14} | {ov:<10} | {co:<10} | {au:<10} | {en:<10} | {hi:<10}")

    lines.append("================================================================================")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry-point for benchmark runner."""
    parser = argparse.ArgumentParser(
        description="Run NiyamSetu RAG Retrieval Benchmark (BM25 vs DenseRetriever)."
    )
    parser.add_argument(
        "--strategy",
        choices=["bm25", "dense", "hybrid", "all"],
        default="bm25",
        help="Retrieval strategy to benchmark (default: bm25).",
    )
    parser.add_argument(
        "--experiment",
        choices=["clause_aware", "paragraph"],
        default="clause_aware",
        help="Experiment corpus: 'clause_aware' (Exp A) or 'paragraph' (Exp B) (default: clause_aware).",
    )
    parser.add_argument(
        "--worksheet",
        type=Path,
        default=DEFAULT_WORKSHEET_PATH,
        help=f"Path to annotation worksheet (default: {DEFAULT_WORKSHEET_PATH}).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=f"Directory to save JSON results (default: {DEFAULT_RESULTS_DIR}).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_E5_MODEL,
        help=f"Dense embedding model identifier (default: {DEFAULT_E5_MODEL}).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Execution device for dense retrieval (default: cpu).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top passages to retrieve per query (default: 10).",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = BenchmarkConfig(
        experiment=args.experiment,
        worksheet_path=args.worksheet,
        results_dir=args.results_dir,
        model_name=args.model_name,
        device=args.device,
        top_k=args.top_k,
    )

    strategies_to_run = (
        ["bm25", "dense", "hybrid"] if args.strategy == "all" else [args.strategy]
    )

    exit_code = 0
    for strat in strategies_to_run:
        print(f"\n>>> Running {strat.upper()} benchmark (Experiment: {config.experiment})...")
        try:
            result, saved_file = run_benchmark(config=config, strategy=strat, save_results=True)
            print(format_summary_table(result))
            if saved_file:
                print(f"Results successfully saved to: {saved_file}")
        except ImportError as exc:
            print(f"\n[ERROR] Dense retrieval dependencies missing: {exc}", file=sys.stderr)
            print(
                "\nTo benchmark DenseRetriever with real multilingual embeddings, install prerequisites:",
                file=sys.stderr,
            )
            print("  pip install sentence-transformers torch\n", file=sys.stderr)
            exit_code = 1
        except Exception as exc:
            print(f"\n[ERROR] Benchmark execution failed for {strat}: {exc}", file=sys.stderr)
            exit_code = 1

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
