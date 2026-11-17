"""Pure-Python retrieval evaluation metrics for the NiyamSetu RAG subsystem.

Provides deterministic, implementation-independent metric implementations:
- Recall@K: fraction of authoritative relevant clauses retrieved in top K.
- Reciprocal Rank (MRR): reciprocal rank of the first relevant clause.
- NDCG@K: Normalized Discounted Cumulative Gain with binary relevance and
  standard logarithmic discounting (log2(i + 1)).
- Aggregate metrics: macro-averaging across queries with subgroup breakdown
  by dataset_group (core / auxiliary) and language (en / hi).

This module relies exclusively on the Python standard library.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Collection, Dict, List, Optional, Sequence, Set, Tuple, Union

# ---------------------------------------------------------------------------
# Input extraction helpers
# ---------------------------------------------------------------------------


def _extract_clause_id(item: Any) -> str:
    """Extract a string clause ID from a string, dataclass, or dictionary."""
    if isinstance(item, str):
        cid = item.strip()
        if not cid:
            raise ValueError("Retrieved clause ID string cannot be empty or whitespace.")
        return cid
    if hasattr(item, "clause_id"):
        cid = getattr(item, "clause_id")
        if not isinstance(cid, str) or not cid.strip():
            raise ValueError(f"Object {item!r} has empty or non-string clause_id.")
        return cid.strip()
    if isinstance(item, dict):
        if "clause_id" in item:
            cid = item["clause_id"]
            if not isinstance(cid, str) or not cid.strip():
                raise ValueError(f"Dict {item!r} has empty or non-string 'clause_id'.")
            return cid.strip()
    raise TypeError(
        f"Cannot extract clause_id from item of type {type(item).__name__}: {item!r}. "
        "Expected str, object with .clause_id (e.g. RetrievedPassage), or dict with 'clause_id'."
    )


def _deduplicate_ordered(items: Sequence[Any]) -> List[str]:
    """Convert sequence of retrieved items to deduplicated clause IDs preserving order."""
    seen: Set[str] = set()
    result: List[str] = []
    for raw in items:
        cid = _extract_clause_id(raw)
        if cid not in seen:
            seen.add(cid)
            result.append(cid)
    return result


def _validate_k(k: int) -> None:
    """Validate that K is an integer >= 1."""
    if not isinstance(k, int) or isinstance(k, bool):
        raise TypeError(f"K must be an integer, got {type(k).__name__}: {k!r}")
    if k < 1:
        raise ValueError(f"K must be >= 1, got {k}")


def _validate_relevant(relevant: Collection[str], query_id: Optional[str] = None) -> Set[str]:
    """Validate ground-truth relevant clause IDs and return as a set."""
    if relevant is None:
        ctx = f" for question {query_id!r}" if query_id else ""
        raise ValueError(f"Relevant clause IDs cannot be None{ctx}.")
    if not isinstance(relevant, (set, list, tuple)):
        ctx = f" for question {query_id!r}" if query_id else ""
        raise TypeError(
            f"Relevant clause IDs must be a collection of strings{ctx}, got {type(relevant).__name__}"
        )
    rel_set: Set[str] = set()
    for item in relevant:
        cid = _extract_clause_id(item)
        rel_set.add(cid)
    if not rel_set:
        ctx = f" for question {query_id!r}" if query_id else ""
        raise ValueError(
            f"Question has empty relevance labels (ground truth must contain at least one relevant clause ID){ctx}."
        )
    return rel_set


# ---------------------------------------------------------------------------
# Core metric functions (single query)
# ---------------------------------------------------------------------------


def recall_at_k(
    retrieved: Sequence[Any],
    relevant: Collection[str],
    k: int,
    query_id: Optional[str] = None,
) -> float:
    """Calculate Recall@K for a single query.

    Recall@K = |{retrieved in top K} ∩ relevant| / |relevant|

    Args:
        retrieved: Ordered sequence of retrieved items or clause IDs.
        relevant: Collection of authoritative ground-truth clause IDs.
        k: Number of top retrieved documents to evaluate (integer >= 1).
        query_id: Optional question ID for descriptive error messages.

    Returns:
        Float value between 0.0 and 1.0.

    Raises:
        ValueError: If k < 1 or relevant collection is empty.
        TypeError: If k or relevant collection has invalid types.
    """
    _validate_k(k)
    rel_set = _validate_relevant(relevant, query_id=query_id)
    if not retrieved:
        return 0.0

    retrieved_cids = _deduplicate_ordered(retrieved)[:k]
    matched = sum(1 for cid in retrieved_cids if cid in rel_set)
    return matched / len(rel_set)


def reciprocal_rank(
    retrieved: Sequence[Any],
    relevant: Collection[str],
    k: Optional[int] = None,
    query_id: Optional[str] = None,
) -> float:
    """Calculate Reciprocal Rank (RR) for a single query.

    RR = 1 / rank of the first relevant retrieved clause, or 0.0 if none found.

    Args:
        retrieved: Ordered sequence of retrieved items or clause IDs.
        relevant: Collection of authoritative ground-truth clause IDs.
        k: Optional cutoff; if specified, only top-K items are evaluated.
        query_id: Optional question ID for descriptive error messages.

    Returns:
        Float value (e.g. 1.0, 0.5, 0.333..., or 0.0).

    Raises:
        ValueError: If k is provided and < 1, or relevant collection is empty.
        TypeError: If relevant collection or k has invalid types.
    """
    if k is not None:
        _validate_k(k)
    rel_set = _validate_relevant(relevant, query_id=query_id)
    if not retrieved:
        return 0.0

    retrieved_cids = _deduplicate_ordered(retrieved)
    if k is not None:
        retrieved_cids = retrieved_cids[:k]

    for rank, cid in enumerate(retrieved_cids, start=1):
        if cid in rel_set:
            return 1.0 / rank
    return 0.0


def dcg_at_k(
    retrieved: Sequence[Any],
    relevant: Collection[str],
    k: int,
    query_id: Optional[str] = None,
) -> float:
    """Calculate Discounted Cumulative Gain (DCG@K) using standard log2 discounting.

    DCG@K = sum_{i=1}^K rel_i / log2(i + 1)
    Binary relevance: rel_i = 1 if retrieved clause is relevant, else 0.

    Args:
        retrieved: Ordered sequence of retrieved items or clause IDs.
        relevant: Collection of authoritative ground-truth clause IDs.
        k: Cutoff rank (integer >= 1).
        query_id: Optional question ID for descriptive error messages.

    Returns:
        Float DCG score >= 0.0.
    """
    _validate_k(k)
    rel_set = _validate_relevant(relevant, query_id=query_id)
    if not retrieved:
        return 0.0

    retrieved_cids = _deduplicate_ordered(retrieved)[:k]
    dcg = 0.0
    for i, cid in enumerate(retrieved_cids, start=1):
        if cid in rel_set:
            dcg += 1.0 / math.log2(i + 1)
    return dcg


def ndcg_at_k(
    retrieved: Sequence[Any],
    relevant: Collection[str],
    k: int,
    query_id: Optional[str] = None,
) -> float:
    """Calculate Normalized Discounted Cumulative Gain (NDCG@K) with binary relevance.

    NDCG@K = DCG@K / IDCG@K
    where IDCG@K = sum_{i=1}^{min(k, |relevant|)} 1 / log2(i + 1)

    Args:
        retrieved: Ordered sequence of retrieved items or clause IDs.
        relevant: Collection of authoritative ground-truth clause IDs.
        k: Cutoff rank (integer >= 1).
        query_id: Optional question ID for descriptive error messages.

    Returns:
        Float value between 0.0 and 1.0.

    Raises:
        ValueError: If k < 1 or relevant collection is empty.
    """
    _validate_k(k)
    rel_set = _validate_relevant(relevant, query_id=query_id)
    if not retrieved:
        return 0.0

    actual_dcg = dcg_at_k(retrieved, rel_set, k, query_id=query_id)
    if actual_dcg == 0.0:
        return 0.0

    # Ideal DCG with binary relevance
    ideal_count = min(k, len(rel_set))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_count + 1))
    if idcg == 0.0:
        return 0.0

    return actual_dcg / idcg


def evaluate_query(
    retrieved: Sequence[Any],
    relevant: Collection[str],
    k_values: Sequence[int] = (1, 3, 5, 10),
    query_id: Optional[str] = None,
) -> Dict[str, float]:
    """Compute all standard retrieval metrics for a single query.

    Args:
        retrieved: Ordered sequence of retrieved items or clause IDs.
        relevant: Collection of authoritative ground-truth clause IDs.
        k_values: Cutoff values for Recall and NDCG.
        query_id: Optional question ID for descriptive error messages.

    Returns:
        Dictionary mapping metric names to score values.
        Example keys: 'recall@1', 'recall@3', 'mrr', 'ndcg@3', etc.
    """
    for k in k_values:
        _validate_k(k)
    rel_set = _validate_relevant(relevant, query_id=query_id)

    metrics: Dict[str, float] = {}
    for k in k_values:
        metrics[f"recall@{k}"] = recall_at_k(retrieved, rel_set, k, query_id=query_id)

    metrics["mrr"] = reciprocal_rank(retrieved, rel_set, query_id=query_id)

    for k in k_values:
        metrics[f"ndcg@{k}"] = ndcg_at_k(retrieved, rel_set, k, query_id=query_id)

    return metrics


# ---------------------------------------------------------------------------
# Dataclasses for Structured Results
# ---------------------------------------------------------------------------


@dataclass
class QueryResult:
    """Evaluation result for a single question."""

    question_id: str
    metrics: Dict[str, float]
    language: str
    dataset_group: str
    num_retrieved: int
    num_relevant: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationSummary:
    """Macro-averaged evaluation metrics with subgroup breakdowns."""

    overall: Dict[str, float]
    by_group: Dict[str, Dict[str, float]]
    by_language: Dict[str, Dict[str, float]]
    question_count: int
    group_counts: Dict[str, int]
    language_counts: Dict[str, int]
    per_query: List[QueryResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall": self.overall,
            "by_group": self.by_group,
            "by_language": self.by_language,
            "question_count": self.question_count,
            "group_counts": self.group_counts,
            "language_counts": self.language_counts,
            "per_query": [q.to_dict() for q in self.per_query],
        }


# ---------------------------------------------------------------------------
# Ground Truth Loading
# ---------------------------------------------------------------------------


def load_ground_truth(
    source: Union[Path, str, Sequence[Dict[str, Any]]],
    use_verified: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Load ground-truth annotations from a worksheet or dataset file.

    Supports:
    1. annotation_worksheet.jsonl (using human_verified_relevant_clause_ids)
    2. standard benchmark JSONL (using relevant_clause_ids)
    3. In-memory sequence of dictionaries

    Args:
        source: File path (JSONL) or sequence of dictionaries.
        use_verified: If True and human_verified_relevant_clause_ids is present,
            use verified IDs; otherwise use relevant_clause_ids.

    Returns:
        Dict mapping question_id -> record dictionary containing 'relevant_clause_ids',
        'dataset_group', 'language', and 'question'.

    Raises:
        FileNotFoundError: If the source path does not exist.
        ValueError: On duplicate question_id, missing fields, or empty relevance labels.
    """
    records: List[Dict[str, Any]] = []

    if isinstance(source, (str, Path)):
        p = Path(source)
        if not p.exists():
            raise FileNotFoundError(f"Ground truth file not found: {p}")
        with open(p, "r", encoding="utf-8") as fh:
            for line_no, raw_line in enumerate(fh, start=1):
                stripped = raw_line.strip()
                if not stripped:
                    continue
                try:
                    records.append(json.loads(stripped))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"JSON parse error on line {line_no} of {p}: {exc}") from exc
    elif isinstance(source, Sequence):
        records = list(source)
    else:
        raise TypeError(f"Unsupported source type: {type(source).__name__}")

    ground_truth: Dict[str, Dict[str, Any]] = {}

    for idx, rec in enumerate(records, start=1):
        if not isinstance(rec, dict):
            raise TypeError(f"Record {idx} must be a dictionary, got {type(rec).__name__}")

        qid = rec.get("question_id")
        if not qid or not isinstance(qid, str):
            raise ValueError(f"Record {idx} missing valid 'question_id'.")
        if qid in ground_truth:
            raise ValueError(f"Duplicate question_id {qid!r} found in ground truth.")

        # Identify relevant clause IDs
        rel_ids: List[str] = []
        if use_verified and "human_verified_relevant_clause_ids" in rec:
            rel_ids = rec["human_verified_relevant_clause_ids"]
        elif "relevant_clause_ids" in rec:
            rel_ids = rec["relevant_clause_ids"]
        elif "candidate_relevant_clause_ids" in rec and not use_verified:
            rel_ids = rec["candidate_relevant_clause_ids"]
        else:
            raise ValueError(
                f"Record {qid!r} has no recognized relevance field "
                "(expected 'human_verified_relevant_clause_ids' or 'relevant_clause_ids')."
            )

        # Validate non-empty relevance
        validated_set = _validate_relevant(rel_ids, query_id=qid)

        ground_truth[qid] = {
            "question_id": qid,
            "question": rec.get("question", ""),
            "language": rec.get("language", "en"),
            "dataset_group": rec.get("dataset_group", "core"),
            "experiment": rec.get("experiment", "clause_aware"),
            "relevant_clause_ids": sorted(list(validated_set)),
        }

    return ground_truth


# ---------------------------------------------------------------------------
# Benchmark Evaluator & Aggregator
# ---------------------------------------------------------------------------


def _macro_average(query_results: Sequence[QueryResult]) -> Dict[str, float]:
    """Calculate macro-averaged metrics across a sequence of QueryResult objects."""
    if not query_results:
        return {}
    metric_names = list(query_results[0].metrics.keys())
    totals: Dict[str, float] = {m: 0.0 for m in metric_names}
    for qr in query_results:
        for m in metric_names:
            totals[m] += qr.metrics.get(m, 0.0)
    count = len(query_results)
    return {m: totals[m] / count for m in metric_names}


def evaluate_benchmark(
    predictions: Union[Dict[str, Sequence[Any]], Sequence[Dict[str, Any]]],
    ground_truth: Union[Path, str, Dict[str, Dict[str, Any]], Sequence[Dict[str, Any]]],
    k_values: Sequence[int] = (1, 3, 5, 10),
    use_verified: bool = True,
) -> EvaluationSummary:
    """Evaluate retrieval predictions against ground truth and produce macro-averaged metrics.

    Args:
        predictions: Dictionary mapping question_id -> sequence of retrieved items/IDs,
            or sequence of dicts with 'question_id' and 'retrieved'.
        ground_truth: File path (JSONL) or parsed ground truth mapping.
        k_values: Cutoff K values to evaluate.
        use_verified: Use verified IDs if loading from worksheet.

    Returns:
        EvaluationSummary containing overall macro-averages, breakdowns by
        dataset_group (core/auxiliary) and language (en/hi), and per-query details.

    Raises:
        ValueError: On duplicate question IDs, missing ground-truth matches,
            invalid K values, or malformed inputs.
        TypeError: On invalid input types.
    """
    for k in k_values:
        _validate_k(k)

    # 1. Normalize ground truth
    gt: Dict[str, Dict[str, Any]]
    if isinstance(ground_truth, dict):
        gt = {}
        for qid, entry in ground_truth.items():
            if not isinstance(entry, dict):
                raise TypeError(f"Ground truth entry for {qid!r} must be dict.")
            rel = entry.get("relevant_clause_ids") or entry.get("human_verified_relevant_clause_ids")
            validated = _validate_relevant(rel, query_id=qid)
            gt[qid] = {
                "question_id": qid,
                "question": entry.get("question", ""),
                "language": entry.get("language", "en"),
                "dataset_group": entry.get("dataset_group", "core"),
                "relevant_clause_ids": sorted(list(validated)),
            }
    else:
        gt = load_ground_truth(ground_truth, use_verified=use_verified)

    if not gt:
        raise ValueError("Ground truth is empty; cannot evaluate.")

    # 2. Normalize predictions
    pred_map: Dict[str, Sequence[Any]] = {}
    if isinstance(predictions, dict):
        pred_map = dict(predictions)
    elif isinstance(predictions, Sequence):
        for idx, item in enumerate(predictions, start=1):
            if not isinstance(item, dict):
                raise TypeError(f"Prediction item {idx} must be a dict.")
            qid = item.get("question_id")
            if not qid or not isinstance(qid, str):
                raise ValueError(f"Prediction item {idx} missing 'question_id'.")
            if qid in pred_map:
                raise ValueError(f"Duplicate question_id {qid!r} found in predictions.")
            ret = (
                item.get("retrieved")
                if "retrieved" in item
                else item.get("retrieved_passages")
                if "retrieved_passages" in item
                else item.get("retrieved_clause_ids")
            )
            if ret is None:
                raise ValueError(f"Prediction item {qid!r} missing retrieved items field.")
            pred_map[qid] = ret
    else:
        raise TypeError(f"Unsupported predictions type: {type(predictions).__name__}")

    # Check for duplicate keys in dictionary if provided as dict
    # (dict naturally has unique keys in python, but check matching against gt)
    for qid in pred_map:
        if qid not in gt:
            raise ValueError(f"Prediction question_id {qid!r} not found in ground truth.")

    # 3. Score each question
    query_results: List[QueryResult] = []
    group_map: Dict[str, List[QueryResult]] = {}
    lang_map: Dict[str, List[QueryResult]] = {}

    for qid in sorted(gt.keys()):
        gt_entry = gt[qid]
        rel_cids = gt_entry["relevant_clause_ids"]
        retrieved_items = pred_map.get(qid, [])

        q_metrics = evaluate_query(
            retrieved=retrieved_items,
            relevant=rel_cids,
            k_values=k_values,
            query_id=qid,
        )

        qr = QueryResult(
            question_id=qid,
            metrics=q_metrics,
            language=gt_entry["language"],
            dataset_group=gt_entry["dataset_group"],
            num_retrieved=len(retrieved_items),
            num_relevant=len(rel_cids),
        )
        query_results.append(qr)

        # Subgroup tracking
        grp = qr.dataset_group
        group_map.setdefault(grp, []).append(qr)

        lang = qr.language
        lang_map.setdefault(lang, []).append(qr)

    overall_metrics = _macro_average(query_results)
    by_group = {grp: _macro_average(items) for grp, items in sorted(group_map.items())}
    by_lang = {lang: _macro_average(items) for lang, items in sorted(lang_map.items())}

    group_counts = {grp: len(items) for grp, items in sorted(group_map.items())}
    lang_counts = {lang: len(items) for lang, items in sorted(lang_map.items())}

    return EvaluationSummary(
        overall=overall_metrics,
        by_group=by_group,
        by_language=by_lang,
        question_count=len(query_results),
        group_counts=group_counts,
        language_counts=lang_counts,
        per_query=query_results,
    )
