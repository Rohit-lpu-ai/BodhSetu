"""Vector validation and prefix utilities for dense embeddings.

Provides reusable checks for:
- Consistent vector dimensions
- Finite numeric values (no NaN or Inf)
- Correct L2 normalization
- Handling of zero-norm vectors
- E5 asymmetric prefix formatting (query: and passage:) without duplication
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "


def format_query_prefix(query: str) -> str:
    """Prepend 'query: ' prefix for E5 asymmetric embedding if not already present.

    Leading and trailing whitespace is stripped. If the text already begins with
    'query: ' (case-insensitive), no duplicate prefix is added. The final prefix
    is always normalized to lowercase 'query: '.
    """
    if not isinstance(query, str):
        raise TypeError(
            f"format_query_prefix() expects a str, got {type(query).__name__}"
        )
    q = query.strip()
    if q.lower().startswith(QUERY_PREFIX):
        return QUERY_PREFIX + q[len(QUERY_PREFIX):]
    return f"{QUERY_PREFIX}{q}"


def format_passage_prefix(text: str) -> str:
    """Prepend 'passage: ' prefix for E5 asymmetric embedding if not already present.

    Leading and trailing whitespace is stripped. If the text already begins with
    'passage: ' (case-insensitive), no duplicate prefix is added. The final prefix
    is always normalized to lowercase 'passage: '.
    """
    if not isinstance(text, str):
        raise TypeError(
            f"format_passage_prefix() expects a str, got {type(text).__name__}"
        )
    t = text.strip()
    if t.lower().startswith(PASSAGE_PREFIX):
        return PASSAGE_PREFIX + t[len(PASSAGE_PREFIX):]
    return f"{PASSAGE_PREFIX}{t}"


def is_finite_number(val: float) -> bool:
    """Return True if val is a finite real number (not NaN, not Inf)."""
    return isinstance(val, (int, float)) and not math.isnan(val) and not math.isinf(val)


def vector_norm(vector: Sequence[float]) -> float:
    """Calculate the Euclidean L2 norm of a vector."""
    return math.sqrt(sum(float(x) * float(x) for x in vector))


def normalize_l2(vector: Sequence[float], eps: float = 1e-12) -> List[float]:
    """Return the unit L2-normalized version of a vector.

    Args:
        vector: Sequence of numeric values.
        eps: Minimum norm threshold to avoid division by zero.

    Returns:
        List of floats with ||v||_2 == 1.0.

    Raises:
        ValueError: If vector has zero or near-zero norm (< eps).
    """
    norm = vector_norm(vector)
    if norm < eps:
        raise ValueError(
            f"Cannot normalize zero-norm vector (L2 norm = {norm:.6e} < {eps:.6e})"
        )
    return [float(x / norm) for x in vector]


def validate_vector(
    vector: Sequence[float],
    expected_dim: Optional[int] = None,
    check_normalized: bool = False,
    norm_tol: float = 1e-3,
) -> None:
    """Validate a single dense vector.

    Checks:
    1. Vector is a sequence of numbers.
    2. Dimension matches expected_dim (if provided).
    3. Every component is a finite numeric value (no NaN, Inf).
    4. If check_normalized is True, asserts ||v||_2 == 1.0 within norm_tol.

    Raises:
        ValueError: On dimension mismatch, non-finite values, or normalization failure.
        TypeError: If vector is not a sequence.
    """
    if not isinstance(vector, (list, tuple)):
        raise TypeError(
            f"Vector must be a sequence of floats, got {type(vector).__name__}"
        )

    if expected_dim is not None and len(vector) != expected_dim:
        raise ValueError(
            f"Vector dimension mismatch: expected {expected_dim}, got {len(vector)}"
        )

    for i, x in enumerate(vector):
        if not is_finite_number(x):
            raise ValueError(f"Non-finite value at index {i}: {x}")

    if check_normalized:
        norm = vector_norm(vector)
        if norm == 0.0:
            raise ValueError("Zero-norm vector cannot be unit-normalized")
        if abs(norm - 1.0) > norm_tol:
            raise ValueError(
                f"Vector is not L2-normalized: ||v||_2 = {norm:.6f} (expected 1.0 ± {norm_tol})"
            )


def validate_vectors(
    vectors: Sequence[Sequence[float]],
    expected_dim: Optional[int] = None,
    check_normalized: bool = False,
    norm_tol: float = 1e-3,
) -> None:
    """Validate a batch of dense vectors.

    Checks:
    1. Empty batch is valid (no-op).
    2. All vectors have identical dimension (matching expected_dim if provided).
    3. Every vector passes validate_vector.

    Raises:
        ValueError: If any vector fails validation or batch has inconsistent dimensions.
    """
    if not vectors:
        return

    dim = expected_dim if expected_dim is not None else len(vectors[0])
    for idx, vec in enumerate(vectors):
        try:
            validate_vector(
                vec,
                expected_dim=dim,
                check_normalized=check_normalized,
                norm_tol=norm_tol,
            )
        except (ValueError, TypeError) as err:
            raise ValueError(f"Vector at index {idx} failed validation: {err}") from err
