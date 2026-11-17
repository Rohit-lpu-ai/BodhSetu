"""Hybrid retrieval: Reciprocal Rank Fusion of BM25 and dense rankings.

Why fuse *ranks* and not scores
-------------------------------
BM25 produces unbounded Lucene scores; the dense retriever produces cosine
similarities in ``[-1, 1]``. They have no common unit, no common scale and no
common distribution, so any weighted sum of the two raw numbers is arbitrary —
change the corpus and the implied weighting changes with it. Reciprocal Rank
Fusion sidesteps the problem entirely by throwing the magnitudes away and
keeping only the ordering each retriever produced:

.. math::

    RRF(d) = \\sum_{i} \\frac{1}{k + \\mathrm{rank}_i(d)}

with ``k = 60`` (Cormack, Clarke & Buettcher 2009, and the value
``Settings.rrf_k`` already defaulted to). ``k`` damps the contribution of the
very top ranks so that a document ranked 1st by one retriever and absent from
the other does not automatically beat a document ranked 2nd and 3rd by both —
which is the behaviour that makes RRF robust when one retriever is wrong.

What this class guarantees
--------------------------
* **Both retrievers see the same query and the same corpus.** ``index()``
  indexes both, from one list of segments, in one call.
* **Union, not intersection.** A clause found by only one retriever is kept,
  scored on the one rank it has. Hybrid can therefore never retrieve *less*
  than its components.
* **Identity is ``clause_id``.** Candidates are merged on it, so the same
  clause reached by two routes becomes one result with a combined score.
* **Depth is independent of width.** ``candidate_depth`` controls how deep each
  component retrieves before fusion; ``top_k`` controls how many survive it.
  Fusing only the top 5 of each would throw away exactly the evidence RRF is
  there to recover.
* **Deterministic.** Ties break on RRF score DESC, then ``ordinal`` ASC, then
  ``clause_id`` ASC — the same rule both component retrievers already use.
* **Degrades honestly.** With no embedder (no ``sentence-transformers``, no
  embedding provider configured), the dense arm is absent and
  ``active_components`` says ``["bm25"]`` rather than the result silently
  pretending to be hybrid.

Nothing here mutates a segment or either component index.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

from rag_module.core.interfaces import BaseEmbedder, BaseRetriever
from rag_module.core.models import ClauseSegment, RetrievedPassage
from rag_module.retrieval.bm25_retriever import BM25Retriever
from rag_module.retrieval.dense_retriever import DenseRetriever

logger = logging.getLogger(__name__)

__all__ = ["HybridRetriever", "DEFAULT_RRF_K", "DEFAULT_CANDIDATE_DEPTH", "rrf_score"]

#: Cormack et al.'s constant, and the value ``Settings.rrf_k`` already carries.
DEFAULT_RRF_K = 60

#: How deep each component retrieves before fusion. Independent of ``top_k``.
DEFAULT_CANDIDATE_DEPTH = 20


def rrf_score(ranks: Sequence[int], k: int = DEFAULT_RRF_K) -> float:
    """Reciprocal rank fusion score for one document.

    Args:
        ranks: 1-based ranks this document achieved, one per retriever that
            returned it. A retriever that did not return it contributes
            nothing — it is simply absent from this sequence.
        k: Smoothing constant. Larger values flatten the advantage of rank 1.

    Raises:
        ValueError: If a rank is not >= 1, or ``k`` is negative. Both would
            silently produce nonsense (a zero or negative denominator), and a
            ranking that is quietly wrong is worse than one that fails.
    """
    if k < 0:
        raise ValueError(f"rrf k must be >= 0, got {k}")
    total = 0.0
    for rank in ranks:
        if rank < 1:
            raise ValueError(f"ranks are 1-based; got {rank}")
        total += 1.0 / (k + rank)
    return total


class HybridRetriever(BaseRetriever):
    """BM25 + dense retrieval fused by Reciprocal Rank Fusion."""

    def __init__(
        self,
        *,
        embedder: Optional[BaseEmbedder] = None,
        bm25: Optional[BM25Retriever] = None,
        dense: Optional[DenseRetriever] = None,
        rrf_k: int = DEFAULT_RRF_K,
        candidate_depth: int = DEFAULT_CANDIDATE_DEPTH,
    ) -> None:
        """Build a hybrid retriever.

        Args:
            embedder: Embedder for the dense arm. ``None`` disables it, and
                the retriever then runs BM25 only and says so.
            bm25: An existing BM25 retriever to reuse. A fresh one otherwise.
            dense: An existing dense retriever to reuse. Built from
                ``embedder`` otherwise.
            rrf_k: The RRF smoothing constant.
            candidate_depth: How many candidates each arm contributes.
        """
        if rrf_k < 0:
            raise ValueError(f"rrf_k must be >= 0, got {rrf_k}")
        if candidate_depth < 1:
            raise ValueError(f"candidate_depth must be >= 1, got {candidate_depth}")

        self.rrf_k = rrf_k
        self.candidate_depth = candidate_depth
        self.bm25: BM25Retriever = bm25 if bm25 is not None else BM25Retriever()

        self.dense: Optional[DenseRetriever]
        if dense is not None:
            self.dense = dense
        elif embedder is not None:
            self.dense = DenseRetriever(embedder=embedder)
        else:
            self.dense = None

        self._segment_count = 0

    # ----------------------------------------------------------------- state

    @property
    def active_components(self) -> List[str]:
        """Which arms will actually contribute to a fusion, in order."""
        return ["bm25", "dense"] if self.dense is not None else ["bm25"]

    @property
    def segment_count(self) -> int:
        return self._segment_count

    @property
    def is_hybrid(self) -> bool:
        """False when the dense arm is unavailable, so a caller can say so."""
        return self.dense is not None

    # ----------------------------------------------------------------- index

    def index(self, segments: List[ClauseSegment]) -> None:
        """Index the same segments in both arms.

        Atomic in the sense that matters: if the dense arm raises (a model that
        will not load, a vector that fails validation), the exception
        propagates rather than leaving a hybrid retriever whose two arms hold
        different corpora. A half-indexed hybrid would rank against evidence it
        could not cite.
        """
        self.bm25.index(segments)
        if self.dense is not None:
            self.dense.index(segments)
        self._segment_count = len(segments)

    # -------------------------------------------------------------- retrieve

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        document_id: Optional[str] = None,
        version_id: Optional[str] = None,
        candidate_depth: Optional[int] = None,
    ) -> List[RetrievedPassage]:
        """Retrieve by fusing both arms' rankings.

        Args:
            query: The user's question. Empty or whitespace returns ``[]``.
            top_k: How many fused results to return.
            document_id: Optional conjunctive filter, applied inside each arm.
            version_id: Optional conjunctive filter, applied inside each arm.
            candidate_depth: Override the per-arm retrieval depth for this call.

        Returns:
            Passages ranked by RRF score, each carrying
            ``retrieval_method="hybrid"`` and, in ``segment.metadata`` — no,
            see below: component detail is returned via
            :meth:`retrieve_with_detail` so that nothing is written onto the
            shared segment objects.
        """
        return [passage for passage, _ in self.retrieve_with_detail(
            query,
            top_k=top_k,
            document_id=document_id,
            version_id=version_id,
            candidate_depth=candidate_depth,
        )]

    def retrieve_with_detail(
        self,
        query: str,
        top_k: int = 5,
        document_id: Optional[str] = None,
        version_id: Optional[str] = None,
        candidate_depth: Optional[int] = None,
    ) -> List[Tuple[RetrievedPassage, Dict[str, object]]]:
        """Like :meth:`retrieve`, plus per-result fusion detail.

        The detail dict carries ``component_ranks`` (which arm put it where)
        and ``component_scores`` (each arm's raw score). It is returned
        alongside the passage rather than written into ``segment.metadata``,
        because the segments are shared with the index and with every other
        query — mutating them would leak one request's ranking into the next.
        """
        if not query or not query.strip():
            return []
        if top_k < 1:
            return []
        if self._segment_count == 0:
            return []

        depth = candidate_depth if candidate_depth is not None else self.candidate_depth
        depth = max(depth, top_k)   # never fuse a window narrower than the output

        runs: Dict[str, List[RetrievedPassage]] = {
            "bm25": self.bm25.retrieve(
                query, top_k=depth, document_id=document_id, version_id=version_id
            )
        }
        if self.dense is not None:
            runs["dense"] = self.dense.retrieve(
                query, top_k=depth, document_id=document_id, version_id=version_id
            )

        # Merge on clause_id. First sighting wins for the segment object; the
        # segments are identical across arms anyway, since both arms indexed
        # the same list.
        segments: Dict[str, ClauseSegment] = {}
        component_ranks: Dict[str, Dict[str, int]] = {}
        component_scores: Dict[str, Dict[str, float]] = {}

        for method, passages in runs.items():
            for passage in passages:
                clause_id = passage.clause_id
                segments.setdefault(clause_id, passage.segment)
                component_ranks.setdefault(clause_id, {})[method] = passage.rank
                component_scores.setdefault(clause_id, {})[method] = passage.score

        if not segments:
            return []

        fused: List[Tuple[float, int, str, ClauseSegment]] = []
        for clause_id, segment in segments.items():
            score = rrf_score(list(component_ranks[clause_id].values()), k=self.rrf_k)
            fused.append((score, segment.ordinal, clause_id, segment))

        # Same deterministic rule both arms use: score DESC, ordinal ASC, id ASC.
        fused.sort(key=lambda row: (-row[0], row[1], row[2]))

        results: List[Tuple[RetrievedPassage, Dict[str, object]]] = []
        for rank, (score, _ordinal, clause_id, segment) in enumerate(fused[:top_k], start=1):
            passage = RetrievedPassage(
                segment=segment,
                score=float(score),
                rank=rank,
                retrieval_method="hybrid" if self.is_hybrid else "bm25",
            )
            detail: Dict[str, object] = {
                "component_ranks": dict(component_ranks[clause_id]),
                "component_scores": {
                    name: float(value)
                    for name, value in component_scores[clause_id].items()
                },
                "rrf_k": self.rrf_k,
                "candidate_depth": depth,
            }
            results.append((passage, detail))

        return results
