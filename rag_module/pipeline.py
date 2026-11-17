"""The Ask pipeline: question in, grounded answer with checked citations out.

    retrieve (hybrid RRF)
      -> sufficiency gate
      -> context construction
      -> generation
      -> citation verification
      -> RAGResponse

Each stage can stop the pipeline, and each stop produces a *stated* outcome
rather than an empty answer: "nothing matched", "the evidence is too weak",
"no model is configured", "the model timed out". A user is never shown a blank
reply with no explanation, and never shown a confident answer that failed a
check.

The sufficiency gate
--------------------
Retrieval always returns something — BM25 will happily rank the least-bad
clause in the corpus for a question about tax law when the corpus is about
banking. The gate exists so the pipeline can say "this isn't in the documents"
*before* a model is asked, which is both cheaper and more reliable than hoping
the model refuses. It is deliberately not a single threshold on a BM25 score:
those are unbounded and corpus-dependent (the BM25 docstring says so). Instead
it uses two IDF-weighted signals that mean something across corpora: how much
of the question's *distinctive* vocabulary the best passage covers, and how
much of it the corpus has never seen at all.

This module holds no state and opens no connections. It is handed a retriever
that is already indexed, so the caller decides corpus lifetime and caching.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from rag_module.citation.verifier import CitationVerifier
from rag_module.core.models import Citation, RAGQuery, RAGResponse, RetrievedPassage
from rag_module.generation.context import (
    DEFAULT_CONTEXT_CHAR_BUDGET,
    DEFAULT_MAX_PASSAGE_CHARS,
)
from rag_module.generation.generator import (
    ExtractiveGenerator,
    GenerationOutcome,
    GroundedGenerator,
)
from rag_module.retrieval.hybrid_retriever import (
    DEFAULT_CANDIDATE_DEPTH,
    HybridRetriever,
)

logger = logging.getLogger(__name__)

__all__ = ["AskPipeline", "SufficiencyVerdict", "assess_sufficiency", "MAX_QUESTION_CHARS"]

#: Longer than this is a pasted document, not a question. Rejected with a
#: message rather than silently truncated, because truncating the middle of a
#: question changes what was asked.
MAX_QUESTION_CHARS = 1000

#: Cosine similarity (multilingual-e5-small) above which dense retrieval alone
#: counts as evidence. Measured on the live corpus: on-topic questions scored
#: 0.84-0.88, off-topic ones 0.75-0.81. Small sample - re-measure on new data.
DENSE_MIN_SIMILARITY = 0.82

#: Words the sufficiency gate ignores when deciding what a question is *about*.
#:
#: Wider than a retrieval stopword list on purpose, and the two lists must not
#: be merged. BM25 deliberately keeps "shall", "must" and "not" because they
#: carry binding legal force and change what a clause means. The gate is asking
#: a different question — is this topic in the collection at all? — and for
#: that, modals are noise: a question about "which banks *must* set up special
#: cells" is topically about banks, cells and SCs/STs, not about "must". Before
#: this list was widened, "must" counted as a missing distinctive term and got
#: a perfectly answerable question refused.
_STOPWORDS = {
    # articles, pronouns, prepositions, conjunctions
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by",
    "for", "from", "has", "have", "had", "i", "if", "in", "into", "is", "it",
    "its", "me", "my", "of", "on", "or", "our", "that", "the", "their", "them",
    "there", "these", "they", "this", "those", "to", "under", "was", "were",
    "with", "within", "you", "your",
    # question words
    "how", "what", "when", "where", "which", "who", "whom", "whose", "why",
    # modals and generic verbs: legal weight in retrieval, noise in topicality
    "can", "could", "do", "does", "did", "may", "might", "must", "need",
    "needs", "shall", "should", "will", "would", "get", "give", "make", "made",
    "take", "use", "used", "say", "says", "tell", "know", "apply", "applies",
    # generic quantifiers and discourse words
    "about", "after", "again", "all", "also", "am", "any", "because", "before",
    "between", "both", "during", "each", "either", "else", "even", "ever",
    "every", "like", "many", "more", "most", "much", "no", "only", "other",
    "others", "out", "over", "per", "same", "some", "such", "than", "then",
    "up", "very", "well", "while",
}


def _content_words(text: str) -> List[str]:
    tokens = re.findall(r"[0-9a-zA-Zऀ-ॿ%().\-/]+", (text or "").lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


def _fold(term: str) -> str:
    """Fold a term to a crude stem, for the "does the corpus mention this" test.

    Not a stemmer and not trying to be one: it strips a regular English plural
    so that a question about "cells" is not reported as a term the corpus has
    never seen when every clause says "cell". A real stemmer would be a new
    dependency for a check that only has to be approximately right — this one
    is wrong in harmless directions (it will fold "process" to "proces", which
    simply means both sides fold the same way and still match).

    Applied to both sides of the comparison, so it can only ever make the gate
    more permissive, never less.
    """
    if len(term) > 4 and term.endswith("ies"):
        return term[:-3] + "y"
    if len(term) > 4 and term.endswith(("ses", "xes", "zes", "ches", "shes")):
        return term[:-2]
    if len(term) > 3 and term.endswith("s") and not term.endswith("ss"):
        return term[:-1]
    return term


@dataclass
class SufficiencyVerdict:
    """Whether the retrieved evidence is worth generating from, and why."""

    sufficient: bool
    reason: str
    matched_passages: int = 0
    term_coverage: float = 0.0
    unseen_term_share: float = 0.0
    unseen_terms: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sufficient": self.sufficient,
            "reason": self.reason,
            "matched_passages": self.matched_passages,
            "term_coverage": round(self.term_coverage, 4),
            "unseen_term_share": round(self.unseen_term_share, 4),
            "unseen_terms": list(self.unseen_terms),
        }


def assess_sufficiency(
    question: str,
    passages: Sequence[RetrievedPassage],
    *,
    idf: Optional[Any] = None,
    min_term_coverage: float = 0.25,
    max_unseen_share: float = 0.35,
) -> SufficiencyVerdict:
    """Decide whether retrieval found anything worth answering from.

    Retrieval always returns *something*: BM25 will rank the least-bad clause
    in a banking corpus for a question about Portuguese tax law, because
    "capital" and "rate" happen to appear in both. The gate exists to catch
    that before a model is asked, which is cheaper and more reliable than
    hoping the model refuses.

    Two signals, both weighted by **inverse document frequency**, because an
    unweighted word count is exactly what lets the Portugal question through —
    it matches on the two commonest words in the question and misses the two
    that identify what is being asked:

    * ``term_coverage`` — the share of the question's IDF mass that the best
      single passage actually contains. Weighting means matching "capital"
      (idf 1.96) counts far less than matching "delhi" (idf 4.64).
    * ``unseen_term_share`` — the share of the question's IDF mass made up of
      words the **corpus has never seen at all**. This is the decisive one. A
      question whose distinctive vocabulary does not exist anywhere in the
      collection is a question the collection cannot answer, regardless of how
      the ranker ordered its least-bad guesses.

    ``idf`` is a callable returning the IDF of a term — in practice
    ``BM25Retriever.term_idf``, which scores an unseen term at the df=0
    ceiling. Without it, both measures fall back to unweighted counts and the
    gate is correspondingly blunter; that is why the pipeline always passes it.
    """
    terms = list(dict.fromkeys(_content_words(question)))

    if not passages:
        # Retrieval found nothing at all, which means no indexed clause
        # contains any of the question's terms. That is the same situation as
        # the unseen-term branch below and deserves the same actionable
        # message rather than a terse "nothing matched" -- the user needs to
        # know *which* words the collection does not have.
        return SufficiencyVerdict(
            sufficient=False,
            reason=(
                "The documents in this collection do not mention "
                + _list_terms(terms)
                + ", so they cannot answer this question. Try a question about "
                "the policies that have been added, or add the relevant "
                "document first."
            )
            if terms
            else "Nothing in the collection matched this question.",
            matched_passages=0,
            unseen_term_share=1.0 if terms else 0.0,
            unseen_terms=list(terms),
        )

    if not terms:
        # A question made entirely of stopwords ("what is it about?"). There is
        # nothing to measure against, so defer to the ranker.
        return SufficiencyVerdict(
            sufficient=True,
            reason="The question carries no distinctive terms; ranking alone was used.",
            matched_passages=len(passages),
        )

    weight = (lambda term: float(idf(term))) if callable(idf) else (lambda term: 1.0)
    weights = {term: max(weight(term), 0.0) for term in terms}
    total_weight = sum(weights.values())
    if total_weight <= 0:
        return SufficiencyVerdict(
            sufficient=True,
            reason="Question terms carry no weight in this corpus; ranking alone was used.",
            matched_passages=len(passages),
        )

    # Which question terms appear anywhere in what was retrieved? Compared on
    # folded forms so a plural in the question still matches a singular in the
    # clause (see _fold).
    def _passage_terms(passage: RetrievedPassage) -> set:
        words = set(_content_words(passage.segment.text))
        if passage.segment.heading:
            words |= set(_content_words(passage.segment.heading))
        if passage.segment.clause_number:
            words |= set(_content_words(passage.segment.clause_number))
        return {_fold(word) for word in words}

    per_passage = [_passage_terms(passage) for passage in passages]
    corpus_terms: set = set()
    for words in per_passage:
        corpus_terms |= words

    folded = {term: _fold(term) for term in terms}
    unseen = [term for term in terms if folded[term] not in corpus_terms]
    unseen_share = sum(weights[t] for t in unseen) / total_weight

    best_coverage = 0.0
    for words in per_passage:
        covered = sum(weights[t] for t in terms if folded[t] in words)
        best_coverage = max(best_coverage, covered / total_weight)

    if unseen_share > max_unseen_share:
        return SufficiencyVerdict(
            sufficient=False,
            reason=(
                "The documents in this collection do not mention "
                + _list_terms(unseen)
                + ", so they cannot answer this question. Try a question about "
                "the policies that have been added, or add the relevant document first."
            ),
            matched_passages=len(passages),
            term_coverage=best_coverage,
            unseen_term_share=unseen_share,
            unseen_terms=unseen,
        )

    if best_coverage < min_term_coverage:
        return SufficiencyVerdict(
            sufficient=False,
            reason=(
                "The passages that came back do not cover enough of what the "
                "question asks about to answer it from the documents."
            ),
            matched_passages=len(passages),
            term_coverage=best_coverage,
            unseen_term_share=unseen_share,
            unseen_terms=unseen,
        )

    return SufficiencyVerdict(
        sufficient=True,
        reason="Retrieved passages cover enough of the question to answer from.",
        matched_passages=len(passages),
        term_coverage=best_coverage,
        unseen_term_share=unseen_share,
        unseen_terms=unseen,
    )


def _list_terms(terms: Sequence[str], limit: int = 4) -> str:
    """Render the unseen terms for a message the user can act on."""
    shown = [f"“{t}”" for t in terms[:limit]]
    if not shown:
        return "the terms in this question"
    if len(terms) > limit:
        return ", ".join(shown) + f" (and {len(terms) - limit} more)"
    if len(shown) == 1:
        return shown[0]
    return ", ".join(shown[:-1]) + " or " + shown[-1]


class AskPipeline:
    """Retrieve, generate and verify. Owns no corpus and no connections."""

    def __init__(
        self,
        retriever: HybridRetriever,
        *,
        generator: Optional[Any] = None,
        verifier: Optional[CitationVerifier] = None,
        document_titles: Optional[Dict[str, str]] = None,
        top_k: int = 6,
        candidate_depth: int = DEFAULT_CANDIDATE_DEPTH,
        context_char_budget: int = DEFAULT_CONTEXT_CHAR_BUDGET,
        max_passage_chars: int = DEFAULT_MAX_PASSAGE_CHARS,
        min_term_coverage: float = 0.25,
    ) -> None:
        self.retriever = retriever
        self.generator = generator if generator is not None else ExtractiveGenerator(
            context_char_budget=context_char_budget,
            max_passage_chars=max_passage_chars,
        )
        self.verifier = verifier or CitationVerifier()
        self.document_titles = dict(document_titles or {})
        self.top_k = top_k
        self.candidate_depth = candidate_depth
        self.min_term_coverage = min_term_coverage

    # ------------------------------------------------------------------ ask

    def ask(self, query: RAGQuery) -> RAGResponse:
        """Answer one question. Never raises for an ordinary failure."""
        started = time.perf_counter()
        question = query.question.strip()

        if len(question) > MAX_QUESTION_CHARS:
            return self._stopped(
                query,
                answer=(
                    f"That question is {len(question)} characters long. Please "
                    f"ask something under {MAX_QUESTION_CHARS} characters — this "
                    "assistant answers questions, it does not summarise pasted "
                    "documents."
                ),
                started=started,
                status="question_too_long",
            )

        document_id = _single(query.document_ids)
        version_id = _single(query.version_ids)

        detailed = self.retriever.retrieve_with_detail(
            question,
            top_k=max(self.top_k, query.top_k),
            document_id=document_id,
            version_id=version_id,
            candidate_depth=self.candidate_depth,
        )
        passages = [passage for passage, _ in detailed]
        fusion_detail = {
            passage.clause_id: detail for passage, detail in detailed
        }

        verdict = assess_sufficiency(
            question,
            passages,
            idf=self.retriever.bm25.term_idf,
            min_term_coverage=self.min_term_coverage,
        )

        # The lexical gate above only counts shared words, so a paraphrase
        # ("get paid" vs "stipend") is refused even when dense retrieval found
        # the right clause. If the meaning-based score is high, trust it.
        if not verdict.sufficient and passages:
            dense_scores = [
                detail.get("component_scores", {}).get("dense")
                for _, detail in detailed
            ]
            best_dense = max((s for s in dense_scores if s is not None), default=None)
            if best_dense is not None and best_dense >= DENSE_MIN_SIMILARITY:
                verdict = SufficiencyVerdict(
                    sufficient=True,
                    reason=(
                        "Semantic retrieval found closely related passages "
                        f"(similarity {best_dense:.2f})."
                    ),
                    matched_passages=len(passages),
                    term_coverage=verdict.term_coverage,
                )

        if not verdict.sufficient:
            return self._stopped(
                query,
                answer=verdict.reason,
                started=started,
                status="insufficient_evidence",
                passages=passages,
                extra={
                    "sufficiency": verdict.to_dict(),
                    "fusion": fusion_detail,
                },
            )

        outcome: GenerationOutcome = self.generator.generate(
            question, passages, document_titles=self.document_titles
        )

        if outcome.failed:
            return self._stopped(
                query,
                answer=_failure_message(outcome),
                started=started,
                status=_failure_status(outcome),
                passages=passages,
                extra={
                    "sufficiency": verdict.to_dict(),
                    "fusion": fusion_detail,
                    "generation": {
                        "mode": outcome.generation_mode,
                        "provider": outcome.provider,
                        "error_code": outcome.error_code,
                        "error_message": outcome.error_message,
                    },
                },
            )

        report = self.verifier.build_citations(
            outcome.answer,
            outcome.marker_to_passage,
            document_titles=self.document_titles,
        )

        # An answer that claims to have evidence but cites none that resolves
        # is the failure mode this whole module exists to prevent. It is not
        # suppressed -- suppressing it hides a real defect -- but it is marked,
        # and the API and UI both show the warning.
        warnings: List[str] = []
        if report.unknown_markers:
            warnings.append(
                "The model referred to "
                f"{len(report.unknown_markers)} source(s) it was not given; "
                "those references were removed."
            )
        if report.uncited and outcome.sufficient_evidence:
            warnings.append(
                "This answer carries no citation that could be checked against "
                "the stored documents. Treat it with caution."
            )
        if outcome.context.dropped_clause_ids:
            warnings.append(
                f"{len(outcome.context.dropped_clause_ids)} lower-ranked "
                "passage(s) did not fit in the model's context."
            )

        answer = _strip_unknown_markers(outcome.answer, report.unknown_markers)

        return RAGResponse(
            query=question,
            answer=answer,
            sufficient_evidence=outcome.sufficient_evidence and not report.uncited,
            citations=report.citations,
            retrieved_passages=passages,
            segmentation_method=query.segmentation_method,
            retrieval_strategy=(
                "hybrid" if self.retriever.is_hybrid else "bm25"
            ),
            latency_ms=(time.perf_counter() - started) * 1000.0,
            metadata={
                "status": "answered",
                "generation": {
                    "mode": outcome.generation_mode,
                    "provider": outcome.provider,
                    "model": outcome.model,
                    "latency_ms": outcome.latency_ms,
                    "prompt_tokens": outcome.prompt_tokens,
                    "completion_tokens": outcome.completion_tokens,
                },
                "retrieval": {
                    "components": self.retriever.active_components,
                    "candidate_depth": self.candidate_depth,
                    "rrf_k": self.retriever.rrf_k,
                    "corpus_segments": self.retriever.segment_count,
                },
                "context": {
                    "markers": {
                        block.marker: block.clause_id
                        for block in outcome.context.blocks
                    },
                    "blocks": len(outcome.context.blocks),
                    "chars": outcome.context.total_chars,
                    "budget_chars": outcome.context.budget_chars,
                    "dropped_clause_ids": outcome.context.dropped_clause_ids,
                    "duplicate_clause_ids": outcome.context.duplicate_clause_ids,
                },
                "citation_report": report.to_dict(),
                "sufficiency": verdict.to_dict(),
                "fusion": fusion_detail,
                "warnings": warnings,
            },
        )

    # ------------------------------------------------------------- helpers

    def _stopped(
        self,
        query: RAGQuery,
        *,
        answer: str,
        started: float,
        status: str,
        passages: Optional[Sequence[RetrievedPassage]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> RAGResponse:
        """A response that stops before, or instead of, a generated answer."""
        metadata: Dict[str, Any] = {
            "status": status,
            "generation": {"mode": "none"},
            "retrieval": {
                "components": self.retriever.active_components,
                "corpus_segments": self.retriever.segment_count,
            },
            "warnings": [],
        }
        metadata.update(extra or {})
        return RAGResponse(
            query=query.question.strip(),
            answer=answer,
            sufficient_evidence=False,
            citations=[],
            retrieved_passages=list(passages or []),
            segmentation_method=query.segmentation_method,
            retrieval_strategy="hybrid" if self.retriever.is_hybrid else "bm25",
            latency_ms=(time.perf_counter() - started) * 1000.0,
            metadata=metadata,
        )


def _single(values: Optional[Sequence[str]]) -> Optional[str]:
    """The component retrievers filter on one id, not a list."""
    if not values:
        return None
    return values[0]


def _failure_status(outcome: GenerationOutcome) -> str:
    return {
        "NO_EVIDENCE": "no_evidence",
        "LLM_PROVIDER_UNAVAILABLE": "provider_unavailable",
        "LLM_TIMEOUT": "generation_timeout",
        "LLM_INVALID_RESPONSE": "invalid_model_response",
    }.get(outcome.error_code or "", "generation_failed")


def _failure_message(outcome: GenerationOutcome) -> str:
    """A message for the reader, not a stack trace.

    The provider's own message is included because it is the actionable part
    ("pull the model first", "check the key") and it is written by this
    codebase, not echoed from a remote body — `_safe_error_detail` in
    providers.py is what keeps relayed text credential-free.
    """
    base = {
        "NO_EVIDENCE": "No passage in the collection matched this question.",
        "LLM_PROVIDER_UNAVAILABLE": (
            "The answer could not be generated because the language model is "
            "not available."
        ),
        "LLM_TIMEOUT": "The language model took too long to answer.",
        "LLM_INVALID_RESPONSE": (
            "The language model returned something this system could not read."
        ),
    }.get(outcome.error_code or "", "The answer could not be generated.")
    detail = (outcome.error_message or "").strip()
    if detail:
        # The provider messages start lowercase because they read as clauses
        # inside an exception; here they are a sentence of their own.
        detail = detail[0].upper() + detail[1:]
        if not detail.endswith((".", "!", "?")):
            detail += "."
    return f"{base} {detail}".strip()


def _strip_unknown_markers(answer: str, unknown: Sequence[str]) -> str:
    """Remove markers the model invented, leaving the prose intact.

    Leaving them in would show the reader a citation link to nothing; removing
    the whole sentence would change what the model said. The marker goes, the
    claim stays, and the warning above tells the reader it happened.
    """
    if not unknown:
        return answer
    cleaned = answer
    for marker in unknown:
        number = marker.lstrip("Ss")
        # Same bracket set the verifier accepts (ASCII, CJK lenticular,
        # fullwidth); stripping only the ASCII form would leave a dead
        # reference on screen in exactly the cases that need removing.
        cleaned = re.sub(
            rf"\s*[\[\u3010\uff3b]\s*S\s*{re.escape(number)}\s*[\]\u3011\uff3d]",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
    return cleaned.strip()
