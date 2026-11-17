"""Answer generation: the LLM path, and the honest path when there is no LLM.

:class:`GroundedGenerator` is the real one. It builds the context, asks the
configured model, and hands the raw answer back for citation verification. It
owns one piece of judgement: deciding whether the model refused. The prompt
asks for a sentinel token, so this is a string check rather than a guess about
tone.

:class:`ExtractiveGenerator` is what runs when no model is configured. It does
**not** write prose — it assembles an answer out of sentences copied verbatim
from the retrieved clauses, with the source markers attached, and labels itself
as such. That distinction is the whole reason it is allowed to exist: a reader
is never shown composed-from-source text and told a model wrote it, and the
API reports ``generation_mode`` so a caller cannot confuse the two either. It
exists because a retrieval system with no answer at all is not a product, and
pretending an unconfigured key produced an answer would be worse than both.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from rag_module.core.models import RetrievedPassage
from rag_module.generation.context import (
    DEFAULT_CONTEXT_CHAR_BUDGET,
    DEFAULT_MAX_PASSAGE_CHARS,
    BuiltContext,
    build_context,
)
from rag_module.generation.prompt import (
    REFUSAL_SENTINEL,
    SYSTEM_PROMPT,
    build_user_prompt,
)
from rag_module.generation.providers import (
    BaseLLMProvider,
    GenerationTimeout,
    InvalidModelResponse,
    LLMError,
    ProviderUnavailable,
)

logger = logging.getLogger(__name__)

__all__ = ["GenerationOutcome", "GroundedGenerator", "ExtractiveGenerator"]


@dataclass
class GenerationOutcome:
    """Everything one generation attempt produced, success or not."""

    answer: str
    sufficient_evidence: bool
    generation_mode: str                     # "llm" | "extractive" | "none"
    context: BuiltContext
    marker_to_passage: Dict[str, RetrievedPassage] = field(default_factory=dict)
    provider: Optional[str] = None
    model: Optional[str] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    latency_ms: Optional[float] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None

    @property
    def failed(self) -> bool:
        return self.error_code is not None


def _marker_map(context: BuiltContext) -> Dict[str, RetrievedPassage]:
    return {block.marker: block.passage for block in context.blocks}


class GroundedGenerator:
    """Generate an answer with a language model, grounded in retrieved text."""

    mode = "llm"

    def __init__(
        self,
        provider: BaseLLMProvider,
        *,
        temperature: float = 0.0,
        max_tokens: int = 800,
        timeout_seconds: float = 60.0,
        context_char_budget: int = DEFAULT_CONTEXT_CHAR_BUDGET,
        max_passage_chars: int = DEFAULT_MAX_PASSAGE_CHARS,
    ) -> None:
        self.provider = provider
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.context_char_budget = context_char_budget
        self.max_passage_chars = max_passage_chars

    def generate(
        self,
        question: str,
        passages: Sequence[RetrievedPassage],
        *,
        document_titles: Optional[Dict[str, str]] = None,
    ) -> GenerationOutcome:
        context = build_context(
            passages,
            budget_chars=self.context_char_budget,
            max_passage_chars=self.max_passage_chars,
        )

        if context.is_empty:
            return GenerationOutcome(
                answer="",
                sufficient_evidence=False,
                generation_mode="none",
                context=context,
                error_code="NO_EVIDENCE",
                error_message=(
                    "No passage in the collection matched this question, so "
                    "there was nothing to answer from."
                ),
            )

        system_prompt = SYSTEM_PROMPT
        user_prompt = build_user_prompt(
            question, context, document_titles=document_titles
        )

        started = time.perf_counter()
        try:
            result = self.provider.complete(
                system_prompt,
                user_prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout_seconds=self.timeout_seconds,
            )
        except LLMError as exc:
            return GenerationOutcome(
                answer="",
                sufficient_evidence=False,
                generation_mode="none",
                context=context,
                marker_to_passage=_marker_map(context),
                provider=getattr(self.provider, "name", None),
                error_code=getattr(exc, "code", "LLM_ERROR"),
                error_message=str(exc),
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )
        except Exception as exc:  # noqa: BLE001 - a provider bug must not 500 the API
            logger.exception("unexpected generation failure")
            return GenerationOutcome(
                answer="",
                sufficient_evidence=False,
                generation_mode="none",
                context=context,
                marker_to_passage=_marker_map(context),
                provider=getattr(self.provider, "name", None),
                error_code="LLM_ERROR",
                error_message=f"the model call failed: {type(exc).__name__}",
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )

        answer = (result.text or "").strip()
        refused = answer.upper().startswith(REFUSAL_SENTINEL) or (
            REFUSAL_SENTINEL in answer.upper()[:200]
        )
        if refused:
            # Strip the sentinel; the explanation after it is the real answer.
            answer = re.sub(
                rf"^\s*{REFUSAL_SENTINEL}\s*:?\s*", "", answer, flags=re.IGNORECASE
            ).strip()
            if not answer:
                answer = (
                    "The documents in this collection do not contain an answer "
                    "to that question."
                )

        return GenerationOutcome(
            answer=answer,
            sufficient_evidence=not refused,
            generation_mode=self.mode,
            context=context,
            marker_to_passage=_marker_map(context),
            provider=result.provider,
            model=result.model,
            latency_ms=result.latency_ms,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
        )


# --------------------------------------------------------------------------- #
# The no-model path
# --------------------------------------------------------------------------- #

#: Sentence-ish split that keeps clause numbering ("3.2 The bank shall…")
#: from being treated as a sentence end.
_SENTENCE_SPLIT = re.compile(r"(?<=[.;:])\s+(?=[A-Z(ऀ-ॿ])")

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for",
    "from", "has", "have", "how", "i", "if", "in", "is", "it", "of", "on", "or",
    "that", "the", "their", "there", "these", "they", "this", "to", "under",
    "was", "what", "when", "where", "which", "who", "why", "will", "with",
    "my", "me", "you", "your", "am", "any", "all",
}


def _content_words(text: str) -> List[str]:
    tokens = re.findall(r"[0-9a-zA-Zऀ-ॿ%().\-/]+", (text or "").lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


class ExtractiveGenerator:
    """Compose an answer from the source sentences themselves. No model.

    Every sentence in the output is copied verbatim from a retrieved clause and
    carries that clause's source marker, so the result is checkable in exactly
    the same way an LLM answer is — and ``generation_mode`` says ``extractive``
    so nothing downstream can present it as model-written prose.
    """

    mode = "extractive"

    def __init__(
        self,
        *,
        max_sentences: int = 6,
        max_passages: int = 4,
        context_char_budget: int = DEFAULT_CONTEXT_CHAR_BUDGET,
        max_passage_chars: int = DEFAULT_MAX_PASSAGE_CHARS,
        reason: Optional[str] = None,
    ) -> None:
        self.max_sentences = max_sentences
        self.max_passages = max_passages
        self.context_char_budget = context_char_budget
        self.max_passage_chars = max_passage_chars
        #: Why there is no model, surfaced to the UI verbatim.
        self.reason = reason or (
            "No language model is configured, so this is assembled from the "
            "source text rather than written."
        )

    def generate(
        self,
        question: str,
        passages: Sequence[RetrievedPassage],
        *,
        document_titles: Optional[Dict[str, str]] = None,
    ) -> GenerationOutcome:
        context = build_context(
            passages,
            budget_chars=self.context_char_budget,
            max_passage_chars=self.max_passage_chars,
        )

        if context.is_empty:
            return GenerationOutcome(
                answer="",
                sufficient_evidence=False,
                generation_mode="none",
                context=context,
                error_code="NO_EVIDENCE",
                error_message=(
                    "No passage in the collection matched this question."
                ),
            )

        query_terms = set(_content_words(question))
        scored: List[tuple] = []

        for position, block in enumerate(context.blocks[: self.max_passages]):
            for sentence in _split_sentences(block.text):
                stripped = sentence.strip()
                if len(stripped) < 30:
                    continue
                terms = set(_content_words(stripped))
                if not terms:
                    continue
                overlap = len(query_terms & terms)
                if overlap == 0:
                    continue
                # Normalise by sentence length so a long paragraph does not
                # win on sheer surface area.
                density = overlap / (len(terms) ** 0.5)
                scored.append((-density, position, stripped, block.marker))

        scored.sort()
        chosen = scored[: self.max_sentences]

        if not chosen:
            # Nothing in the retrieved text shares vocabulary with the
            # question. Saying so is the right answer, not quoting at random.
            return GenerationOutcome(
                answer=(
                    "The documents retrieved for this question do not contain a "
                    "passage that addresses it. The collection may not cover "
                    "this topic, or the question may use different words from "
                    "the documents."
                ),
                sufficient_evidence=False,
                generation_mode=self.mode,
                context=context,
                marker_to_passage=_marker_map(context),
                provider=None,
                model=None,
            )

        # Restore document order within the answer so it reads coherently.
        chosen.sort(key=lambda row: (row[1], row[2]))

        lines = [
            "Here is what the official documents say, in their own words:",
            "",
        ]
        for _score, _position, sentence, marker in chosen:
            lines.append(f"- {sentence} [{marker}]")
        lines.append("")
        lines.append(f"({self.reason})")

        return GenerationOutcome(
            answer="\n".join(lines),
            sufficient_evidence=True,
            generation_mode=self.mode,
            context=context,
            marker_to_passage=_marker_map(context),
            provider=None,
            model=None,
        )


def _split_sentences(text: str) -> List[str]:
    parts = _SENTENCE_SPLIT.split(text or "")
    out: List[str] = []
    for part in parts:
        # A clause can be one unbroken 900-character paragraph; cut it so a
        # single sentence cannot swallow the whole answer.
        if len(part) > 400:
            out.extend(_hard_wrap(part, 400))
        else:
            out.append(part)
    return out


def _hard_wrap(text: str, limit: int) -> List[str]:
    chunks: List[str] = []
    remaining = text
    while len(remaining) > limit:
        window = remaining[:limit]
        cut = window.rfind(" ")
        if cut <= limit // 2:
            cut = limit
        chunks.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    if remaining:
        chunks.append(remaining)
    return chunks
