"""Citation extraction and verification.

A RAG answer is only worth as much as the reader's ability to check it, so
this module treats every citation as guilty until it resolves. Three
independent checks, in order:

1. **Does the source marker exist?** The model is only ever shown ``[S1]``…
   markers, so a marker it did not receive is a hallucinated one and is
   dropped. This is why the prompt uses markers rather than clause ids: the
   model cannot invent ``CLS-0000166`` because it never sees an id to pattern
   off.
2. **Does the clause exist in the retrieved context?** The marker is resolved
   to a real :class:`ClauseSegment` that retrieval actually returned — never
   to a corpus lookup, which would let an answer cite evidence the pipeline
   never considered.
3. **Is the quote really in that clause?** The quote is checked against the
   stored segment text verbatim, and then under whitespace normalisation.

Why two levels of quote matching
--------------------------------
The corpus comes out of ``pypdf`` on bilingual RBI PDFs, where a word can
arrive broken across a line (``"Mast er Circular"``) and punctuation can
arrive as mojibake. A strict substring test would reject quotes that are
genuinely present in the source as extracted. So a normalised match is
recorded as ``AUDITED_SPAN_MATCH`` — a real, weaker verification level that the
API and the UI both surface — rather than being silently promoted to
``STRICT_VERBATIM`` or silently dropped. The distinction is the point: the
reader is told which kind of check passed.

``page_number`` and ``char_offset`` are filled only for single-page segments,
because NiyamSetu's offsets index each page's own ``normalized_text`` and
there is no continuous range across a page break (``ClauseSegment.char_start``
raises for exactly that reason).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from rag_module.core.models import Citation, ClauseSegment, RetrievedPassage

__all__ = [
    "MARKER_PATTERN",
    "extract_markers",
    "CitationVerifier",
    "CitationReport",
    "DEFAULT_QUOTE_CHARS",
]

#: The bracket characters a model may actually use around a source marker.
#:
#: The prompt asks for ``[S1]``. Models do not reliably comply, and the failure
#: is invisible until it costs you every citation in an answer: ``gpt-oss-120b``
#: on Groq emits the CJK lenticular form ``\u3010S2\u3011``, so an ASCII-only pattern
#: found zero markers, dropped all citations, and the pipeline correctly but
#: uselessly reported "this answer carries no citation that could be checked".
#:
#: Accepting the variants is not laxity: the marker still has to resolve to a
#: clause that retrieval actually returned, and the quote still has to verify
#: against the stored text. All this widens is the punctuation, which carries
#: no meaning.
_OPEN_BRACKETS = "\\[\u3010\uff3b"
_CLOSE_BRACKETS = "\\]\u3011\uff3d"

#: ``[S1]``, ``\u3010S1\u3011``, ``\uff3bS1\uff3d``, ``[S12]``, and the grouped forms models
#: reach for: ``[S1, S2]`` and ``[S1][S2]``. Matching the grouped form stops a
#: perfectly good citation being discarded over a comma.
MARKER_PATTERN = re.compile(
    rf"[{_OPEN_BRACKETS}]\s*(S\s*\d+(?:\s*[,;]\s*S?\s*\d+)*)\s*[{_CLOSE_BRACKETS}]",
    re.IGNORECASE,
)

_SINGLE_MARKER = re.compile(r"S?\s*(\d+)", re.IGNORECASE)

#: How much of the clause to quote back as the supporting excerpt. Long enough
#: to be evidence, short enough to read. Cut on a word boundary.
DEFAULT_QUOTE_CHARS = 320


def extract_markers(text: str) -> List[str]:
    """Every distinct source marker in an answer, in order of first appearance.

    Returns canonical ``"S1"`` forms. Grouped markers (``[S1, S2]``) expand to
    their members.
    """
    seen: Set[str] = set()
    ordered: List[str] = []
    for group in MARKER_PATTERN.findall(text or ""):
        for number in _SINGLE_MARKER.findall(group):
            marker = f"S{int(number)}"
            if marker not in seen:
                seen.add(marker)
                ordered.append(marker)
    return ordered


def _normalise(text: str) -> str:
    """Whitespace-insensitive, case-insensitive form for the audited match.

    Collapses every run of whitespace — which is what makes a word broken
    across a line break (``"Mast er"``) still fail, while a quote that merely
    re-wrapped succeeds. Deliberately does NOT strip punctuation: dropping it
    would let a quote match a clause that says the opposite.
    """
    return " ".join((text or "").split()).lower()


def _trim_to_word(text: str, limit: int) -> str:
    """Cut at or before ``limit`` without splitting a word.

    **No ellipsis is appended.** The returned string has to stay a verbatim
    substring of the clause, because that is exactly what the verification
    step checks and what CLAUDE.md §12.2 requires of a ``quote``. Appending
    "…" made every quote drawn from a clause longer than ``limit`` fail both
    the strict and the normalised match and get dropped — so a long clause
    could be used by the answer and then silently lose its citation. Found by
    asking a real question against the live corpus.

    A caller that wants to show the reader the quote was cut can compare its
    length against the segment's own text; the API does.
    """
    if len(text) <= limit:
        return text
    window = text[:limit]
    space = window.rfind(" ")
    if space > limit // 2:
        window = window[:space]
    return window.rstrip()


@dataclass
class CitationReport:
    """What verification concluded, including what it threw away."""

    citations: List[Citation] = field(default_factory=list)
    #: Markers the model used that were never supplied to it.
    unknown_markers: List[str] = field(default_factory=list)
    #: Clause ids whose quote did not resolve against the stored text.
    rejected_clause_ids: List[str] = field(default_factory=list)
    #: True when the answer made factual claims but cited nothing valid.
    uncited: bool = False

    @property
    def verified_count(self) -> int:
        return sum(
            1
            for c in self.citations
            if c.verification_status in {"STRICT_VERBATIM", "AUDITED_SPAN_MATCH"}
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "unknown_markers": list(self.unknown_markers),
            "rejected_clause_ids": list(self.rejected_clause_ids),
            "uncited": self.uncited,
            "verified_count": self.verified_count,
            "citation_count": len(self.citations),
        }


class CitationVerifier:
    """Turns source markers in an answer into verified :class:`Citation` objects."""

    def __init__(self, *, quote_chars: int = DEFAULT_QUOTE_CHARS) -> None:
        if quote_chars < 1:
            raise ValueError(f"quote_chars must be >= 1, got {quote_chars}")
        self.quote_chars = quote_chars

    # ------------------------------------------------------------ one check

    def verify_quote(
        self, quote: str, segment: ClauseSegment
    ) -> Tuple[str, Optional[str]]:
        """Classify a quote against a segment's stored text.

        Returns:
            ``(verification_status, extracted_raw_quote)``. The raw quote is
            the span as it actually appears in the source when a match was
            found under normalisation but not verbatim — i.e. the evidence for
            why the weaker status was given.
        """
        if not quote or not quote.strip():
            return "REJECTED", None

        source = segment.text or ""
        if quote in source:
            return "STRICT_VERBATIM", None

        norm_quote = _normalise(quote)
        norm_source = _normalise(source)
        if norm_quote and norm_quote in norm_source:
            return "AUDITED_SPAN_MATCH", _recover_raw_span(source, quote)

        return "REJECTED", None

    # --------------------------------------------------------- whole answer

    def build_citations(
        self,
        answer: str,
        marker_to_passage: Dict[str, RetrievedPassage],
        *,
        document_titles: Optional[Dict[str, str]] = None,
    ) -> CitationReport:
        """Extract the markers an answer used and verify each one.

        Args:
            answer: The generated answer text.
            marker_to_passage: The markers that were actually supplied, mapped
                to the passages they stood for. A marker outside this mapping
                is hallucinated and is reported, never resolved.
            document_titles: Used only to carry a readable title back to the
                caller; it never affects verification.

        Returns:
            A :class:`CitationReport`.
        """
        report = CitationReport()
        markers = extract_markers(answer)

        for marker in markers:
            passage = marker_to_passage.get(marker)
            if passage is None:
                report.unknown_markers.append(marker)
                continue

            segment = passage.segment
            quote = _trim_to_word(segment.text.strip(), self.quote_chars)
            status, raw = self.verify_quote(quote, segment)

            if status == "REJECTED":
                # The quote is drawn from the segment itself, so this can only
                # fire on an empty or whitespace-only segment -- which the
                # ClauseSegment validator already forbids. Reported rather than
                # asserted, because a corpus can change under us.
                report.rejected_clause_ids.append(segment.clause_id)
                continue

            page_number, char_offset = _single_page_offsets(segment)

            report.citations.append(
                Citation(
                    document_id=segment.document_id,
                    version_id=segment.version_id,
                    clause_id=segment.clause_id,
                    page_start=segment.page_start,
                    page_end=segment.page_end,
                    quote=quote,
                    page_number=page_number,
                    verification_status=status,
                    extracted_raw_quote=raw,
                    char_offset=char_offset,
                )
            )

        report.uncited = bool(answer.strip()) and not report.citations
        return report


def _single_page_offsets(
    segment: ClauseSegment,
) -> Tuple[Optional[int], Optional[Tuple[int, int]]]:
    """Page-scoped offsets, for single-page segments only.

    A multi-page segment has one offset range *per page*, each starting at 0 in
    that page's own ``normalized_text``. There is no single ``(start, end)``
    that means anything across the break, so none is claimed.
    """
    if segment.page_start != segment.page_end:
        return None, None
    try:
        start, end = segment.char_span_for_page(segment.page_start)
    except (ValueError, KeyError):
        return None, None
    return segment.page_start, (int(start), int(end))


def _recover_raw_span(source: str, quote: str) -> Optional[str]:
    """Find the span in ``source`` that matched ``quote`` under normalisation.

    Walks the source once, building the normalised form and remembering which
    source index each normalised character came from, so the matched window can
    be mapped back to the original characters — including whatever extraction
    damage made the verbatim check fail in the first place. That damaged text
    is the evidence for the weaker verification status, so it is worth
    recovering rather than guessing.
    """
    norm_quote = _normalise(quote)
    if not norm_quote:
        return None

    normalised_chars: List[str] = []
    origins: List[int] = []
    previous_was_space = True
    for index, char in enumerate(source):
        if char.isspace():
            if previous_was_space:
                continue
            normalised_chars.append(" ")
            origins.append(index)
            previous_was_space = True
        else:
            normalised_chars.append(char.lower())
            origins.append(index)
            previous_was_space = False

    normalised = "".join(normalised_chars)
    position = normalised.find(norm_quote)
    if position < 0:
        return None

    start = origins[position]
    end_index = position + len(norm_quote) - 1
    end = origins[end_index] + 1 if end_index < len(origins) else len(source)
    return source[start:end]
