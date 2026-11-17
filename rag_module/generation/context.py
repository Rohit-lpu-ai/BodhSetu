"""Context construction: turning retrieved passages into a groundable prompt.

The context block is the only thing the model is allowed to treat as true, so
how it is built decides whether the answer can be checked afterwards. Five
rules, each of which exists because breaking it makes verification impossible:

1. **One passage, one source marker.** Every block is introduced by
   ``[S1] … [S2] …``. The model is told to cite those markers, which means a
   citation maps back to exactly one ``clause_id`` — no fuzzy matching of a
   quoted sentence against 363 clauses, and no way for the model to invent a
   clause id, because it never sees one it could invent from.
2. **Clause boundaries are preserved.** A passage is never cut mid-clause to
   fit a budget; it is either included whole or dropped, and dropped ones are
   reported. A truncated clause would make a verbatim quote check fail against
   text that was genuinely in the source.
3. **Metadata travels with the text.** Document, version, clause number,
   heading and page range sit in the block header, so the model can say *which*
   policy and *which page* a statement comes from without being asked to guess.
4. **No duplicates.** The same ``clause_id`` appears once even if two
   retrievers surfaced it, and near-identical text (the paragraph and
   clause-aware views of one passage) is collapsed, because duplicated evidence
   reads to a language model as corroboration it has not got.
5. **Budget in characters, not tokens.** No tokenizer is available without
   adding a dependency, and a character budget with a conservative ratio is
   predictable and auditable. Over-budget passages are dropped from the tail —
   the lowest-ranked evidence — never silently from the middle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from rag_module.core.models import ClauseSegment, RetrievedPassage

__all__ = [
    "ContextBlock",
    "BuiltContext",
    "build_context",
    "DEFAULT_CONTEXT_CHAR_BUDGET",
    "DEFAULT_MAX_PASSAGE_CHARS",
]

#: Total characters of passage text allowed in one prompt. ~4 chars/token puts
#: this near 3,000 tokens of evidence, which every usable chat model can hold
#: alongside the system prompt and the answer.
DEFAULT_CONTEXT_CHAR_BUDGET = 12_000

#: A single clause longer than this is a table or an annexure dump rather than
#: a provision. It is included up to this length with an explicit marker, so
#: the model is told the passage was cut rather than inferring a clause ended.
DEFAULT_MAX_PASSAGE_CHARS = 3_000

_TRUNCATION_MARKER = " […passage truncated…]"


@dataclass
class ContextBlock:
    """One passage as it appears in the prompt, with its source marker."""

    marker: str                 # "S1", "S2", ...
    passage: RetrievedPassage
    text: str                   # the text actually placed in the prompt
    truncated: bool = False

    @property
    def segment(self) -> ClauseSegment:
        return self.passage.segment

    @property
    def clause_id(self) -> str:
        return self.passage.clause_id

    def render(self, document_titles: Optional[Dict[str, str]] = None) -> str:
        """Render this block for the prompt."""
        segment = self.segment
        titles = document_titles or {}
        title = titles.get(segment.document_id) or segment.document_id

        pages = (
            f"page {segment.page_start}"
            if segment.page_start == segment.page_end
            else f"pages {segment.page_start}-{segment.page_end}"
        )
        header_bits = [f"[{self.marker}]", title]
        if segment.clause_number:
            header_bits.append(f"clause {segment.clause_number}")
        if segment.heading:
            header_bits.append(f"— {segment.heading}")
        header_bits.append(f"({pages})")

        return f"{' '.join(header_bits)}\n{self.text}"


@dataclass
class BuiltContext:
    """The assembled context plus everything needed to audit it afterwards."""

    blocks: List[ContextBlock] = field(default_factory=list)
    dropped_clause_ids: List[str] = field(default_factory=list)
    duplicate_clause_ids: List[str] = field(default_factory=list)
    total_chars: int = 0
    budget_chars: int = DEFAULT_CONTEXT_CHAR_BUDGET

    @property
    def is_empty(self) -> bool:
        return not self.blocks

    def by_marker(self) -> Dict[str, ContextBlock]:
        return {block.marker: block for block in self.blocks}

    def by_clause_id(self) -> Dict[str, ContextBlock]:
        return {block.clause_id: block for block in self.blocks}

    def render(self, document_titles: Optional[Dict[str, str]] = None) -> str:
        """The context exactly as the model will see it."""
        return "\n\n".join(block.render(document_titles) for block in self.blocks)


def _normalise_for_dedup(text: str) -> str:
    """Collapse whitespace and case so two views of one passage compare equal."""
    return " ".join(text.lower().split())


def build_context(
    passages: Sequence[RetrievedPassage],
    *,
    budget_chars: int = DEFAULT_CONTEXT_CHAR_BUDGET,
    max_passage_chars: int = DEFAULT_MAX_PASSAGE_CHARS,
    max_passages: Optional[int] = None,
) -> BuiltContext:
    """Assemble retrieved passages into a numbered, deduplicated context.

    Args:
        passages: Retrieved passages, **already in relevance order**. Order is
            respected: the budget is spent on the best evidence first.
        budget_chars: Total passage characters allowed.
        max_passage_chars: Per-passage cap before truncation.
        max_passages: Optional hard cap on how many blocks to emit.

    Returns:
        A :class:`BuiltContext`. Passages that did not fit are listed in
        ``dropped_clause_ids`` rather than vanishing, so the caller can report
        that the evidence was trimmed.
    """
    if budget_chars < 1:
        raise ValueError(f"budget_chars must be >= 1, got {budget_chars}")
    if max_passage_chars < 1:
        raise ValueError(f"max_passage_chars must be >= 1, got {max_passage_chars}")

    built = BuiltContext(budget_chars=budget_chars)
    seen_clause_ids: set = set()
    seen_texts: set = set()
    used = 0

    for passage in passages:
        if max_passages is not None and len(built.blocks) >= max_passages:
            built.dropped_clause_ids.append(passage.clause_id)
            continue

        clause_id = passage.clause_id
        if clause_id in seen_clause_ids:
            built.duplicate_clause_ids.append(clause_id)
            continue

        body = passage.segment.text.strip()
        if not body:
            built.dropped_clause_ids.append(clause_id)
            continue

        fingerprint = _normalise_for_dedup(body)
        if fingerprint in seen_texts:
            # The paragraph and clause-aware views of the same words. Keeping
            # both would read to the model as two independent sources.
            built.duplicate_clause_ids.append(clause_id)
            continue

        truncated = False
        if len(body) > max_passage_chars:
            body = body[:max_passage_chars].rstrip() + _TRUNCATION_MARKER
            truncated = True

        if used + len(body) > budget_chars:
            # Out of room. Everything from here down is lower-ranked, so it is
            # dropped rather than squeezed in at the cost of better evidence.
            built.dropped_clause_ids.append(clause_id)
            continue

        seen_clause_ids.add(clause_id)
        seen_texts.add(fingerprint)
        used += len(body)
        built.blocks.append(
            ContextBlock(
                marker=f"S{len(built.blocks) + 1}",
                passage=passage,
                text=body,
                truncated=truncated,
            )
        )

    built.total_chars = used
    return built
