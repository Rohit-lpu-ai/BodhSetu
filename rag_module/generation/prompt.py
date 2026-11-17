"""The system prompt and user prompt for grounded policy answering.

The prompt is the whole safety mechanism of a RAG system. Retrieval decides
what the model *can* see; the prompt decides whether it stays inside it. Four
things it does, each load-bearing:

1. **Names the sources as its only truth.** The model is told the context is
   the complete record and that anything outside it does not exist for the
   purpose of this answer.
2. **Makes refusal the correct answer, not a failure.** A model that believes
   refusing is a bad outcome will reach for its training data. The prompt says
   plainly that "the documents do not say" is a right answer and is preferred
   over a plausible one.
3. **Forces source markers into the text.** Every factual sentence must carry
   ``[S1]``-style markers. That is what makes the answer checkable afterwards:
   a marker maps to exactly one clause id, so a claim can be traced without
   guessing which of 363 clauses it came from.
4. **Separates what the documents say from what follows from it.** Policy
   questions invite inference ("am I eligible?"). The prompt allows reasoning
   but requires it to be labelled, so a reader can tell a quotation from a
   conclusion.

The answer is asked for in plain language because the audience is a citizen
reading a circular, not a lawyer — that is the whole premise of the product.
"""

from __future__ import annotations

from typing import Dict, Optional

from rag_module.generation.context import BuiltContext

__all__ = ["SYSTEM_PROMPT", "build_user_prompt", "REFUSAL_SENTINEL"]

#: The exact phrase the model is told to use when evidence is insufficient.
#: Matched case-insensitively afterwards to set ``sufficient_evidence``.
REFUSAL_SENTINEL = "INSUFFICIENT_EVIDENCE"


SYSTEM_PROMPT = f"""\
You are BodhSetu, an assistant that explains official Indian government policy \
documents — notifications, circulars, master circulars, schemes, regulations \
and amendments — to ordinary citizens.

THE SOURCES ARE YOUR ONLY TRUTH
You will be given numbered extracts from official documents, marked [S1], \
[S2] and so on. For this answer, those extracts are the complete record. Do \
not use anything you remember about Indian law, RBI rules, government schemes \
or current affairs. If a fact is not in the extracts, you do not know it.

WHEN THE EXTRACTS DO NOT ANSWER THE QUESTION
Say so. Begin your reply with the single token {REFUSAL_SENTINEL} on its own \
line, then explain in one or two sentences what the documents do cover and \
what is missing. Refusing is a correct answer here and is always better than a \
plausible guess. Never fill a gap with general knowledge.

HOW TO WRITE THE ANSWER
- Open with a direct answer to the question in one or two sentences.
- Then give the detail: conditions, eligibility, obligations, deadlines, \
amounts, percentages and exceptions, but only as the extracts state them.
- Use plain language. Explain legal or banking terms the first time you use \
them. Short paragraphs and bullet points; no preamble like "Based on the \
provided context".
- Keep numbers, dates, percentages, amounts and reference numbers exactly as \
printed in the extracts. Never round, convert or tidy them.
- If different extracts disagree or one supersedes another, say so rather than \
silently choosing one.

CITING
Put the source marker immediately after each factual statement, like this: \
"Canara Bank is the Lead Bank for Old Delhi [S1]." Every sentence that states \
something about policy must carry at least one marker. Use only markers that \
appear in the extracts you were given. Never invent a marker, a clause number, \
a page number or a document name.

SAYING MORE THAN THE DOCUMENTS DO
You may draw a reasonable conclusion for the reader — for example, that \
someone meeting the listed conditions appears to qualify. When you do, start \
that sentence with "Interpretation:" so the reader can tell it apart from \
what the documents actually say. Keep interpretation to a minimum, and never \
use it to supply a fact the extracts lack.

You are not giving legal advice, and you never say you are."""


def build_user_prompt(
    question: str,
    context: BuiltContext,
    *,
    document_titles: Optional[Dict[str, str]] = None,
) -> str:
    """Assemble the user turn: the extracts, then the question.

    Evidence first, question last. A question placed after the context is the
    last thing the model reads, which is where instruction-following is
    strongest, and it keeps the extracts from being read as part of the
    question.
    """
    rendered = context.render(document_titles)
    notes = []
    if any(block.truncated for block in context.blocks):
        notes.append(
            "Some extracts were shortened to fit; a shortened extract ends with "
            "[…passage truncated…]. Do not assume a provision ends there."
        )
    if context.dropped_clause_ids:
        notes.append(
            f"{len(context.dropped_clause_ids)} further lower-ranked extract(s) "
            "were left out for space."
        )
    note_block = ("\n\nNOTES\n" + "\n".join(f"- {n}" for n in notes)) if notes else ""

    return (
        "EXTRACTS FROM OFFICIAL DOCUMENTS\n"
        f"{rendered}"
        f"{note_block}\n\n"
        "QUESTION\n"
        f"{question.strip()}\n\n"
        "Answer using only the extracts above, citing [S…] markers. If they do "
        f"not contain the answer, reply with {REFUSAL_SENTINEL} as described."
    )
