"""Citation verification and the end-to-end Ask pipeline.

The citation tests are the ones that matter most: everything else in the
system can be wrong and still be recoverable, but a citation that resolves to
text the source does not contain makes the whole product untrustworthy. So the
cases below include the ones that must *fail* — an invented marker, a quote
that is not in the clause, a page that is not the clause's page.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rag_module.citation.verifier import (  # noqa: E402
    CitationVerifier,
    extract_markers,
)
from rag_module.core.models import (  # noqa: E402
    Citation,
    ClauseSegment,
    PageProvenance,
    RAGQuery,
    RetrievedPassage,
)
from rag_module.generation.generator import ExtractiveGenerator, GroundedGenerator  # noqa: E402
from rag_module.generation.providers import MockProvider, ProviderUnavailable  # noqa: E402
from rag_module.pipeline import (  # noqa: E402
    MAX_QUESTION_CHARS,
    AskPipeline,
    assess_sufficiency,
)
from rag_module.retrieval.hybrid_retriever import HybridRetriever  # noqa: E402

CLAUSE_TEXT = (
    "3.2 Banks shall set up special cells at their head offices to monitor the "
    "flow of credit to Scheduled Castes and Scheduled Tribes, and shall submit "
    "a quarterly return to the Reserve Bank."
)


def seg(clause_id: str, text: str, *, ordinal: int = 0, pages=None, **kw) -> ClauseSegment:
    pages = pages or [PageProvenance(page_number=7, char_start=100, char_end=100 + len(text))]
    return ClauseSegment(
        document_id=kw.pop("document_id", "DOC-0001"),
        version_id=kw.pop("version_id", "VER-0002"),
        clause_id=clause_id,
        fingerprint=f"SEG-{clause_id}",
        ordinal=ordinal,
        segment_type="clause",
        text=text,
        page_start=pages[0].page_number,
        page_end=pages[-1].page_number,
        pages=pages,
        **kw,
    )


def passage(segment: ClauseSegment, rank: int = 1) -> RetrievedPassage:
    return RetrievedPassage(
        segment=segment, score=1.0 / rank, rank=rank, retrieval_method="hybrid"
    )


class TestMarkerExtraction(unittest.TestCase):
    def test_simple_markers(self) -> None:
        self.assertEqual(extract_markers("a [S1] b [S2]"), ["S1", "S2"])

    def test_grouped_markers_expand(self) -> None:
        self.assertEqual(extract_markers("a [S1, S2] b"), ["S1", "S2"])

    def test_adjacent_markers(self) -> None:
        self.assertEqual(extract_markers("a [S1][S3]"), ["S1", "S3"])

    def test_duplicates_collapse_preserving_order(self) -> None:
        self.assertEqual(extract_markers("[S2] x [S1] y [S2]"), ["S2", "S1"])

    def test_whitespace_inside_brackets(self) -> None:
        self.assertEqual(extract_markers("a [ S1 ]"), ["S1"])

    def test_case_insensitive(self) -> None:
        self.assertEqual(extract_markers("a [s4]"), ["S4"])

    def test_no_markers(self) -> None:
        self.assertEqual(extract_markers("no citations here"), [])

    def test_empty_and_none_safe(self) -> None:
        self.assertEqual(extract_markers(""), [])
        self.assertEqual(extract_markers(None), [])


class TestQuoteVerification(unittest.TestCase):
    def setUp(self) -> None:
        self.verifier = CitationVerifier()
        self.segment = seg("CLS-1", CLAUSE_TEXT)

    def test_an_exact_substring_is_strict(self) -> None:
        status, raw = self.verifier.verify_quote("special cells", self.segment)
        self.assertEqual(status, "STRICT_VERBATIM")
        self.assertIsNone(raw)

    def test_a_rewrapped_quote_is_audited_not_rejected(self) -> None:
        """Extraction puts line breaks inside clauses; that must not fail."""
        status, raw = self.verifier.verify_quote(
            "special    cells   at their head offices", self.segment
        )
        self.assertEqual(status, "AUDITED_SPAN_MATCH")
        self.assertIn("special cells", raw or "")

    def test_case_differences_are_audited(self) -> None:
        status, _ = self.verifier.verify_quote("SPECIAL CELLS", self.segment)
        self.assertEqual(status, "AUDITED_SPAN_MATCH")

    def test_text_not_in_the_clause_is_rejected(self) -> None:
        status, _ = self.verifier.verify_quote(
            "Banks shall waive all penalties", self.segment
        )
        self.assertEqual(status, "REJECTED")

    def test_a_negation_cannot_sneak_through(self) -> None:
        """Punctuation is not stripped, so "shall not" cannot match "shall"."""
        status, _ = self.verifier.verify_quote(
            "Banks shall not set up special cells", self.segment
        )
        self.assertEqual(status, "REJECTED")

    def test_empty_quote_is_rejected(self) -> None:
        self.assertEqual(self.verifier.verify_quote("", self.segment)[0], "REJECTED")
        self.assertEqual(self.verifier.verify_quote("   ", self.segment)[0], "REJECTED")

    def test_quote_chars_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            CitationVerifier(quote_chars=0)


class TestBuildCitations(unittest.TestCase):
    def setUp(self) -> None:
        self.verifier = CitationVerifier()
        self.segment = seg("CLS-1", CLAUSE_TEXT, clause_number="3.2")
        self.markers = {"S1": passage(self.segment)}

    def test_a_valid_marker_produces_a_verified_citation(self) -> None:
        report = self.verifier.build_citations("Banks must do X [S1].", self.markers)
        self.assertEqual(len(report.citations), 1)
        citation = report.citations[0]
        self.assertEqual(citation.clause_id, "CLS-1")
        self.assertEqual(citation.document_id, "DOC-0001")
        self.assertEqual(citation.version_id, "VER-0002")
        self.assertEqual(citation.verification_status, "STRICT_VERBATIM")
        self.assertIn(citation.quote, CLAUSE_TEXT)

    def test_an_invented_marker_is_dropped_and_reported(self) -> None:
        report = self.verifier.build_citations("Claim [S9].", self.markers)
        self.assertEqual(report.citations, [])
        self.assertEqual(report.unknown_markers, ["S9"])

    def test_a_mix_keeps_the_good_and_reports_the_bad(self) -> None:
        report = self.verifier.build_citations("A [S1]. B [S7].", self.markers)
        self.assertEqual(len(report.citations), 1)
        self.assertEqual(report.unknown_markers, ["S7"])

    def test_an_answer_with_no_markers_is_flagged_uncited(self) -> None:
        report = self.verifier.build_citations("Confident unsourced claim.", self.markers)
        self.assertTrue(report.uncited)
        self.assertEqual(report.citations, [])

    def test_page_offsets_come_from_the_real_provenance(self) -> None:
        report = self.verifier.build_citations("X [S1].", self.markers)
        citation = report.citations[0]
        self.assertEqual(citation.page_number, 7)
        self.assertEqual(citation.char_offset, (100, 100 + len(CLAUSE_TEXT)))

    def test_a_multi_page_clause_claims_no_single_offset(self) -> None:
        """Offsets restart at 0 on each page; there is no range across a break."""
        pages = [
            PageProvenance(page_number=3, char_start=900, char_end=1000),
            PageProvenance(page_number=4, char_start=0, char_end=120),
        ]
        spanning = seg("CLS-2", CLAUSE_TEXT, pages=pages)
        report = self.verifier.build_citations("X [S1].", {"S1": passage(spanning)})
        citation = report.citations[0]
        self.assertEqual(citation.page_start, 3)
        self.assertEqual(citation.page_end, 4)
        self.assertIsNone(citation.page_number)
        self.assertIsNone(citation.char_offset)

    def test_a_long_clause_still_yields_a_verbatim_quote(self) -> None:
        """Regression: the quote trimmer used to append an ellipsis, which made
        every quote from a clause longer than the limit fail verification and
        silently lose its citation."""
        long_segment = seg("CLS-3", "The bank shall comply. " * 80)
        report = self.verifier.build_citations("X [S1].", {"S1": passage(long_segment)})
        self.assertEqual(len(report.citations), 1)
        self.assertEqual(report.citations[0].verification_status, "STRICT_VERBATIM")
        self.assertIn(report.citations[0].quote, long_segment.text)
        self.assertNotIn("…", report.citations[0].quote)

    def test_verified_count_matches(self) -> None:
        report = self.verifier.build_citations("A [S1].", self.markers)
        self.assertEqual(report.verified_count, 1)
        self.assertEqual(report.to_dict()["citation_count"], 1)


class TestSufficiency(unittest.TestCase):
    def setUp(self) -> None:
        self.passages = [passage(seg("CLS-1", CLAUSE_TEXT))]

    def test_no_passages_is_insufficient(self) -> None:
        verdict = assess_sufficiency("anything", [])
        self.assertFalse(verdict.sufficient)

    def test_an_on_topic_question_passes(self) -> None:
        verdict = assess_sufficiency(
            "What special cells must banks set up for Scheduled Castes?",
            self.passages,
        )
        self.assertTrue(verdict.sufficient)

    def test_an_off_topic_question_is_refused_and_names_the_missing_terms(self) -> None:
        verdict = assess_sufficiency(
            "How do I renew my passport in Portugal?", self.passages
        )
        self.assertFalse(verdict.sufficient)
        self.assertIn("passport", verdict.unseen_terms)
        self.assertIn("passport", verdict.reason)

    def test_modals_do_not_count_as_missing_topic_words(self) -> None:
        """Regression: "must" counted as unseen and refused a good question."""
        verdict = assess_sufficiency(
            "Which banks must set up special cells?", self.passages
        )
        self.assertTrue(verdict.sufficient)
        self.assertNotIn("must", verdict.unseen_terms)

    def test_a_plural_matches_a_singular_in_the_source(self) -> None:
        verdict = assess_sufficiency("head offices and returns", self.passages)
        self.assertNotIn("offices", verdict.unseen_terms)

    def test_a_stopword_only_question_defers_to_the_ranker(self) -> None:
        verdict = assess_sufficiency("what is this about?", self.passages)
        self.assertTrue(verdict.sufficient)

    def test_idf_weighting_changes_the_verdict(self) -> None:
        """Without weighting, matching two common words looks like coverage."""
        unweighted = assess_sufficiency(
            "credit flow in Portugal under the passport scheme", self.passages
        )
        weighted = assess_sufficiency(
            "credit flow in Portugal under the passport scheme",
            self.passages,
            idf=lambda term: 10.0 if term in {"portugal", "passport"} else 0.5,
        )
        self.assertGreater(weighted.unseen_term_share, unweighted.unseen_term_share)


class TestAskPipeline(unittest.TestCase):
    def _pipeline(self, generator=None, corpus=None) -> AskPipeline:
        retriever = HybridRetriever()
        retriever.index(corpus if corpus is not None else [
            seg("CLS-1", CLAUSE_TEXT, ordinal=1, clause_number="3.2"),
            seg("CLS-2", "Leverage ratio requirements for commercial banks.", ordinal=2),
        ])
        return AskPipeline(retriever, generator=generator)

    def test_an_answered_question_carries_verified_citations(self) -> None:
        provider = MockProvider(reply="Banks must set up special cells [S1].")
        pipeline = self._pipeline(GroundedGenerator(provider))
        result = pipeline.ask(RAGQuery(question="What special cells must banks set up?"))
        self.assertEqual(result.metadata["status"], "answered")
        self.assertTrue(result.sufficient_evidence)
        self.assertEqual(len(result.citations), 1)
        self.assertEqual(result.citations[0].clause_id, "CLS-1")

    def test_an_invented_marker_is_stripped_from_the_answer(self) -> None:
        provider = MockProvider(reply="Claim one [S1]. Claim two [S9].")
        pipeline = self._pipeline(GroundedGenerator(provider))
        result = pipeline.ask(RAGQuery(question="What special cells must banks set up?"))
        self.assertNotIn("[S9]", result.answer)
        self.assertIn("[S1]", result.answer)
        self.assertTrue(any("not given" in w for w in result.metadata["warnings"]))

    def test_an_uncited_answer_loses_its_sufficiency_claim(self) -> None:
        provider = MockProvider(reply="Banks must do this. No source given.")
        pipeline = self._pipeline(GroundedGenerator(provider))
        result = pipeline.ask(RAGQuery(question="What special cells must banks set up?"))
        self.assertFalse(result.sufficient_evidence)
        self.assertTrue(any("caution" in w for w in result.metadata["warnings"]))

    def test_an_off_topic_question_never_reaches_the_model(self) -> None:
        provider = MockProvider(reply="I will happily make something up.")
        pipeline = self._pipeline(GroundedGenerator(provider))
        result = pipeline.ask(RAGQuery(question="How do I renew my passport?"))
        self.assertEqual(result.metadata["status"], "insufficient_evidence")
        self.assertEqual(provider.calls, [])
        self.assertNotIn("make something up", result.answer)

    def test_a_provider_failure_is_a_stated_outcome(self) -> None:
        provider = MockProvider(fail_with=ProviderUnavailable("no key configured"))
        pipeline = self._pipeline(GroundedGenerator(provider))
        result = pipeline.ask(RAGQuery(question="What special cells must banks set up?"))
        self.assertEqual(result.metadata["status"], "provider_unavailable")
        self.assertFalse(result.sufficient_evidence)
        self.assertIn("No key configured", result.answer)
        # The evidence is still returned, so the UI can show it.
        self.assertTrue(result.retrieved_passages)

    def test_an_over_long_question_is_refused_without_retrieval(self) -> None:
        pipeline = self._pipeline()
        result = pipeline.ask(RAGQuery(question="x " * (MAX_QUESTION_CHARS)))
        self.assertEqual(result.metadata["status"], "question_too_long")
        self.assertEqual(result.retrieved_passages, [])

    def test_an_empty_corpus_answers_without_crashing(self) -> None:
        retriever = HybridRetriever()
        retriever.index([])
        pipeline = AskPipeline(retriever)
        result = pipeline.ask(RAGQuery(question="anything at all"))
        self.assertFalse(result.sufficient_evidence)
        self.assertEqual(result.retrieved_passages, [])

    def test_the_extractive_default_is_labelled(self) -> None:
        pipeline = self._pipeline()
        result = pipeline.ask(RAGQuery(question="What special cells must banks set up?"))
        self.assertEqual(result.metadata["generation"]["mode"], "extractive")
        self.assertIsNone(result.metadata["generation"]["provider"])

    def test_strategy_reports_bm25_when_there_is_no_dense_arm(self) -> None:
        pipeline = self._pipeline()
        result = pipeline.ask(RAGQuery(question="What special cells must banks set up?"))
        self.assertEqual(result.retrieval_strategy, "bm25")

    def test_metadata_records_the_context_markers(self) -> None:
        provider = MockProvider(reply="X [S1].")
        pipeline = self._pipeline(GroundedGenerator(provider))
        result = pipeline.ask(RAGQuery(question="What special cells must banks set up?"))
        markers = result.metadata["context"]["markers"]
        self.assertIn("S1", markers)
        self.assertEqual(markers["S1"], "CLS-1")

    def test_a_document_filter_restricts_the_evidence(self) -> None:
        corpus = [
            seg("CLS-1", CLAUSE_TEXT, ordinal=1),
            seg("CLS-9", CLAUSE_TEXT.replace("3.2", "9.9"), ordinal=2,
                document_id="DOC-0002"),
        ]
        pipeline = self._pipeline(corpus=corpus)
        result = pipeline.ask(
            RAGQuery(
                question="What special cells must banks set up?",
                document_ids=["DOC-0002"],
            )
        )
        self.assertTrue(result.retrieved_passages)
        self.assertTrue(
            all(p.document_id == "DOC-0002" for p in result.retrieved_passages)
        )

    def test_latency_is_recorded(self) -> None:
        pipeline = self._pipeline()
        result = pipeline.ask(RAGQuery(question="What special cells must banks set up?"))
        self.assertIsNotNone(result.latency_ms)
        self.assertGreaterEqual(result.latency_ms, 0.0)

    def test_the_response_serialises(self) -> None:
        provider = MockProvider(reply="X [S1].")
        pipeline = self._pipeline(GroundedGenerator(provider))
        payload = pipeline.ask(
            RAGQuery(question="What special cells must banks set up?")
        ).to_dict()
        self.assertIn("answer", payload)
        self.assertIn("citations", payload)
        self.assertIn("retrieved_passages", payload)


if __name__ == "__main__":
    unittest.main()


class TestRealWorldMarkerBrackets(unittest.TestCase):
    """Bracket forms models actually emit, as opposed to the one we ask for.

    Regression, found the first time a real hosted model was called:
    `openai/gpt-oss-120b` on Groq writes the CJK lenticular form
    \u3010S2\u3011. An ASCII-only pattern matched nothing, so a perfectly good
    answer lost every citation and the pipeline reported "no citation that
    could be checked" -- correct behaviour reached from a wrong premise, which
    is the worst kind of bug to have in a verification layer.

    Widening the brackets loosens punctuation only. A marker still has to
    resolve to a clause retrieval actually returned, and its quote still has
    to verify against the stored text.
    """

    def setUp(self) -> None:
        self.verifier = CitationVerifier()
        self.markers = {"S1": passage(seg("CLS-1", CLAUSE_TEXT))}

    def test_cjk_lenticular_brackets(self) -> None:
        self.assertEqual(extract_markers("Banks must act\u3010S1\u3011."), ["S1"])

    def test_fullwidth_brackets(self) -> None:
        self.assertEqual(extract_markers("Banks must act\uff3bS1\uff3d."), ["S1"])

    def test_space_between_s_and_number(self) -> None:
        self.assertEqual(extract_markers("Banks must act [S 1]."), ["S1"])

    def test_narrow_no_break_space_before_the_marker(self) -> None:
        """gpt-oss puts U+202F between the word and the bracket."""
        self.assertEqual(extract_markers("quarterly\u202f[S1]."), ["S1"])

    def test_a_lenticular_marker_produces_a_verified_citation(self) -> None:
        report = self.verifier.build_citations("Banks must act\u3010S1\u3011.", self.markers)
        self.assertEqual(len(report.citations), 1)
        self.assertEqual(report.citations[0].verification_status, "STRICT_VERBATIM")
        self.assertFalse(report.uncited)

    def test_an_invented_lenticular_marker_is_stripped(self) -> None:
        from rag_module.pipeline import _strip_unknown_markers

        cleaned = _strip_unknown_markers("Real\u3010S1\u3011. Invented\u3010S9\u3011.", ["S9"])
        self.assertNotIn("S9", cleaned)
        self.assertIn("S1", cleaned)

    def test_mixed_bracket_styles_in_one_answer(self) -> None:
        self.assertEqual(
            extract_markers("a [S1] b \u3010S2\u3011 c \uff3bS3\uff3d"), ["S1", "S2", "S3"]
        )

    def test_a_bare_bracket_is_still_not_a_marker(self) -> None:
        self.assertEqual(extract_markers("see clause [3.2] and \u3010note\u3011"), [])
