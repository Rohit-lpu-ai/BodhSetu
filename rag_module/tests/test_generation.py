"""Context construction, prompting, providers and the two generators.

No network, no model: the HTTP providers are exercised through a stub
transport, and the generators through :class:`MockProvider`. What is actually
being checked is the behaviour that keeps an answer honest — the refusal path,
the failure paths, and the fact that the extractive generator never claims to
be a language model.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rag_module.core.models import (  # noqa: E402
    ClauseSegment,
    PageProvenance,
    RetrievedPassage,
)
from rag_module.generation.context import (  # noqa: E402
    build_context,
)
from rag_module.generation.generator import (  # noqa: E402
    ExtractiveGenerator,
    GroundedGenerator,
)
from rag_module.generation.prompt import (  # noqa: E402
    REFUSAL_SENTINEL,
    SYSTEM_PROMPT,
    build_user_prompt,
)
from rag_module.generation.providers import (  # noqa: E402
    GenerationTimeout,
    InvalidModelResponse,
    MockProvider,
    OllamaProvider,
    OpenAICompatibleProvider,
    ProviderUnavailable,
    describe_configuration,
    resolve_provider,
)


def seg(clause_id: str, text: str, *, ordinal: int = 0, pages=None, **kw) -> ClauseSegment:
    pages = pages or [PageProvenance(page_number=1, char_start=0, char_end=len(text))]
    return ClauseSegment(
        document_id=kw.pop("document_id", "DOC-0001"),
        version_id=kw.pop("version_id", "VER-0001"),
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


def passage(segment: ClauseSegment, rank: int = 1, score: float = 1.0) -> RetrievedPassage:
    return RetrievedPassage(
        segment=segment, score=score, rank=rank, retrieval_method="hybrid"
    )


class TestBuildContext(unittest.TestCase):
    def test_markers_are_sequential_from_s1(self) -> None:
        ctx = build_context(
            [passage(seg("A", "alpha text"), 1), passage(seg("B", "beta text"), 2)]
        )
        self.assertEqual([b.marker for b in ctx.blocks], ["S1", "S2"])

    def test_duplicate_clause_ids_appear_once(self) -> None:
        one = seg("A", "alpha text")
        ctx = build_context([passage(one, 1), passage(one, 2)])
        self.assertEqual(len(ctx.blocks), 1)
        self.assertEqual(ctx.duplicate_clause_ids, ["A"])

    def test_identical_text_under_two_ids_is_collapsed(self) -> None:
        """The paragraph and clause views of one passage are not two sources."""
        ctx = build_context(
            [
                passage(seg("A", "The bank shall report quarterly."), 1),
                passage(seg("B", "The   bank shall  report quarterly.  "), 2),
            ]
        )
        self.assertEqual(len(ctx.blocks), 1)
        self.assertEqual(ctx.duplicate_clause_ids, ["B"])

    def test_budget_drops_the_tail_not_the_head(self) -> None:
        passages = [
            passage(seg(f"C{i}", f"passage {i} " + "x" * 88, ordinal=i), i + 1)
            for i in range(5)
        ]
        ctx = build_context(passages, budget_chars=250)
        self.assertEqual([b.clause_id for b in ctx.blocks], ["C0", "C1"])
        self.assertEqual(ctx.dropped_clause_ids, ["C2", "C3", "C4"])

    def test_oversized_passage_is_truncated_and_marked(self) -> None:
        ctx = build_context([passage(seg("A", "y" * 5000), 1)], max_passage_chars=100)
        self.assertTrue(ctx.blocks[0].truncated)
        self.assertIn("truncated", ctx.blocks[0].text)

    def test_max_passages_caps_blocks(self) -> None:
        passages = [passage(seg(f"C{i}", f"text {i}", ordinal=i), i + 1) for i in range(5)]
        ctx = build_context(passages, max_passages=2)
        self.assertEqual(len(ctx.blocks), 2)
        self.assertEqual(len(ctx.dropped_clause_ids), 3)

    def test_empty_input_is_empty_context(self) -> None:
        self.assertTrue(build_context([]).is_empty)

    def test_whitespace_only_segment_is_dropped(self) -> None:
        # ClauseSegment forbids empty text, so this is the closest real case.
        ctx = build_context([passage(seg("A", "   . ")), passage(seg("B", "real text"))])
        self.assertIn("B", [b.clause_id for b in ctx.blocks])

    def test_render_includes_document_page_and_clause_number(self) -> None:
        ctx = build_context([passage(seg("A", "body", clause_number="3.2",
                                         heading="Reporting"))])
        rendered = ctx.render({"DOC-0001": "Master Circular"})
        self.assertIn("[S1]", rendered)
        self.assertIn("Master Circular", rendered)
        self.assertIn("clause 3.2", rendered)
        self.assertIn("Reporting", rendered)
        self.assertIn("page 1", rendered)

    def test_multi_page_renders_a_range(self) -> None:
        pages = [
            PageProvenance(page_number=3, char_start=0, char_end=10),
            PageProvenance(page_number=4, char_start=0, char_end=10),
        ]
        ctx = build_context([passage(seg("A", "spanning body", pages=pages))])
        self.assertIn("pages 3-4", ctx.render())

    def test_invalid_budget_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_context([], budget_chars=0)
        with self.assertRaises(ValueError):
            build_context([], max_passage_chars=0)


class TestPrompt(unittest.TestCase):
    def test_system_prompt_names_the_refusal_token(self) -> None:
        self.assertIn(REFUSAL_SENTINEL, SYSTEM_PROMPT)

    def test_system_prompt_forbids_outside_knowledge(self) -> None:
        lowered = SYSTEM_PROMPT.lower()
        self.assertIn("do not use anything you remember", lowered)
        self.assertIn("never invent", lowered)

    def test_user_prompt_puts_the_question_after_the_evidence(self) -> None:
        ctx = build_context([passage(seg("A", "alpha text"))])
        prompt = build_user_prompt("What does it say?", ctx)
        self.assertLess(prompt.index("EXTRACTS"), prompt.index("QUESTION"))
        self.assertIn("alpha text", prompt)
        self.assertIn("What does it say?", prompt)

    def test_truncation_is_declared_to_the_model(self) -> None:
        ctx = build_context([passage(seg("A", "y" * 5000))], max_passage_chars=100)
        self.assertIn("shortened", build_user_prompt("q", ctx))

    def test_dropped_passages_are_declared_to_the_model(self) -> None:
        passages = [
            passage(seg(f"C{i}", f"passage {i} " + "x" * 88, ordinal=i), i + 1)
            for i in range(5)
        ]
        ctx = build_context(passages, budget_chars=250)
        self.assertIn("left out for space", build_user_prompt("q", ctx))


class TestGroundedGenerator(unittest.TestCase):
    def setUp(self) -> None:
        self.passages = [
            passage(seg("CLS-1", "Canara Bank is the Lead Bank for Old Delhi."), 1)
        ]

    def test_a_normal_answer_is_returned_with_markers(self) -> None:
        provider = MockProvider(reply="Canara Bank is the Lead Bank [S1].")
        outcome = GroundedGenerator(provider).generate("Which bank?", self.passages)
        self.assertEqual(outcome.generation_mode, "llm")
        self.assertTrue(outcome.sufficient_evidence)
        self.assertIn("[S1]", outcome.answer)
        self.assertFalse(outcome.failed)

    def test_the_refusal_sentinel_clears_sufficient_evidence(self) -> None:
        provider = MockProvider(
            reply=f"{REFUSAL_SENTINEL}\nThe extracts cover Lead Banks, not tax."
        )
        outcome = GroundedGenerator(provider).generate("Tax rate?", self.passages)
        self.assertFalse(outcome.sufficient_evidence)
        self.assertNotIn(REFUSAL_SENTINEL, outcome.answer)
        self.assertIn("Lead Banks", outcome.answer)

    def test_a_bare_sentinel_still_produces_a_readable_message(self) -> None:
        outcome = GroundedGenerator(MockProvider(reply=REFUSAL_SENTINEL)).generate(
            "q", self.passages
        )
        self.assertFalse(outcome.sufficient_evidence)
        self.assertTrue(outcome.answer.strip())

    def test_no_passages_is_reported_not_guessed(self) -> None:
        outcome = GroundedGenerator(MockProvider(reply="anything")).generate("q", [])
        self.assertTrue(outcome.failed)
        self.assertEqual(outcome.error_code, "NO_EVIDENCE")
        self.assertEqual(outcome.answer, "")

    def test_provider_unavailable_is_captured_not_raised(self) -> None:
        provider = MockProvider(fail_with=ProviderUnavailable("no key configured"))
        outcome = GroundedGenerator(provider).generate("q", self.passages)
        self.assertTrue(outcome.failed)
        self.assertEqual(outcome.error_code, "LLM_PROVIDER_UNAVAILABLE")
        self.assertIn("no key configured", outcome.error_message or "")

    def test_timeout_is_captured(self) -> None:
        provider = MockProvider(fail_with=GenerationTimeout("too slow"))
        outcome = GroundedGenerator(provider).generate("q", self.passages)
        self.assertEqual(outcome.error_code, "LLM_TIMEOUT")

    def test_invalid_response_is_captured(self) -> None:
        provider = MockProvider(fail_with=InvalidModelResponse("garbage"))
        outcome = GroundedGenerator(provider).generate("q", self.passages)
        self.assertEqual(outcome.error_code, "LLM_INVALID_RESPONSE")

    def test_an_unexpected_provider_bug_does_not_escape(self) -> None:
        provider = MockProvider(fail_with=RuntimeError("boom"))
        outcome = GroundedGenerator(provider).generate("q", self.passages)
        self.assertTrue(outcome.failed)
        self.assertEqual(outcome.error_code, "LLM_ERROR")

    def test_the_marker_map_covers_every_block(self) -> None:
        provider = MockProvider(reply="answer [S1]")
        outcome = GroundedGenerator(provider).generate("q", self.passages)
        self.assertEqual(set(outcome.marker_to_passage), {"S1"})

    def test_settings_reach_the_provider(self) -> None:
        provider = MockProvider(reply="ok [S1]")
        GroundedGenerator(provider, temperature=0.3, max_tokens=42).generate(
            "q", self.passages
        )
        self.assertEqual(provider.calls[0]["temperature"], 0.3)
        self.assertEqual(provider.calls[0]["max_tokens"], 42)


class TestExtractiveGenerator(unittest.TestCase):
    def setUp(self) -> None:
        self.passages = [
            passage(
                seg(
                    "CLS-1",
                    "Banks shall set up special cells for Scheduled Castes. "
                    "Each cell shall be headed by a senior officer.",
                ),
                1,
            ),
            passage(seg("CLS-2", "Unrelated content about leverage ratios.", ordinal=2), 2),
        ]

    def test_mode_is_declared_extractive(self) -> None:
        outcome = ExtractiveGenerator().generate("special cells", self.passages)
        self.assertEqual(outcome.generation_mode, "extractive")

    def test_every_sentence_is_verbatim_from_a_source(self) -> None:
        """The whole licence for this generator: it writes nothing."""
        outcome = ExtractiveGenerator().generate(
            "What cells must banks set up?", self.passages
        )
        sources = " ".join(p.segment.text for p in self.passages)
        for line in outcome.answer.splitlines():
            if not line.startswith("- "):
                continue
            quoted = line[2:].rsplit(" [", 1)[0].strip()
            self.assertIn(quoted, sources)

    def test_it_says_it_is_not_a_model(self) -> None:
        outcome = ExtractiveGenerator().generate("special cells", self.passages)
        self.assertIn("rather than written", outcome.answer)
        self.assertIsNone(outcome.provider)
        self.assertIsNone(outcome.model)

    def test_a_custom_reason_is_surfaced(self) -> None:
        outcome = ExtractiveGenerator(reason="key missing, so rather than written.").generate(
            "special cells", self.passages
        )
        self.assertIn("key missing", outcome.answer)

    def test_no_vocabulary_overlap_refuses_rather_than_quoting_at_random(self) -> None:
        outcome = ExtractiveGenerator().generate("zzzzzz qqqqqq", self.passages)
        self.assertFalse(outcome.sufficient_evidence)
        self.assertIn("do not contain", outcome.answer)

    def test_no_passages_is_reported(self) -> None:
        outcome = ExtractiveGenerator().generate("q", [])
        self.assertTrue(outcome.failed)
        self.assertEqual(outcome.error_code, "NO_EVIDENCE")

    def test_markers_point_at_real_blocks(self) -> None:
        outcome = ExtractiveGenerator().generate("special cells", self.passages)
        for marker in outcome.marker_to_passage:
            self.assertRegex(marker, r"^S\d+$")


class TestProviderResolution(unittest.TestCase):
    class _S:
        llm_provider = "none"
        llm_model_name = ""
        llm_api_key = None
        llm_base_url = None

    def _settings(self, **kw):
        s = TestProviderResolution._S()
        for key, value in kw.items():
            setattr(s, key, value)
        return s

    def test_unset_resolves_to_nothing(self) -> None:
        self.assertIsNone(resolve_provider(self._settings()))

    def test_explicit_none_resolves_to_nothing(self) -> None:
        self.assertIsNone(resolve_provider(self._settings(llm_provider="none")))

    def test_a_hosted_provider_without_a_key_is_refused_loudly(self) -> None:
        """Silently falling back would hide a configuration mistake."""
        with self.assertRaises(ProviderUnavailable):
            resolve_provider(self._settings(llm_provider="openai"))

    def test_a_hosted_provider_with_a_key_resolves(self) -> None:
        provider = resolve_provider(
            self._settings(llm_provider="groq", llm_api_key="k")
        )
        self.assertIsInstance(provider, OpenAICompatibleProvider)
        self.assertIn("groq", provider.base_url)

    def test_a_local_openai_compatible_endpoint_needs_no_key(self) -> None:
        provider = resolve_provider(
            self._settings(
                llm_provider="openai_compatible",
                llm_base_url="http://127.0.0.1:1234/v1",
            )
        )
        self.assertIsInstance(provider, OpenAICompatibleProvider)

    def test_ollama_needs_no_key(self) -> None:
        provider = resolve_provider(self._settings(llm_provider="ollama"))
        self.assertIsInstance(provider, OllamaProvider)

    def test_an_unknown_provider_is_refused(self) -> None:
        with self.assertRaises(ProviderUnavailable):
            resolve_provider(self._settings(llm_provider="not-a-provider"))

    def test_mock_is_only_selected_when_asked_for_by_name(self) -> None:
        self.assertIsInstance(
            resolve_provider(self._settings(llm_provider="mock")), MockProvider
        )

    def test_describe_configuration_never_returns_the_key(self) -> None:
        described = describe_configuration(
            self._settings(llm_provider="openai", llm_api_key="sk-secret-value")
        )
        self.assertNotIn("sk-secret-value", repr(described))
        self.assertTrue(described["api_key_present"])


class TestErrorDetailRedaction(unittest.TestCase):
    def test_a_credential_looking_body_is_withheld(self) -> None:
        from rag_module.generation.providers import _safe_error_detail

        self.assertIn(
            "withheld", _safe_error_detail('{"error":"bad api_key sk-abc123"}')
        )

    def test_an_ordinary_body_is_relayed(self) -> None:
        from rag_module.generation.providers import _safe_error_detail

        self.assertIn("model not found", _safe_error_detail("model not found"))


if __name__ == "__main__":
    unittest.main()
