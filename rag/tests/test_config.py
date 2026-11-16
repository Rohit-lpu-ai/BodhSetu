"""Unit tests for RAG configuration."""

import os
import unittest
from unittest.mock import patch

from rag_module.config.settings import Settings, get_settings, reset_settings


class TestSettings(unittest.TestCase):
    def setUp(self) -> None:
        reset_settings()

    def tearDown(self) -> None:
        reset_settings()

    def test_default_settings(self) -> None:
        cfg = Settings()
        self.assertEqual(cfg.backend_base_url, "http://127.0.0.1:8000/api")
        self.assertEqual(cfg.default_segmentation_method, "clause_aware")
        self.assertEqual(cfg.default_retrieval_strategy, "hybrid")
        self.assertEqual(cfg.retrieval_top_k, 5)
        self.assertEqual(cfg.rrf_k, 60)
        self.assertEqual(cfg.embedding_tier, "tier1")
        self.assertEqual(cfg.embedding_model_name, "intfloat/multilingual-e5-small")
        self.assertEqual(cfg.llm_provider, "mock")
        self.assertIsNone(cfg.llm_api_key)

    @patch.dict(
        os.environ,
        {
            "NIYAMSETU_RAG_BACKEND_URL": "http://custom-host:9000/api/",
            "NIYAMSETU_RAG_TOP_K": "10",
            "NIYAMSETU_RAG_EMBEDDING_TIER": "tier2",
            "NIYAMSETU_RAG_LLM_PROVIDER": "ollama",
        },
    )
    def test_environment_variable_overrides(self) -> None:
        cfg = Settings.from_env()
        self.assertEqual(cfg.backend_base_url, "http://custom-host:9000/api")
        self.assertEqual(cfg.retrieval_top_k, 10)
        self.assertEqual(cfg.embedding_tier, "tier2")
        self.assertEqual(cfg.embedding_model_name, "BAAI/bge-m3")
        self.assertEqual(cfg.llm_provider, "ollama")

    def test_get_settings_caches_instance(self) -> None:
        s1 = get_settings()
        s2 = get_settings()
        self.assertIs(s1, s2)


if __name__ == "__main__":
    unittest.main()

