"""Configuration management for NiyamSetu RAG module.

Loads settings from environment variables with sensible defaults designed for
lightweight, reproducible execution. No external configuration libraries are required.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Settings:
    """Independent configuration for the RAG module.

    All properties can be overridden by environment variables prefixed with
    ``NIYAMSETU_RAG_`` or standard names.
    """

    # Backend connection
    backend_base_url: str = "http://127.0.0.1:8000/api"
    http_timeout_seconds: float = 10.0
    http_max_retries: int = 2

    # Retrieval parameters
    default_segmentation_method: str = "clause_aware"
    default_retrieval_strategy: str = "hybrid"
    retrieval_top_k: int = 5
    rrf_k: int = 60  # Smoothing constant for Reciprocal Rank Fusion

    # Embedding model configuration
    # Tier 1 (Default development & fast CPU): intfloat/multilingual-e5-small
    # Tier 2 (Benchmark & final report): BAAI/bge-m3
    embedding_tier: str = "tier1"
    embedding_model_name: str = "intfloat/multilingual-e5-small"
    embedding_device: str = "cpu"
    embedding_batch_size: int = 32

    # Generation & Sufficiency
    llm_provider: str = "mock"  # "mock" | "ollama" | "api"
    llm_model_name: str = "mock-model"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 512
    llm_api_key: Optional[str] = None
    llm_base_url: Optional[str] = None

    # Offline / Fixture path
    fixture_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent
        / "client"
        / "fixtures"
    )

    @classmethod
    def from_env(cls) -> Settings:
        """Instantiate Settings by reading environment variables."""

        def _get_str(name: str, alt_name: str, default: str) -> str:
            return os.environ.get(name, os.environ.get(alt_name, default))

        def _get_float(name: str, alt_name: str, default: float) -> float:
            val = os.environ.get(name, os.environ.get(alt_name))
            if val is not None:
                try:
                    return float(val)
                except ValueError:
                    pass
            return default

        def _get_int(name: str, alt_name: str, default: int) -> int:
            val = os.environ.get(name, os.environ.get(alt_name))
            if val is not None:
                try:
                    return int(val)
                except ValueError:
                    pass
            return default

        tier = _get_str("NIYAMSETU_RAG_EMBEDDING_TIER", "EMBEDDING_TIER", "tier1").lower()
        default_model = (
            "BAAI/bge-m3" if tier == "tier2" else "intfloat/multilingual-e5-small"
        )

        fixture_path_str = os.environ.get("NIYAMSETU_RAG_FIXTURE_DIR")
        fixture_path = (
            Path(fixture_path_str).resolve()
            if fixture_path_str
            else Path(__file__).resolve().parent.parent / "client" / "fixtures"
        )

        return cls(
            backend_base_url=_get_str(
                "NIYAMSETU_RAG_BACKEND_URL",
                "NIYAMSETU_BACKEND_URL",
                "http://127.0.0.1:8000/api",
            ).rstrip("/"),
            http_timeout_seconds=_get_float(
                "NIYAMSETU_RAG_HTTP_TIMEOUT", "HTTP_TIMEOUT", 10.0
            ),
            http_max_retries=_get_int(
                "NIYAMSETU_RAG_HTTP_MAX_RETRIES", "HTTP_MAX_RETRIES", 2
            ),
            default_segmentation_method=_get_str(
                "NIYAMSETU_RAG_DEFAULT_METHOD", "DEFAULT_METHOD", "clause_aware"
            ),
            default_retrieval_strategy=_get_str(
                "NIYAMSETU_RAG_RETRIEVAL_STRATEGY", "RETRIEVAL_STRATEGY", "hybrid"
            ),
            retrieval_top_k=_get_int(
                "NIYAMSETU_RAG_TOP_K", "RETRIEVAL_TOP_K", 5
            ),
            rrf_k=_get_int("NIYAMSETU_RAG_RRF_K", "RRF_K", 60),
            embedding_tier=tier,
            embedding_model_name=_get_str(
                "NIYAMSETU_RAG_EMBEDDING_MODEL", "EMBEDDING_MODEL", default_model
            ),
            embedding_device=_get_str(
                "NIYAMSETU_RAG_EMBEDDING_DEVICE", "EMBEDDING_DEVICE", "cpu"
            ),
            embedding_batch_size=_get_int(
                "NIYAMSETU_RAG_BATCH_SIZE", "EMBEDDING_BATCH_SIZE", 32
            ),
            llm_provider=_get_str(
                "NIYAMSETU_RAG_LLM_PROVIDER", "LLM_PROVIDER", "mock"
            ),
            llm_model_name=_get_str(
                "NIYAMSETU_RAG_LLM_MODEL", "LLM_MODEL", "mock-model"
            ),
            llm_temperature=_get_float(
                "NIYAMSETU_RAG_LLM_TEMPERATURE", "LLM_TEMPERATURE", 0.0
            ),
            llm_max_tokens=_get_int(
                "NIYAMSETU_RAG_LLM_MAX_TOKENS", "LLM_MAX_TOKENS", 512
            ),
            llm_api_key=os.environ.get("NIYAMSETU_RAG_LLM_API_KEY", os.environ.get("LLM_API_KEY")),
            llm_base_url=os.environ.get("NIYAMSETU_RAG_LLM_BASE_URL", os.environ.get("LLM_BASE_URL")),
            fixture_dir=fixture_path,
        )


_DEFAULT_SETTINGS: Optional[Settings] = None


def get_settings() -> Settings:
    """Return the global Settings instance, initializing from environment if needed."""
    global _DEFAULT_SETTINGS
    if _DEFAULT_SETTINGS is None:
        _DEFAULT_SETTINGS = Settings.from_env()
    return _DEFAULT_SETTINGS


def reset_settings() -> None:
    """Reset cached settings (primarily for test isolation)."""
    global _DEFAULT_SETTINGS
    _DEFAULT_SETTINGS = None

