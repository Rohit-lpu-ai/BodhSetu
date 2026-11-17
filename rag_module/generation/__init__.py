"""Answer generation: context construction, prompting, providers, generators."""

from rag_module.generation.context import (
    DEFAULT_CONTEXT_CHAR_BUDGET,
    DEFAULT_MAX_PASSAGE_CHARS,
    BuiltContext,
    ContextBlock,
    build_context,
)
from rag_module.generation.generator import (
    ExtractiveGenerator,
    GenerationOutcome,
    GroundedGenerator,
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
    LLMResult,
    MockProvider,
    OllamaProvider,
    OpenAICompatibleProvider,
    ProviderUnavailable,
    describe_configuration,
    resolve_provider,
)

__all__ = [
    "BuiltContext",
    "ContextBlock",
    "build_context",
    "DEFAULT_CONTEXT_CHAR_BUDGET",
    "DEFAULT_MAX_PASSAGE_CHARS",
    "SYSTEM_PROMPT",
    "REFUSAL_SENTINEL",
    "build_user_prompt",
    "BaseLLMProvider",
    "OpenAICompatibleProvider",
    "OllamaProvider",
    "MockProvider",
    "LLMResult",
    "LLMError",
    "ProviderUnavailable",
    "GenerationTimeout",
    "InvalidModelResponse",
    "resolve_provider",
    "describe_configuration",
    "GroundedGenerator",
    "ExtractiveGenerator",
    "GenerationOutcome",
]
