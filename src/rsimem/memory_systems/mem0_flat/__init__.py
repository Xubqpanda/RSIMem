"""Host-neutral Mem0-flat prompt and completion contracts."""

from .prompts import (
    FACT_EXTRACTION_PROMPT,
    INTERNAL_OPERATION_PROMPT,
    MEMBASE_COMMIT,
    MEMBASE_LICENSE,
    MEMBASE_PROMPT_PATH,
    MEMBASE_SOURCE_DIGEST,
    PROMPT_CONTRACT_SCHEMA_VERSION,
    SEMANTIC_RETRIEVAL_SCORER_PROMPT,
    CompletionResult,
    FakeCompletionClient,
    PromptArtifact,
    PromptSourceProvenance,
    PromptTemplate,
    RenderedPrompt,
    build_prompt_catalog,
)

__all__ = [
    "CompletionResult",
    "FACT_EXTRACTION_PROMPT",
    "FakeCompletionClient",
    "INTERNAL_OPERATION_PROMPT",
    "MEMBASE_COMMIT",
    "MEMBASE_LICENSE",
    "MEMBASE_PROMPT_PATH",
    "MEMBASE_SOURCE_DIGEST",
    "PROMPT_CONTRACT_SCHEMA_VERSION",
    "PromptArtifact",
    "PromptSourceProvenance",
    "PromptTemplate",
    "RenderedPrompt",
    "SEMANTIC_RETRIEVAL_SCORER_PROMPT",
    "build_prompt_catalog",
]
