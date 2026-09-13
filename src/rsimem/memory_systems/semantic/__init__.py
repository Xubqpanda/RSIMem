"""Semantic memory systems and their capability declarations."""

from .mem0_flat import (
    MEM0_FLAT_EXTRACTION_SLOT,
    MEM0_FLAT_PROMPT_ADAPTER_ID,
    Mem0FlatPromptAdapter,
    Mem0FlatSemanticPolicy,
)
from .hermes_native import HermesSemanticBackend, semantic_artifact_id

__all__ = [
    "MEM0_FLAT_EXTRACTION_SLOT",
    "MEM0_FLAT_PROMPT_ADAPTER_ID",
    "Mem0FlatPromptAdapter",
    "Mem0FlatSemanticPolicy",
    "HermesSemanticBackend",
    "semantic_artifact_id",
]
