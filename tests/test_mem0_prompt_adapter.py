from __future__ import annotations

from dataclasses import replace

import pytest

from rsimem.memory.prompt_components import (
    PromptAdapterRegistry,
    PromptComponentArtifact,
)
from rsimem.memory_systems.mem0_flat import (
    MEM0_FLAT_EXTRACTION_SLOT,
    MEM0_FLAT_EXTRACTION_SLOT_ID,
    FakeCompletionClient,
    Mem0FlatPromptAdapter,
    Mem0FlatSemanticPolicy,
)


def _bound_policy(
    adapter: Mem0FlatPromptAdapter,
    registry: PromptAdapterRegistry,
    artifact: PromptComponentArtifact,
):
    fingerprint = registry.bind(MEM0_FLAT_EXTRACTION_SLOT_ID, artifact)
    policy = Mem0FlatSemanticPolicy(
        FakeCompletionClient({}),
        fact_prompt=adapter.bound_template(fingerprint),
        extraction_binding=fingerprint,
    )
    return fingerprint, policy


def test_mem0_adapter_binds_root_and_independent_extraction_replacement() -> None:
    adapter = Mem0FlatPromptAdapter()
    registry = PromptAdapterRegistry()
    registry.register(adapter)
    root = registry.root_artifact(MEM0_FLAT_EXTRACTION_SLOT_ID)
    root_binding, root_policy = _bound_policy(adapter, registry, root)
    child = PromptComponentArtifact.create(
        slot=MEM0_FLAT_EXTRACTION_SLOT,
        version="candidate-v2",
        policy_body=(
            "Extract stable user-provided preferences, facts, and constraints. "
            "Exclude temporary or unsupported claims."
        ),
        parent_artifact_id=root.artifact_id,
    )
    child_binding, child_policy = _bound_policy(adapter, registry, child)

    assert root_binding.binding_id != child_binding.binding_id
    assert root_policy.fact_prompt.binding_fingerprint == root_binding.binding_id
    assert child_policy.fact_prompt.binding_fingerprint == child_binding.binding_id
    assert root_policy.semantic_manifest.extraction_component_digest == root.body_digest
    assert child_policy.semantic_manifest.extraction_component_digest == child.body_digest
    assert root_policy.semantic_manifest.update_component_digest == (
        child_policy.semantic_manifest.update_component_digest
    )
    assert root_policy.semantic_manifest.retrieval_component_digest == (
        child_policy.semantic_manifest.retrieval_component_digest
    )
    assert root_policy.semantic_manifest.composite_digest != (
        child_policy.semantic_manifest.composite_digest
    )


def test_mem0_adapter_requires_registration_slot_and_actual_bound_template() -> None:
    adapter = Mem0FlatPromptAdapter()
    registry = PromptAdapterRegistry()
    with pytest.raises(KeyError, match="unregistered prompt slot"):
        registry.root_artifact(MEM0_FLAT_EXTRACTION_SLOT_ID)
    registry.register(adapter)
    root = registry.root_artifact(MEM0_FLAT_EXTRACTION_SLOT_ID)
    with pytest.raises(KeyError, match="unregistered prompt slot"):
        registry.bind("python.module:POLICY_FACT_EXTRACTION_PROMPT", root)
    with pytest.raises(ValueError, match="artifact ID mismatch"):
        registry.bind(
            MEM0_FLAT_EXTRACTION_SLOT_ID,
            replace(root, source_provenance="python.module:source-only"),
        )

    registry.bind(MEM0_FLAT_EXTRACTION_SLOT_ID, root)
    other_adapter = Mem0FlatPromptAdapter()
    other_registry = PromptAdapterRegistry()
    other_registry.register(other_adapter)
    child = PromptComponentArtifact.create(
        slot=MEM0_FLAT_EXTRACTION_SLOT,
        version="other-v2",
        policy_body="Extract stable facts only.",
        parent_artifact_id=root.artifact_id,
    )
    wrong_binding = other_registry.bind(MEM0_FLAT_EXTRACTION_SLOT_ID, child)
    with pytest.raises(KeyError, match="binding is not active"):
        adapter.bound_template(wrong_binding)


def test_mem0_policy_rejects_unbound_or_mismatched_extraction_fingerprint() -> None:
    adapter = Mem0FlatPromptAdapter()
    registry = PromptAdapterRegistry()
    registry.register(adapter)
    root = registry.root_artifact(MEM0_FLAT_EXTRACTION_SLOT_ID)
    binding = registry.bind(MEM0_FLAT_EXTRACTION_SLOT_ID, root)
    template = adapter.bound_template(binding)

    with pytest.raises(ValueError, match="does not match fact prompt"):
        Mem0FlatSemanticPolicy(
            FakeCompletionClient({}),
            extraction_binding=binding,
        )
    with pytest.raises(ValueError, match="does not match fact prompt"):
        Mem0FlatSemanticPolicy(
            FakeCompletionClient({}),
            fact_prompt=replace(template, binding_fingerprint=None),
            extraction_binding=binding,
        )
