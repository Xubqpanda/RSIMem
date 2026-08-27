from __future__ import annotations

from dataclasses import replace

import pytest

from rsimem.memory import MemoryKind
from rsimem.memory.prompt_components import (
    MemoryPromptAdapter,
    PromptAdapterRegistry,
    PromptBindingFingerprint,
    PromptComponentArtifact,
    PromptPolicyStage,
    PromptSlotDescriptor,
    SemanticPolicyManifest,
    content_digest,
    text_digest,
)


WRAPPER = "Policy:\n{policy_body}\nSource:\n{source_messages}\n"


def _slot(
    *,
    slot_id: str = "fake.semantic.extraction",
    owner: str = "fake-memory-adapter-v1",
) -> PromptSlotDescriptor:
    return PromptSlotDescriptor(
        slot_id=slot_id,
        memory_kind=MemoryKind.SEMANTIC,
        policy_stage=PromptPolicyStage.EXTRACTION,
        input_schema_digest=content_digest({"required": ["source_messages"]}),
        output_schema_digest=content_digest({"required": ["facts"]}),
        frozen_wrapper_digest=text_digest(WRAPPER),
        model_profile="fixture-model-v1",
        owner_adapter_id=owner,
        required_placeholders=("policy_body", "source_messages"),
    )


class _FakeAdapter(MemoryPromptAdapter):
    adapter_id = "fake-memory-adapter-v1"

    def __init__(self, slot: PromptSlotDescriptor | None = None) -> None:
        self.slot = slot or _slot()
        self.root = PromptComponentArtifact.create(
            slot=self.slot,
            version="root-v1",
            policy_body="Extract stable user preferences only.",
            source_provenance="fixture-root",
        )
        self.bound: PromptComponentArtifact | None = None

    def list_slots(self) -> tuple[PromptSlotDescriptor, ...]:
        return (self.slot,)

    def root_artifact(self, slot_id: str) -> PromptComponentArtifact:
        if slot_id != self.slot.slot_id:
            raise KeyError(slot_id)
        return self.root

    def validate_replacement(
        self,
        slot_id: str,
        artifact: PromptComponentArtifact,
    ) -> None:
        if slot_id != self.slot.slot_id:
            raise KeyError(slot_id)
        if len(artifact.policy_body) > 1_000:
            raise ValueError("fake prompt body is oversized")

    def bind(
        self,
        slot_id: str,
        artifact: PromptComponentArtifact,
    ) -> PromptBindingFingerprint:
        self.validate_replacement(slot_id, artifact)
        self.bound = artifact
        rendered = WRAPPER.format(
            policy_body=artifact.policy_body,
            source_messages="{source_messages}",
        )
        return PromptBindingFingerprint.create(
            adapter_id=self.adapter_id,
            slot_id=slot_id,
            slot_contract_digest=self.slot.contract_digest,
            artifact_id=artifact.artifact_id,
            artifact_body_digest=artifact.body_digest,
            rendered_template_digest=text_digest(rendered),
        )

    def render(self, source_messages: str) -> str:
        if self.bound is None:
            raise RuntimeError("fake prompt slot is not bound")
        return WRAPPER.format(
            policy_body=self.bound.policy_body,
            source_messages=source_messages,
        )


def test_host_neutral_fake_adapter_root_replacement_render_and_fingerprint() -> None:
    adapter = _FakeAdapter()
    registry = PromptAdapterRegistry()
    registry.register(adapter)
    root = registry.root_artifact(adapter.slot.slot_id)
    root_binding = registry.bind(adapter.slot.slot_id, root)

    assert isinstance(adapter, MemoryPromptAdapter)
    assert adapter.render("message-a").startswith("Policy:\nExtract stable")
    assert root_binding.artifact_id == root.artifact_id
    assert root_binding.rendered_template_digest == text_digest(
        WRAPPER.format(
            policy_body=root.policy_body,
            source_messages="{source_messages}",
        )
    )

    replacement = PromptComponentArtifact.create(
        slot=adapter.slot,
        version="candidate-v2",
        policy_body="Extract durable preferences and explicit constraints.",
        parent_artifact_id=root.artifact_id,
    )
    replacement_binding = registry.bind(adapter.slot.slot_id, replacement)
    assert replacement_binding != root_binding
    assert replacement.policy_body in adapter.render("message-b")


def test_prompt_registry_fails_closed_on_missing_duplicate_and_wrong_owner() -> None:
    adapter = _FakeAdapter()
    registry = PromptAdapterRegistry()
    with pytest.raises(KeyError, match="unregistered prompt slot"):
        registry.root_artifact(adapter.slot.slot_id)
    registry.register(adapter)
    with pytest.raises(ValueError, match="already registered"):
        registry.register(_FakeAdapter())
    wrong_slot = _slot(slot_id="fake.semantic.other")
    wrong_artifact = PromptComponentArtifact.create(
        slot=wrong_slot,
        version="wrong-v1",
        policy_body="Wrong slot policy.",
    )
    with pytest.raises(ValueError, match="another slot"):
        registry.bind(adapter.slot.slot_id, wrong_artifact)

    wrong_owner_slot = _slot(owner="different-adapter-v1")
    with pytest.raises(ValueError, match="owner"):
        PromptAdapterRegistry().register(_FakeAdapter(wrong_owner_slot))
    with pytest.raises(ValueError, match="artifact ID mismatch"):
        replace(adapter.root, source_provenance="python.module:symbol")


def _manifest(extraction_id: str, extraction_digest: str) -> SemanticPolicyManifest:
    return SemanticPolicyManifest.create(
        route="hermes-native-semantic",
        boundary="task-completed-v1",
        backend="hermes-native-semantic",
        framework_version="mem0-flat-framework-v1",
        model_profile="fixture-model-v1",
        extraction_component_id=extraction_id,
        extraction_component_digest=extraction_digest,
        update_component_id="mem0-flat.semantic.update.root-v1",
        update_component_digest="1" * 64,
        retrieval_component_id="mem0-flat.semantic.retrieval.root-v1",
        retrieval_component_digest="2" * 64,
    )


def test_composite_manifest_changes_only_with_extraction_component() -> None:
    adapter = _FakeAdapter()
    root = adapter.root
    child = PromptComponentArtifact.create(
        slot=adapter.slot,
        version="child-v2",
        policy_body="Extract durable preferences and constraints.",
        parent_artifact_id=root.artifact_id,
    )
    parent_manifest = _manifest(root.artifact_id, root.body_digest)
    child_manifest = _manifest(child.artifact_id, child.body_digest)

    assert parent_manifest.update_component_digest == child_manifest.update_component_digest
    assert parent_manifest.retrieval_component_digest == (
        child_manifest.retrieval_component_digest
    )
    assert parent_manifest.extraction_component_digest != (
        child_manifest.extraction_component_digest
    )
    assert parent_manifest.composite_digest != child_manifest.composite_digest
    assert parent_manifest.composite_policy_version != (
        child_manifest.composite_policy_version
    )
