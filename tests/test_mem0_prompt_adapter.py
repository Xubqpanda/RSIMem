from __future__ import annotations

import json
from dataclasses import replace

import pytest

from rsimem.lifecycle import RawResourceUsage
from rsimem.memory.extraction_policy_artifact import (
    ExtractionGenerationProvenance,
    ExtractionPolicyRule,
    ExtractionPromptPolicyArtifact,
    ExtractionRuleEdit,
    ExtractionRuleEditAction,
    serialize_extraction_prompt_artifact,
)
from rsimem.memory.extraction_policy_store import (
    ExtractionPolicyState,
    JsonExtractionPolicyStore,
)
from rsimem.memory.prompt_components import (
    PromptAdapterRegistry,
    PromptComponentArtifact,
)
from rsimem.memory_systems.mem0_flat import (
    MEM0_FLAT_EXTRACTION_MAX_BODY_CHARS,
    MEM0_FLAT_EXTRACTION_SLOT,
    MEM0_FLAT_EXTRACTION_SLOT_ID,
    POLICY_FACT_EXTRACTION_FROZEN_WRAPPER,
    POLICY_FACT_EXTRACTION_ROOT_BODY,
    FakeCompletionClient,
    Mem0FlatPromptAdapter,
    Mem0FlatSemanticPolicy,
    compile_policy_fact_extraction_template,
)


EXPECTED_ROOT_COMPONENT_ID = (
    "prompt-component.9058cc5994f5ee77da51e21131aa0198be48d148"
)


class _FakeExtractionArtifactLoader:
    def load(self, artifact: ExtractionPromptPolicyArtifact) -> tuple[str, str]:
        component = artifact.to_prompt_component(MEM0_FLAT_EXTRACTION_SLOT)
        rendered = POLICY_FACT_EXTRACTION_FROZEN_WRAPPER.replace(
            "$policy_body",
            component.policy_body,
            1,
        )
        return component.artifact_id, rendered


def _generation_provenance() -> ExtractionGenerationProvenance:
    return ExtractionGenerationProvenance(
        optimizer_model="optimizer-model-v1",
        optimizer_config_digest="4" * 64,
        training_corpus_id="corpus-v1",
        training_cutoff="cutoff-v1",
        proposal_request_digest="5" * 64,
        completion_digest="6" * 64,
        usage=RawResourceUsage(
            input_tokens=10,
            output_tokens=2,
            model_requests=1,
        ),
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


def test_mem0_adapter_exports_reloadable_exact_root_policy() -> None:
    adapter = Mem0FlatPromptAdapter()
    root_component = adapter.root_artifact(MEM0_FLAT_EXTRACTION_SLOT_ID)
    root_policy = adapter.export_root_policy_artifact(
        MEM0_FLAT_EXTRACTION_SLOT_ID
    )

    assert root_policy.compiled_body == POLICY_FACT_EXTRACTION_ROOT_BODY
    assert root_policy.max_body_chars == MEM0_FLAT_EXTRACTION_MAX_BODY_CHARS
    assert tuple(rule.rule_id for rule in root_policy.spec.rules) == (
        "durable-candidates",
        "future-useful-scope",
        "source-safety-exclusions",
        "standalone-candidates",
        "output-schema",
    )
    assert tuple(rule.rule_id for rule in root_policy.spec.rules if rule.protected) == (
        "durable-candidates",
        "source-safety-exclusions",
        "output-schema",
    )
    assert root_component.artifact_id == EXPECTED_ROOT_COMPONENT_ID
    assert root_component.body_digest == root_policy.body_digest

    loaded = ExtractionPromptPolicyArtifact.from_payload(json.loads(
        serialize_extraction_prompt_artifact(root_policy)
    ))
    fingerprint = adapter.bind_policy_artifact(
        MEM0_FLAT_EXTRACTION_SLOT_ID,
        loaded,
    )
    assert fingerprint.artifact_id == EXPECTED_ROOT_COMPONENT_ID
    assert adapter.bound_template(fingerprint).template == (
        compile_policy_fact_extraction_template(POLICY_FACT_EXTRACTION_ROOT_BODY)
    )

    fake_component_id, fake_rendered = _FakeExtractionArtifactLoader().load(loaded)
    assert fake_component_id == EXPECTED_ROOT_COMPONENT_ID
    assert fake_rendered == adapter.bound_template(fingerprint).template


def test_mem0_adapter_binds_rich_child_without_changing_frozen_components() -> None:
    adapter = Mem0FlatPromptAdapter()
    root = adapter.export_root_policy_artifact(MEM0_FLAT_EXTRACTION_SLOT_ID)
    child = ExtractionPromptPolicyArtifact.create_child(
        parent=root,
        policy_version="candidate-v2",
        edits=(ExtractionRuleEdit(
            "edit.expand-durable-scope",
            ExtractionRuleEditAction.REPLACE,
            "future-useful-scope",
            ExtractionPolicyRule(
                "future-useful-scope",
                "Keep user-supplied durable facts, preferences, rules, and constraints "
                "that can help a future task.",
            ),
        ),),
        generation_provenance=_generation_provenance(),
    )

    root_binding = adapter.bind_policy_artifact(
        MEM0_FLAT_EXTRACTION_SLOT_ID,
        root,
    )
    child_binding = adapter.bind_policy_artifact(
        MEM0_FLAT_EXTRACTION_SLOT_ID,
        child,
    )
    root_policy = Mem0FlatSemanticPolicy(
        FakeCompletionClient({}),
        fact_prompt=adapter.bound_template(root_binding),
        extraction_binding=root_binding,
    )
    child_policy = Mem0FlatSemanticPolicy(
        FakeCompletionClient({}),
        fact_prompt=adapter.bound_template(child_binding),
        extraction_binding=child_binding,
    )

    assert child_binding.artifact_body_digest == child.body_digest
    assert root_policy.semantic_manifest.extraction_component_digest != (
        child_policy.semantic_manifest.extraction_component_digest
    )
    assert root_policy.semantic_manifest.update_component_digest == (
        child_policy.semantic_manifest.update_component_digest
    )
    assert root_policy.semantic_manifest.retrieval_component_digest == (
        child_policy.semantic_manifest.retrieval_component_digest
    )


def test_mem0_root_durability_rule_cannot_be_replaced() -> None:
    root = Mem0FlatPromptAdapter().export_root_policy_artifact(
        MEM0_FLAT_EXTRACTION_SLOT_ID
    )
    with pytest.raises(ValueError, match="protected"):
        ExtractionPromptPolicyArtifact.create_child(
            parent=root,
            policy_version="candidate-weakened-v2",
            edits=(ExtractionRuleEdit(
                "edit.weaken-durability",
                ExtractionRuleEditAction.REPLACE,
                "durable-candidates",
                ExtractionPolicyRule(
                    "durable-candidates",
                    "Extract any information from the experience.",
                ),
            ),),
            generation_provenance=_generation_provenance(),
        )


def test_mem0_active_child_renders_identically_after_store_restart(tmp_path) -> None:
    adapter = Mem0FlatPromptAdapter()
    root = adapter.export_root_policy_artifact(MEM0_FLAT_EXTRACTION_SLOT_ID)
    child = ExtractionPromptPolicyArtifact.create_child(
        parent=root,
        policy_version="candidate-restart-v2",
        edits=(ExtractionRuleEdit(
            "edit.restart-scope",
            ExtractionRuleEditAction.REPLACE,
            "future-useful-scope",
            ExtractionPolicyRule(
                "future-useful-scope",
                "Keep durable user facts and preferences that can help later tasks.",
            ),
        ),),
        generation_provenance=_generation_provenance(),
    )
    store_path = tmp_path / "extraction-policy-store.json"
    store = JsonExtractionPolicyStore(
        store_path,
        trusted_root=root,
        slot=MEM0_FLAT_EXTRACTION_SLOT,
    )
    store.register(child)
    store.transition(
        child.artifact_id,
        to_state=ExtractionPolicyState.ACTIVE,
        transition_id="transition.activate-restart-v2",
        reason_code="validation_passed",
    )
    before_binding = adapter.bind_policy_artifact(
        MEM0_FLAT_EXTRACTION_SLOT_ID,
        child,
    )
    before_template = adapter.bound_template(before_binding)

    restarted_store = JsonExtractionPolicyStore(
        store_path,
        trusted_root=root,
        slot=MEM0_FLAT_EXTRACTION_SLOT,
    )
    restarted_adapter = Mem0FlatPromptAdapter()
    after_binding = restarted_adapter.bind_policy_artifact(
        MEM0_FLAT_EXTRACTION_SLOT_ID,
        restarted_store.active_or_root(),
    )
    after_template = restarted_adapter.bound_template(after_binding)

    assert after_binding == before_binding
    assert after_template.template == before_template.template
    assert after_template.artifact.template_digest == (
        before_template.artifact.template_digest
    )


def test_mem0_policy_artifact_bridge_fails_closed_on_contract_drift() -> None:
    adapter = Mem0FlatPromptAdapter()
    root = adapter.export_root_policy_artifact(MEM0_FLAT_EXTRACTION_SLOT_ID)
    wrong_slot = replace(
        MEM0_FLAT_EXTRACTION_SLOT,
        frozen_wrapper_digest="9" * 64,
    )
    wrong_artifact = ExtractionPromptPolicyArtifact.create_root(
        slot=wrong_slot,
        policy_version="root-v1",
        spec=root.spec,
        max_body_chars=MEM0_FLAT_EXTRACTION_MAX_BODY_CHARS,
        source_provenance="fixture-wrong-wrapper",
    )

    with pytest.raises(KeyError, match="unknown Mem0-flat prompt slot"):
        adapter.export_root_policy_artifact("other.semantic.extraction")
    with pytest.raises(ValueError, match="runtime slot"):
        adapter.bind_policy_artifact(
            MEM0_FLAT_EXTRACTION_SLOT_ID,
            wrong_artifact,
        )
    with pytest.raises(TypeError, match="wrong type"):
        adapter.bind_policy_artifact(
            MEM0_FLAT_EXTRACTION_SLOT_ID,
            adapter.root_artifact(MEM0_FLAT_EXTRACTION_SLOT_ID),  # type: ignore[arg-type]
        )
