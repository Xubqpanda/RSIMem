from __future__ import annotations

from rsimem.memory.extraction_projection import ExtractionActivationFingerprint
from rsimem.memory.live_writeback import (
    ExtractionPromptRuntimeScope,
    ExtractionRuntimeBinding,
)
from rsimem.memory.prompt_components import SemanticPolicyManifest, content_digest
from rsimem.memory.extraction_policy_artifact import ExtractionPromptPolicyArtifact
from rsimem.memory_systems.mem0_flat.policy import ExtractionInvocationFingerprint


def extraction_activation_fixture(
    *,
    compilation_id: str,
    extraction_operation_id: str,
    component_artifact_id: str,
    component_artifact_digest: str,
    parsed_output_digest: str,
    persisted_artifact_ids: tuple[str, ...] = (),
    mutation_ids: tuple[str, ...] = (),
    policy_artifact_id: str | None = None,
    policy_artifact_digest: str | None = None,
    matched_validation: bool = False,
    semantic_policy: SemanticPolicyManifest | None = None,
    policy_artifact: ExtractionPromptPolicyArtifact | None = None,
) -> ExtractionActivationFingerprint:
    policy_id = (
        policy_artifact.artifact_id
        if policy_artifact is not None
        else policy_artifact_id
    ) or (
        f"extraction-prompt.fixture.{content_digest(component_artifact_id)[:24]}"
    )
    policy_digest = (
        policy_artifact.artifact_digest
        if policy_artifact is not None
        else policy_artifact_digest
    ) or content_digest({
        "fixture": "extraction-policy",
        "component": component_artifact_id,
    })
    binding_id = (
        f"prompt-binding.fixture.{content_digest((component_artifact_id, component_artifact_digest))[:24]}"
    )
    template_digest = content_digest({
        "fixture": "rendered-template",
        "component": component_artifact_id,
    })
    model_profile = (
        semantic_policy.model_profile
        if semantic_policy is not None
        else "semantic-ingestion-default-v1"
    )
    semantic_policy = semantic_policy or SemanticPolicyManifest.create(
        route="hermes-native-semantic",
        boundary="task-completed-v1",
        backend="hermes-native-semantic",
        framework_version="mem0-flat-framework-v1",
        model_profile=model_profile,
        extraction_component_id=component_artifact_id,
        extraction_component_digest=component_artifact_digest,
        update_component_id="prompt-component.update.fixture",
        update_component_digest=content_digest("fixture-update-component"),
        retrieval_component_id="retrieval-config.fixture",
        retrieval_component_digest=content_digest("fixture-retrieval-component"),
    )
    binding = ExtractionRuntimeBinding(
        deployment_scope=(
            ExtractionPromptRuntimeScope.MATCHED_VALIDATION
            if matched_validation
            else ExtractionPromptRuntimeScope.ROOT_STATIC
        ),
        policy_artifact_id=policy_id,
        policy_artifact_digest=policy_digest,
        policy_version=(
            policy_artifact.policy_version
            if policy_artifact is not None
            else "candidate-v1" if matched_validation else "root-v1"
        ),
        component_artifact_id=component_artifact_id,
        component_body_digest=component_artifact_digest,
        binding_id=binding_id,
        adapter_id="mem0-flat-prompt-adapter-v1",
        slot_id=(
            policy_artifact.slot_id
            if policy_artifact is not None
            else "mem0-flat.semantic.extraction"
        ),
        slot_contract_digest=(
            policy_artifact.slot_contract_digest
            if policy_artifact is not None
            else content_digest("fixture-slot-contract")
        ),
        frozen_wrapper_digest=(
            policy_artifact.frozen_wrapper_digest
            if policy_artifact is not None
            else content_digest("fixture-frozen-wrapper")
        ),
        input_schema_digest=(
            policy_artifact.input_schema_digest
            if policy_artifact is not None
            else content_digest("fixture-input-schema")
        ),
        output_schema_digest=(
            policy_artifact.output_schema_digest
            if policy_artifact is not None
            else content_digest("fixture-output-schema")
        ),
        rendered_template_digest=template_digest,
        model_profile=model_profile,
        trial_id="extraction-trial.fixture" if matched_validation else None,
    )
    invocation = ExtractionInvocationFingerprint(
        render_id=(
            f"render.fixture.{content_digest((compilation_id, component_artifact_id))[:24]}"
        ),
        render_input_digest=content_digest({
            "fixture": "render-input",
            "compilation_id": compilation_id,
        }),
        rendered_template_digest=template_digest,
        model_output_digest=content_digest({
            "fixture": "model-output",
            "parsed_output_digest": parsed_output_digest,
        }),
        binding_id=binding_id,
    )
    return ExtractionActivationFingerprint.create(
        compilation_id=compilation_id,
        extraction_operation_id=extraction_operation_id,
        runtime_binding=binding,
        semantic_policy=semantic_policy,
        invocation=invocation,
        parsed_output_digest=parsed_output_digest,
        mutation_ids=mutation_ids,
        persisted_artifact_ids=persisted_artifact_ids,
    )
