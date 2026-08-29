from __future__ import annotations

import json
from dataclasses import replace

import pytest

from rsimem.lifecycle import RawResourceUsage
from rsimem.memory.contracts import MemoryKind
from rsimem.memory.extraction_policy_artifact import (
    EXTRACTION_POLICY_COMPILER_DIGEST,
    ExtractionGenerationProvenance,
    ExtractionPolicyRule,
    ExtractionPolicySpec,
    ExtractionPromptPolicyArtifact,
    ExtractionRuleEdit,
    ExtractionRuleEditAction,
    apply_extraction_rule_edits,
    compile_extraction_policy_spec,
    serialize_extraction_prompt_artifact,
)
from rsimem.memory.evidence_planes import EvidencePlane, EvidenceSourceKind
from rsimem.memory.prompt_components import (
    PromptPolicyStage,
    PromptSlotDescriptor,
    content_digest,
)


def _slot() -> PromptSlotDescriptor:
    return PromptSlotDescriptor(
        slot_id="fixture.semantic.extraction",
        memory_kind=MemoryKind.SEMANTIC,
        policy_stage=PromptPolicyStage.EXTRACTION,
        input_schema_digest="1" * 64,
        output_schema_digest="2" * 64,
        frozen_wrapper_digest="3" * 64,
        model_profile="fixture-model-v1",
        owner_adapter_id="fixture-adapter-v1",
        required_placeholders=(
            "exit_evidence",
            "policy_body",
            "source_messages",
        ),
    )


def _spec() -> ExtractionPolicySpec:
    return ExtractionPolicySpec((
        ExtractionPolicyRule("scope", "Extract durable user facts."),
        ExtractionPolicyRule(
            "safety",
            "Exclude unsupported and temporary claims.",
            protected=True,
        ),
        ExtractionPolicyRule("format", "Return concise facts."),
    ))


def _provenance() -> ExtractionGenerationProvenance:
    return ExtractionGenerationProvenance(
        optimizer_model="optimizer-model-v1",
        optimizer_config_digest="4" * 64,
        training_corpus_id="corpus.train-v1",
        training_cutoff="operation.cutoff-v1",
        proposal_request_digest="5" * 64,
        completion_digest="6" * 64,
        usage=RawResourceUsage(
            input_tokens=100,
            output_tokens=20,
            model_requests=1,
            duration_ms=50,
        ),
    )


def _root() -> ExtractionPromptPolicyArtifact:
    return ExtractionPromptPolicyArtifact.create_root(
        slot=_slot(),
        policy_version="root-v1",
        spec=_spec(),
        max_body_chars=1_000,
        source_provenance="fixture-root-export-v1",
    )


def test_root_and_child_artifacts_round_trip_and_replay_exactly() -> None:
    root = _root()
    edits = (
        ExtractionRuleEdit(
            "edit.replace-scope",
            ExtractionRuleEditAction.REPLACE,
            "scope",
            ExtractionPolicyRule(
                "scope",
                "Extract durable user preferences, facts, and constraints.",
            ),
        ),
        ExtractionRuleEdit(
            "edit.add-conflict",
            ExtractionRuleEditAction.ADD,
            None,
            ExtractionPolicyRule(
                "conflict",
                "Prefer explicit current evidence over older claims.",
            ),
            after_rule_id="scope",
        ),
    )
    child = ExtractionPromptPolicyArtifact.create_child(
        parent=root,
        policy_version="candidate-v2",
        edits=edits,
        generation_provenance=_provenance(),
    )

    assert root.parent_artifact_id is None
    assert child.parent_artifact_id == root.artifact_id
    assert child.parent_spec_digest == root.spec.spec_digest
    assert child.generation_provenance is not None
    assert child.generation_provenance.evidence_plane is EvidencePlane.PURE_PROCESS
    assert child.generation_provenance.evidence_source is EvidenceSourceKind.RUNTIME_OBSERVATION
    assert child.compiler_digest == EXTRACTION_POLICY_COMPILER_DIGEST
    assert child.compiled_body == compile_extraction_policy_spec(
        apply_extraction_rule_edits(root.spec, edits)
    )
    serialized = serialize_extraction_prompt_artifact(child)
    assert ExtractionPromptPolicyArtifact.from_payload(json.loads(serialized)) == child
    component = child.to_prompt_component(_slot())
    assert component.policy_body == child.compiled_body
    assert component.body_digest == child.body_digest
    assert child.artifact_id in component.source_provenance
    assert "fixture-adapter" not in serialized


@pytest.mark.parametrize(
    "edits",
    (
        (
            ExtractionRuleEdit(
                "edit.unknown",
                ExtractionRuleEditAction.DELETE,
                "missing",
                None,
            ),
        ),
        (
            ExtractionRuleEdit(
                "edit.protected",
                ExtractionRuleEditAction.DELETE,
                "safety",
                None,
            ),
        ),
        (
            ExtractionRuleEdit(
                "edit.noop",
                ExtractionRuleEditAction.REPLACE,
                "scope",
                ExtractionPolicyRule("scope", "Extract durable user facts."),
            ),
        ),
    ),
)
def test_invalid_rule_edit_sequences_fail_closed(edits) -> None:
    with pytest.raises(ValueError):
        apply_extraction_rule_edits(_spec(), edits)


def test_duplicate_rule_and_edit_ids_fail_closed() -> None:
    with pytest.raises(ValueError, match="duplicate rule"):
        ExtractionPolicySpec((
            ExtractionPolicyRule("same", "First."),
            ExtractionPolicyRule("same", "Second."),
        ))
    edit = ExtractionRuleEdit(
        "edit.same",
        ExtractionRuleEditAction.DELETE,
        "format",
        None,
    )
    with pytest.raises(ValueError, match="duplicate edit"):
        apply_extraction_rule_edits(_spec(), (edit, edit))


def test_generation_provenance_rejects_diagnostic_plane() -> None:
    with pytest.raises(ValueError, match="pure_process runtime evidence"):
        _provenance_with_plane(EvidencePlane.BENCHMARK_AUDIT)


def _provenance_with_plane(plane: EvidencePlane) -> ExtractionGenerationProvenance:
    return ExtractionGenerationProvenance(
        optimizer_model="optimizer-model-v1",
        optimizer_config_digest="4" * 64,
        training_corpus_id="corpus.train-v1",
        training_cutoff="operation.cutoff-v1",
        proposal_request_digest="5" * 64,
        completion_digest="6" * 64,
        usage=RawResourceUsage(input_tokens=100, output_tokens=20, model_requests=1),
        evidence_plane=plane,
        evidence_source=(
            EvidenceSourceKind.BENCHMARK_CONTRACT
            if plane is EvidencePlane.BENCHMARK_AUDIT
            else EvidenceSourceKind.RUNTIME_OBSERVATION
        ),
    )


def test_artifact_tampering_oversize_schema_and_slot_drift_fail_closed() -> None:
    root = _root()
    with pytest.raises(ValueError, match="does not replay"):
        replace(root, compiled_body=root.compiled_body + "\nChanged.")
    with pytest.raises(ValueError, match="artifact digest"):
        replace(root, artifact_digest="f" * 64)
    with pytest.raises(ValueError, match="oversized"):
        replace(root, max_body_chars=1)
    with pytest.raises(ValueError, match="unsupported"):
        replace(root, schema_version=2)
    with pytest.raises(ValueError, match="runtime slot"):
        root.to_prompt_component(replace(_slot(), frozen_wrapper_digest="9" * 64))
    malformed = root.payload()
    malformed["unknown"] = True
    with pytest.raises(ValueError, match="malformed"):
        ExtractionPromptPolicyArtifact.from_payload(malformed)


def test_placeholder_controls_and_generated_lineage_are_strict() -> None:
    with pytest.raises(ValueError, match="template controls"):
        ExtractionPolicyRule("escape", "Read $source_messages directly.")
    root = _root()
    edit = ExtractionRuleEdit(
        "edit.replace",
        ExtractionRuleEditAction.REPLACE,
        "format",
        ExtractionPolicyRule("format", "Return minimal standalone facts."),
    )
    child = ExtractionPromptPolicyArtifact.create_child(
        parent=root,
        policy_version="candidate-v2",
        edits=(edit,),
        generation_provenance=_provenance(),
    )
    with pytest.raises(ValueError, match="lineage"):
        replace(child, parent_spec_digest=None)
    assert content_digest(root.spec.payload()) == root.spec.spec_digest
