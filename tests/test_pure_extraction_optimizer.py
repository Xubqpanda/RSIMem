from __future__ import annotations

from dataclasses import replace
import json

import pytest

from rsimem.memory.extraction_optimizer_corpus import (
    OptimizerDelayedEvidence,
    OptimizerExtractedFact,
    OptimizerSourceMessage,
)
from rsimem.memory.extraction_optimizer_builder import ExtractionFactContent
from rsimem.memory.extraction_source import ExtractionSourceProjection
from rsimem.memory.extraction_feedback import (
    ExtractedFactEvidence,
    ExtractionSetStatus,
    ExtractionSourceEvidence,
    FactDisposition,
)
from rsimem.memory.extraction_policy_artifact import ExtractionPromptPolicyArtifact
from rsimem.memory.optimizer_content_boundary import OptimizerSecretBoundary
from rsimem.memory.prompt_components import content_digest, text_digest
from rsimem.memory.pure_extraction import (
    PureExtractionFeedbackRecord,
    PureExtractionOptimizerExample,
    PureExtractionOptimizerCorpus,
    PureExtractionSourceRecord,
)
from rsimem.memory.pure_extraction_optimizer import (
    JsonPureExtractionOptimizerContentCaptureStore,
    PureExtractionOptimizerContentCapture,
    build_pure_extraction_optimizer_gate_request,
    build_pure_extraction_optimizer_request,
)
from rsimem.memory.extraction_optimizer_contracts import (
    ExtractionOptimizerConfig,
)
from rsimem.memory.extraction_prompt_optimizer import (
    CapturedExtractionOptimizerClient,
    ExtractionOptimizerDecision,
    ExtractionPromptOptimizer,
)
from rsimem.memory.opportunity import OpportunityEvidence, OpportunitySurface
from rsimem.memory.use_attribution import MemoryUseEvidence, OutcomeEvidenceKind
from rsimem.memory_systems.mem0_flat import (
    MEM0_FLAT_EXTRACTION_SLOT_ID,
    Mem0FlatPromptAdapter,
)
from extraction_fingerprint_support import extraction_activation_fixture


def _fixture() -> tuple[
    ExtractionPromptPolicyArtifact,
    PureExtractionOptimizerExample,
    PureExtractionOptimizerContentCapture,
]:
    parent = Mem0FlatPromptAdapter().export_root_policy_artifact(
        MEM0_FLAT_EXTRACTION_SLOT_ID
    )
    message = {
        "segment_id": "segment.pure-request",
        "source_message_id": "message.pure-request",
        "role": "user",
        "content": "The user prefers concise status updates.",
        "segment_kind": "message",
        "tool_call_id": None,
        "content_truncated": False,
    }
    projection_payload = {
        "schema_version": 1,
        "schema": "completed-task-extraction-source-v1",
        "snapshot_id": "snapshot.pure-request",
        "task_id": "task.pure-request",
        "context_revision": "revision.pure-request",
        "messages": [message],
        "source_message_ids": [message["source_message_id"]],
        "source_segment_ids": [message["segment_id"]],
        "omitted_segment_ids": [],
        "truncated_segment_ids": [],
        "max_content_chars": 1_000,
        "projected_content_chars": len(message["content"]),
    }
    projection = ExtractionSourceProjection.from_payload({
        **projection_payload,
        "projection_id": "extraction-source." + content_digest(projection_payload)[:40],
        "projection_digest": content_digest(projection_payload),
    })
    fact_text = "The user prefers concise status updates."
    fact_content = ExtractionFactContent("fact.pure-request", fact_text, True, None)
    extraction_output_digest = content_digest([fact_content.trace_payload()])
    extraction_artifact_id = "component.pure-request"
    activation = extraction_activation_fixture(
        compilation_id="compilation.pure-request",
        extraction_operation_id="extraction-set.pure-request",
        component_artifact_id=extraction_artifact_id,
        component_artifact_digest=parent.body_digest,
        parsed_output_digest=extraction_output_digest,
        persisted_artifact_ids=("artifact.pure-request",),
        mutation_ids=("mutation.pure-request",),
        policy_artifact=parent,
    )
    source_evidence = ExtractionSourceEvidence(
        "source.pure-request",
        projection.projection_digest,
        "extraction-set.pure-request",
        ExtractionSetStatus.NONEMPTY,
        ("preference.status.concise",),
        (ExtractedFactEvidence(
            "fact.pure-request",
            ("preference.status.concise",),
            FactDisposition.PERSISTED,
            artifact_id="artifact.pure-request",
        ),),
    )
    source = PureExtractionSourceRecord.create(
        source_projection_id=projection.projection_id,
        source_projection_digest=projection.projection_digest,
        context_revision=projection.context_revision,
        extraction_set_id=source_evidence.extraction_set_id,
        extraction_artifact_id=extraction_artifact_id,
        extraction_artifact_digest=parent.body_digest,
        extraction_output_digest=extraction_output_digest,
        source=source_evidence,
        activation=activation,
        provenance_id="provenance.pure-request",
    )
    opportunity = OpportunityEvidence.create(
        source_surface=OpportunitySurface.USER_REQUEST,
        semantic_requirement="preference.status.concise",
        observation_time="2026-08-20T00:00:00Z",
        operation_id="op.opportunity.pure-request",
        provenance_id=source.provenance_id,
        source_payload={"request": "status"},
    )
    use = MemoryUseEvidence.create(
        artifact_ids=("artifact.pure-request",),
        retrieved_artifact_ids=("artifact.pure-request",),
        retrieval_operation_id="op.retrieval.pure-request",
        injected_artifact_ids=("artifact.pure-request",),
        injection_operation_id="op.injection.pure-request",
        used_artifact_ids=("artifact.pure-request",),
        downstream_operation_id="op.use.pure-request",
        outcome_operation_id="op.outcome.pure-request",
        outcome_kind=OutcomeEvidenceKind.STATE_TRANSITION,
        outcome_success=True,
        observation_cutoff="2026-08-21T00:00:00Z",
        provenance_id=source.provenance_id,
    )
    feedback = PureExtractionFeedbackRecord.derive_from_evidence(
        source=source,
        opportunity=opportunity,
        memory_use=use,
        observation_window="window.pure-request",
        provenance_id=source.provenance_id,
    )
    example = PureExtractionOptimizerExample.from_records(source, feedback)
    boundary = OptimizerSecretBoundary()
    source_messages = tuple(
        OptimizerSourceMessage(
            item.segment_id,
            item.source_message_id,
            item.role,
            item.segment_kind.value,
            item.tool_call_id,
            item.content_truncated,
            boundary.project(item.content),
        )
        for item in projection.messages
    )
    extracted_facts = (
        OptimizerExtractedFact(
            fact_content.fact_id,
            boundary.project(fact_content.content),
            text_digest(fact_content.content),
            True,
            None,
            ("preference.status.concise",),
            FactDisposition.PERSISTED,
            "artifact.pure-request",
        ),
    )
    delayed = OptimizerDelayedEvidence(
        "observation.pure-request",
        "2026-08-19T00:00:00Z",
        "2026-08-21T00:00:00Z",
        opportunity.evidence_id,
        opportunity.operation_id,
        use.downstream_operation_id,
        use.outcome_operation_id,
        boundary.project("The user asks for a status update."),
        boundary.project("A concise status update is produced."),
        boundary.project(json.dumps({"completed": True}, sort_keys=True)),
    )
    capture = PureExtractionOptimizerContentCapture(
        example.example_id,
        "logical-case.pure-request",
        ("physical-observation.pure-request",),
        source,
        projection,
        source_messages,
        extracted_facts,
        delayed,
    )
    return parent, example, capture


def _corpus(example: PureExtractionOptimizerExample) -> PureExtractionOptimizerCorpus:
    return PureExtractionOptimizerCorpus.create(
        split="train",
        observation_cutoff="2026-08-22T00:00:00Z",
        examples=(example,),
        process_signal_gate="ready",
        process_signal_protocol_id="protocol.pure-request",
        process_signal_case_digest="a" * 64,
        process_signal_case_count=2,
        process_signal_optimization_count=2,
    )


def test_pure_optimizer_request_is_content_bound_and_replay_stable() -> None:
    parent, example, capture = _fixture()
    corpus = _corpus(example)
    first = build_pure_extraction_optimizer_request(
        parent,
        corpus,
        captures=(capture,),
    )
    second = build_pure_extraction_optimizer_request(
        parent,
        corpus,
        captures=(capture,),
    )
    assert first == second
    payload = json.loads(first.input_json)
    assert payload["evidence_groups"]["useful"][0]["replica_count"] == 1
    assert payload["process_signal"]["optimization_count"] == 2
    serialized = json.dumps(payload, ensure_ascii=True)
    assert "family_id" not in serialized
    assert "stage" not in serialized
    assert "official_score" not in serialized
    assert "The user prefers concise status updates." in serialized


def test_pure_optimizer_request_requires_actionable_content_capture() -> None:
    parent, example, _ = _fixture()
    with pytest.raises(ValueError, match="content capture"):
        build_pure_extraction_optimizer_request(parent, _corpus(example))


def test_pure_optimizer_request_fails_closed_without_ready_gate() -> None:
    parent, example, capture = _fixture()
    blocked = PureExtractionOptimizerCorpus.create(
        split="train",
        observation_cutoff="2026-08-22T00:00:00Z",
        examples=(example,),
        process_signal_gate="no_signal",
    )
    with pytest.raises(ValueError, match="ready process-signal gate"):
        build_pure_extraction_optimizer_request(
            parent,
            blocked,
            captures=(capture,),
        )


def test_pure_optimizer_gate_request_is_provider_ineligible() -> None:
    parent, example, _ = _fixture()
    corpus = PureExtractionOptimizerCorpus.create(
        split="train",
        observation_cutoff="2026-08-22T00:00:00Z",
        examples=(example,),
        process_signal_gate="no_signal",
    )
    request = build_pure_extraction_optimizer_gate_request(
        parent,
        corpus,
        reason_codes=("no_optimization_process_signal",),
    )
    assert request.provider_eligible is False
    assert json.loads(request.input_json)["request_mode"] == "deterministic_signal_gate"


def test_pure_optimizer_propose_uses_pure_request_boundary() -> None:
    parent, example, capture = _fixture()
    corpus = _corpus(example)
    config = ExtractionOptimizerConfig(minimum_actionable_primary_examples=1)
    client = CapturedExtractionOptimizerClient(
        json.dumps({
            "decision": "PROPOSE",
            "reason_codes": ["abstract_reusable_rule"],
            "edits": [{
                "edit_id": "edit.pure-request",
                "action": "ADD_RULE",
                "target_rule_id": None,
                "rule_id": "rule.pure-request",
                "rule_text": "Preserve durable reusable information.",
                "after_rule_id": None,
                "evidence_example_ids": [example.example_id],
                "reason_codes": ["abstract_reusable_rule"],
            }],
        })
    )
    result = ExtractionPromptOptimizer(client, config=config).propose_pure(
        parent,
        corpus,
        captures=(capture,),
    )
    assert result.decision is ExtractionOptimizerDecision.PROPOSE
    assert result.candidate is not None
    assert len(client.requests) == 1
    assert json.loads(client.requests[0].input_json)["evidence_groups"]["useful"]


def test_pure_optimizer_capture_rejects_evaluation_text() -> None:
    parent, example, capture = _fixture()
    with pytest.raises(ValueError, match="forbidden evaluation"):
        build_pure_extraction_optimizer_request(
            parent,
            _corpus(example),
            captures=(replace(
                capture,
                source_messages=(replace(
                    capture.source_messages[0],
                    content=OptimizerSecretBoundary().project("official grader answer"),
                ),),
            ),),
        )


def test_pure_optimizer_capture_binds_source_structure() -> None:
    parent, example, capture = _fixture()
    tampered = replace(
        capture,
        source_messages=(replace(capture.source_messages[0], role="assistant"),),
    )
    with pytest.raises(ValueError, match="source structure"):
        build_pure_extraction_optimizer_request(
            parent,
            _corpus(example),
            captures=(tampered,),
        )


def test_pure_optimizer_capture_binds_fact_lineage() -> None:
    parent, example, capture = _fixture()
    tampered = replace(
        capture,
        extracted_facts=(replace(
            capture.extracted_facts[0],
            persisted_artifact_id="artifact.foreign",
        ),),
    )
    with pytest.raises(ValueError, match="fact lineage"):
        build_pure_extraction_optimizer_request(
            parent,
            _corpus(example),
            captures=(tampered,),
        )


def test_pure_optimizer_capture_store_is_restart_safe_and_conflict_checked(tmp_path) -> None:
    _, example, capture = _fixture()
    path = tmp_path / "owner" / "pure-captures.jsonl"
    store = JsonPureExtractionOptimizerContentCaptureStore(path)
    assert store.append(capture) is True
    assert store.append(capture) is False
    assert JsonPureExtractionOptimizerContentCaptureStore(path).records() == (capture,)
    assert path.stat().st_mode & 0o777 == 0o600

    conflicting = PureExtractionOptimizerContentCapture(
        capture.example_id,
        capture.logical_case_id,
        ("physical-observation.other",),
        capture.source_record,
        capture.source_projection,
        capture.source_messages,
        capture.extracted_facts,
        capture.delayed_evidence,
    )
    with pytest.raises(ValueError, match="conflicting"):
        store.append(conflicting)

    duplicate_observation = PureExtractionOptimizerContentCapture(
        "pure-extraction-example.other",
        "logical-case.other",
        capture.physical_observation_ids,
        capture.source_record,
        capture.source_projection,
        capture.source_messages,
        capture.extracted_facts,
        capture.delayed_evidence,
    )
    with pytest.raises(ValueError, match="physical observation"):
        store.append(duplicate_observation)

    payload = capture.payload()
    payload["family_id"] = "SM02_forbidden"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed"):
        JsonPureExtractionOptimizerContentCaptureStore(path).records()

    # A private content capture must never be readable through a permissive
    # file mode after restart.
    path.write_text(json.dumps(capture.payload()) + "\n", encoding="utf-8")
    path.chmod(0o644)
    with pytest.raises(PermissionError, match="permissions"):
        JsonPureExtractionOptimizerContentCaptureStore(path).records()
