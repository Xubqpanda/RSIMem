from __future__ import annotations

import json
from dataclasses import replace

import pytest

from rsimem.lifecycle import (
    EvaluationTrigger,
    HermesLifecycleConfig,
    HermesLifecycleDryRunRuntime,
    TaskLifecycleState,
)
from rsimem.memory.executor import MutationExecutionStatus
from rsimem.memory.extraction_feedback import (
    ExtractionQualityIssue,
    ExtractionSetStatus,
    FactDisposition,
)
from rsimem.memory.extraction_projection import Mem0FlatExtractionSourceProjector
from rsimem.memory.extraction_projection import JsonExtractionSourceRecordStore
from rsimem.memory.evidence_planes import EvidencePlane, EvidenceSourceKind
from rsimem.memory.live_writeback import StaticSemanticWritebackRuntime
from rsimem.memory_systems.mem0_flat import (
    FakeCompletionClient,
    POLICY_FACT_EXTRACTION_PROMPT,
    POLICY_INTERNAL_OPERATION_PROMPT,
)


FAMILY = "SM01_preference_adoption"
TSV_KEY = "preference.summary.tsv"
PREFERENCE = "Always use TSV with owner, priority, task, and due_date."


def _lifecycle(tmp_path):
    runtime = HermesLifecycleDryRunRuntime(
        HermesLifecycleConfig(evaluator_mode="deterministic"),
        run_id="run-projection",
        episode_id="episode-projection",
        session_id="session-projection",
        task_id="SM01_LEARN_A_001",
        variant="static-rsimem",
        trace_id="trace-projection",
        receipt_path=tmp_path / "lifecycle-receipts.json",
        evidence_path=tmp_path / "lifecycle-evidence.jsonl",
        family_id=FAMILY,
        stage="learn_a",
    )
    return runtime.process(
        (
            {"id": 1, "role": "user", "content": "Always use TSV output."},
            {"id": 2, "role": "assistant", "content": "Understood."},
        ),
        trigger=EvaluationTrigger.TASK_COMPLETED,
        task_state=TaskLifecycleState.COMPLETED,
        source_ref="hermes_state:session:projection",
    )


def _compile(tmp_path, *, facts: tuple[str, ...], action: str = "add"):
    responses = {
        POLICY_FACT_EXTRACTION_PROMPT.artifact.prompt_id: json.dumps({
            "facts": list(facts),
        }),
        POLICY_INTERNAL_OPERATION_PROMPT.artifact.prompt_id: json.dumps({
            "operations": [{
                "fact_index": index,
                "action": action,
                "candidate_id": None,
            } for index in range(len(facts))],
        }),
    }
    runtime = StaticSemanticWritebackRuntime(
        tmp_path / "hermes-home",
        FakeCompletionClient(responses),
        operation_evidence_path=tmp_path / "operations.jsonl",
        mutation_receipt_path=tmp_path / "hermes-home" / "receipts.json",
    )
    boundary = runtime.process(_lifecycle(tmp_path))[0]
    return runtime, boundary


@pytest.mark.parametrize(
    ("facts", "action", "expected_status", "expected_disposition"),
    (
        (
            (PREFERENCE,),
            "add",
            ExtractionSetStatus.NONEMPTY,
            FactDisposition.PERSISTED,
        ),
        (
            (PREFERENCE,),
            "none",
            ExtractionSetStatus.NONE,
            FactDisposition.NONE,
        ),
        (
            (),
            "add",
            ExtractionSetStatus.EMPTY,
            None,
        ),
        (
            ("For this task, use TSV temporarily.",),
            "add",
            ExtractionSetStatus.FILTERED,
            FactDisposition.FILTERED,
        ),
    ),
)
def test_projector_retains_real_compilation_outcomes(
    tmp_path,
    facts,
    action,
    expected_status,
    expected_disposition,
) -> None:
    runtime, boundary = _compile(tmp_path, facts=facts, action=action)
    try:
        source = Mem0FlatExtractionSourceProjector().project(
            boundary,
            runtime.policy,
            family_id=FAMILY,
            available_semantic_keys=(TSV_KEY,),
        )
        assert source.status == expected_status
        assert source.source_projection_digest == (
            boundary.writeback.ingestion.source_digest
        )
        if expected_disposition is None:
            assert source.facts == ()
        else:
            assert source.facts[0].disposition == expected_disposition
            assert source.facts[0].semantic_keys == (TSV_KEY,)
            if expected_disposition == FactDisposition.PERSISTED:
                assert source.facts[0].artifact_id is not None
            else:
                assert source.facts[0].artifact_id is None
    finally:
        runtime.close()


def test_projector_marks_uncommitted_mutation_and_unsupported_fact(tmp_path) -> None:
    runtime, boundary = _compile(tmp_path, facts=(PREFERENCE,))
    try:
        assert boundary.writeback is not None
        execution = replace(
            boundary.writeback.executions[0],
            status=MutationExecutionStatus.FAILED,
            artifact_id=None,
            revision=None,
        )
        failed_writeback = replace(
            boundary.writeback,
            executions=(execution,),
            logical_exit=False,
            source_retained=True,
            reason_code="mutation_not_committed",
        )
        failed_boundary = replace(boundary, writeback=failed_writeback)
        source = Mem0FlatExtractionSourceProjector().project(
            failed_boundary,
            runtime.policy,
            family_id=FAMILY,
            available_semantic_keys=(),
        )
        assert source.status == ExtractionSetStatus.MUTATION_FAILED
        assert source.facts[0].disposition == FactDisposition.MUTATION_FAILED
        assert source.facts[0].quality_issue == ExtractionQualityIssue.UNSUPPORTED
    finally:
        runtime.close()


def test_projector_rejects_duplicate_without_original_trace_result(tmp_path) -> None:
    runtime, boundary = _compile(tmp_path, facts=(PREFERENCE,))
    try:
        with pytest.raises(ValueError, match="original writeback"):
            Mem0FlatExtractionSourceProjector().project(
                replace(boundary, duplicate=True, writeback=None),
                runtime.policy,
                family_id=FAMILY,
                available_semantic_keys=(TSV_KEY,),
            )
    finally:
        runtime.close()


def test_source_record_store_is_restart_safe_and_fails_closed(tmp_path) -> None:
    runtime, boundary = _compile(tmp_path, facts=(PREFERENCE,))
    path = tmp_path / "source-records.jsonl"
    try:
        record = Mem0FlatExtractionSourceProjector().project_record(
            boundary,
            runtime.policy,
            runtime.extraction_runtime_binding,
            family_id=FAMILY,
            stage="learn_a",
            available_semantic_keys=(TSV_KEY,),
        )
        assert record.extraction_artifact_digest == (
            runtime.policy.semantic_manifest.extraction_component_digest
        )
        assert record.activation.runtime_binding == (
            runtime.extraction_runtime_binding
        )
        assert record.activation.invocation.render_input_digest == (
            runtime.policy.extraction_invocation(
                boundary.writeback.ingestion.idempotency_key
            ).render_input_digest
        )
        assert record.activation.semantic_policy == runtime.policy.semantic_manifest
        assert record.activation.persisted_artifact_ids == record.artifact_ids
        assert record.evidence_plane is EvidencePlane.BENCHMARK_AUDIT
        assert record.evidence_source is EvidenceSourceKind.BENCHMARK_CONTRACT
        with pytest.raises(ValueError, match="family-bound extraction source"):
            replace(
                record,
                evidence_plane=EvidencePlane.PURE_PROCESS,
                evidence_source=EvidenceSourceKind.RUNTIME_OBSERVATION,
            )
        with pytest.raises(ValueError, match="activation fingerprint differs"):
            replace(record, extraction_output_digest="0" * 64)
        store = JsonExtractionSourceRecordStore(path)
        assert store.append(record) is True
        assert JsonExtractionSourceRecordStore(path).append(record) is False
        candidates = JsonExtractionSourceRecordStore(path).candidates(
            family_id=FAMILY,
            artifact_ids=record.artifact_ids,
            opportunity_semantic_keys=(TSV_KEY,),
        )
        assert candidates == (record,)
        serialized = path.read_text(encoding="utf-8")
        assert PREFERENCE not in serialized
        assert runtime.extraction_policy_artifact.compiled_body not in serialized
        assert "render_input_digest" in serialized
        assert "model_output_digest" in serialized
    finally:
        runtime.close()

    path.write_text("not-json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed extraction source"):
        JsonExtractionSourceRecordStore(path).records()


def test_extraction_source_store_rejects_symlinked_paths(tmp_path: Path) -> None:
    target = tmp_path / "target.jsonl"
    target.write_text("sentinel\n", encoding="utf-8")
    path = tmp_path / "sources.jsonl"
    path.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        JsonExtractionSourceRecordStore(path).records()
    assert target.read_text(encoding="utf-8") == "sentinel\n"


def test_extraction_source_store_rejects_symlinked_lock(tmp_path: Path) -> None:
    path = tmp_path / "sources.jsonl"
    lock_target = tmp_path / "lock-target"
    lock_target.write_text("", encoding="utf-8")
    path.with_suffix(path.suffix + ".lock").symlink_to(lock_target)
    with pytest.raises(ValueError, match="lock.*symlink"):
        JsonExtractionSourceRecordStore(path).records()


def test_empty_source_record_remains_joinable_without_memory_artifact(tmp_path) -> None:
    runtime, boundary = _compile(tmp_path, facts=())
    try:
        record = Mem0FlatExtractionSourceProjector().project_record(
            boundary,
            runtime.policy,
            runtime.extraction_runtime_binding,
            family_id=FAMILY,
            stage="learn_a",
            available_semantic_keys=(TSV_KEY,),
        )
        assert record.artifact_ids == ()
        store = JsonExtractionSourceRecordStore(tmp_path / "empty-sources.jsonl")
        store.append(record)
        assert store.candidates(
            family_id=FAMILY,
            artifact_ids=(),
            opportunity_semantic_keys=(TSV_KEY,),
        ) == (record,)
    finally:
        runtime.close()
