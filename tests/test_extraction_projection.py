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
