from __future__ import annotations

import json
from dataclasses import replace

import pytest

from rsimem.lifecycle import (
    DryRunStatus,
    EvaluationTrigger,
    HermesLifecycleConfig,
    HermesLifecycleDryRunRuntime,
    TaskLifecycleState,
)
from rsimem.memory import MemoryKind, MemoryQuery
from rsimem.memory.live_writeback import (
    STATIC_SEMANTIC_WRITEBACK_SCHEMA_VERSION,
    StaticSemanticWritebackConfig,
    StaticSemanticWritebackMode,
    StaticSemanticWritebackRuntime,
)
from rsimem.memory.operation_graph import audit_operation_evidence
from rsimem.memory.receipts import MutationReceiptStatus
from rsimem.memory_systems.mem0_flat import (
    FakeCompletionClient,
    POLICY_FACT_EXTRACTION_PROMPT,
    POLICY_INTERNAL_OPERATION_PROMPT,
)


PREFERENCE = "Use TSV with owner, priority, task, and due_date."


def _lifecycle(tmp_path):
    runtime = HermesLifecycleDryRunRuntime(
        HermesLifecycleConfig(evaluator_mode="deterministic"),
        run_id="run-static",
        episode_id="episode-static",
        session_id="session-static",
        task_id="SM01-static",
        variant="static-rsimem",
        trace_id="trace-static",
        receipt_path=tmp_path / "lifecycle-receipts.json",
        evidence_path=tmp_path / "lifecycle-evidence.jsonl",
        family_id="SM01_preference_adoption",
        stage="learn",
    )
    return runtime.process(
        (
            {"id": 1, "role": "user", "content": "Always use TSV output."},
            {"id": 2, "role": "assistant", "content": "Understood."},
        ),
        trigger=EvaluationTrigger.TASK_COMPLETED,
        task_state=TaskLifecycleState.COMPLETED,
        source_ref="hermes_state:session:static",
    )


def _client(*, fact_response: str | None = None) -> FakeCompletionClient:
    return FakeCompletionClient({
        POLICY_FACT_EXTRACTION_PROMPT.artifact.prompt_id: (
            fact_response
            if fact_response is not None
            else json.dumps({"facts": [PREFERENCE]})
        ),
        POLICY_INTERNAL_OPERATION_PROMPT.artifact.prompt_id: json.dumps({
            "operations": [{
                "fact_index": 0,
                "action": "add",
                "candidate_id": None,
            }],
        }),
    })


def _runtime(tmp_path, client):
    return StaticSemanticWritebackRuntime(
        tmp_path / "hermes-home",
        client,
        operation_evidence_path=tmp_path / "episode" / "operations.jsonl",
        mutation_receipt_path=tmp_path / "hermes-home" / "rsimem-receipts.json",
    )


def test_static_config_is_default_disabled_and_strict() -> None:
    assert STATIC_SEMANTIC_WRITEBACK_SCHEMA_VERSION == 1
    assert StaticSemanticWritebackConfig().enabled is False
    assert StaticSemanticWritebackConfig.from_mapping({
        "mode": "static",
        "timeout_seconds": 12,
        "max_output_tokens": 512,
    }).mode == StaticSemanticWritebackMode.STATIC
    with pytest.raises(ValueError, match="unknown static semantic"):
        StaticSemanticWritebackConfig.from_mapping({"provider_seed": 7})


def test_static_runtime_commits_restart_duplicate_and_emits_content_free_evidence(
    tmp_path,
) -> None:
    lifecycle = _lifecycle(tmp_path)
    first = _runtime(tmp_path, _client())
    result = first.process(lifecycle)[0]

    assert result.snapshot_id == lifecycle.snapshot.snapshot_id
    assert result.plan_id == lifecycle.plans[0].plan_id
    assert result.writeback.logical_exit is True
    assert result.writeback.source_retained is False
    assert result.writeback.executions[0].status.value == "committed"
    assert first.receipts.all()[0].status == MutationReceiptStatus.COMMITTED
    hits = first.registry.resolve(MemoryKind.SEMANTIC).query(MemoryQuery(
        MemoryKind.SEMANTIC,
        "",
        namespace="user",
        limit=10,
    ))
    assert [hit.artifact.content for hit in hits] == [PREFERENCE]
    assert PREFERENCE not in json.dumps(result.observer_evidence(), sort_keys=True)
    assert audit_operation_evidence(
        first.operation_log.events,
        forbidden_values=(PREFERENCE, "Always use TSV output."),
    ) == ()
    first.close()
    first.close()

    restarted = _runtime(tmp_path, _client())
    replay = restarted.process(lifecycle)[0]
    assert replay.writeback.logical_exit is True
    assert replay.writeback.executions[0].status.value == "duplicate"
    assert len(restarted.receipts.all()) == 1
    serialized = (tmp_path / "hermes-home" / "rsimem-receipts.json").read_text(
        encoding="utf-8"
    )
    assert PREFERENCE not in serialized
    assert "Always use TSV output." not in serialized
    restarted.close()


def test_static_runtime_rejects_unvalidated_or_mismatched_lifecycle(tmp_path) -> None:
    lifecycle = _lifecycle(tmp_path)
    runtime = _runtime(tmp_path, _client())
    rejected = replace(lifecycle.receipts[0], status=DryRunStatus.REJECTED)
    with pytest.raises(ValueError, match="requires a validated plan"):
        runtime.process(replace(lifecycle, receipts=(rejected,)))
    with pytest.raises(ValueError, match="one-to-one"):
        runtime.process(replace(lifecycle, receipts=()))
    assert runtime.receipts.all() == ()
    runtime.close()


def test_static_runtime_policy_failure_retains_source_without_mutation(tmp_path) -> None:
    lifecycle = _lifecycle(tmp_path)
    runtime = _runtime(tmp_path, _client(fact_response="not json"))
    result = runtime.process(lifecycle)[0]

    assert result.writeback.logical_exit is False
    assert result.writeback.source_retained is True
    assert result.writeback.reason_code == "ingestion_not_successful"
    assert result.writeback.executions == ()
    assert runtime.receipts.all() == ()
    assert not (tmp_path / "hermes-home" / "memories" / "USER.md").exists()
    runtime.close()
