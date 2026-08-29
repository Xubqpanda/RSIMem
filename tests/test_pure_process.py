from __future__ import annotations

import json

import pytest

from rsimem.memory.evidence_planes import (
    EvidencePlane,
    EvidenceSourceKind,
    require_optimizer_plane,
    validate_plane_source,
    validate_pure_process_payload,
)
from rsimem.memory.policy_contracts import PolicyLayer
from rsimem.memory.process_feedback import ProcessEvent, ProcessEventKind, ProcessEventStatus
from rsimem.memory.pure_process import (
    JsonPureProcessCorpusStore,
    PureProcessCorpus,
)


def _event() -> ProcessEvent:
    return ProcessEvent.create(
        kind=ProcessEventKind.TOOL_RESULT,
        status=ProcessEventStatus.SUCCESS,
        run_id="run.pure-v1",
        variant="native+ledger",
        trace_id="trace.pure-v1",
        episode_id="episode.pure-v1",
        session_id="session.pure-v1",
        task_id="task.pure-v1",
        host_event_id="event.pure-v1",
        source_revision="revision.pure-v1",
        input_payload={"tool_name_digest": "a" * 64},
        output_payload={"success": True},
        policy_decision_id="decision.pure-v1",
        policy_layer=PolicyLayer.COMMIT,
        lineage_id="lineage.pure-v1",
        execution_receipt_ids=("receipt.pure-v1",),
        family_id="SM02_constraint_retention",
        stage="eval_far",
    )


def test_pure_process_projection_strips_benchmark_identity_and_replays(tmp_path) -> None:
    corpus = PureProcessCorpus.create((_event(),))
    payload = corpus.payload()
    serialized = json.dumps(payload, ensure_ascii=True)
    for forbidden in (
        "family_id", "familyId", "stage", "grader", "answer_key",
        "official_score", "task_score", "hidden_expectation",
    ):
        assert forbidden not in serialized
    assert payload["evidence_plane"] == EvidencePlane.PURE_PROCESS.value
    store = JsonPureProcessCorpusStore(tmp_path / "pure-process.json")
    assert store.put(corpus)[1] is True
    assert store.put(corpus)[1] is False
    assert JsonPureProcessCorpusStore(tmp_path / "pure-process.json").get() == corpus


def test_pure_process_payload_rejects_evaluation_fields() -> None:
    with pytest.raises(ValueError, match="forbidden evaluation fields"):
        validate_pure_process_payload({"nested": {"officialScore": 0.5}})


def test_previous_pure_process_schema_is_not_silently_migrated() -> None:
    corpus = PureProcessCorpus.create((_event(),))
    payload = corpus.payload()
    payload["schema_version"] = 1
    payload["schema"] = "rsimem-pure-process-corpus-v1"
    with pytest.raises(ValueError, match="malformed pure-process corpus|unsupported"):
        PureProcessCorpus.from_payload(payload)


def test_evidence_plane_source_identity_is_least_privilege() -> None:
    assert validate_plane_source(
        EvidencePlane.PURE_PROCESS,
        EvidenceSourceKind.RUNTIME_OBSERVATION,
    ) == (EvidencePlane.PURE_PROCESS, EvidenceSourceKind.RUNTIME_OBSERVATION)
    assert validate_plane_source(
        EvidencePlane.BENCHMARK_AUDIT,
        EvidenceSourceKind.BENCHMARK_CONTRACT,
    )[0] == EvidencePlane.BENCHMARK_AUDIT
    with pytest.raises(ValueError, match="plane and source identity"):
        validate_plane_source(
            EvidencePlane.PURE_PROCESS,
            EvidenceSourceKind.BENCHMARK_CONTRACT,
        )


@pytest.mark.parametrize("plane", (EvidencePlane.BENCHMARK_AUDIT, EvidencePlane.FINAL_EVALUATION))
def test_optimizer_rejects_non_pure_process_plane(plane) -> None:
    with pytest.raises(ValueError, match="optimizer requires pure_process"):
        require_optimizer_plane(plane)
