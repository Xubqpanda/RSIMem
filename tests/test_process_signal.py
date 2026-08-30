from __future__ import annotations

import json
from dataclasses import replace

import pytest

from rsimem.memory.evidence_planes import EvidencePlane
from rsimem.memory.process_signal import (
    JsonProcessSignalCaseStore,
    ProcessSignalCase,
    ProcessSignalCaseStatus,
    build_process_signal_cases,
    census_process_signal_cases,
)
from rsimem.memory.process_feedback import ProcessEvent, ProcessEventKind, ProcessEventStatus
from rsimem.memory.pure_process import PureProcessCorpus


def _case(*, logical: str = "logical-case.fixture.v1", complete: bool = True, attribution: bool = True, hypothesis: str | None = "a" * 64, physical: str = "physical-observation.1", stage_diagnosis_observed: bool = True) -> ProcessSignalCase:
    return ProcessSignalCase.create(
        logical_case_id=logical,
        physical_observation_ids=(physical,),
        source_observed=True,
        extraction_observed=True,
        persistence_observed=True,
        retrieval_observed=True,
        exposure_observed=True,
        outcome_observed=True,
        extraction_attributable=attribution,
        abstract_hypothesis_digest=hypothesis,
        observation_complete=complete,
        stage_diagnosis_observed=stage_diagnosis_observed,
    )


def test_signal_statuses_are_conservative_and_replayable() -> None:
    signal = _case()
    assert signal.status == ProcessSignalCaseStatus.OPTIMIZATION_SIGNAL
    assert ProcessSignalCase.from_payload(json.loads(json.dumps(signal.payload()))) == signal
    assert census_process_signal_cases((signal,)).payload()["logicalCaseCount"] == 1

    diagnostic = _case(hypothesis=None, physical="physical-observation.2")
    assert diagnostic.status == ProcessSignalCaseStatus.DIAGNOSTIC_ONLY
    observable = _case(
        attribution=False,
        physical="physical-observation.3",
        stage_diagnosis_observed=False,
    )
    assert observable.status == ProcessSignalCaseStatus.OBSERVABLE_ONLY
    censored = _case(complete=False, physical="physical-observation.4")
    assert censored.status == ProcessSignalCaseStatus.CENSORED
    invalid = ProcessSignalCase.create(
        logical_case_id="logical-case.fixture.v1",
        physical_observation_ids=("physical-observation.5",),
        source_observed=True, extraction_observed=True, persistence_observed=True,
        retrieval_observed=True, exposure_observed=True, outcome_observed=True,
        extraction_attributable=True, abstract_hypothesis_digest="a" * 64,
        observation_complete=True, invalid_reason_code="schema_mismatch",
    )
    assert invalid.status == ProcessSignalCaseStatus.INVALID


def test_repeated_logical_case_counts_one_case_and_conflict_is_ambiguous() -> None:
    first = _case(physical="physical-observation.1")
    second = _case(hypothesis=None, physical="physical-observation.2")
    census = census_process_signal_cases((first, second))
    assert census.physical_observation_count == 2
    assert census.logical_case_count == 1
    assert census.conflict_case_count == 1
    assert census.payload()["replicateConsistentCaseCount"] == 0
    assert census.payload()["replicateConsistency"] == 0.0
    assert census.status_counts == {"ambiguous": 1}
    assert census.payload()["conflictRate"] == 1.0


def test_hypothesis_conflict_is_ambiguous_even_when_statuses_match() -> None:
    first = _case(physical="physical-observation.hypothesis.1", hypothesis="a" * 64)
    second = _case(physical="physical-observation.hypothesis.2", hypothesis="b" * 64)
    census = census_process_signal_cases((first, second))
    assert census.status_counts == {"ambiguous": 1}
    assert census.conflict_case_count == 1
    assert census.payload()["conflictRate"] == 1.0


def test_census_tracks_hypothesis_support_across_distinct_logical_cases() -> None:
    hypothesis = "a" * 64
    first = _case(
        logical="logical-case.hypothesis-a",
        physical="physical-observation.hypothesis-a",
        hypothesis=hypothesis,
    )
    second = _case(
        logical="logical-case.hypothesis-b",
        physical="physical-observation.hypothesis-b",
        hypothesis=hypothesis,
    )
    census = census_process_signal_cases((first, second))
    assert census.optimization_hypothesis_case_counts == {hypothesis: 2}
    assert census.payload()["optimizationHypothesisCaseCounts"] == {hypothesis: 2}


def test_census_does_not_count_conflicting_or_replicated_cases_as_support() -> None:
    hypothesis = "a" * 64
    replica = _case(
        logical="logical-case.replicated",
        physical="physical-observation.replicated",
        hypothesis=hypothesis,
    )
    conflict = _case(
        logical="logical-case.replicated",
        physical="physical-observation.replicated-conflict",
        hypothesis=None,
    )
    census = census_process_signal_cases((replica, conflict))
    assert census.optimization_hypothesis_case_counts == {}
    assert census.conflict_case_count == 1


def test_census_rejects_hypothesis_support_without_optimization_cases() -> None:
    with pytest.raises(ValueError, match="exceed optimization cases"):
        from rsimem.memory.process_signal import ProcessSignalCaseCensus

        ProcessSignalCaseCensus(
            physical_observation_count=2,
            logical_case_count=2,
            status_counts={"observable_only": 2},
            conflict_case_count=0,
            optimization_hypothesis_case_counts={"a" * 64: 1},
        )


def test_non_pure_plane_and_duplicate_physical_ids_fail_closed() -> None:
    with pytest.raises(ValueError, match="plane and source identity"):
        replace(_case(), evidence_plane="benchmark_audit")
    first = _case(physical="physical-observation.1")
    with pytest.raises(ValueError, match="physical observation IDs must be unique"):
        ProcessSignalCase.create(
            logical_case_id=first.logical_case_id,
            physical_observation_ids=("physical-observation.1", "physical-observation.1"),
            source_observed=True, extraction_observed=True, persistence_observed=True,
            retrieval_observed=True, exposure_observed=True, outcome_observed=True,
            extraction_attributable=True, abstract_hypothesis_digest="a" * 64,
            observation_complete=True,
        )


def test_census_rejects_same_physical_observation_in_two_cases() -> None:
    first = _case(physical="physical-observation.1")
    second = _case(logical="logical-case.other.v1", physical="physical-observation.1")
    with pytest.raises(ValueError, match="duplicate physical observation identity"):
        census_process_signal_cases((first, second))


def test_census_counts_all_physical_observations_inside_one_case() -> None:
    case = ProcessSignalCase.create(
        logical_case_id="logical-case.multi.v1",
        physical_observation_ids=("physical-observation.1", "physical-observation.2"),
        source_observed=True, extraction_observed=True, persistence_observed=True,
        retrieval_observed=True, exposure_observed=True, outcome_observed=True,
        extraction_attributable=True, abstract_hypothesis_digest="a" * 64,
        observation_complete=True,
    )
    assert census_process_signal_cases((case,)).physical_observation_count == 2
    assert census_process_signal_cases((case,)).payload()["replicateConsistency"] == 1.0


@pytest.mark.parametrize(
    ("kwargs", "error"),
    (
        (
            {
                "physical_observation_count": 1,
                "logical_case_count": 1,
                "status_counts": {"future_status": 1},
                "conflict_case_count": 0,
            },
            "status key",
        ),
        (
            {
                "physical_observation_count": 0,
                "logical_case_count": 1,
                "status_counts": {"observable_only": 1},
                "conflict_case_count": 0,
            },
            "less than logical",
        ),
        (
            {
                "physical_observation_count": 1,
                "logical_case_count": 0,
                "status_counts": {},
                "conflict_case_count": 0,
            },
            "require at least one logical",
        ),
        (
            {
                "physical_observation_count": True,
                "logical_case_count": 1,
                "status_counts": {"observable_only": 1},
                "conflict_case_count": 0,
            },
            "integer",
        ),
    ),
)
def test_census_contract_rejects_untrusted_statistics(kwargs, error: str) -> None:
    from rsimem.memory.process_signal import ProcessSignalCaseCensus

    with pytest.raises((TypeError, ValueError), match=error):
        ProcessSignalCaseCensus(**kwargs)


def test_process_signal_case_store_replays_logical_census(tmp_path) -> None:
    path = tmp_path / "process-signal.jsonl"
    first = _case(physical="physical-observation.store.1")
    second = _case(hypothesis=None, physical="physical-observation.store.2")
    store = JsonProcessSignalCaseStore(path)
    assert store.append(first) is True
    assert store.append(first) is False
    assert store.append(second) is True
    replay = JsonProcessSignalCaseStore(path)
    assert {case.case_id for case in replay.records()} == {first.case_id, second.case_id}
    census = replay.census()
    assert census.physical_observation_count == 2
    assert census.logical_case_count == 1
    assert census.conflict_case_count == 1


def test_process_signal_case_store_replay_order_is_canonical(tmp_path) -> None:
    first = _case(physical="physical-observation.order.1")
    second = _case(hypothesis=None, physical="physical-observation.order.2")
    path = tmp_path / "process-signal-order.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(item.payload(), sort_keys=True)
            for item in (second, first)
        ) + "\n",
        encoding="utf-8",
    )
    records = JsonProcessSignalCaseStore(path).records()
    assert tuple(item.case_id for item in records) == tuple(
        sorted((first.case_id, second.case_id))
    )


def test_process_signal_case_store_fails_closed_on_symlink(tmp_path) -> None:
    target = tmp_path / "target.jsonl"
    target.write_text("", encoding="utf-8")
    path = tmp_path / "process-signal.jsonl"
    path.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        JsonProcessSignalCaseStore(path).records()


def test_process_signal_case_store_rejects_symlinked_lock(tmp_path) -> None:
    path = tmp_path / "process-signal.jsonl"
    lock_target = tmp_path / "lock-target"
    lock_target.write_text("", encoding="utf-8")
    path.with_name(path.name + ".lock").symlink_to(lock_target)
    with pytest.raises(ValueError, match="lock.*symlink"):
        JsonProcessSignalCaseStore(path).records()


def test_build_process_signal_cases_separates_logical_and_physical_identity() -> None:
    common = dict(
        run_id="run.builder.v1",
        variant="native+ledger",
        trace_id="trace.builder.v1",
        episode_id="episode.builder.v1",
        session_id="session.builder.v1",
        task_id="task.builder.v1",
        host_event_id="event.builder.v1",
        source_revision="revision.builder.v1",
    )
    events = tuple(
        ProcessEvent.create(
            kind=kind,
            status=ProcessEventStatus.SUCCESS,
            input_payload={"kind": kind.value},
            output_payload={"ok": True},
            **common,
        )
        for kind in (
            ProcessEventKind.SOURCE_SELECTION,
            ProcessEventKind.EXTRACTION,
            ProcessEventKind.COMMIT,
            ProcessEventKind.RETRIEVAL,
            ProcessEventKind.EXPOSURE,
            ProcessEventKind.TASK_OUTCOME,
        )
    )
    first = build_process_signal_cases(
        events,
        frozen_policy_digest="a" * 64,
        source_task_template_id="source-template.v1",
        future_task_template_id="future-template.v1",
        observation_window="window.v1",
        replicate_id="replicate.1",
    )
    second = build_process_signal_cases(
        events,
        frozen_policy_digest="a" * 64,
        source_task_template_id="source-template.v1",
        future_task_template_id="future-template.v1",
        observation_window="window.v1",
        replicate_id="replicate.2",
    )
    assert first[0].logical_case_id == second[0].logical_case_id
    assert first[0].physical_observation_ids != second[0].physical_observation_ids


def test_build_process_signal_cases_ignore_physical_task_ids_when_source_trace_matches() -> None:
    def make_events(task_id: str, run_id: str) -> tuple[ProcessEvent, ...]:
        common = dict(
            run_id=run_id,
            variant="native+ledger",
            trace_id=f"trace.{run_id}",
            episode_id=f"episode.{run_id}",
            session_id=f"session.{run_id}",
            task_id=task_id,
            host_event_id=f"event.{run_id}",
            source_revision="revision.same-source",
        )
        return tuple(
            ProcessEvent.create(
                kind=kind,
                status=ProcessEventStatus.SUCCESS,
                input_digest=("1" * 64 if kind is ProcessEventKind.SOURCE_SELECTION else "2" * 64),
                output_digest=("3" * 64 if kind is ProcessEventKind.SOURCE_SELECTION else "4" * 64),
                input_payload={},
                output_payload={},
                **common,
            )
            for kind in (
                ProcessEventKind.SOURCE_SELECTION,
                ProcessEventKind.EXTRACTION,
                ProcessEventKind.COMMIT,
                ProcessEventKind.RETRIEVAL,
                ProcessEventKind.EXPOSURE,
                ProcessEventKind.TASK_OUTCOME,
            )
        )

    first = build_process_signal_cases(
        make_events("task.physical.one", "run.physical.one"),
        frozen_policy_digest="a" * 64,
        source_task_template_id="source-template.same",
        future_task_template_id="future-template.same",
        observation_window="window.same",
        replicate_id="replicate.one",
    )
    second = build_process_signal_cases(
        make_events("task.physical.two", "run.physical.two"),
        frozen_policy_digest="a" * 64,
        source_task_template_id="source-template.same",
        future_task_template_id="future-template.same",
        observation_window="window.same",
        replicate_id="replicate.two",
    )
    assert first[0].logical_case_id == second[0].logical_case_id


def test_extraction_output_variation_does_not_split_logical_case() -> None:
    """Replicate output differences remain physical observations of one case."""

    def make_events(extraction_output: str) -> tuple[ProcessEvent, ...]:
        common = dict(
            run_id="run.physical-output",
            variant="native+ledger",
            trace_id="trace.physical-output",
            episode_id="episode.physical-output",
            session_id="session.physical-output",
            task_id="task.physical-output",
            host_event_id="event.physical-output",
            source_revision="revision.same-source",
        )
        events = []
        for kind in (
            ProcessEventKind.SOURCE_SELECTION,
            ProcessEventKind.EXTRACTION,
            ProcessEventKind.COMMIT,
            ProcessEventKind.RETRIEVAL,
            ProcessEventKind.EXPOSURE,
            ProcessEventKind.TASK_OUTCOME,
        ):
            output_payload = (
                {"selected": ["segment.user.v1"]}
                if kind is ProcessEventKind.SOURCE_SELECTION
                else {"extraction": extraction_output}
                if kind is ProcessEventKind.EXTRACTION
                else {"kind": kind.value}
            )
            events.append(ProcessEvent.create(
                kind=kind,
                status=ProcessEventStatus.SUCCESS,
                input_payload={"source": "same"},
                output_payload=output_payload,
                **common,
            ))
        return tuple(events)

    first = build_process_signal_cases(
        make_events("facts-a"),
        frozen_policy_digest="a" * 64,
        source_task_template_id="source-template.same-output",
        future_task_template_id="future-template.same-output",
        observation_window="window.same-output",
        replicate_id="replicate.output-a",
    )
    second = build_process_signal_cases(
        make_events("facts-b"),
        frozen_policy_digest="a" * 64,
        source_task_template_id="source-template.same-output",
        future_task_template_id="future-template.same-output",
        observation_window="window.same-output",
        replicate_id="replicate.output-b",
    )
    assert first[0].logical_case_id == second[0].logical_case_id
    assert first[0].physical_observation_ids != second[0].physical_observation_ids


def test_process_signal_case_protocol_binding_is_replay_stable() -> None:
    case = _case()
    bound = ProcessSignalCase.create(
        logical_case_id=case.logical_case_id,
        physical_observation_ids=("physical-observation.bound.v1",),
        source_observed=True,
        extraction_observed=True,
        persistence_observed=True,
        retrieval_observed=True,
        exposure_observed=True,
        outcome_observed=True,
        extraction_attributable=False,
        abstract_hypothesis_digest=None,
        observation_complete=True,
        analysis_protocol_id="signal-protocol.bound.v1",
        replicate_id="replicate.1",
        observation_window="completed-task.v1",
    )
    assert ProcessSignalCase.from_payload(bound.payload()) == bound
    with pytest.raises(ValueError, match="metadata must be complete"):
        ProcessSignalCase.create(
            logical_case_id=case.logical_case_id,
            physical_observation_ids=("physical-observation.bound.v2",),
            source_observed=True,
            extraction_observed=True,
            persistence_observed=True,
            retrieval_observed=True,
            exposure_observed=True,
            outcome_observed=True,
            extraction_attributable=False,
            abstract_hypothesis_digest=None,
            observation_complete=True,
            analysis_protocol_id="signal-protocol.bound.v1",
            replicate_id="replicate.1",
            observation_window=None,
        )


def test_projection_from_process_events_never_infers_extraction_attribution() -> None:
    common = dict(
        run_id="run.signal.v1", variant="native", trace_id="trace.signal.v1",
        episode_id="episode.signal.v1", session_id="session.signal.v1",
        task_id="task.signal.v1", host_event_id="event.signal.v1",
        source_revision="revision.signal.v1", execution_receipt_ids=("receipt.signal.v1",),
    )
    events = tuple(
        ProcessEvent.create(
            kind=kind,
            status=ProcessEventStatus.SUCCESS,
            input_payload={"kind": kind.value},
            output_payload={"ok": True},
            **common,
        )
        for kind in (
            ProcessEventKind.SOURCE_SELECTION,
            ProcessEventKind.EXTRACTION,
            ProcessEventKind.COMMIT,
            ProcessEventKind.RETRIEVAL,
            ProcessEventKind.EXPOSURE,
            ProcessEventKind.TASK_OUTCOME,
        )
    )
    case = ProcessSignalCase.from_process_events(
        logical_case_id="logical-case.events.v1",
        physical_observation_ids=("physical-observation.events.v1",),
        events=events,
    )
    assert case.status == ProcessSignalCaseStatus.OBSERVABLE_ONLY
    assert case.extraction_attributable is False


def test_signal_projection_uses_pure_runtime_events_and_strips_benchmark_identity() -> None:
    common = dict(
        run_id="run.signal-plane.v1", variant="native", trace_id="trace.signal-plane.v1",
        episode_id="episode.signal-plane.v1", session_id="session.signal-plane.v1",
        task_id="task.signal-plane.v1", host_event_id="event.signal-plane.v1",
        source_revision="revision.signal-plane.v1", execution_receipt_ids=("receipt.signal-plane.v1",),
    )
    audit_event = ProcessEvent.create(
        kind=ProcessEventKind.EXTRACTION,
        status=ProcessEventStatus.SUCCESS,
        input_payload={"kind": "extraction"},
        output_payload={"ok": True},
        family_id="SM02_constraint_retention",
        stage="eval_near",
        **common,
    )
    pure_events = PureProcessCorpus.create((audit_event,)).events
    case = ProcessSignalCase.from_process_events(
        logical_case_id="logical-case.signal-plane.v1",
        physical_observation_ids=("physical-observation.signal-plane.v1",),
        events=pure_events,
    )
    assert case.status is ProcessSignalCaseStatus.OBSERVABLE_ONLY
    assert all(event.evidence_plane is EvidencePlane.PURE_PROCESS for event in pure_events)
    with pytest.raises(ValueError, match="pure_process runtime events"):
        ProcessSignalCase.from_process_events(
            logical_case_id="logical-case.signal-plane-audit.v1",
            physical_observation_ids=("physical-observation.signal-plane-audit.v1",),
            events=(audit_event,),
        )
