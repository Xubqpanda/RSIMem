from __future__ import annotations

import json
from dataclasses import replace

import pytest

from rsimem.memory.process_signal import (
    JsonProcessSignalCaseStore,
    ProcessSignalCase,
    ProcessSignalCaseStatus,
    census_process_signal_cases,
)
from rsimem.memory.process_feedback import ProcessEvent, ProcessEventKind, ProcessEventStatus


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
    assert census.status_counts == {"ambiguous": 1}


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
