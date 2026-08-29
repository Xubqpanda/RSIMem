from __future__ import annotations

import json
from dataclasses import replace

import pytest

from rsimem.memory.process_signal import (
    ProcessSignalCase,
    ProcessSignalCaseStatus,
    census_process_signal_cases,
)


def _case(*, logical: str = "logical-case.fixture.v1", complete: bool = True, attribution: bool = True, hypothesis: str | None = "a" * 64, physical: str = "physical-observation.1") -> ProcessSignalCase:
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
    )


def test_signal_statuses_are_conservative_and_replayable() -> None:
    signal = _case()
    assert signal.status == ProcessSignalCaseStatus.OPTIMIZATION_SIGNAL
    assert ProcessSignalCase.from_payload(json.loads(json.dumps(signal.payload()))) == signal
    assert census_process_signal_cases((signal,)).payload()["logicalCaseCount"] == 1

    diagnostic = _case(hypothesis=None, physical="physical-observation.2")
    assert diagnostic.status == ProcessSignalCaseStatus.DIAGNOSTIC_ONLY
    observable = _case(attribution=False, physical="physical-observation.3")
    assert observable.status == ProcessSignalCaseStatus.OBSERVABLE_ONLY
    censored = _case(complete=False, physical="physical-observation.4")
    assert censored.status == ProcessSignalCaseStatus.CENSORED


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
