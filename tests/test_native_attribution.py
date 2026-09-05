from rsimem.native_attribution import (
    FailureSurface,
    NativeAttributionExpectation,
    attribute_native_observation,
)
from rsimem.native_observation import NativeLifecycleEventType, ObservationStatus
from test_native_execution_audit import _fixture
from rsimem.native_observation import extract_native_observations


def _expectation(task_id, *events):
    return NativeAttributionExpectation(
        contract_id="application-contract.fixture",
        contract_digest="a" * 64,
        task_id=task_id,
        required_events=events,
    )


def test_attribution_without_application_expectation_is_unresolved(tmp_path) -> None:
    run = _fixture(tmp_path)
    observation = extract_native_observations(run=run, output_root=tmp_path)[0]
    value = attribute_native_observation(observation, None)
    assert value.primary_failure_surface is FailureSurface.UNRESOLVED
    assert not value.is_actionable
    assert "application_expectation_not_registered" in value.evidence_refs


def test_single_missing_formation_maps_to_candidate_repair_axis(tmp_path) -> None:
    run = _fixture(tmp_path)
    observation = extract_native_observations(run=run, output_root=tmp_path)[0]
    value = attribute_native_observation(
        observation,
        _expectation(observation.task_id, NativeLifecycleEventType.FORMATION),
    )
    assert value.primary_failure_surface is FailureSurface.FORMATION_MISSING
    assert value.candidate_repair_axis == "formation"
    assert value.is_actionable
    assert value.review_status == "candidate_pending_review"


def test_application_gap_stays_unresolved_not_ignored(tmp_path) -> None:
    run = _fixture(tmp_path)
    observation = extract_native_observations(run=run, output_root=tmp_path)[0]
    value = attribute_native_observation(
        observation,
        _expectation(observation.task_id, NativeLifecycleEventType.APPLICATION),
    )
    assert value.primary_failure_surface is FailureSurface.UNRESOLVED
    assert "required_application_not_observed_without_safe_mapping" in value.evidence_refs


def test_multiple_missing_axes_are_not_forced_to_one_primary(tmp_path) -> None:
    run = _fixture(tmp_path)
    observation = extract_native_observations(run=run, output_root=tmp_path)[0]
    value = attribute_native_observation(
        observation,
        _expectation(
            observation.task_id,
            NativeLifecycleEventType.FORMATION,
            NativeLifecycleEventType.COMMIT,
        ),
    )
    assert value.primary_failure_surface is FailureSurface.UNRESOLVED
    assert not value.is_actionable
