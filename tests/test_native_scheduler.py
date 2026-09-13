from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from experiments.legacy.native.native_attribution_protocol import NativeAttributionRepairProtocol
from experiments.legacy.native.native_attribution_run import build_native_attribution_manifest
from experiments.legacy.native.native_scheduler import (
    BoundedNativeRunScheduler,
    InfrastructureFailure,
    NativeRunObservation,
    NativeRunOutcomeStore,
    classify_infrastructure_failure,
)


def _runs(root, count=3):
    manifest = build_native_attribution_manifest(
        batch_id="scheduler-fixture", protocol=NativeAttributionRepairProtocol.create(),
        past_bench_root=root / "benchmarks" / "past-bench",
        rsimem_commit="commit.rsimem.fixture", past_bench_commit="commit.past.fixture",
        replicate_count=1,
    )
    return manifest.runs[:count]


@pytest.mark.parametrize(
    ("observation", "failure"),
    [
        (NativeRunObservation(1, True, True, provider_status=429), InfrastructureFailure.PROVIDER_RATE_LIMIT),
        (NativeRunObservation(1, True, True, provider_status=503), InfrastructureFailure.PROVIDER_SERVER_ERROR),
        (NativeRunObservation(1, True, True, error_code="timeout"), InfrastructureFailure.PROVIDER_CONNECTION),
        (NativeRunObservation(1, True, True, error_code="service_failure"), InfrastructureFailure.SERVICE_FAILURE),
        (NativeRunObservation(0, True, False), InfrastructureFailure.STATE_IDENTITY_MISMATCH),
        (NativeRunObservation(0, False, True), InfrastructureFailure.USAGE_INCOMPLETE),
        (NativeRunObservation(1, True, True), InfrastructureFailure.RUNNER_FAILURE),
        (NativeRunObservation(0, True, True), None),
    ],
)
def test_infrastructure_classification(observation, failure) -> None:
    assert classify_infrastructure_failure(observation) is failure


def test_scheduler_retries_only_provider_failures_and_accepts_complete_retry() -> None:
    root = Path(__file__).resolve().parents[1]
    run = _runs(root, 1)[0]
    observations = iter((
        NativeRunObservation(1, True, True, provider_status=503),
        NativeRunObservation(0, True, True),
    ))
    outcome = BoundedNativeRunScheduler(max_retries=2).run(
        (run,), lambda _run, _attempt: next(observations)
    )[0]
    assert outcome.status == "accepted"
    assert outcome.enters_quality_denominator
    assert len(outcome.attempts) == 2
    assert outcome.attempts[0].retryable
    assert outcome.attempts[0].provider_status == 503
    assert outcome.attempts[0].error_code is None


def test_usage_failure_is_retained_and_excluded_without_retry() -> None:
    root = Path(__file__).resolve().parents[1]
    run = _runs(root, 1)[0]
    calls = 0
    def execute(_run, _attempt):
        nonlocal calls
        calls += 1
        return NativeRunObservation(0, False, True)
    outcome = BoundedNativeRunScheduler(max_retries=3).run((run,), execute)[0]
    assert outcome.status == "infrastructure_attempt"
    assert not outcome.enters_quality_denominator
    assert outcome.attempts[-1].failure is InfrastructureFailure.USAGE_INCOMPLETE
    assert outcome.attempts[-1].provider_status is None
    assert calls == 1


def test_scheduler_enforces_concurrency_bound() -> None:
    root = Path(__file__).resolve().parents[1]
    runs = _runs(root, 4)
    lock = threading.Lock()
    active = 0
    peak = 0
    def execute(_run, _attempt):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        return NativeRunObservation(0, True, True)
    outcomes = BoundedNativeRunScheduler(max_concurrency=2).run(runs, execute)
    assert len(outcomes) == 4
    assert all(value.status == "accepted" for value in outcomes)
    assert peak == 2


def test_executor_exception_is_runner_failure_and_not_retried() -> None:
    root = Path(__file__).resolve().parents[1]
    run = _runs(root, 1)[0]
    outcome = BoundedNativeRunScheduler(max_retries=3).run(
        (run,), lambda _run, _attempt: (_ for _ in ()).throw(RuntimeError("boom"))
    )[0]
    assert len(outcome.attempts) == 1
    assert outcome.attempts[0].failure is InfrastructureFailure.RUNNER_FAILURE


def test_outcome_store_is_idempotent_and_filters_infrastructure(tmp_path) -> None:
    root = Path(__file__).resolve().parents[1]
    first, second = _runs(root, 2)
    scheduler = BoundedNativeRunScheduler()
    accepted = scheduler.run(
        (first,), lambda _run, _attempt: NativeRunObservation(0, True, True)
    )[0]
    excluded = scheduler.run(
        (second,), lambda _run, _attempt: NativeRunObservation(0, False, True)
    )[0]
    store = NativeRunOutcomeStore(tmp_path / "outcomes")
    assert store.put(accepted) is True
    assert store.put(accepted) is False
    assert store.put(excluded) is True
    assert [value["run_id"] for value in store.accepted()] == [first.run_id]
    assert NativeRunOutcomeStore(tmp_path / "outcomes").accepted()[0]["attempts"][0]["provider_status"] is None


def test_outcome_store_rejects_conflicting_receipt(tmp_path) -> None:
    root = Path(__file__).resolve().parents[1]
    run = _runs(root, 1)[0]
    scheduler = BoundedNativeRunScheduler()
    accepted = scheduler.run(
        (run,), lambda _run, _attempt: NativeRunObservation(0, True, True)
    )[0]
    excluded = scheduler.run(
        (run,), lambda _run, _attempt: NativeRunObservation(0, False, True)
    )[0]
    store = NativeRunOutcomeStore(tmp_path / "outcomes")
    store.put(accepted)
    with pytest.raises(ValueError, match="different outcome"):
        store.put(excluded)
