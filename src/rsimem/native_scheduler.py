"""Bounded scheduler and infrastructure classification for native runs."""

from __future__ import annotations

import concurrent.futures
import fcntl
import hashlib
import json
import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Callable, Sequence

from .native_attribution_run import NativeAttributionRunSpec


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


class InfrastructureFailure(StrEnum):
    PROVIDER_RATE_LIMIT = "provider_rate_limit"
    PROVIDER_SERVER_ERROR = "provider_server_error"
    PROVIDER_CONNECTION = "provider_connection"
    SERVICE_FAILURE = "service_failure"
    FIXTURE_IDENTITY_MISMATCH = "fixture_identity_mismatch"
    STATE_IDENTITY_MISMATCH = "state_identity_mismatch"
    USAGE_INCOMPLETE = "usage_incomplete"
    RUNNER_FAILURE = "runner_failure"


_RETRYABLE = {
    InfrastructureFailure.PROVIDER_RATE_LIMIT,
    InfrastructureFailure.PROVIDER_SERVER_ERROR,
    InfrastructureFailure.PROVIDER_CONNECTION,
}


@dataclass(frozen=True, slots=True)
class NativeRunObservation:
    return_code: int
    usage_complete: bool
    identity_verified: bool
    provider_status: int | None = None
    error_code: str | None = None

    def __post_init__(self) -> None:
        if type(self.return_code) is not int:
            raise ValueError("return code must be int")
        if type(self.usage_complete) is not bool or type(self.identity_verified) is not bool:
            raise ValueError("usage and identity status must be bool")
        if self.provider_status is not None and (
            type(self.provider_status) is not int or not 100 <= self.provider_status <= 599
        ):
            raise ValueError("provider status must be an HTTP status")


def classify_infrastructure_failure(
    observation: NativeRunObservation,
) -> InfrastructureFailure | None:
    if observation.error_code == "runner_failure":
        return InfrastructureFailure.RUNNER_FAILURE
    if observation.error_code == "fixture_identity_mismatch":
        return InfrastructureFailure.FIXTURE_IDENTITY_MISMATCH
    if observation.error_code == "state_identity_mismatch" or not observation.identity_verified:
        return InfrastructureFailure.STATE_IDENTITY_MISMATCH
    if observation.error_code == "service_failure":
        return InfrastructureFailure.SERVICE_FAILURE
    if observation.provider_status == 429:
        return InfrastructureFailure.PROVIDER_RATE_LIMIT
    if observation.provider_status is not None and observation.provider_status >= 500:
        return InfrastructureFailure.PROVIDER_SERVER_ERROR
    if observation.error_code in {"connection_error", "timeout"}:
        return InfrastructureFailure.PROVIDER_CONNECTION
    if not observation.usage_complete:
        return InfrastructureFailure.USAGE_INCOMPLETE
    if observation.return_code != 0:
        return InfrastructureFailure.RUNNER_FAILURE
    return None


@dataclass(frozen=True, slots=True)
class NativeRunAttempt:
    attempt_id: str
    run_id: str
    attempt: int
    failure: InfrastructureFailure | None
    retryable: bool
    accepted: bool
    return_code: int
    usage_complete: bool
    identity_verified: bool

    @classmethod
    def create(
        cls, *, run_id: str, attempt: int, observation: NativeRunObservation
    ) -> "NativeRunAttempt":
        failure = classify_infrastructure_failure(observation)
        identity = {
            "run_id": run_id, "attempt": attempt,
            "failure": failure.value if failure else None,
            "retryable": failure in _RETRYABLE,
            "accepted": failure is None,
            "return_code": observation.return_code,
            "usage_complete": observation.usage_complete,
            "identity_verified": observation.identity_verified,
        }
        return cls(
            attempt_id="native-attempt." + _digest(identity)[:40],
            run_id=run_id,
            attempt=attempt,
            failure=failure,
            retryable=failure in _RETRYABLE,
            accepted=failure is None,
            return_code=observation.return_code,
            usage_complete=observation.usage_complete,
            identity_verified=observation.identity_verified,
        )

    def payload(self) -> dict[str, object]:
        return {
            "attempt_id": self.attempt_id, "run_id": self.run_id,
            "attempt": self.attempt,
            "failure": self.failure.value if self.failure else None,
            "retryable": self.retryable, "accepted": self.accepted,
            "return_code": self.return_code,
            "usage_complete": self.usage_complete,
            "identity_verified": self.identity_verified,
        }


@dataclass(frozen=True, slots=True)
class NativeRunOutcome:
    run_id: str
    status: str
    attempts: tuple[NativeRunAttempt, ...]

    def __post_init__(self) -> None:
        if self.status not in {"accepted", "infrastructure_attempt"}:
            raise ValueError("native run outcome status is invalid")
        if not self.attempts or any(item.run_id != self.run_id for item in self.attempts):
            raise ValueError("native run attempts do not match outcome")
        if self.status == "accepted" and not self.attempts[-1].accepted:
            raise ValueError("accepted outcome requires accepted terminal attempt")
        if self.status == "infrastructure_attempt" and self.attempts[-1].accepted:
            raise ValueError("infrastructure outcome cannot contain accepted terminal attempt")

    @property
    def enters_quality_denominator(self) -> bool:
        return self.status == "accepted"

    def payload(self) -> dict[str, object]:
        return {
            "run_id": self.run_id, "status": self.status,
            "enters_quality_denominator": self.enters_quality_denominator,
            "attempts": [attempt.payload() for attempt in self.attempts],
        }


class NativeRunOutcomeStore:
    """Append-once per-run outcome store for restart-safe audit."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).expanduser().resolve()

    def put(self, outcome: NativeRunOutcome) -> bool:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{outcome.run_id}.json"
        lock_path = self.root / f"{outcome.run_id}.lock"
        serialized = _canonical(outcome.payload()) + "\n"
        with lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if path.is_symlink() or lock_path.is_symlink():
                raise ValueError("native outcome store cannot be symlinked")
            if path.exists():
                if path.read_text(encoding="utf-8") != serialized:
                    raise ValueError("native run already has a different outcome")
                return False
            temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
            temporary.write_text(serialized, encoding="utf-8")
            temporary.replace(path)
            return True

    def accepted(self) -> tuple[dict[str, object], ...]:
        values: list[dict[str, object]] = []
        if not self.root.exists():
            return ()
        for path in sorted(self.root.glob("native-run.*.json")):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError("native run outcome is unreadable") from exc
            if not isinstance(value, dict) or value.get("run_id") != path.stem:
                raise ValueError("native run outcome identity mismatch")
            if value.get("status") == "accepted" and value.get("enters_quality_denominator") is True:
                values.append(value)
            elif value.get("status") not in {"accepted", "infrastructure_attempt"}:
                raise ValueError("native run outcome status is malformed")
        return tuple(values)


Executor = Callable[[NativeAttributionRunSpec, int], NativeRunObservation]


class BoundedNativeRunScheduler:
    def __init__(self, *, max_concurrency: int = 1, max_retries: int = 2) -> None:
        if type(max_concurrency) is not int or max_concurrency < 1:
            raise ValueError("max concurrency must be positive")
        if type(max_retries) is not int or max_retries < 0:
            raise ValueError("max retries must be nonnegative")
        self.max_concurrency = max_concurrency
        self.max_retries = max_retries

    def _execute_one(
        self, run: NativeAttributionRunSpec, executor: Executor
    ) -> NativeRunOutcome:
        attempts: list[NativeRunAttempt] = []
        for ordinal in range(1, self.max_retries + 2):
            try:
                observation = executor(run, ordinal)
            except Exception:
                observation = NativeRunObservation(
                    return_code=-1, usage_complete=False, identity_verified=False,
                    error_code="runner_failure",
                )
            attempt = NativeRunAttempt.create(
                run_id=run.run_id, attempt=ordinal, observation=observation
            )
            attempts.append(attempt)
            if attempt.accepted:
                return NativeRunOutcome(run.run_id, "accepted", tuple(attempts))
            if not attempt.retryable:
                break
        return NativeRunOutcome(run.run_id, "infrastructure_attempt", tuple(attempts))

    def run(
        self, runs: Sequence[NativeAttributionRunSpec], executor: Executor
    ) -> tuple[NativeRunOutcome, ...]:
        values = tuple(runs)
        if not values or len({run.run_id for run in values}) != len(values):
            raise ValueError("scheduler requires unique native runs")
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_concurrency) as pool:
            futures = {pool.submit(self._execute_one, run, executor): index for index, run in enumerate(values)}
            ordered: list[NativeRunOutcome | None] = [None] * len(values)
            for future in concurrent.futures.as_completed(futures):
                ordered[futures[future]] = future.result()
        return tuple(value for value in ordered if value is not None)


__all__ = [
    "BoundedNativeRunScheduler", "InfrastructureFailure", "NativeRunAttempt",
    "NativeRunObservation", "NativeRunOutcome", "NativeRunOutcomeStore",
    "classify_infrastructure_failure",
]
