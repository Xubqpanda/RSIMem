"""PAST-Bench implementation of the host-neutral benchmark adapter."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

import yaml

from .adapter_contracts import (
    AdapterResult,
    AdapterStatus,
    BenchmarkAdapter,
    BenchmarkPublicEvent,
    BenchmarkSplit,
    BenchmarkTaskRequest,
    FinalEvaluationRecord,
    content_digest,
)
from .memory.family_matrix import PastFamilyMatrix


class PastBenchAdapter:
    """Expose task identity while keeping grader/reference data final-only.

    The runner remains responsible for executing an agent.  This adapter
    supplies deterministic case enumeration and reset/step identities; a final
    evaluator callback is required explicitly when a score is requested.
    """

    def __init__(
        self,
        benchmark_root: Path,
        family_matrix: PastFamilyMatrix,
        *,
        split_family_ids: Mapping[BenchmarkSplit, Sequence[str]] | None = None,
        final_score_digest_provider: Callable[[BenchmarkTaskRequest], str] | None = None,
    ) -> None:
        self.root = benchmark_root.expanduser().resolve()
        self.family_matrix = family_matrix
        if split_family_ids is None:
            raise ValueError("PAST-Bench adapter requires frozen split family IDs")
        self._split_family_ids = {
            BenchmarkSplit(role): tuple(family_ids)
            for role, family_ids in split_family_ids.items()
        }
        if set(self._split_family_ids) != set(BenchmarkSplit):
            raise ValueError("PAST-Bench split family IDs must cover train, validation, and final")
        for role, family_ids in self._split_family_ids.items():
            if not family_ids or len(family_ids) != len(set(family_ids)):
                raise ValueError(f"{role.value} split family IDs must be nonempty and unique")
            for family_id in family_ids:
                self.family_matrix.spec_for(family_id)
        self._final_score_digest_provider = final_score_digest_provider
        if not self.root.is_dir():
            raise ValueError("PAST-Bench root must be a directory")

    def _task_root(self, family_id: str) -> Path:
        spec = self.family_matrix.spec_for(family_id)
        return self.root / spec.task_root

    @staticmethod
    def _task_digest(path: Path) -> str:
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise ValueError("PAST task file cannot be read") from exc
        return hashlib.sha256(raw).hexdigest()

    @staticmethod
    def _task_public_identity(path: Path) -> tuple[str, str]:
        """Read only task identity fields; do not return prompt/grader values."""

        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise ValueError("PAST task YAML cannot be parsed") from exc
        if not isinstance(payload, dict):
            raise ValueError("PAST task YAML must be an object")
        task_id = payload.get("task_id")
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError("PAST task identity is missing")
        return task_id, PastBenchAdapter._task_digest(path)

    def enumerate_cases(self, split: BenchmarkSplit) -> Sequence[BenchmarkTaskRequest]:
        role = BenchmarkSplit(split)
        family_ids = tuple(self.family_matrix.spec_for(family_id) for family_id in self._split_family_ids[role])
        requests: list[BenchmarkTaskRequest] = []
        for spec in family_ids:
            task_root = self._task_root(spec.family_id)
            if not task_root.is_dir():
                continue
            for task_path in sorted(task_root.glob("*/task.yaml")):
                task_id, digest = self._task_public_identity(task_path)
                identity = {
                    "relative_task": str(task_path.relative_to(self.root)),
                    "task_id": task_id,
                    "split": role.value,
                }
                case_digest = content_digest(identity)
                requests.append(BenchmarkTaskRequest(
                    case_id=f"past-case.{case_digest[:40]}",
                    split=role,
                    task_template_id=f"past-template.{digest[:40]}",
                    seed=f"past-seed.{case_digest[40:64]}",
                    tool_budget=32,
                    max_turns=20,
                ))
        return tuple(requests)

    def _find_request_path(self, request: BenchmarkTaskRequest) -> Path:
        # Re-enumeration is deterministic and avoids carrying a mutable path
        # through the host/method boundary.
        matches = [item for item in self.enumerate_cases(request.split) if item.case_id == request.case_id]
        if len(matches) != 1:
            raise ValueError("benchmark case is not registered in the matrix")
        # The adapter intentionally does not expose task path in the request;
        # reset/step use the case identity only and remain metadata operations.
        return self.root

    def reset(self, request: BenchmarkTaskRequest) -> AdapterResult:
        self._find_request_path(request)
        operation_id = f"operation.past.reset.{request.case_id}"
        return AdapterResult(AdapterStatus.SUPPORTED, operation_id)

    def step(self, request: BenchmarkTaskRequest) -> tuple[AdapterResult, BenchmarkPublicEvent]:
        self._find_request_path(request)
        state_digest = hashlib.sha256(
            json.dumps({"case_id": request.case_id, "split": request.split.value}, sort_keys=True).encode()
        ).hexdigest()
        event = BenchmarkPublicEvent(
            event_id=f"benchmark-event.ready.{request.case_id}",
            case_id=request.case_id,
            stage="task",
            event_type="task.ready",
            public_state_digest=state_digest,
        )
        return AdapterResult(AdapterStatus.SUPPORTED, f"operation.past.step.{request.case_id}", output_digest=state_digest), event

    def evaluate_final(self, request: BenchmarkTaskRequest) -> FinalEvaluationRecord:
        self._find_request_path(request)
        if self._final_score_digest_provider is None:
            raise ValueError("final score provider is required at the final evaluation boundary")
        score_digest = self._final_score_digest_provider(request)
        return FinalEvaluationRecord(
            evaluation_id=f"evaluation.past.{request.case_id}",
            case_id=request.case_id,
            metric_id="past_bench.official_task_metric.v1",
            score_digest=score_digest,
        )


__all__ = ["PastBenchAdapter"]
