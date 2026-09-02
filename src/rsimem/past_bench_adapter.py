"""PAST-Bench implementation of the host-neutral benchmark adapter."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

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


_TRACE_DIGEST = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class PastExecutionTrace:
    """Content-free trace exported by one completed PAST runtime response.

    This contract is intentionally independent of Pydantic's runtime protocol
    classes so the benchmark runtime can evolve without becoming an RSIMem
    dependency.  It retains only digests and counters needed to compare a
    matched execution; raw final output, prompts, memory content, and scores
    remain outside the adapter boundary.
    """

    trace_id: str
    case_id: str
    terminal_status: str
    final_output_digest: str
    usage_digest: str
    process_event_count: int
    process_feedback_digest: str | None
    host_event_count: int
    host_state_digest: str | None
    host_projection_digest: str | None

    def __post_init__(self) -> None:
        if self.trace_id != (
            "past-execution-trace."
            + content_digest(self.identity_payload())[:40]
        ):
            raise ValueError("PAST execution trace ID mismatch")
        if self.terminal_status not in {"acting", "finished", "error"}:
            raise ValueError("PAST execution trace status is invalid")
        for value, name in (
            (self.final_output_digest, "final output"),
            (self.usage_digest, "usage"),
        ):
            if not isinstance(value, str) or _TRACE_DIGEST.fullmatch(value) is None:
                raise ValueError(f"PAST execution trace {name} digest is invalid")
        for value, name in (
            (self.process_event_count, "process event count"),
            (self.host_event_count, "host event count"),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"PAST execution trace {name} is invalid")
        for value, name in (
            (self.process_feedback_digest, "process feedback"),
            (self.host_state_digest, "host state"),
            (self.host_projection_digest, "host projection"),
        ):
            if value is not None and (
                not isinstance(value, str) or _TRACE_DIGEST.fullmatch(value) is None
            ):
                raise ValueError(f"PAST execution trace {name} digest is invalid")
        if self.host_event_count == 0 and any((
            self.host_state_digest is not None,
            self.host_projection_digest is not None,
        )):
            raise ValueError("host digests require host events")
        if self.host_event_count and (
            self.host_state_digest is None or self.host_projection_digest is None
        ):
            raise ValueError("host events require host digests")

    def identity_payload(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "terminal_status": self.terminal_status,
            "final_output_digest": self.final_output_digest,
            "usage_digest": self.usage_digest,
            "process_event_count": self.process_event_count,
            "process_feedback_digest": self.process_feedback_digest,
            "host_event_count": self.host_event_count,
            "host_state_digest": self.host_state_digest,
            "host_projection_digest": self.host_projection_digest,
        }

    @property
    def matched_projection_digest(self) -> str:
        """Variant-neutral identity for output, raw usage, and host projection."""

        return content_digest({
            "case_id": self.case_id,
            "terminal_status": self.terminal_status,
            "final_output_digest": self.final_output_digest,
            "usage_digest": self.usage_digest,
            "host_event_count": self.host_event_count,
            "host_projection_digest": self.host_projection_digest,
        })

    @classmethod
    def from_runtime_response(
        cls,
        request: BenchmarkTaskRequest,
        response: object,
    ) -> "PastExecutionTrace":
        status = getattr(response, "status", None)
        if not isinstance(status, str):
            raise TypeError("PAST runtime response status must be text")
        final_output = getattr(response, "final_output", None)
        if final_output is not None and not isinstance(final_output, str):
            raise TypeError("PAST runtime final output must be text or null")
        usage = getattr(response, "usage", None)
        usage_payload = {
            name: getattr(usage, name, None)
            for name in (
                "input_tokens", "output_tokens", "cache_read_tokens",
                "cache_write_tokens", "reasoning_tokens", "request_count",
                "retry_count", "usage_complete",
            )
        }
        for name, value in usage_payload.items():
            if name == "usage_complete":
                if type(value) is not bool:
                    raise TypeError("PAST runtime usage completeness must be bool")
            elif value is not None and (type(value) is not int or value < 0):
                raise ValueError("PAST runtime usage value is invalid")
        process_ids = tuple(getattr(response, "process_feedback_event_ids", ()))
        host_ids = tuple(getattr(response, "host_event_ids", ()))
        for values, name in ((process_ids, "process event IDs"), (host_ids, "host event IDs")):
            if len(values) != len(set(values)) or any(
                not isinstance(value, str) or not value.strip() for value in values
            ):
                raise ValueError(f"PAST runtime {name} are invalid")
        values: dict[str, Any] = {
            "case_id": request.case_id,
            "terminal_status": status,
            "final_output_digest": content_digest(final_output),
            "usage_digest": content_digest(usage_payload),
            "process_event_count": len(process_ids),
            "process_feedback_digest": getattr(response, "process_feedback_digest", None),
            "host_event_count": len(host_ids),
            "host_state_digest": getattr(response, "host_state_digest", None),
            "host_projection_digest": getattr(response, "host_projection_digest", None),
        }
        return cls(
            trace_id="past-execution-trace." + content_digest(values)[:40],
            **values,
        )


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
        matches = [item for item in self.enumerate_cases(request.split) if item == request]
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


__all__ = ["PastBenchAdapter", "PastExecutionTrace"]
