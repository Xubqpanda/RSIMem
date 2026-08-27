"""Opt-in lifecycle dry-run runtime for persisted Hermes sessions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Callable, Mapping, Sequence

from ..memory.contracts import MemoryKind
from .contracts import (
    CompletionStatus,
    ContextAction,
    ContextEvaluation,
    ContextEvaluationRequest,
    EvaluationSignal,
    EvaluationTrigger,
    MemoryScope,
    TemporalValidity,
    WritebackAction,
)
from .controller import LifecycleController
from .evaluators import JsonLlmContextEvaluator
from .hermes import HermesStateSnapshotCollector, snapshot_to_evaluation_request
from .snapshot import ContextSnapshot, TaskLifecycleState
from .writeback import (
    DryRunReceipt,
    JsonIdempotencyReceiptStore,
    WritebackCoordinator,
    WritebackPlan,
)


def _stable_hash(prefix: str, value: object, *, length: int = 40) -> str:
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return f"{prefix}_{hashlib.sha256(encoded.encode('utf-8')).hexdigest()[:length]}"


class HermesLifecycleEvaluatorMode(StrEnum):
    DISABLED = "disabled"
    DETERMINISTIC = "deterministic"
    INJECTED_JSON = "injected_json"


@dataclass(frozen=True, slots=True)
class HermesLifecycleConfig:
    evaluator_mode: HermesLifecycleEvaluatorMode = HermesLifecycleEvaluatorMode.DISABLED
    policy_version: str = "phase1-dry-run-v1"
    compiler_version: str = "uncompiled-v0"
    timeout_seconds: float = 30.0
    max_output_tokens: int = 4096

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "evaluator_mode",
            HermesLifecycleEvaluatorMode(self.evaluator_mode),
        )
        if not self.policy_version.strip() or not self.compiler_version.strip():
            raise ValueError("lifecycle policy and compiler versions must not be empty")
        if self.timeout_seconds <= 0:
            raise ValueError("lifecycle evaluator timeout must be positive")
        if self.max_output_tokens < 1:
            raise ValueError("lifecycle evaluator max_output_tokens must be positive")

    @property
    def enabled(self) -> bool:
        return self.evaluator_mode != HermesLifecycleEvaluatorMode.DISABLED

    @classmethod
    def from_mapping(cls, value: Mapping[str, object] | None) -> "HermesLifecycleConfig":
        value = value or {}
        allowed = {
            "evaluator_mode",
            "policy_version",
            "compiler_version",
            "timeout_seconds",
            "max_output_tokens",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(
                "unknown Hermes lifecycle configuration fields: "
                + ", ".join(sorted(unknown))
            )
        return cls(
            evaluator_mode=HermesLifecycleEvaluatorMode(
                str(value.get("evaluator_mode") or "disabled")
            ),
            policy_version=str(value.get("policy_version") or "phase1-dry-run-v1"),
            compiler_version=str(value.get("compiler_version") or "uncompiled-v0"),
            timeout_seconds=float(value.get("timeout_seconds") or 30.0),
            max_output_tokens=int(value.get("max_output_tokens") or 4096),
        )


class DeterministicHermesDryRunEvaluator:
    """Select the first completed user segment as a semantic candidate."""

    name = "deterministic-hermes-dry-run"

    def __init__(self, *, policy_version: str, compiler_version: str) -> None:
        self.policy_version = policy_version
        self.compiler_version = compiler_version

    def evaluate(self, request: ContextEvaluationRequest) -> ContextEvaluation:
        candidate = next(
            (
                segment.segment_id
                for segment in request.segments
                if segment.role == "user"
                and segment.completed
                and segment.segment_id not in request.active_segment_ids
            ),
            None,
        )
        signals = tuple(
            EvaluationSignal(
                segment_id=segment.segment_id,
                context_action=(
                    ContextAction.EVICT
                    if segment.segment_id == candidate
                    else ContextAction.RETAIN
                ),
                writeback_action=(
                    WritebackAction.ADD
                    if segment.segment_id == candidate
                    else WritebackAction.DEFER
                ),
                memory_kind=(
                    MemoryKind.SEMANTIC
                    if segment.segment_id == candidate
                    else None
                ),
                utility_estimate=1.0 if segment.segment_id == candidate else 0.0,
                confidence=1.0,
                completion_status=(
                    CompletionStatus.COMPLETED
                    if segment.completed
                    else CompletionStatus.IN_PROGRESS
                ),
                completion_evidence=(
                    ("host_segment_completed",)
                    if segment.completed
                    else ("host_segment_unresolved",)
                ),
                safe_to_evict=(segment.segment_id == candidate),
                unresolved_state=None if segment.completed else "host_unresolved",
                scope=(MemoryScope.USER if segment.segment_id == candidate else MemoryScope.TASK),
                temporal_validity=(
                    TemporalValidity.DURABLE
                    if segment.segment_id == candidate
                    else TemporalValidity.CURRENT
                ),
                provenance=(
                    str(request.metadata.get("snapshot_id") or "snapshot_unknown"),
                    segment.segment_id,
                ),
                reusable_facts=(
                    ("compile_from_source_segment",)
                    if segment.segment_id == candidate
                    else ()
                ),
                compiler_version=self.compiler_version,
                reason_codes=(
                    ("host_completed_user_candidate",)
                    if segment.segment_id == candidate
                    else ("no_action",)
                ),
            )
            for segment in request.segments
        )
        return ContextEvaluation(
            evaluation_id=request.evaluation_id,
            evaluator=self.name,
            trigger=request.trigger,
            signals=signals,
            policy_version=self.policy_version,
            input_chars=sum(len(segment.content) for segment in request.segments),
        )


@dataclass(frozen=True, slots=True)
class HermesLifecycleDryRunResult:
    snapshot: ContextSnapshot
    evaluation: ContextEvaluation
    plans: tuple[WritebackPlan, ...]
    receipts: tuple[DryRunReceipt, ...]


class HermesLifecycleDryRunRuntime:
    """Build, evaluate, validate, and reserve plans without applying them."""

    def __init__(
        self,
        config: HermesLifecycleConfig,
        *,
        run_id: str,
        episode_id: str,
        session_id: str,
        task_id: str,
        variant: str,
        trace_id: str,
        receipt_path: Path,
        evidence_path: Path,
        family_id: str | None = None,
        stage: str | None = None,
        injected_complete: Callable[[str], str] | None = None,
    ) -> None:
        if not config.enabled:
            raise ValueError("disabled lifecycle mode must not construct a runtime")
        if (
            config.evaluator_mode == HermesLifecycleEvaluatorMode.INJECTED_JSON
            and injected_complete is None
        ):
            raise ValueError("injected_json lifecycle mode requires an injected client")
        self.config = config
        self.run_id = run_id
        self.episode_id = episode_id
        self.session_id = session_id
        self.task_id = task_id
        self._collector = HermesStateSnapshotCollector()
        self._processed: dict[str, HermesLifecycleDryRunResult] = {}
        from ..ledger import LifecycleLedgerObserver

        self.observer = LifecycleLedgerObserver(
            variant=variant,
            trace_id=trace_id,
            family_id=family_id,
            stage=stage,
            output_path=evidence_path,
        )
        if config.evaluator_mode == HermesLifecycleEvaluatorMode.DETERMINISTIC:
            evaluator = DeterministicHermesDryRunEvaluator(
                policy_version=config.policy_version,
                compiler_version=config.compiler_version,
            )
        else:
            assert injected_complete is not None
            evaluator = JsonLlmContextEvaluator(
                injected_complete,
                compiler_version=config.compiler_version,
            )
        self._controller = LifecycleController(evaluator)
        self._coordinator = WritebackCoordinator(
            receipt_store=JsonIdempotencyReceiptStore(receipt_path),
            observers=(self.observer,),
        )

    def process(
        self,
        rows: Sequence[Mapping[str, object]],
        *,
        trigger: EvaluationTrigger,
        task_state: TaskLifecycleState,
        source_ref: str,
    ) -> HermesLifecycleDryRunResult:
        trigger = EvaluationTrigger(trigger)
        if trigger not in {
            EvaluationTrigger.TASK_COMPLETED,
            EvaluationTrigger.SESSION_END,
        }:
            raise ValueError("live Hermes dry-run only accepts explicit host boundaries")
        task_state = TaskLifecycleState(task_state)
        if (
            trigger == EvaluationTrigger.TASK_COMPLETED
            and task_state != TaskLifecycleState.COMPLETED
        ):
            raise ValueError("task_completed boundary requires completed host task state")
        if trigger == EvaluationTrigger.SESSION_END and task_state == TaskLifecycleState.ACTIVE:
            raise ValueError("session_end boundary cannot carry active host task state")
        snapshot = self._collector.collect(
            rows,
            run_id=self.run_id,
            episode_id=self.episode_id,
            session_id=self.session_id,
            task_id=self.task_id,
            task_state=task_state,
            lifecycle_state=trigger.value,
            source_ref=source_ref,
        )
        evaluation_id = _stable_hash(
            "evaluation",
            {
                "snapshot_id": snapshot.snapshot_id,
                "context_revision": snapshot.context_revision,
                "trigger": trigger.value,
                "policy_version": self.config.policy_version,
            },
        )
        existing = self._processed.get(evaluation_id)
        if existing is not None:
            return existing

        self.observer.record_snapshot(snapshot)
        request = snapshot_to_evaluation_request(
            snapshot,
            evaluation_id=evaluation_id,
            trigger=trigger,
            turn_index=sum(1 for row in rows if row.get("role") == "user"),
            policy_version=self.config.policy_version,
        )
        try:
            evaluation = self._controller.evaluate(request, force=True)
        except Exception as exc:
            self.observer.record_evaluation(
                snapshot,
                evaluation_id=evaluation_id,
                trigger=trigger.value,
                evaluator=self._controller.evaluator.name,
                policy_version=self.config.policy_version,
                status="rejected",
                reason_codes=(f"evaluator_{type(exc).__name__.lower()}",),
            )
            raise
        assert evaluation is not None
        if evaluation.policy_version != self.config.policy_version:
            evaluation = replace(evaluation, policy_version=self.config.policy_version)
        self.observer.record_evaluation(
            snapshot,
            evaluation_id=evaluation_id,
            trigger=trigger.value,
            evaluator=evaluation.evaluator,
            policy_version=evaluation.policy_version,
            status="accepted",
        )
        plans = self._coordinator.create_plans(snapshot, evaluation)
        receipts = tuple(self._coordinator.dry_run(plan, snapshot) for plan in plans)
        result = HermesLifecycleDryRunResult(snapshot, evaluation, plans, receipts)
        self._processed[evaluation_id] = result
        return result
