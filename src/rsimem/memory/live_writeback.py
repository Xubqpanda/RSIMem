"""Opt-in static semantic writeback at validated live Hermes boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping

from ..lifecycle import DryRunStatus, HermesLifecycleDryRunResult, PlanMemoryAction
from .backends import build_hermes_native_registry
from .contracts import MemoryExperience, MemoryKind, MemoryMessage, MemoryObserver
from .executor import TransactionalMutationExecutor
from .ingestion import (
    SemanticIngestionCoordinator,
    SemanticPolicyRegistry,
    build_semantic_ingest_request,
)
from .operation_graph import (
    AppendOnlyOperationEvidenceLog,
    AtomicOperationRecorder,
)
from .receipts import JsonMutationReceiptStore
from .semantic_loop import SemanticWritebackLoop, SemanticWritebackLoopResult
from .validation import MutationValidator
from ..memory_systems.mem0_flat.policy import (
    FlatSemanticCandidateReader,
    Mem0FlatSemanticPolicy,
)
from ..memory_systems.mem0_flat.prompts import CompletionClient


STATIC_SEMANTIC_WRITEBACK_SCHEMA_VERSION = 1


class StaticSemanticWritebackMode(StrEnum):
    DISABLED = "disabled"
    STATIC = "static"


@dataclass(frozen=True, slots=True)
class StaticSemanticWritebackConfig:
    mode: StaticSemanticWritebackMode = StaticSemanticWritebackMode.DISABLED
    timeout_seconds: float = 30.0
    max_output_tokens: int = 4096
    schema_version: int = STATIC_SEMANTIC_WRITEBACK_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != STATIC_SEMANTIC_WRITEBACK_SCHEMA_VERSION:
            raise ValueError("unsupported static semantic writeback schema version")
        object.__setattr__(self, "mode", StaticSemanticWritebackMode(self.mode))
        if self.timeout_seconds <= 0 or self.max_output_tokens < 1:
            raise ValueError("static semantic model limits must be positive")

    @property
    def enabled(self) -> bool:
        return self.mode == StaticSemanticWritebackMode.STATIC

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object] | None,
    ) -> "StaticSemanticWritebackConfig":
        value = value or {}
        allowed = {"mode", "timeout_seconds", "max_output_tokens"}
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(
                "unknown static semantic writeback fields: "
                + ", ".join(sorted(unknown))
            )
        return cls(
            mode=StaticSemanticWritebackMode(
                str(value.get("mode") or StaticSemanticWritebackMode.DISABLED)
            ),
            timeout_seconds=float(value.get("timeout_seconds") or 30.0),
            max_output_tokens=int(value.get("max_output_tokens") or 4096),
        )


@dataclass(frozen=True, slots=True)
class StaticSemanticBoundaryResult:
    snapshot_id: str
    evaluation_id: str
    plan_id: str
    writeback: SemanticWritebackLoopResult
    schema_version: int = STATIC_SEMANTIC_WRITEBACK_SCHEMA_VERSION

    def observer_evidence(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "snapshot_id": self.snapshot_id,
            "evaluation_id": self.evaluation_id,
            "plan_id": self.plan_id,
            "writeback": self.writeback.observer_evidence(),
        }


class StaticSemanticWritebackRuntime:
    """Execute fixed Mem0-flat semantic writeback inside one isolated PAST home."""

    def __init__(
        self,
        hermes_home: Path,
        completion_client: CompletionClient,
        *,
        operation_evidence_path: Path,
        mutation_receipt_path: Path,
        observer: MemoryObserver | None = None,
        ingestion_observer: Any | None = None,
    ) -> None:
        self.hermes_home = hermes_home.expanduser().resolve()
        self.registry = build_hermes_native_registry(self.hermes_home)
        self.receipts = JsonMutationReceiptStore(mutation_receipt_path)
        self.operation_log = AppendOnlyOperationEvidenceLog(operation_evidence_path)
        self.operation_recorder = AtomicOperationRecorder(self.operation_log)
        self.policy = Mem0FlatSemanticPolicy(
            completion_client,
            operation_recorder=self.operation_recorder,
        )
        self.candidates = FlatSemanticCandidateReader(
            self.registry,
            ownership=self.receipts,
        )
        policies = SemanticPolicyRegistry()
        policies.register(self.policy)
        self.coordinator = SemanticIngestionCoordinator(
            policies,
            provider=self.policy.descriptor.provider,
        )
        self.executor = TransactionalMutationExecutor(
            self.registry,
            MutationValidator(self.registry, target_resolver=self.receipts),
            self.receipts,
            enabled=True,
            isolated_fixture=True,
            operation_recorder=self.operation_recorder,
        )
        self.loop = SemanticWritebackLoop(
            self.coordinator,
            self.policy,
            self.candidates,
            self.executor,
            observer=observer,
            operation_recorder=self.operation_recorder,
        )
        if ingestion_observer is not None and not callable(
            getattr(ingestion_observer, "record_ingestion", None)
        ):
            raise TypeError("ingestion observer must provide record_ingestion")
        self.ingestion_observer = ingestion_observer
        self._results: list[StaticSemanticBoundaryResult] = []
        self._closed = False

    @property
    def results(self) -> tuple[StaticSemanticBoundaryResult, ...]:
        return tuple(self._results)

    def process(
        self,
        lifecycle: HermesLifecycleDryRunResult,
    ) -> tuple[StaticSemanticBoundaryResult, ...]:
        if self._closed:
            raise RuntimeError("static semantic writeback runtime is closed")
        if len(lifecycle.plans) != len(lifecycle.receipts):
            raise ValueError("lifecycle plans and dry-run receipts must be one-to-one")
        receipts = {receipt.plan_id: receipt for receipt in lifecycle.receipts}
        if len(receipts) != len(lifecycle.receipts):
            raise ValueError("lifecycle dry-run receipt plan IDs must be unique")
        if set(receipts) != {plan.plan_id for plan in lifecycle.plans}:
            raise ValueError("lifecycle dry-run receipts must match plan IDs")

        snapshot = lifecycle.snapshot
        experience = MemoryExperience(
            experience_id=f"experience.{snapshot.snapshot_id}",
            session_id=snapshot.session_id,
            task_id=snapshot.task_id,
            outcome="completed",
            messages=tuple(
                MemoryMessage(segment.role, segment.content)
                for segment in snapshot.segments
            ),
        )
        added = []
        for plan in lifecycle.plans:
            if (
                plan.memory_kind != MemoryKind.SEMANTIC
                or plan.memory_action
                not in {PlanMemoryAction.ADD, PlanMemoryAction.UPDATE}
            ):
                continue
            receipt = receipts[plan.plan_id]
            if receipt.status not in {
                DryRunStatus.ACCEPTED,
                DryRunStatus.DUPLICATE,
            }:
                raise ValueError("static semantic writeback requires a validated plan")
            if receipt.mutation_id is None:
                raise ValueError("validated lifecycle plan requires a mutation ID")
            request = build_semantic_ingest_request(
                snapshot,
                plan,
                experience,
                policy_version=self.policy.descriptor.policy_version,
                framework_version=self.policy.descriptor.framework_version,
            )
            writeback = self.loop.run(
                request,
                current_source_revision=snapshot.context_revision,
            )
            if self.ingestion_observer is not None and writeback.ingestion is not None:
                self.ingestion_observer.record_ingestion(request, writeback.ingestion)
            result = StaticSemanticBoundaryResult(
                snapshot.snapshot_id,
                lifecycle.evaluation.evaluation_id,
                plan.plan_id,
                writeback,
            )
            self._results.append(result)
            added.append(result)
        return tuple(added)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.registry.close()
