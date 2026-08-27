"""Opt-in static semantic writeback at validated live Hermes boundaries."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping

from ..lifecycle import (
    ContextSnapshot,
    DryRunStatus,
    HermesLifecycleDryRunResult,
    PlanMemoryAction,
)
from .adaptive_mem0_binding import (
    ActiveAdaptiveMem0Binder,
    AdaptiveMem0Binding,
    TrustedAdaptiveMem0Parameter,
)
from .adaptive_policy_store import JsonAdaptivePolicyStore
from .backends import build_hermes_native_registry
from .contracts import MemoryExperience, MemoryKind, MemoryMessage, MemoryObserver
from .executor import TransactionalMutationExecutor
from .future_trace import SemanticFeedbackContract
from .ingestion import (
    SemanticIngestionCoordinator,
    SemanticIngestRequest,
    SemanticPolicyRegistry,
    build_completed_task_semantic_ingest_request,
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
from ..memory_systems.mem0_flat.utility_gate import FrozenMem0UtilityGate


STATIC_SEMANTIC_WRITEBACK_SCHEMA_VERSION = 1
SEMANTIC_COMPILATION_RECEIPT_SCHEMA_VERSION = 1


class StaticSemanticWritebackMode(StrEnum):
    DISABLED = "disabled"
    STATIC = "static"
    STATIC_UTILITY = "static_utility"
    ADAPTIVE_UTILITY = "adaptive_utility"


class SemanticCompilationReceiptStatus(StrEnum):
    PENDING = "pending"
    COMPLETED = "completed"


@dataclass(frozen=True, slots=True)
class SemanticCompilationReceipt:
    compilation_id: str
    request_digest: str
    snapshot_id: str
    context_revision: str
    status: SemanticCompilationReceiptStatus
    ingestion_execution_id: str | None = None
    mutation_ids: tuple[str, ...] = ()
    schema_version: int = SEMANTIC_COMPILATION_RECEIPT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "status",
            SemanticCompilationReceiptStatus(self.status),
        )
        if self.schema_version != SEMANTIC_COMPILATION_RECEIPT_SCHEMA_VERSION:
            raise ValueError("unsupported semantic compilation receipt schema")
        if any(not value.strip() for value in (
            self.compilation_id,
            self.snapshot_id,
            self.context_revision,
        )):
            raise ValueError("semantic compilation receipt identity is incomplete")
        if len(self.request_digest) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.request_digest
        ):
            raise ValueError("semantic compilation request digest is invalid")
        if self.status == SemanticCompilationReceiptStatus.PENDING and (
            self.ingestion_execution_id is not None or self.mutation_ids
        ):
            raise ValueError("pending semantic compilation cannot carry results")
        if self.status == SemanticCompilationReceiptStatus.COMPLETED and (
            self.ingestion_execution_id is None
        ):
            raise ValueError("completed semantic compilation requires ingestion identity")

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "compilation_id": self.compilation_id,
            "request_digest": self.request_digest,
            "snapshot_id": self.snapshot_id,
            "context_revision": self.context_revision,
            "status": self.status.value,
            "ingestion_execution_id": self.ingestion_execution_id,
            "mutation_ids": list(self.mutation_ids),
        }


class JsonSemanticCompilationReceiptStore:
    """Atomically reserve content-free completed-task compilation identities."""

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().resolve()

    @contextmanager
    def _lock(self, operation: int):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_suffix(self.path.suffix + ".lock").open(
            "w", encoding="utf-8"
        ) as lock:
            fcntl.flock(lock.fileno(), operation)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _read_unlocked(self) -> dict[str, SemanticCompilationReceipt]:
        if not self.path.exists():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("malformed semantic compilation receipt store") from exc
        if not isinstance(payload, dict) or set(payload) != {
            "schema_version", "receipts"
        } or payload["schema_version"] != SEMANTIC_COMPILATION_RECEIPT_SCHEMA_VERSION:
            raise ValueError("unsupported semantic compilation receipt store")
        if not isinstance(payload["receipts"], dict):
            raise ValueError("malformed semantic compilation receipts")
        receipts = {}
        expected = set(SemanticCompilationReceipt.__dataclass_fields__)
        for key, raw in payload["receipts"].items():
            if not isinstance(raw, dict) or set(raw) != expected:
                raise ValueError("malformed semantic compilation receipt")
            try:
                receipt = SemanticCompilationReceipt(
                    **{**raw, "mutation_ids": tuple(raw["mutation_ids"])}
                )
            except (TypeError, ValueError) as exc:
                raise ValueError("malformed semantic compilation receipt") from exc
            if key != receipt.compilation_id:
                raise ValueError("semantic compilation receipt key mismatch")
            receipts[key] = receipt
        return receipts

    def _write_unlocked(
        self,
        receipts: dict[str, SemanticCompilationReceipt],
    ) -> None:
        payload = {
            "schema_version": SEMANTIC_COMPILATION_RECEIPT_SCHEMA_VERSION,
            "receipts": {
                key: value.payload() for key, value in sorted(receipts.items())
            },
        }
        fd, temporary = tempfile.mkstemp(prefix=".semantic-compilation.", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=True, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        except BaseException:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise

    def reserve(
        self,
        receipt: SemanticCompilationReceipt,
    ) -> tuple[SemanticCompilationReceipt, bool]:
        if receipt.status != SemanticCompilationReceiptStatus.PENDING:
            raise ValueError("semantic compilation reservation must be pending")
        with self._lock(fcntl.LOCK_EX):
            receipts = self._read_unlocked()
            existing = receipts.get(receipt.compilation_id)
            if existing is not None:
                if (
                    existing.request_digest != receipt.request_digest
                    or existing.snapshot_id != receipt.snapshot_id
                    or existing.context_revision != receipt.context_revision
                ):
                    raise ValueError("semantic compilation receipt identity conflict")
                return existing, False
            receipts[receipt.compilation_id] = receipt
            self._write_unlocked(receipts)
            return receipt, True

    def complete(
        self,
        receipt: SemanticCompilationReceipt,
    ) -> SemanticCompilationReceipt:
        if receipt.status != SemanticCompilationReceiptStatus.COMPLETED:
            raise ValueError("semantic compilation completion must be terminal")
        with self._lock(fcntl.LOCK_EX):
            receipts = self._read_unlocked()
            current = receipts.get(receipt.compilation_id)
            if current is None or current.request_digest != receipt.request_digest:
                raise ValueError("semantic compilation reservation is missing")
            if current.status == SemanticCompilationReceiptStatus.COMPLETED:
                if current != receipt:
                    raise ValueError("semantic compilation completion conflicts")
                return current
            receipts[receipt.compilation_id] = receipt
            self._write_unlocked(receipts)
            return receipt


@dataclass(frozen=True, slots=True)
class StaticSemanticWritebackConfig:
    mode: StaticSemanticWritebackMode = StaticSemanticWritebackMode.DISABLED
    timeout_seconds: float = 30.0
    max_output_tokens: int = 4096
    adaptive_policy_store_path: str | None = None
    adaptive_trusted_roots: tuple[str, ...] = ()
    adaptive_parameters: tuple[TrustedAdaptiveMem0Parameter, ...] = ()
    feedback_contract: SemanticFeedbackContract = SemanticFeedbackContract.DISABLED
    schema_version: int = STATIC_SEMANTIC_WRITEBACK_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != STATIC_SEMANTIC_WRITEBACK_SCHEMA_VERSION:
            raise ValueError("unsupported static semantic writeback schema version")
        object.__setattr__(self, "mode", StaticSemanticWritebackMode(self.mode))
        if self.timeout_seconds <= 0 or self.max_output_tokens < 1:
            raise ValueError("static semantic model limits must be positive")
        object.__setattr__(
            self,
            "adaptive_trusted_roots",
            tuple(self.adaptive_trusted_roots),
        )
        object.__setattr__(self, "adaptive_parameters", tuple(self.adaptive_parameters))
        object.__setattr__(
            self,
            "feedback_contract",
            SemanticFeedbackContract(self.feedback_contract),
        )
        adaptive_fields = bool(
            self.adaptive_policy_store_path
            or self.adaptive_trusted_roots
            or self.adaptive_parameters
        )
        if self.adaptive_enabled and (
            not self.adaptive_policy_store_path
            or not self.adaptive_trusted_roots
            or not self.adaptive_parameters
        ):
            raise ValueError("adaptive semantic writeback configuration is incomplete")
        if not self.adaptive_enabled and adaptive_fields:
            raise ValueError("adaptive semantic fields require adaptive_utility mode")
        if not self.enabled and self.feedback_contract != SemanticFeedbackContract.DISABLED:
            raise ValueError("semantic feedback contract requires writeback mode")

    @property
    def enabled(self) -> bool:
        return self.mode != StaticSemanticWritebackMode.DISABLED

    @property
    def utility_enabled(self) -> bool:
        return self.mode in {
            StaticSemanticWritebackMode.STATIC_UTILITY,
            StaticSemanticWritebackMode.ADAPTIVE_UTILITY,
        }

    @property
    def adaptive_enabled(self) -> bool:
        return self.mode == StaticSemanticWritebackMode.ADAPTIVE_UTILITY

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object] | None,
    ) -> "StaticSemanticWritebackConfig":
        value = value or {}
        allowed = {
            "mode",
            "timeout_seconds",
            "max_output_tokens",
            "adaptive_policy_store_path",
            "adaptive_trusted_roots",
            "adaptive_parameters",
            "feedback_contract",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(
                "unknown static semantic writeback fields: "
                + ", ".join(sorted(unknown))
            )
        parameter_values = value.get("adaptive_parameters") or ()
        if not isinstance(parameter_values, (list, tuple)):
            raise ValueError("adaptive semantic parameters must be a list")
        parameters = []
        for item in parameter_values:
            if not isinstance(item, Mapping):
                raise ValueError("adaptive semantic parameter must be an object")
            if set(item) != {
                "parameter_id",
                "name",
                "prompt_ref",
                "baseline_value",
            }:
                raise ValueError("adaptive semantic parameter fields are invalid")
            parameters.append(TrustedAdaptiveMem0Parameter(
                parameter_id=str(item["parameter_id"]),
                name=str(item["name"]),
                prompt_ref=str(item["prompt_ref"]),
                baseline_value=float(item["baseline_value"]),
            ))
        roots = value.get("adaptive_trusted_roots") or ()
        if not isinstance(roots, (list, tuple)):
            raise ValueError("adaptive trusted roots must be a list")
        return cls(
            mode=StaticSemanticWritebackMode(
                str(value.get("mode") or StaticSemanticWritebackMode.DISABLED)
            ),
            timeout_seconds=float(value.get("timeout_seconds") or 30.0),
            max_output_tokens=int(value.get("max_output_tokens") or 4096),
            adaptive_policy_store_path=(
                str(value["adaptive_policy_store_path"])
                if value.get("adaptive_policy_store_path")
                else None
            ),
            adaptive_trusted_roots=tuple(str(item) for item in roots),
            adaptive_parameters=tuple(parameters),
            feedback_contract=SemanticFeedbackContract(
                str(value.get("feedback_contract") or SemanticFeedbackContract.DISABLED)
            ),
        )


@dataclass(frozen=True, slots=True)
class StaticSemanticBoundaryResult:
    snapshot_id: str
    compilation_id: str
    writeback: SemanticWritebackLoopResult | None
    receipt: SemanticCompilationReceipt | None = None
    duplicate: bool = False
    schema_version: int = STATIC_SEMANTIC_WRITEBACK_SCHEMA_VERSION

    def observer_evidence(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "snapshot_id": self.snapshot_id,
            "compilation_id": self.compilation_id,
            "writeback": (
                self.writeback.observer_evidence()
                if self.writeback is not None
                else None
            ),
            "receipt": self.receipt.payload() if self.receipt is not None else None,
            "duplicate": self.duplicate,
        }

    @property
    def plan_id(self) -> str:
        """Compatibility alias for legacy deterministic fixtures."""

        return self.compilation_id


class StaticSemanticWritebackRuntime:
    """Execute fixed Mem0-flat semantic writeback inside one isolated PAST home."""

    def __init__(
        self,
        hermes_home: Path,
        completion_client: CompletionClient,
        *,
        operation_evidence_path: Path,
        mutation_receipt_path: Path,
        compilation_receipt_path: Path | None = None,
        observer: MemoryObserver | None = None,
        ingestion_observer: Any | None = None,
        utility_gate: FrozenMem0UtilityGate | None = None,
        adaptive_policy_store: JsonAdaptivePolicyStore | None = None,
        adaptive_parameters: tuple[TrustedAdaptiveMem0Parameter, ...] = (),
        require_adaptive_policy: bool = False,
    ) -> None:
        self.hermes_home = hermes_home.expanduser().resolve()
        self.registry = build_hermes_native_registry(self.hermes_home)
        self.receipts = JsonMutationReceiptStore(mutation_receipt_path)
        self.compilation_receipts = JsonSemanticCompilationReceiptStore(
            compilation_receipt_path
            or mutation_receipt_path.with_name("semantic_compilation_receipts.json")
        )
        self.operation_log = AppendOnlyOperationEvidenceLog(operation_evidence_path)
        self.operation_recorder = AtomicOperationRecorder(self.operation_log)
        base_gate = utility_gate
        if adaptive_policy_store is not None and base_gate is None:
            base_gate = FrozenMem0UtilityGate()
        base_policy = Mem0FlatSemanticPolicy(
            completion_client,
            operation_recorder=self.operation_recorder,
            utility_gate=base_gate,
        )
        self.adaptive_binding: AdaptiveMem0Binding | None = None
        if adaptive_policy_store is None:
            self.utility_gate = base_gate
            self.policy = base_policy
        else:
            self.adaptive_binding = ActiveAdaptiveMem0Binder(
                adaptive_parameters
            ).bind(
                adaptive_policy_store,
                base_gate or FrozenMem0UtilityGate(),
                expected_parent_policy_version=(
                    base_policy.descriptor.policy_version
                ),
            )
            if require_adaptive_policy and not self.adaptive_binding.adaptive:
                raise ValueError("adaptive semantic writeback requires an active policy")
            self.utility_gate = self.adaptive_binding.gate
            self.policy = (
                Mem0FlatSemanticPolicy(
                    completion_client,
                    operation_recorder=self.operation_recorder,
                    utility_gate=self.utility_gate,
                    descriptor_policy_version=(
                        self.adaptive_binding.actual_policy_version
                    ),
                )
                if self.adaptive_binding.adaptive
                else base_policy
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
        self._results_by_compilation: dict[str, StaticSemanticBoundaryResult] = {}
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
            result = self._execute(snapshot, request)
            added.append(result)
        return tuple(added)

    def process_completed_snapshot(
        self,
        snapshot: ContextSnapshot,
    ) -> tuple[StaticSemanticBoundaryResult, ...]:
        """Compile one trusted completed task without a lifecycle evaluator."""

        if self._closed:
            raise RuntimeError("static semantic writeback runtime is closed")
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
        request = build_completed_task_semantic_ingest_request(
            snapshot,
            experience,
            policy_version=self.policy.descriptor.policy_version,
            framework_version=self.policy.descriptor.framework_version,
        )
        existing = self._results_by_compilation.get(
            request.provenance.compilation_id
        )
        if existing is not None:
            return (existing,)
        request_digest = hashlib.sha256(
            json.dumps(
                request.canonical_payload(),
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        pending = SemanticCompilationReceipt(
            compilation_id=request.provenance.compilation_id,
            request_digest=request_digest,
            snapshot_id=snapshot.snapshot_id,
            context_revision=snapshot.context_revision,
            status=SemanticCompilationReceiptStatus.PENDING,
        )
        receipt, reserved = self.compilation_receipts.reserve(pending)
        if not reserved:
            duplicate = StaticSemanticBoundaryResult(
                snapshot.snapshot_id,
                receipt.compilation_id,
                None,
                receipt=receipt,
                duplicate=True,
            )
            self._results_by_compilation[receipt.compilation_id] = duplicate
            self._results.append(duplicate)
            return (duplicate,)
        result = self._execute(snapshot, request)
        if result.writeback is None or result.writeback.ingestion is None:
            raise ValueError("semantic compilation produced no ingestion result")
        completed = self.compilation_receipts.complete(SemanticCompilationReceipt(
            compilation_id=receipt.compilation_id,
            request_digest=receipt.request_digest,
            snapshot_id=receipt.snapshot_id,
            context_revision=receipt.context_revision,
            status=SemanticCompilationReceiptStatus.COMPLETED,
            ingestion_execution_id=result.writeback.ingestion.execution_id,
            mutation_ids=tuple(
                execution.mutation_id for execution in result.writeback.executions
            ),
        ))
        result = StaticSemanticBoundaryResult(
            result.snapshot_id,
            result.compilation_id,
            result.writeback,
            receipt=completed,
        )
        self._results_by_compilation[result.compilation_id] = result
        self._results[-1] = result
        return (result,)

    def _execute(
        self,
        snapshot: ContextSnapshot,
        request: SemanticIngestRequest,
    ) -> StaticSemanticBoundaryResult:
        writeback = self.loop.run(
            request,
            current_source_revision=snapshot.context_revision,
        )
        if self.ingestion_observer is not None and writeback.ingestion is not None:
            self.ingestion_observer.record_ingestion(request, writeback.ingestion)
            if self.utility_gate is not None:
                record_utility = getattr(
                    self.ingestion_observer,
                    "record_utility_decisions",
                    None,
                )
                if callable(record_utility):
                    record_utility(
                        request,
                        writeback.ingestion,
                        self.utility_gate.observer_evidence(
                            request.idempotency_key
                        ),
                    )
        result = StaticSemanticBoundaryResult(
            snapshot.snapshot_id,
            request.provenance.compilation_id,
            writeback,
        )
        existing = self._results_by_compilation.get(result.compilation_id)
        if existing is not None:
            if existing != result:
                raise ValueError("semantic compilation identity conflict")
            return existing
        self._results_by_compilation[result.compilation_id] = result
        self._results.append(result)
        return result

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.registry.close()
