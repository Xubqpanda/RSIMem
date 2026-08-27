"""Content-free deterministic-first attribution over atomic memory operations."""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Mapping, Protocol, Sequence, runtime_checkable

from ..lifecycle import RawResourceUsage
from .operation_graph import (
    ArtifactKind,
    OperationGraph,
    OperationKind,
    OperationRecord,
    OperationStatus,
)


ATTRIBUTION_SCHEMA_VERSION = 1
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$")
_REASON_CODE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _stable_id(prefix: str, value: object) -> str:
    digest = hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()
    return f"{prefix}.{digest[:40]}"


class FailureCategory(StrEnum):
    EXTRACTION_MISS = "extraction_miss"
    WRONG_UPDATE_TARGET = "wrong_update_target"
    DUPLICATE_ADD = "duplicate_add"
    RETRIEVAL_MISS = "retrieval_miss"
    RETRIEVED_BUT_UNUSED = "retrieved_but_unused"
    UNRESOLVED_TASK_FAILURE = "unresolved_task_failure"


class AttributionMethod(StrEnum):
    DETERMINISTIC_CONTRACT = "deterministic_contract"
    DETERMINISTIC_RECEIPT = "deterministic_receipt"
    DETERMINISTIC_EXPOSURE = "deterministic_exposure"
    MODEL = "model"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True, slots=True)
class AttributionBudget:
    max_calls: int = 0
    max_input_tokens: int = 0
    max_output_tokens: int = 0
    max_wall_time_ms: int = 0
    max_candidate_operations: int = 64

    def __post_init__(self) -> None:
        if any(value < 0 for value in (
            self.max_calls,
            self.max_input_tokens,
            self.max_output_tokens,
            self.max_wall_time_ms,
            self.max_candidate_operations,
        )):
            raise ValueError("attribution budgets must not be negative")


@dataclass(frozen=True, slots=True)
class AttributionWindow:
    window_id: str
    version: str
    cutoff_operation_id: str
    visible_operation_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not all(_IDENTIFIER.fullmatch(value) for value in (
            self.window_id,
            self.version,
            self.cutoff_operation_id,
        )):
            raise ValueError("attribution window identity is invalid")
        if (
            not self.visible_operation_ids
            or self.cutoff_operation_id not in self.visible_operation_ids
            or len(self.visible_operation_ids) != len(set(self.visible_operation_ids))
            or any(
                not _IDENTIFIER.fullmatch(value)
                for value in self.visible_operation_ids
            )
        ):
            raise ValueError("attribution window operation IDs are invalid")


@dataclass(frozen=True, slots=True)
class ModelAttributionRequest:
    window_id: str
    operation_records: tuple[Mapping[str, object], ...]
    allowed_operation_ids: tuple[str, ...]
    allowed_artifact_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ModelAttributionResponse:
    category: FailureCategory
    operation_ids: tuple[str, ...]
    artifact_ids: tuple[str, ...]
    confidence: float
    usage: RawResourceUsage

    def __post_init__(self) -> None:
        object.__setattr__(self, "category", FailureCategory(self.category))
        if self.category == FailureCategory.UNRESOLVED_TASK_FAILURE:
            raise ValueError("attribution model cannot return unresolved as a finding")
        if not self.operation_ids:
            raise ValueError("attribution model response requires operation IDs")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("attribution confidence must be between zero and one")
        if self.usage.input_tokens is None or self.usage.output_tokens is None:
            raise ValueError("attribution model usage requires token accounting")
        if self.usage.model_requests != 1:
            raise ValueError("attribution model response must account for one request")


@runtime_checkable
class AttributionModelClient(Protocol):
    def attribute(self, request: ModelAttributionRequest) -> ModelAttributionResponse: ...


class AttributionModelError(RuntimeError):
    def __init__(self, reason_code: str, usage: RawResourceUsage) -> None:
        if not _REASON_CODE.fullmatch(reason_code):
            raise ValueError("attribution model error reason must be machine-readable")
        if usage.model_requests != 1:
            raise ValueError("attribution model error must account for one request")
        self.reason_code = reason_code
        self.usage = usage
        super().__init__(reason_code)


@dataclass(frozen=True, slots=True)
class AttributionRecord:
    attribution_id: str
    category: FailureCategory
    method: AttributionMethod
    candidate_operation_ids: tuple[str, ...]
    artifact_ids: tuple[str, ...]
    policy_parameter_ids: tuple[str, ...]
    evidence_window_id: str
    evidence_window_version: str
    confidence: float
    reason_code: str
    model_usage: RawResourceUsage = RawResourceUsage()
    schema_version: int = ATTRIBUTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTRIBUTION_SCHEMA_VERSION:
            raise ValueError("unsupported attribution record schema version")
        object.__setattr__(self, "category", FailureCategory(self.category))
        object.__setattr__(self, "method", AttributionMethod(self.method))
        if not self.attribution_id or not self.evidence_window_id:
            raise ValueError("attribution identity is incomplete")
        if not self.candidate_operation_ids:
            raise ValueError("attribution requires candidate operation IDs")
        for values in (
            self.candidate_operation_ids,
            self.artifact_ids,
            self.policy_parameter_ids,
        ):
            if len(values) != len(set(values)):
                raise ValueError("attribution references must be unique")
            if any(not _IDENTIFIER.fullmatch(value) for value in values):
                raise ValueError("attribution references must be stable identifiers")
        if not all(_IDENTIFIER.fullmatch(value) for value in (
            self.attribution_id,
            self.evidence_window_id,
            self.evidence_window_version,
        )):
            raise ValueError("attribution identity must use stable identifiers")
        if not _REASON_CODE.fullmatch(self.reason_code):
            raise ValueError("attribution reason_code must be machine-readable")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("attribution confidence must be between zero and one")
        if (
            self.method == AttributionMethod.MODEL
            and self.model_usage.model_requests != 1
        ):
            raise ValueError("model attribution must carry one accounted request")
        if (
            self.method != AttributionMethod.MODEL
            and self.model_usage.model_requests not in {0, 1}
        ):
            raise ValueError("unresolved attribution usage is inconsistent")

    def observer_evidence(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "attribution_id": self.attribution_id,
            "category": self.category.value,
            "method": self.method.value,
            "candidate_operation_ids": list(self.candidate_operation_ids),
            "artifact_ids": list(self.artifact_ids),
            "policy_parameter_ids": list(self.policy_parameter_ids),
            "evidence_window_id": self.evidence_window_id,
            "evidence_window_version": self.evidence_window_version,
            "confidence": self.confidence,
            "reason_code": self.reason_code,
            "model_usage": self.model_usage.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class AttributionReport:
    window: AttributionWindow
    records: tuple[AttributionRecord, ...]
    model_call_count: int
    model_usage: RawResourceUsage
    disabled: bool
    schema_version: int = ATTRIBUTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTRIBUTION_SCHEMA_VERSION:
            raise ValueError("unsupported attribution report schema version")
        if self.model_call_count < 0:
            raise ValueError("attribution model call count must not be negative")
        if self.model_usage.model_requests != self.model_call_count:
            raise ValueError("attribution report model usage must match call count")
        if self.disabled and (self.records or self.model_call_count):
            raise ValueError("disabled attribution cannot produce findings or model calls")

    def observer_evidence(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "window_id": self.window.window_id,
            "window_version": self.window.version,
            "cutoff_operation_id": self.window.cutoff_operation_id,
            "visible_operation_ids": list(self.window.visible_operation_ids),
            "records": [record.observer_evidence() for record in self.records],
            "model_call_count": self.model_call_count,
            "model_usage": self.model_usage.to_dict(),
            "disabled": self.disabled,
        }


_REASON_MAP = {
    "extraction_miss": (
        FailureCategory.EXTRACTION_MISS,
        AttributionMethod.DETERMINISTIC_CONTRACT,
    ),
    "wrong_update_target": (
        FailureCategory.WRONG_UPDATE_TARGET,
        AttributionMethod.DETERMINISTIC_CONTRACT,
    ),
    "hallucinated_candidate_target": (
        FailureCategory.WRONG_UPDATE_TARGET,
        AttributionMethod.DETERMINISTIC_CONTRACT,
    ),
    "unknown_owner_target": (
        FailureCategory.WRONG_UPDATE_TARGET,
        AttributionMethod.DETERMINISTIC_CONTRACT,
    ),
    "duplicate_add": (
        FailureCategory.DUPLICATE_ADD,
        AttributionMethod.DETERMINISTIC_RECEIPT,
    ),
    "add_ownership_ambiguous": (
        FailureCategory.DUPLICATE_ADD,
        AttributionMethod.DETERMINISTIC_RECEIPT,
    ),
    "retrieval_miss": (
        FailureCategory.RETRIEVAL_MISS,
        AttributionMethod.DETERMINISTIC_EXPOSURE,
    ),
    "retrieved_but_unused": (
        FailureCategory.RETRIEVED_BUT_UNUSED,
        AttributionMethod.DETERMINISTIC_EXPOSURE,
    ),
}
_CATEGORY_KINDS = {
    FailureCategory.EXTRACTION_MISS: {OperationKind.FACT_EXTRACTION},
    FailureCategory.WRONG_UPDATE_TARGET: {
        OperationKind.INTERNAL_OPERATION_DECISION,
        OperationKind.TARGET_RESOLUTION,
    },
    FailureCategory.DUPLICATE_ADD: {
        OperationKind.VALIDATION,
        OperationKind.MUTATION,
    },
    FailureCategory.RETRIEVAL_MISS: {OperationKind.RETRIEVAL},
    FailureCategory.RETRIEVED_BUT_UNUSED: {OperationKind.USE},
}


class DeterministicFirstAttributor:
    """Attribute observable failures without reading raw or future evidence."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        window_version: str = "operation-window-v1",
        model_client: AttributionModelClient | None = None,
        model_enabled: bool = False,
        budget: AttributionBudget = AttributionBudget(),
    ) -> None:
        self.enabled = enabled
        self.window_version = window_version
        self.model_client = model_client
        self.model_enabled = model_enabled
        self.budget = budget
        self._model_calls = 0
        self._model_input_tokens = 0
        self._model_output_tokens = 0
        self._model_wall_time_ms = 0
        self._model_usages: list[RawResourceUsage] = []
        self._unknown_model_usage = False
        self._seen_model_windows: set[str] = set()

    @property
    def policy_update_usage(self) -> RawResourceUsage:
        if not self._model_usages:
            return RawResourceUsage()

        def total(name: str) -> int | None:
            values = [getattr(usage, name) for usage in self._model_usages]
            return None if any(value is None for value in values) else sum(values)

        return RawResourceUsage(
            input_tokens=total("input_tokens"),
            output_tokens=total("output_tokens"),
            cache_read_tokens=total("cache_read_tokens"),
            cache_write_tokens=total("cache_write_tokens"),
            reasoning_tokens=total("reasoning_tokens"),
            model_requests=sum(item.model_requests for item in self._model_usages),
            retry_count=sum(item.retry_count for item in self._model_usages),
            duration_ms=self._model_wall_time_ms,
            storage_bytes=sum(item.storage_bytes for item in self._model_usages),
        )

    def attribute_batch(
        self,
        items: Sequence[tuple[OperationGraph, str | None]],
        *,
        sample_rate: float = 1.0,
        sample_key: str = "attribution-batch-v1",
    ) -> tuple[AttributionReport, ...]:
        if not 0.0 <= sample_rate <= 1.0:
            raise ValueError("attribution batch sample_rate must be between zero and one")
        selected = []
        seen_windows = set()
        for graph, cutoff in items:
            window, _ = self._window(graph.operations, cutoff)
            if window.window_id in seen_windows:
                continue
            seen_windows.add(window.window_id)
            sample_digest = hashlib.sha256(
                f"{sample_key}:{window.window_id}".encode("utf-8")
            ).hexdigest()
            sample_value = int(sample_digest[:16], 16) / 0xFFFFFFFFFFFFFFFF
            if sample_value >= sample_rate:
                continue
            selected.append(self.attribute(graph, cutoff_operation_id=cutoff))
        return tuple(selected)

    def attribute(
        self,
        graph: OperationGraph,
        *,
        cutoff_operation_id: str | None = None,
    ) -> AttributionReport:
        window, visible = self._window(graph.operations, cutoff_operation_id)
        if not self.enabled:
            return AttributionReport(window, (), 0, RawResourceUsage(), True)

        artifacts = {item.artifact_id: item for item in graph.artifacts}
        by_id = {item.operation_id: item for item in visible}
        records = []
        for operation in visible:
            mapped = _REASON_MAP.get(operation.reason_code or "")
            if mapped is None:
                continue
            category, method = mapped
            if operation.kind not in _CATEGORY_KINDS[category]:
                continue
            candidates = self._candidate_operations(operation, category, by_id)
            records.append(self._record(
                window,
                category,
                method,
                candidates,
                by_id,
                artifacts,
                reason_code=operation.reason_code or category.value,
                confidence=1.0,
            ))

        if records:
            return AttributionReport(
                window,
                tuple(records),
                0,
                RawResourceUsage(),
                False,
            )

        outcome_failures = tuple(
            item
            for item in visible
            if item.kind == OperationKind.DOWNSTREAM_OUTCOME
            and item.status in {OperationStatus.FAILED, OperationStatus.REJECTED}
        )
        if not outcome_failures:
            return AttributionReport(window, (), 0, RawResourceUsage(), False)

        outcome = outcome_failures[-1]
        if self._can_call_model(window, visible):
            request = self._model_request(window, visible, artifacts)
            started = time.perf_counter_ns()
            try:
                response = self.model_client.attribute(request)  # type: ignore[union-attr]
            except AttributionModelError as exc:
                elapsed = max(0, (time.perf_counter_ns() - started) // 1_000_000)
                self._account_model(window.window_id, exc.usage, elapsed)
                return self._unresolved_report(
                    window,
                    outcome,
                    by_id,
                    artifacts,
                    reason_code=exc.reason_code,
                    model_call_count=1,
                    model_usage=exc.usage,
                )
            except Exception:
                elapsed = max(0, (time.perf_counter_ns() - started) // 1_000_000)
                usage = RawResourceUsage(
                    model_requests=1,
                    duration_ms=elapsed,
                )
                self._account_model(window.window_id, usage, elapsed)
                return self._unresolved_report(
                    window,
                    outcome,
                    by_id,
                    artifacts,
                    reason_code="model_attribution_exception",
                    model_call_count=1,
                    model_usage=usage,
                )
            elapsed = max(0, (time.perf_counter_ns() - started) // 1_000_000)
            self._account_model(window.window_id, response.usage, elapsed)
            try:
                self._validate_model_response(response, request)
            except ValueError:
                return self._unresolved_report(
                    window,
                    outcome,
                    by_id,
                    artifacts,
                    reason_code="invalid_model_attribution",
                    model_call_count=1,
                    model_usage=response.usage,
                )
            if self._model_budget_exceeded():
                return self._unresolved_report(
                    window,
                    outcome,
                    by_id,
                    artifacts,
                    reason_code="model_budget_exhausted",
                    model_call_count=1,
                    model_usage=response.usage,
                )
            record = self._record(
                window,
                response.category,
                AttributionMethod.MODEL,
                response.operation_ids,
                by_id,
                artifacts,
                reason_code="model_attribution",
                confidence=response.confidence,
                artifact_ids=response.artifact_ids,
                model_usage=response.usage,
            )
            return AttributionReport(window, (record,), 1, response.usage, False)

        reason = "model_attribution_disabled"
        if self.model_enabled and self.model_client is not None:
            reason = (
                "model_budget_exhausted"
                if self._model_eligible(visible)
                else "model_sample_ineligible"
            )
        return self._unresolved_report(
            window,
            outcome,
            by_id,
            artifacts,
            reason_code=reason,
            model_call_count=0,
            model_usage=RawResourceUsage(),
        )

    def _unresolved_report(
        self,
        window: AttributionWindow,
        outcome: OperationRecord,
        operations: Mapping[str, OperationRecord],
        artifacts: Mapping[str, object],
        *,
        reason_code: str,
        model_call_count: int,
        model_usage: RawResourceUsage,
    ) -> AttributionReport:
        unresolved = self._record(
            window,
            FailureCategory.UNRESOLVED_TASK_FAILURE,
            AttributionMethod.UNRESOLVED,
            (outcome.operation_id,),
            operations,
            artifacts,
            reason_code=reason_code,
            confidence=0.0,
            model_usage=model_usage,
        )
        return AttributionReport(
            window,
            (unresolved,),
            model_call_count,
            model_usage,
            False,
        )

    def _window(
        self,
        operations: Sequence[OperationRecord],
        cutoff_operation_id: str | None,
    ) -> tuple[AttributionWindow, tuple[OperationRecord, ...]]:
        if not operations:
            raise ValueError("attribution requires operation evidence")
        cutoff = cutoff_operation_id or operations[-1].operation_id
        indexes = [
            index for index, item in enumerate(operations) if item.operation_id == cutoff
        ]
        if len(indexes) != 1:
            raise ValueError("attribution cutoff must identify one observable operation")
        visible = tuple(operations[: indexes[0] + 1])
        visible_ids = tuple(item.operation_id for item in visible)
        identity = {
            "schema_version": ATTRIBUTION_SCHEMA_VERSION,
            "version": self.window_version,
            "cutoff_operation_id": cutoff,
            "visible_operation_ids": visible_ids,
        }
        return AttributionWindow(
            _stable_id("attribution-window", identity),
            self.window_version,
            cutoff,
            visible_ids,
        ), visible

    @staticmethod
    def _ancestors(
        operation_ids: Sequence[str],
        by_id: Mapping[str, OperationRecord],
    ) -> tuple[str, ...]:
        selected = set(operation_ids)
        pending = list(operation_ids)
        while pending:
            operation = by_id.get(pending.pop())
            if operation is None:
                continue
            for parent in operation.parent_operation_ids:
                if parent not in selected:
                    selected.add(parent)
                    pending.append(parent)
        return tuple(item for item in by_id if item in selected)

    def _candidate_operations(
        self,
        operation: OperationRecord,
        category: FailureCategory,
        by_id: Mapping[str, OperationRecord],
    ) -> tuple[str, ...]:
        if category == FailureCategory.RETRIEVED_BUT_UNUSED:
            ancestors = self._ancestors((operation.operation_id,), by_id)
            retrievals = tuple(
                item
                for item in ancestors
                if by_id[item].kind == OperationKind.RETRIEVAL
                and by_id[item].status == OperationStatus.SUCCESS
            )
            return (*retrievals, operation.operation_id)
        if category == FailureCategory.WRONG_UPDATE_TARGET:
            parents = tuple(
                parent
                for parent in operation.parent_operation_ids
                if by_id.get(parent) is not None
                and by_id[parent].kind == OperationKind.INTERNAL_OPERATION_DECISION
            )
            return (*parents, operation.operation_id)
        if category == FailureCategory.DUPLICATE_ADD:
            ancestors = self._ancestors((operation.operation_id,), by_id)
            decisions = tuple(
                item
                for item in ancestors
                if by_id[item].kind == OperationKind.INTERNAL_OPERATION_DECISION
            )
            return (*decisions, operation.operation_id)
        return (operation.operation_id,)

    def _record(
        self,
        window: AttributionWindow,
        category: FailureCategory,
        method: AttributionMethod,
        operation_ids: tuple[str, ...],
        operations: Mapping[str, OperationRecord],
        artifacts: Mapping[str, object],
        *,
        reason_code: str,
        confidence: float,
        artifact_ids: tuple[str, ...] | None = None,
        model_usage: RawResourceUsage = RawResourceUsage(),
    ) -> AttributionRecord:
        graph_artifacts = {
            key: value for key, value in artifacts.items() if hasattr(value, "kind")
        }
        referenced_ids = tuple(dict.fromkeys(
            artifact_id
            for operation_id in operation_ids
            for artifact_id in (
                *operations[operation_id].input_artifact_ids,
                *operations[operation_id].output_artifact_ids,
            )
        ))
        if artifact_ids is None:
            artifact_ids = tuple(
                artifact_id
                for artifact_id in referenced_ids
                if artifact_id in graph_artifacts
                and graph_artifacts[artifact_id].kind != ArtifactKind.POLICY_PARAMETER
            )
        policy_ids = tuple(
            artifact_id
            for artifact_id in referenced_ids
            if artifact_id in graph_artifacts
            and graph_artifacts[artifact_id].kind == ArtifactKind.POLICY_PARAMETER
        )
        identity = {
            "category": category.value,
            "method": method.value,
            "operation_ids": operation_ids,
            "artifact_ids": artifact_ids,
            "policy_parameter_ids": policy_ids,
            "window_id": window.window_id,
            "reason_code": reason_code,
        }
        return AttributionRecord(
            _stable_id("attribution", identity),
            category,
            method,
            operation_ids,
            artifact_ids,
            policy_ids,
            window.window_id,
            window.version,
            confidence,
            reason_code,
            model_usage,
        )

    def _can_call_model(
        self,
        window: AttributionWindow,
        visible: Sequence[OperationRecord],
    ) -> bool:
        if not self.model_enabled or self.model_client is None:
            return False
        if self._unknown_model_usage:
            return False
        if not self._model_eligible(visible):
            return False
        if window.window_id in self._seen_model_windows:
            return False
        if len(visible) > self.budget.max_candidate_operations:
            return False
        return (
            self._model_calls < self.budget.max_calls
            and self._model_input_tokens < self.budget.max_input_tokens
            and self._model_output_tokens < self.budget.max_output_tokens
            and self._model_wall_time_ms < self.budget.max_wall_time_ms
        )

    @staticmethod
    def _model_eligible(visible: Sequence[OperationRecord]) -> bool:
        return any(
            item.kind == OperationKind.INJECTION
            and item.status == OperationStatus.SUCCESS
            for item in visible
        ) and not any(
            item.kind == OperationKind.DOWNSTREAM_OUTCOME
            and item.status == OperationStatus.NONE
            and item.reason_code == "observation_censored"
            for item in visible
        )

    def _model_budget_exceeded(self) -> bool:
        return (
            self._model_calls > self.budget.max_calls
            or self._model_input_tokens > self.budget.max_input_tokens
            or self._model_output_tokens > self.budget.max_output_tokens
            or self._model_wall_time_ms > self.budget.max_wall_time_ms
        )

    @staticmethod
    def _model_request(
        window: AttributionWindow,
        visible: Sequence[OperationRecord],
        artifacts: Mapping[str, object],
    ) -> ModelAttributionRequest:
        records = tuple(MappingProxyType({
            "operation_id": item.operation_id,
            "kind": item.kind.value,
            "status": item.status.value,
            "reason_code": item.reason_code,
            "parent_operation_ids": item.parent_operation_ids,
            "input_artifact_ids": item.input_artifact_ids,
            "output_artifact_ids": item.output_artifact_ids,
            "policy_version": item.context.policy_version,
        }) for item in visible)
        return ModelAttributionRequest(
            window.window_id,
            records,
            tuple(item.operation_id for item in visible),
            tuple(artifacts),
        )

    @staticmethod
    def _validate_model_response(
        response: ModelAttributionResponse,
        request: ModelAttributionRequest,
    ) -> None:
        if not set(response.operation_ids).issubset(request.allowed_operation_ids):
            raise ValueError("attribution model returned an unknown operation")
        if not set(response.artifact_ids).issubset(request.allowed_artifact_ids):
            raise ValueError("attribution model returned an unknown artifact")

    def _account_model(
        self,
        window_id: str,
        usage: RawResourceUsage,
        elapsed_ms: int,
    ) -> None:
        self._model_calls += 1
        self._model_input_tokens += usage.input_tokens or 0
        self._model_output_tokens += usage.output_tokens or 0
        self._model_wall_time_ms += elapsed_ms
        self._model_usages.append(usage)
        if usage.input_tokens is None or usage.output_tokens is None:
            self._unknown_model_usage = True
        self._seen_model_windows.add(window_id)
