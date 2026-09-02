"""Host-neutral Stage 2 adapter contracts.

The interfaces in this module deliberately carry identities and digests rather
than host payloads.  Benchmark, host, method, and feedback implementations can
therefore be replaced independently without giving a method access to grader
or answer material.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from .memory.contracts import MemoryKind
from .memory.evidence_planes import EvidencePlane
from .memory.lifecycle_surfaces import MemoryLifecycleSurface


ADAPTER_CONTRACT_SCHEMA_VERSION = 1
ADAPTER_CONTRACT_SCHEMA = "rsimem-adapter-contract-v1"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:+/-]{0,255}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def content_digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _id(value: object, name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{name} must be a stable identifier")
    return value


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{name} must be a sha256 digest")
    return value


def _ids(values: Sequence[str], name: str, *, allow_empty: bool = True) -> tuple[str, ...]:
    result = tuple(values)
    if not allow_empty and not result:
        raise ValueError(f"{name} must not be empty")
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must not contain duplicates")
    for value in result:
        _id(value, name)
    return result


def _frozen(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType(dict(value or {}))


class AdapterStatus(StrEnum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    STALE = "stale"


class BenchmarkSplit(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    FINAL = "final"


class HostEventKind(StrEnum):
    SESSION_STARTED = "session_started"
    TURN_COMPLETED = "turn_completed"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    MEMORY_RETRIEVAL = "memory_retrieval"
    MEMORY_EXPOSURE = "memory_exposure"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"


class FeedbackCondition(StrEnum):
    F0 = "F0"
    F1 = "F1"
    F2 = "F2"
    F3 = "F3"
    F4 = "F4"
    F5 = "F5"


@dataclass(frozen=True, slots=True)
class AdapterResult:
    status: AdapterStatus
    operation_id: str
    reason_code: str | None = None
    output_digest: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", AdapterStatus(self.status))
        _id(self.operation_id, "adapter operation ID")
        if self.reason_code is not None:
            _id(self.reason_code, "adapter reason code")
        if self.output_digest is not None:
            _sha(self.output_digest, "adapter output digest")


@dataclass(frozen=True, slots=True)
class BenchmarkTaskRequest:
    case_id: str
    split: BenchmarkSplit
    task_template_id: str
    seed: str
    tool_budget: int
    max_turns: int

    def __post_init__(self) -> None:
        _id(self.case_id, "benchmark case ID")
        object.__setattr__(self, "split", BenchmarkSplit(self.split))
        _id(self.task_template_id, "task template ID")
        _id(self.seed, "benchmark seed")
        if type(self.tool_budget) is not int or self.tool_budget < 1:
            raise ValueError("benchmark tool budget must be positive")
        if type(self.max_turns) is not int or self.max_turns < 1:
            raise ValueError("benchmark max turns must be positive")


@dataclass(frozen=True, slots=True)
class BenchmarkPublicEvent:
    """Public benchmark transition; no grader/reference fields are allowed."""

    event_id: str
    case_id: str
    stage: str
    event_type: str
    public_state_digest: str
    output_digest: str | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _id(self.event_id, "benchmark event ID")
        _id(self.case_id, "benchmark case ID")
        _id(self.stage, "benchmark stage")
        _id(self.event_type, "benchmark event type")
        _sha(self.public_state_digest, "benchmark public state digest")
        if self.output_digest is not None:
            _sha(self.output_digest, "benchmark output digest")
        validate_public_payload(self.attributes)
        object.__setattr__(self, "attributes", _frozen(self.attributes))


@dataclass(frozen=True, slots=True)
class FinalEvaluationRecord:
    """Final-only score record, kept outside method and pure-process views."""

    evaluation_id: str
    case_id: str
    metric_id: str
    score_digest: str
    evidence_plane: EvidencePlane = EvidencePlane.FINAL_EVALUATION

    def __post_init__(self) -> None:
        _id(self.evaluation_id, "final evaluation ID")
        _id(self.case_id, "final evaluation case ID")
        _id(self.metric_id, "final metric ID")
        _sha(self.score_digest, "final score digest")
        object.__setattr__(self, "evidence_plane", EvidencePlane(self.evidence_plane))
        if self.evidence_plane is not EvidencePlane.FINAL_EVALUATION:
            raise ValueError("final evaluation must use final_evaluation plane")


@dataclass(frozen=True, slots=True)
class HostCapabilities:
    memory_kinds: tuple[MemoryKind, ...]
    tool_call_result_closure: bool
    usage_accounting: bool
    restart: bool
    context_snapshot: bool
    native_bypass: bool

    def __post_init__(self) -> None:
        kinds = tuple(MemoryKind(value) for value in self.memory_kinds)
        if set(kinds) != set(MemoryKind):
            raise ValueError("host must declare all three memory kinds")
        object.__setattr__(self, "memory_kinds", kinds)
        for name in ("tool_call_result_closure", "usage_accounting", "restart", "context_snapshot", "native_bypass"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"host capability {name} must be bool")

    def payload(self) -> dict[str, object]:
        return {
            "memory_kinds": [value.value for value in self.memory_kinds],
            "tool_call_result_closure": self.tool_call_result_closure,
            "usage_accounting": self.usage_accounting,
            "restart": self.restart,
            "context_snapshot": self.context_snapshot,
            "native_bypass": self.native_bypass,
        }


@dataclass(frozen=True, slots=True)
class CanonicalHostEvent:
    event_id: str
    session_id: str
    task_id: str
    kind: HostEventKind
    revision: str
    input_ids: tuple[str, ...] = ()
    output_ids: tuple[str, ...] = ()
    memory_kind: MemoryKind | None = None
    surface: MemoryLifecycleSurface | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _id(self.event_id, "host event ID")
        _id(self.session_id, "host session ID")
        _id(self.task_id, "host task ID")
        object.__setattr__(self, "kind", HostEventKind(self.kind))
        _id(self.revision, "host event revision")
        object.__setattr__(self, "input_ids", _ids(self.input_ids, "host input IDs"))
        object.__setattr__(self, "output_ids", _ids(self.output_ids, "host output IDs"))
        if self.memory_kind is not None:
            object.__setattr__(self, "memory_kind", MemoryKind(self.memory_kind))
        if self.surface is not None:
            object.__setattr__(self, "surface", MemoryLifecycleSurface(self.surface))
        validate_public_payload(self.attributes)
        object.__setattr__(self, "attributes", _frozen(self.attributes))

    def payload(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "session_id": self.session_id,
            "task_id": self.task_id,
            "kind": self.kind.value,
            "revision": self.revision,
            "input_ids": list(self.input_ids),
            "output_ids": list(self.output_ids),
            "memory_kind": self.memory_kind.value if self.memory_kind else None,
            "surface": self.surface.value if self.surface else None,
            "attributes": dict(self.attributes),
        }

    @classmethod
    def from_payload(cls, value: object) -> "CanonicalHostEvent":
        fields = {
            "event_id", "session_id", "task_id", "kind", "revision",
            "input_ids", "output_ids", "memory_kind", "surface", "attributes",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError("malformed canonical host event")
        if not isinstance(value["input_ids"], list) or not isinstance(value["output_ids"], list):
            raise ValueError("canonical host event IDs must be lists")
        if not isinstance(value["attributes"], Mapping):
            raise ValueError("canonical host event attributes must be an object")
        try:
            event = cls(
                event_id=value["event_id"],
                session_id=value["session_id"],
                task_id=value["task_id"],
                kind=value["kind"],
                revision=value["revision"],
                input_ids=tuple(value["input_ids"]),
                output_ids=tuple(value["output_ids"]),
                memory_kind=value["memory_kind"],
                surface=value["surface"],
                attributes=dict(value["attributes"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("malformed canonical host event") from exc
        if event.payload() != dict(value):
            raise ValueError("non-canonical host event")
        return event


@dataclass(frozen=True, slots=True)
class MethodCapabilities:
    method_id: str
    primary_kind: MemoryKind
    secondary_kind: MemoryKind | None
    transform: str | None
    owned_surfaces: tuple[MemoryLifecycleSurface, ...]
    required_feedback: tuple[FeedbackCondition, ...]
    required_host_capabilities: tuple[str, ...]
    state_schema: str
    lineage_schema: str
    online_update: bool
    validation: bool
    rollback: bool

    def __post_init__(self) -> None:
        _id(self.method_id, "method ID")
        object.__setattr__(self, "primary_kind", MemoryKind(self.primary_kind))
        if self.secondary_kind is not None:
            object.__setattr__(self, "secondary_kind", MemoryKind(self.secondary_kind))
            if self.secondary_kind == self.primary_kind:
                raise ValueError("method secondary kind must differ from primary")
        if self.transform is not None:
            _id(self.transform, "method transform")
        if (self.secondary_kind is None) != (self.transform is None):
            raise ValueError("method secondary kind and transform must be paired")
        surfaces = tuple(MemoryLifecycleSurface(value) for value in self.owned_surfaces)
        if len(surfaces) != len(set(surfaces)):
            raise ValueError("owned surfaces must be unique")
        object.__setattr__(self, "owned_surfaces", surfaces)
        feedback = tuple(FeedbackCondition(value) for value in self.required_feedback)
        if len(feedback) != len(set(feedback)):
            raise ValueError("required feedback conditions must be unique")
        object.__setattr__(self, "required_feedback", feedback)
        object.__setattr__(self, "required_host_capabilities", _ids(self.required_host_capabilities, "required host capabilities"))
        _id(self.state_schema, "method state schema")
        _id(self.lineage_schema, "method lineage schema")
        for name in ("online_update", "validation", "rollback"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"method capability {name} must be bool")

    def payload(self) -> dict[str, object]:
        return {
            "schema": ADAPTER_CONTRACT_SCHEMA,
            "schema_version": ADAPTER_CONTRACT_SCHEMA_VERSION,
            "method_id": self.method_id,
            "primary_kind": self.primary_kind.value,
            "secondary_kind": self.secondary_kind.value if self.secondary_kind else None,
            "transform": self.transform,
            "owned_surfaces": [value.value for value in self.owned_surfaces],
            "required_feedback": [value.value for value in self.required_feedback],
            "required_host_capabilities": list(self.required_host_capabilities),
            "state_schema": self.state_schema,
            "lineage_schema": self.lineage_schema,
            "online_update": self.online_update,
            "validation": self.validation,
            "rollback": self.rollback,
        }


@dataclass(frozen=True, slots=True)
class MethodRunIdentity:
    run_id: str
    session_id: str
    task_id: str
    state_revision: str

    def __post_init__(self) -> None:
        _id(self.run_id, "method run ID")
        _id(self.session_id, "method session ID")
        _id(self.task_id, "method task ID")
        _id(self.state_revision, "method state revision")


@dataclass(frozen=True, slots=True)
class MethodStateSnapshot:
    state_id: str
    revision: str
    state_schema: str
    state_digest: str
    active: bool

    def __post_init__(self) -> None:
        _id(self.state_id, "method state ID")
        _id(self.revision, "method state revision")
        _id(self.state_schema, "method state schema")
        _sha(self.state_digest, "method state digest")
        if type(self.active) is not bool:
            raise ValueError("method state active must be bool")


@dataclass(frozen=True, slots=True)
class MethodUpdate:
    update_id: str
    target_surface: MemoryLifecycleSurface
    affected_artifact_ids: tuple[str, ...]
    base_revision: str
    observation_cutoff: str
    expected_behavior_change: str
    state_digest: str

    def __post_init__(self) -> None:
        _id(self.update_id, "method update ID")
        object.__setattr__(self, "target_surface", MemoryLifecycleSurface(self.target_surface))
        object.__setattr__(self, "affected_artifact_ids", _ids(self.affected_artifact_ids, "affected artifact IDs", allow_empty=False))
        _id(self.base_revision, "update base revision")
        _id(self.observation_cutoff, "update observation cutoff")
        _id(self.expected_behavior_change, "expected behavior change")
        _sha(self.state_digest, "method update state digest")


_FORBIDDEN_FEEDBACK_KEYS = frozenset({
    "family_id", "familyid", "grader", "answer", "answer_key", "hidden_expectation",
    "official_score", "official_evaluation", "task_score", "score", "reference",
    "raw_payload", "payload", "metadata", "pointer", "source_messages",
})
_FORBIDDEN_FEEDBACK_KEYS_NORMALIZED = frozenset(
    re.sub(r"[^a-z0-9]", "", key.lower())
    for key in _FORBIDDEN_FEEDBACK_KEYS
)


_FEEDBACK_ALLOWLIST: Mapping[FeedbackCondition, frozenset[str]] = {
    FeedbackCondition.F0: frozenset(),
    FeedbackCondition.F1: frozenset({"terminal_outcome"}),
    FeedbackCondition.F2: frozenset({"terminal_outcome", "trajectory"}),
    FeedbackCondition.F3: frozenset({"terminal_outcome", "trajectory", "canonical_events"}),
    FeedbackCondition.F4: frozenset({"terminal_outcome", "trajectory", "canonical_events", "provenance_joins"}),
    FeedbackCondition.F5: frozenset({"terminal_outcome", "trajectory", "canonical_events", "provenance_joins", "counterfactual_replay"}),
}


def _normalized_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def validate_public_payload(value: object) -> None:
    """Reject benchmark answer/grader material at the public adapter boundary."""

    def walk(item: object) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                if _normalized_key(key) in _FORBIDDEN_FEEDBACK_KEYS_NORMALIZED:
                    raise ValueError("benchmark public payload contains forbidden field")
                walk(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                walk(child)

    walk(value)


@dataclass(frozen=True, slots=True)
class FeedbackView:
    condition: FeedbackCondition
    observation_cutoff: str
    values: Mapping[str, Any] = field(default_factory=dict)
    schema: str = ADAPTER_CONTRACT_SCHEMA
    schema_version: int = ADAPTER_CONTRACT_SCHEMA_VERSION
    evidence_plane: EvidencePlane = EvidencePlane.PURE_PROCESS

    def __post_init__(self) -> None:
        object.__setattr__(self, "condition", FeedbackCondition(self.condition))
        _id(self.observation_cutoff, "feedback observation cutoff")
        if self.schema != ADAPTER_CONTRACT_SCHEMA or self.schema_version != ADAPTER_CONTRACT_SCHEMA_VERSION:
            raise ValueError("unsupported feedback view schema")
        object.__setattr__(self, "evidence_plane", EvidencePlane(self.evidence_plane))
        if self.evidence_plane is not EvidencePlane.PURE_PROCESS:
            raise ValueError("method feedback must use pure_process plane")
        values = dict(self.values)
        allowed = _FEEDBACK_ALLOWLIST[self.condition]
        normalized = {_normalized_key(key): key for key in values}
        forbidden = _FORBIDDEN_FEEDBACK_KEYS_NORMALIZED.intersection(normalized)
        if forbidden:
            raise ValueError("feedback view contains forbidden field")
        unknown = set(values) - allowed
        if unknown:
            raise ValueError("feedback field is not allowed for condition")
        validate_public_payload(values)
        object.__setattr__(self, "values", _frozen(values))

    @property
    def allowed_fields(self) -> frozenset[str]:
        return _FEEDBACK_ALLOWLIST[self.condition]

    @property
    def value_digest(self) -> str:
        return content_digest({"condition": self.condition.value, "values": dict(self.values)})

    def payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "condition": self.condition.value,
            "observation_cutoff": self.observation_cutoff,
            "evidence_plane": self.evidence_plane.value,
            "allowed_fields": sorted(self.allowed_fields),
            "value_digest": self.value_digest,
        }


@runtime_checkable
class BenchmarkAdapter(Protocol):
    def enumerate_cases(self, split: BenchmarkSplit) -> Sequence[BenchmarkTaskRequest]: ...

    def reset(self, request: BenchmarkTaskRequest) -> AdapterResult: ...

    def step(self, request: BenchmarkTaskRequest) -> tuple[AdapterResult, BenchmarkPublicEvent]: ...

    def evaluate_final(self, request: BenchmarkTaskRequest) -> FinalEvaluationRecord: ...


@runtime_checkable
class HostAdapter(Protocol):
    @property
    def capabilities(self) -> HostCapabilities: ...

    def prepare_session(self, run: MethodRunIdentity) -> AdapterResult: ...

    def observe_event(self, event: CanonicalHostEvent) -> AdapterResult: ...

    def snapshot_state(self) -> MethodStateSnapshot: ...

    def restart(self, run: MethodRunIdentity) -> AdapterResult: ...


class DeterministicHostAdapter:
    """Content-free host fixture shared by semantic/episodic/procedural tests."""

    def __init__(self, capabilities: HostCapabilities) -> None:
        self._capabilities = capabilities
        self._run: MethodRunIdentity | None = None
        self._events: dict[str, CanonicalHostEvent] = {}

    @property
    def capabilities(self) -> HostCapabilities:
        return self._capabilities

    def prepare_session(self, run: MethodRunIdentity) -> AdapterResult:
        if self._run is not None and self._run != run:
            return AdapterResult(AdapterStatus.UNSUPPORTED, "operation.host.prepare", "active_run")
        self._run = run
        return AdapterResult(AdapterStatus.SUPPORTED, "operation.host.prepare")

    def observe_event(self, event: CanonicalHostEvent) -> AdapterResult:
        if self._run is not None and (event.session_id != self._run.session_id or event.task_id != self._run.task_id):
            return AdapterResult(AdapterStatus.REJECTED, "operation.host.observe", "session_task_mismatch")
        if event.event_id in self._events:
            return AdapterResult(AdapterStatus.REJECTED, "operation.host.observe", "duplicate_event")
        self._events[event.event_id] = event
        return AdapterResult(AdapterStatus.SUPPORTED, "operation.host.observe")

    def snapshot_state(self) -> MethodStateSnapshot:
        digest = content_digest({
            "events": [event.event_id for event in sorted(self._events.values(), key=lambda item: item.event_id)],
        })
        return MethodStateSnapshot(
            state_id="state.host.fixture.v1",
            revision=f"revision.host.{digest[:24]}",
            state_schema="host.state.fixture.v1",
            state_digest=digest,
            active=self._run is not None,
        )

    def restart(self, run: MethodRunIdentity) -> AdapterResult:
        if not self._capabilities.restart:
            return AdapterResult(AdapterStatus.UNSUPPORTED, "operation.host.restart", "restart_unsupported")
        if self._run != run:
            return AdapterResult(AdapterStatus.REJECTED, "operation.host.restart", "run_not_prepared")
        return AdapterResult(AdapterStatus.SUPPORTED, "operation.host.restart")


@runtime_checkable
class MemoryMethodAdapter(Protocol):
    def describe_capabilities(self) -> MethodCapabilities: ...

    def prepare_run(self, run: MethodRunIdentity) -> AdapterResult: ...

    def start_episode(self, run: MethodRunIdentity) -> AdapterResult: ...

    def observe_event(self, event: CanonicalHostEvent) -> AdapterResult: ...

    def finalize_episode(self, run: MethodRunIdentity) -> AdapterResult: ...

    def snapshot_state(self) -> MethodStateSnapshot: ...

    def propose_update(self, feedback: FeedbackView) -> tuple[AdapterResult, MethodUpdate | None]: ...

    def validate_update(self, update: MethodUpdate) -> AdapterResult: ...

    def activate_update(self, update: MethodUpdate) -> AdapterResult: ...

    def rollback_update(self, update: MethodUpdate) -> AdapterResult: ...


class DeterministicMemoryMethodAdapter:
    """Small stateful adapter used by the Stage 2 contract harness.

    It records only event identities and digests.  The implementation is
    intentionally method-neutral: one instance owns one memory kind and never
    mutates another kind's state.
    """

    def __init__(self, capabilities: MethodCapabilities) -> None:
        self._capabilities = capabilities
        self._run: MethodRunIdentity | None = None
        self._events: set[str] = set()
        self._revision = "revision.initial"
        self._active_update: str | None = None
        self._updates: dict[str, MethodUpdate] = {}

    def describe_capabilities(self) -> MethodCapabilities:
        return self._capabilities

    def _operation(self, name: str) -> str:
        return f"operation.{self._capabilities.method_id}.{name}"

    def _state_digest(self) -> str:
        return content_digest({
            "method_id": self._capabilities.method_id,
            "primary_kind": self._capabilities.primary_kind.value,
            "revision": self._revision,
            "events": sorted(self._events),
            "active_update": self._active_update,
        })

    def prepare_run(self, run: MethodRunIdentity) -> AdapterResult:
        if self._run is not None and self._run != run:
            return AdapterResult(AdapterStatus.UNSUPPORTED, self._operation("prepare"), "active_run")
        self._run = run
        return AdapterResult(AdapterStatus.SUPPORTED, self._operation("prepare"))

    def start_episode(self, run: MethodRunIdentity) -> AdapterResult:
        if self._run != run:
            return AdapterResult(AdapterStatus.REJECTED, self._operation("start"), "run_not_prepared")
        return AdapterResult(AdapterStatus.SUPPORTED, self._operation("start"))

    def observe_event(self, event: CanonicalHostEvent) -> AdapterResult:
        if event.memory_kind is not None and event.memory_kind != self._capabilities.primary_kind:
            return AdapterResult(AdapterStatus.UNSUPPORTED, self._operation("observe"), "kind_mismatch")
        if event.event_id in self._events:
            return AdapterResult(AdapterStatus.REJECTED, self._operation("observe"), "duplicate_event")
        self._events.add(event.event_id)
        return AdapterResult(AdapterStatus.SUPPORTED, self._operation("observe"))

    def finalize_episode(self, run: MethodRunIdentity) -> AdapterResult:
        if self._run != run:
            return AdapterResult(AdapterStatus.REJECTED, self._operation("finalize"), "run_not_prepared")
        return AdapterResult(AdapterStatus.SUPPORTED, self._operation("finalize"), output_digest=self._state_digest())

    def snapshot_state(self) -> MethodStateSnapshot:
        return MethodStateSnapshot(
            state_id=f"state.{self._capabilities.method_id}",
            revision=self._revision,
            state_schema=self._capabilities.state_schema,
            state_digest=self._state_digest(),
            active=self._active_update is not None,
        )

    def propose_update(self, feedback: FeedbackView) -> tuple[AdapterResult, MethodUpdate | None]:
        if not self._capabilities.owned_surfaces:
            return AdapterResult(AdapterStatus.UNSUPPORTED, self._operation("propose"), "surface_not_owned"), None
        if self._capabilities.required_feedback:
            required = max(self._capabilities.required_feedback, key=lambda item: list(FeedbackCondition).index(item))
            if list(FeedbackCondition).index(feedback.condition) < list(FeedbackCondition).index(required):
                return AdapterResult(AdapterStatus.REJECTED, self._operation("propose"), "insufficient_feedback"), None
        digest = content_digest({
            "method_id": self._capabilities.method_id,
            "feedback_digest": feedback.value_digest,
            "base_revision": self._revision,
        })
        update = MethodUpdate(
            update_id=f"update.{self._capabilities.method_id}.{digest[:32]}",
            target_surface=self._capabilities.owned_surfaces[0],
            affected_artifact_ids=(f"artifact.{digest[:32]}",),
            base_revision=self._revision,
            observation_cutoff=feedback.observation_cutoff,
            expected_behavior_change=f"method_state.{digest[:24]}",
            state_digest=self._state_digest(),
        )
        self._updates[update.update_id] = update
        return AdapterResult(AdapterStatus.ACCEPTED, self._operation("propose"), output_digest=digest), update

    def validate_update(self, update: MethodUpdate) -> AdapterResult:
        if update.target_surface not in self._capabilities.owned_surfaces:
            return AdapterResult(AdapterStatus.UNSUPPORTED, self._operation("validate"), "surface_not_owned")
        if update.base_revision != self._revision:
            return AdapterResult(AdapterStatus.STALE, self._operation("validate"), "stale_revision")
        if update.update_id not in self._updates:
            return AdapterResult(AdapterStatus.REJECTED, self._operation("validate"), "unknown_update")
        return AdapterResult(AdapterStatus.ACCEPTED, self._operation("validate"))

    def activate_update(self, update: MethodUpdate) -> AdapterResult:
        if self._active_update == update.update_id:
            return AdapterResult(AdapterStatus.ACCEPTED, self._operation("activate"), "duplicate_activation")
        validation = self.validate_update(update)
        if validation.status is AdapterStatus.STALE or validation.status is AdapterStatus.UNSUPPORTED:
            return validation
        if validation.status is not AdapterStatus.ACCEPTED:
            return AdapterResult(AdapterStatus.REJECTED, self._operation("activate"), validation.reason_code)
        self._revision = f"revision.{content_digest(update.update_id)[:32]}"
        self._active_update = update.update_id
        self._updates[update.update_id] = update
        return AdapterResult(AdapterStatus.ACCEPTED, self._operation("activate"), output_digest=self._state_digest())

    def rollback_update(self, update: MethodUpdate) -> AdapterResult:
        if self._active_update != update.update_id:
            return AdapterResult(AdapterStatus.REJECTED, self._operation("rollback"), "update_not_active")
        self._revision = f"revision.rollback.{content_digest(update.update_id)[:24]}"
        self._active_update = None
        return AdapterResult(AdapterStatus.ACCEPTED, self._operation("rollback"), output_digest=self._state_digest())


__all__ = [
    "ADAPTER_CONTRACT_SCHEMA",
    "ADAPTER_CONTRACT_SCHEMA_VERSION",
    "AdapterResult",
    "AdapterStatus",
    "BenchmarkAdapter",
    "BenchmarkPublicEvent",
    "BenchmarkSplit",
    "BenchmarkTaskRequest",
    "CanonicalHostEvent",
    "DeterministicHostAdapter",
    "FeedbackCondition",
    "FeedbackView",
    "FinalEvaluationRecord",
    "HostAdapter",
    "HostCapabilities",
    "HostEventKind",
    "MemoryMethodAdapter",
    "DeterministicMemoryMethodAdapter",
    "MethodCapabilities",
    "MethodRunIdentity",
    "MethodStateSnapshot",
    "MethodUpdate",
    "content_digest",
    "validate_public_payload",
]
