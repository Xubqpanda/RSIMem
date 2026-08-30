"""Host-neutral contracts for the six RSIMem memory-policy layers.

The policy layer is deliberately independent from Hermes, PAST-Bench and any
particular memory backend.  A host adapter supplies :class:`TriggerEvent` and
the runtime records the decisions below.  The contracts are intentionally
content-light: source and extraction payloads are represented by digests and
stable identifiers, while content-bearing material remains in the owner
controlled compiler/optimizer boundary.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence, ClassVar, runtime_checkable


MEMORY_POLICY_CONTRACT_SCHEMA_VERSION = 1
MEMORY_POLICY_CONTRACT_SCHEMA = "rsimem-memory-policy-v1"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


def canonical_json(value: object) -> str:
    """Return the canonical JSON representation used by policy identities."""

    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def content_digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _require_identifier(value: object, name: str) -> None:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{name} must be a stable identifier")


def _require_digest(value: object, name: str) -> None:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{name} must be sha256")


def _frozen_metadata(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType(dict(value or {}))


def _tuple_strings(values: Sequence[str], name: str) -> tuple[str, ...]:
    result = tuple(values)
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must not contain duplicates")
    if any(not isinstance(value, str) or not value.strip() for value in result):
        raise ValueError(f"{name} must contain non-empty strings")
    return result


def _identity_value(value: object) -> object:
    """Make enum/tuple values safe and deterministic in an identity payload."""

    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _identity_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_identity_value(item) for item in value]
    digest = getattr(value, "digest", None)
    if isinstance(digest, str) and _DIGEST.fullmatch(digest):
        return {"digest": digest}
    return value


class PolicyLayer(StrEnum):
    TRIGGER = "trigger"
    SOURCE_SELECTION = "source_selection"
    EXTRACTION = "extraction"
    ADMISSION = "admission"
    COMMIT = "commit"
    EXPOSURE = "exposure"


FORMATION_LAYERS = (
    PolicyLayer.TRIGGER,
    PolicyLayer.SOURCE_SELECTION,
    PolicyLayer.EXTRACTION,
    PolicyLayer.ADMISSION,
    PolicyLayer.COMMIT,
)


class DecisionAction(StrEnum):
    """The only control actions a policy may emit."""

    RUN = "RUN"
    SKIP = "SKIP"
    DEFER = "DEFER"


POLICY_DECISION_CONTRACT_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class PolicyDecisionContract:
    """Host-neutral schema declaration for one policy-layer decision.

    The concrete decision dataclasses below enforce these fields at runtime;
    this small, content-free declaration makes the contract inspectable by
    feasibility reports and replay tooling without importing a host adapter.
    """

    contract_id: str
    layer: PolicyLayer
    decision_type: str
    required_fields: tuple[str, ...]
    allowed_actions: tuple[DecisionAction, ...] = (
        DecisionAction.RUN,
        DecisionAction.SKIP,
        DecisionAction.DEFER,
    )
    schema_version: int = POLICY_DECISION_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != POLICY_DECISION_CONTRACT_SCHEMA_VERSION:
            raise ValueError("unsupported policy decision contract schema")
        _require_identifier(self.contract_id, "policy decision contract ID")
        object.__setattr__(self, "layer", PolicyLayer(self.layer))
        _require_identifier(self.decision_type, "policy decision type")
        fields = _tuple_strings(self.required_fields, "policy decision fields")
        if not fields:
            raise ValueError("policy decision contract requires fields")
        object.__setattr__(self, "required_fields", fields)
        actions = tuple(DecisionAction(value) for value in self.allowed_actions)
        if not actions or len(actions) != len(set(actions)):
            raise ValueError("policy decision contract actions must be unique")
        object.__setattr__(self, "allowed_actions", actions)
        expected = f"policy-decision-contract.{self.layer.value}.v1"
        if self.contract_id != expected:
            raise ValueError("policy decision contract ID mismatch")

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "contract_id": self.contract_id,
            "layer": self.layer.value,
            "decision_type": self.decision_type,
            "required_fields": list(self.required_fields),
            "allowed_actions": [value.value for value in self.allowed_actions],
        }

    @classmethod
    def from_payload(cls, value: object) -> "PolicyDecisionContract":
        fields = {
            "schema_version", "contract_id", "layer", "decision_type",
            "required_fields", "allowed_actions",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError("malformed policy decision contract")
        if not isinstance(value["required_fields"], list) or not isinstance(
            value["allowed_actions"], list
        ):
            raise ValueError("malformed policy decision contract collections")
        try:
            result = cls(
                contract_id=value["contract_id"],
                layer=value["layer"],
                decision_type=value["decision_type"],
                required_fields=tuple(value["required_fields"]),
                allowed_actions=tuple(value["allowed_actions"]),
                schema_version=value["schema_version"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("malformed policy decision contract") from exc
        if result.payload() != dict(value):
            raise ValueError("non-canonical policy decision contract")
        return result


_COMMON_DECISION_FIELDS = (
    "decision_id", "policy_version", "source_revision", "input_digest",
    "output_digest", "action", "execution_status", "reason_codes",
    "lineage_id",
)


_DECISION_CONTRACTS: Mapping[PolicyLayer, PolicyDecisionContract] = {
    PolicyLayer.TRIGGER: PolicyDecisionContract(
        "policy-decision-contract.trigger.v1",
        PolicyLayer.TRIGGER,
        "TriggerDecision",
        _COMMON_DECISION_FIELDS + ("next_eligible_boundary", "duplicate_suppressed"),
    ),
    PolicyLayer.SOURCE_SELECTION: PolicyDecisionContract(
        "policy-decision-contract.source_selection.v1",
        PolicyLayer.SOURCE_SELECTION,
        "SourceSelectionDecision",
        _COMMON_DECISION_FIELDS + (
            "projection_mode", "selected_segment_ids", "skipped_segment_ids",
            "rejected_segment_ids", "source_digest", "truncation", "safety",
        ),
    ),
    PolicyLayer.EXTRACTION: PolicyDecisionContract(
        "policy-decision-contract.extraction.v1",
        PolicyLayer.EXTRACTION,
        "ExtractionDecision",
        _COMMON_DECISION_FIELDS + ("candidate_fact_ids", "source_digest", "request_id"),
    ),
    PolicyLayer.ADMISSION: PolicyDecisionContract(
        "policy-decision-contract.admission.v1",
        PolicyLayer.ADMISSION,
        "AdmissionDecision",
        _COMMON_DECISION_FIELDS + (
            "mutation_kind", "candidate_fact_ids", "accepted_fact_ids",
            "filtered_fact_ids", "backend_revision", "target_artifact_ids",
            "update_supported", "safety",
        ),
    ),
    PolicyLayer.COMMIT: PolicyDecisionContract(
        "policy-decision-contract.commit.v1",
        PolicyLayer.COMMIT,
        "CommitDecision",
        _COMMON_DECISION_FIELDS + (
            "commit_mode", "mutation_ids", "expected_revision",
            "execution_boundary", "final_receipt_id", "safety",
        ),
    ),
    PolicyLayer.EXPOSURE: PolicyDecisionContract(
        "policy-decision-contract.exposure.v1",
        PolicyLayer.EXPOSURE,
        "ExposureDecision",
        _COMMON_DECISION_FIELDS + (
            "exposure_mode", "selected_artifact_ids", "ordering",
            "injection_position", "budget_tokens", "injection_receipt_id",
        ),
    ),
}


def decision_contract_for_layer(
    layer: PolicyLayer | str,
) -> PolicyDecisionContract:
    """Return the immutable decision contract for one policy layer."""

    try:
        return _DECISION_CONTRACTS[PolicyLayer(layer)]
    except (KeyError, ValueError) as exc:
        raise ValueError("unknown policy decision layer") from exc


class ExecutionStatus(StrEnum):
    PENDING = "pending"
    EXECUTED = "executed"
    SKIPPED = "skipped"
    DEFERRED = "deferred"
    FAILED = "failed"
    REJECTED = "rejected"


class ProjectionMode(StrEnum):
    WHOLE_COMPLETED_TASK = "whole_completed_task"
    SELECTED_COMPLETED_SEGMENTS = "selected_completed_segments"
    INCREMENTAL_REVISION = "incremental_revision"


class MutationKind(StrEnum):
    NONE = "NONE"
    ADD = "ADD"
    UPDATE = "UPDATE"
    DELETE = "DELETE"


class CommitMode(StrEnum):
    IMMEDIATE = "immediate"
    DEFERRED = "deferred"
    RETRY = "retry"


class ExposureMode(StrEnum):
    EAGER_SYSTEM_PROMPT = "eager_system_prompt"
    SELECTIVE_RETRIEVAL = "selective_retrieval"
    TOOL_MEDIATED = "tool_mediated"
    NOT_EXPOSED = "not_exposed"


class PolicyArtifactKind(StrEnum):
    FIXED = "fixed"
    SINGLE_LAYER_ADAPTIVE = "single_layer_adaptive"
    JOINT = "joint"


@dataclass(frozen=True, slots=True)
class TriggerEvent:
    """A host event from which a formation/exposure decision may be made."""

    event_id: str
    event_type: str
    source_revision: str
    input_digest: str
    session_id: str | None = None
    task_id: str | None = None
    turn_id: str | None = None
    turn_index: int | None = None
    supported: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = MEMORY_POLICY_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != MEMORY_POLICY_CONTRACT_SCHEMA_VERSION:
            raise ValueError("unsupported trigger event schema version")
        for value, name in ((self.event_id, "trigger event ID"), (self.event_type, "event type")):
            _require_identifier(value, name)
        if not isinstance(self.source_revision, str) or not self.source_revision.strip():
            raise ValueError("trigger source revision must not be empty")
        _require_digest(self.input_digest, "trigger input digest")
        for value, name in ((self.session_id, "session ID"), (self.task_id, "task ID"), (self.turn_id, "turn ID")):
            if value is not None:
                _require_identifier(value, name)
        if self.turn_index is not None and (type(self.turn_index) is not int or self.turn_index < 0):
            raise ValueError("turn index must be a non-negative integer")
        if type(self.supported) is not bool:
            raise ValueError("trigger supported must be bool")
        object.__setattr__(self, "metadata", _frozen_metadata(self.metadata))

    @classmethod
    def create(
        cls,
        *,
        event_type: str,
        source_revision: str,
        input_payload: object,
        session_id: str | None = None,
        task_id: str | None = None,
        turn_id: str | None = None,
        turn_index: int | None = None,
        supported: bool = True,
        metadata: Mapping[str, Any] | None = None,
    ) -> "TriggerEvent":
        digest = content_digest(input_payload)
        identity = {
            "schema_version": MEMORY_POLICY_CONTRACT_SCHEMA_VERSION,
            "event_type": event_type,
            "source_revision": source_revision,
            "input_digest": digest,
            "session_id": session_id,
            "task_id": task_id,
            "turn_id": turn_id,
            "turn_index": turn_index,
        }
        return cls(
            event_id=f"event.{content_digest(identity)[:40]}",
            event_type=event_type,
            source_revision=source_revision,
            input_digest=digest,
            session_id=session_id,
            task_id=task_id,
            turn_id=turn_id,
            turn_index=turn_index,
            supported=supported,
            metadata=metadata or {},
        )


@dataclass(frozen=True, slots=True)
class SafetyBoundary:
    """Runtime-owned safety facts which policy actions cannot override."""

    active_segment_ids: tuple[str, ...] = ()
    current_turn_id: str | None = None
    current_turn_segment_ids: tuple[str, ...] = ()
    tool_closures: tuple[tuple[str, ...], ...] = ()
    schema_valid: bool = True
    cas_valid: bool = True
    transaction_valid: bool = True
    rollback_supported: bool = True
    credentials_safe: bool = True
    writer_identity_verified: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "active_segment_ids", _tuple_strings(self.active_segment_ids, "active segment IDs"))
        if self.current_turn_id is not None:
            _require_identifier(self.current_turn_id, "current turn ID")
        object.__setattr__(self, "current_turn_segment_ids", _tuple_strings(self.current_turn_segment_ids, "current turn segment IDs"))
        closures: list[tuple[str, ...]] = []
        members: set[str] = set()
        for closure in self.tool_closures:
            normalized = _tuple_strings(closure, "tool closure")
            if len(normalized) < 1:
                raise ValueError("tool closure must contain at least one segment")
            if members.intersection(normalized):
                raise ValueError("tool closure segments must be disjoint")
            members.update(normalized)
            closures.append(normalized)
        object.__setattr__(self, "tool_closures", tuple(closures))
        for name in ("schema_valid", "cas_valid", "transaction_valid", "rollback_supported", "credentials_safe", "writer_identity_verified"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be bool")

    @property
    def safe(self) -> bool:
        return all((self.schema_valid, self.cas_valid, self.transaction_valid, self.rollback_supported, self.credentials_safe, self.writer_identity_verified))

    def payload(self) -> dict[str, object]:
        return {
            "active_segment_ids": list(self.active_segment_ids),
            "current_turn_id": self.current_turn_id,
            "current_turn_segment_ids": list(self.current_turn_segment_ids),
            "tool_closures": [list(item) for item in self.tool_closures],
            "schema_valid": self.schema_valid,
            "cas_valid": self.cas_valid,
            "transaction_valid": self.transaction_valid,
            "rollback_supported": self.rollback_supported,
            "credentials_safe": self.credentials_safe,
            "writer_identity_verified": self.writer_identity_verified,
        }

    @property
    def digest(self) -> str:
        return content_digest(self.payload())

    def validate_selected(self, selected_segment_ids: Sequence[str]) -> None:
        selected = set(_tuple_strings(selected_segment_ids, "selected segment IDs"))
        protected = set(self.active_segment_ids)
        # Some hosts expose the current turn as a segment-level pointer;
        # retain that representation in addition to the explicit segment set.
        if self.current_turn_id is not None:
            protected.add(self.current_turn_id)
        protected.update(self.current_turn_segment_ids)
        if protected.intersection(selected):
            raise ValueError("source selection includes active/current segment")
        for closure in self.tool_closures:
            overlap = selected.intersection(closure)
            if overlap and overlap != set(closure):
                raise ValueError("source selection splits a tool call/result closure")

    def require_safe(self) -> None:
        if not self.safe:
            raise ValueError("policy action rejected by runtime safety boundary")


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    """Common auditable envelope shared by all six decision types."""

    decision_id: str
    policy_version: str
    source_revision: str
    input_digest: str
    output_digest: str
    action: DecisionAction
    execution_status: ExecutionStatus
    reason_codes: tuple[str, ...]
    lineage_id: str
    schema_version: int = MEMORY_POLICY_CONTRACT_SCHEMA_VERSION
    trigger_event_id: str | None = None
    execution_receipt_id: str | None = None

    layer: ClassVar[PolicyLayer]

    def __post_init__(self) -> None:
        if self.schema_version != MEMORY_POLICY_CONTRACT_SCHEMA_VERSION:
            raise ValueError("unsupported policy decision schema version")
        _require_identifier(self.decision_id, "decision ID")
        _require_identifier(self.policy_version, "policy version")
        if not isinstance(self.source_revision, str) or not self.source_revision.strip():
            raise ValueError("decision source revision must not be empty")
        _require_digest(self.input_digest, "decision input digest")
        _require_digest(self.output_digest, "decision output digest")
        object.__setattr__(self, "action", DecisionAction(self.action))
        object.__setattr__(self, "execution_status", ExecutionStatus(self.execution_status))
        reasons = _tuple_strings(self.reason_codes, "reason codes")
        if not reasons:
            raise ValueError("decision requires at least one reason code")
        object.__setattr__(self, "reason_codes", reasons)
        _require_identifier(self.lineage_id, "lineage ID")
        if self.trigger_event_id is not None:
            _require_identifier(self.trigger_event_id, "trigger event ID")
        if self.execution_receipt_id is not None:
            _require_identifier(self.execution_receipt_id, "execution receipt ID")
        expected_status = {
            DecisionAction.SKIP: ExecutionStatus.SKIPPED,
            DecisionAction.DEFER: ExecutionStatus.DEFERRED,
        }.get(self.action)
        if expected_status is not None and self.execution_status != expected_status:
            raise ValueError(f"{self.action.value} decision must have {expected_status.value} status")
        if self.action == DecisionAction.RUN and self.execution_status in {
            ExecutionStatus.SKIPPED,
            ExecutionStatus.DEFERRED,
        }:
            raise ValueError("RUN decision cannot have non-executing status")

    @property
    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "layer": self.layer.value,
            "policy_version": self.policy_version,
            "source_revision": self.source_revision,
            "action": self.action.value,
            "input_digest": self.input_digest,
            "output_digest": self.output_digest,
            "reason_codes": list(self.reason_codes),
            "lineage_id": self.lineage_id,
            "extra": self.extra_identity_payload,
        }

    @property
    def extra_identity_payload(self) -> dict[str, object]:
        return {}

    @property
    def canonical_id(self) -> str:
        return f"decision.{self.layer.value}.{content_digest(self.identity_payload)[:40]}"

    @property
    def is_canonical(self) -> bool:
        return self.decision_id == self.canonical_id

    @classmethod
    def create(
        cls,
        *,
        policy_version: str,
        source_revision: str,
        input_payload: object,
        output_payload: object,
        action: DecisionAction | str,
        execution_status: ExecutionStatus | str,
        reason_codes: Sequence[str],
        lineage_id: str,
        trigger_event_id: str | None = None,
        execution_receipt_id: str | None = None,
        **extra: object,
    ) -> "PolicyDecision":
        input_digest = content_digest(input_payload)
        output_digest = content_digest(output_payload)
        # Construct once so subclass fields participate in the canonical
        # identity.  The temporary ID only needs to satisfy the identifier
        # grammar during validation; it is replaced with the derived ID below.
        instance = cls(
            decision_id="decision.pending",
            policy_version=policy_version,
            source_revision=source_revision,
            input_digest=input_digest,
            output_digest=output_digest,
            action=action,
            execution_status=execution_status,
            reason_codes=tuple(reason_codes),
            lineage_id=lineage_id,
            trigger_event_id=trigger_event_id,
            execution_receipt_id=execution_receipt_id,
            **extra,
        )
        object.__setattr__(instance, "decision_id", instance.canonical_id)
        return instance


@dataclass(frozen=True, slots=True)
class TriggerDecision(PolicyDecision):
    layer: ClassVar[PolicyLayer] = PolicyLayer.TRIGGER
    next_eligible_boundary: str | None = None
    duplicate_suppressed: bool = False

    @property
    def extra_identity_payload(self) -> dict[str, object]:
        return {
            "next_eligible_boundary": self.next_eligible_boundary,
            "duplicate_suppressed": self.duplicate_suppressed,
        }

    def __post_init__(self) -> None:
        super(TriggerDecision, self).__post_init__()
        if self.action == DecisionAction.DEFER:
            if self.next_eligible_boundary is None or not self.next_eligible_boundary.strip():
                raise ValueError("defer trigger requires next eligible boundary")
        elif self.next_eligible_boundary is not None:
            raise ValueError("next eligible boundary is only valid for defer")
        if type(self.duplicate_suppressed) is not bool:
            raise ValueError("duplicate_suppressed must be bool")


@dataclass(frozen=True, slots=True)
class SourceSelectionDecision(PolicyDecision):
    layer: ClassVar[PolicyLayer] = PolicyLayer.SOURCE_SELECTION
    projection_mode: ProjectionMode = ProjectionMode.WHOLE_COMPLETED_TASK
    selected_segment_ids: tuple[str, ...] = ()
    skipped_segment_ids: tuple[str, ...] = ()
    rejected_segment_ids: tuple[str, ...] = ()
    source_digest: str = ""
    truncation: bool = False
    safety: SafetyBoundary = field(default_factory=SafetyBoundary, repr=False, compare=False)

    @property
    def extra_identity_payload(self) -> dict[str, object]:
        return {
            "projection_mode": self.projection_mode.value,
            "selected_segment_ids": list(self.selected_segment_ids),
            "skipped_segment_ids": list(self.skipped_segment_ids),
            "rejected_segment_ids": list(self.rejected_segment_ids),
            "source_digest": self.source_digest,
            "truncation": self.truncation,
            "safety": {"digest": self.safety.digest},
        }

    def __post_init__(self) -> None:
        super(SourceSelectionDecision, self).__post_init__()
        object.__setattr__(self, "projection_mode", ProjectionMode(self.projection_mode))
        for name in ("selected_segment_ids", "skipped_segment_ids", "rejected_segment_ids"):
            object.__setattr__(self, name, _tuple_strings(getattr(self, name), name))
        groups = [set(self.selected_segment_ids), set(self.skipped_segment_ids), set(self.rejected_segment_ids)]
        if any(groups[i].intersection(groups[j]) for i in range(3) for j in range(i + 1, 3)):
            raise ValueError("source selection segment sets must be disjoint")
        _require_digest(self.source_digest, "source digest")
        if type(self.truncation) is not bool:
            raise ValueError("truncation must be bool")
        if not isinstance(self.safety, SafetyBoundary):
            raise ValueError("source selection safety must be a SafetyBoundary")
        self.safety.validate_selected(self.selected_segment_ids)
        if self.action != DecisionAction.RUN and self.selected_segment_ids:
            raise ValueError("skip/defer source selection cannot select segments")

    @classmethod
    def create(cls, *, selected_segment_ids: Sequence[str], skipped_segment_ids: Sequence[str] = (), rejected_segment_ids: Sequence[str] = (), projection_mode: ProjectionMode | str = ProjectionMode.WHOLE_COMPLETED_TASK, source_payload: object | None = None, source_digest: str | None = None, safety: SafetyBoundary | None = None, **kwargs: object) -> "SourceSelectionDecision":
        selected = tuple(selected_segment_ids)
        if safety is not None:
            safety.validate_selected(selected)
        digest = source_digest or content_digest({"selected_segment_ids": list(selected), "source_payload": source_payload})
        return super(SourceSelectionDecision, cls).create(
            selected_segment_ids=selected,
            skipped_segment_ids=tuple(skipped_segment_ids),
            rejected_segment_ids=tuple(rejected_segment_ids),
            projection_mode=ProjectionMode(projection_mode),
            source_digest=digest,
            safety=safety or SafetyBoundary(),
            **kwargs,
        )


@dataclass(frozen=True, slots=True)
class ExtractionDecision(PolicyDecision):
    layer: ClassVar[PolicyLayer] = PolicyLayer.EXTRACTION
    candidate_fact_ids: tuple[str, ...] = ()
    source_digest: str = ""
    request_id: str | None = None

    @property
    def extra_identity_payload(self) -> dict[str, object]:
        return {
            "candidate_fact_ids": list(self.candidate_fact_ids),
            "source_digest": self.source_digest,
            "request_id": self.request_id,
        }

    def __post_init__(self) -> None:
        super(ExtractionDecision, self).__post_init__()
        object.__setattr__(self, "candidate_fact_ids", _tuple_strings(self.candidate_fact_ids, "candidate fact IDs"))
        _require_digest(self.source_digest, "extraction source digest")
        if self.request_id is not None:
            _require_identifier(self.request_id, "extraction request ID")
        if self.action != DecisionAction.RUN and self.candidate_fact_ids:
            raise ValueError("skip/defer extraction cannot produce fact candidates")


@dataclass(frozen=True, slots=True)
class AdmissionDecision(PolicyDecision):
    layer: ClassVar[PolicyLayer] = PolicyLayer.ADMISSION
    mutation_kind: MutationKind = MutationKind.NONE
    candidate_fact_ids: tuple[str, ...] = ()
    accepted_fact_ids: tuple[str, ...] = ()
    filtered_fact_ids: tuple[str, ...] = ()
    backend_revision: str | None = None
    target_artifact_ids: tuple[str, ...] = ()
    update_supported: bool = True
    safety: SafetyBoundary = field(default_factory=SafetyBoundary, repr=False, compare=False)

    @property
    def extra_identity_payload(self) -> dict[str, object]:
        return {
            "mutation_kind": self.mutation_kind.value,
            "candidate_fact_ids": list(self.candidate_fact_ids),
            "accepted_fact_ids": list(self.accepted_fact_ids),
            "filtered_fact_ids": list(self.filtered_fact_ids),
            "backend_revision": self.backend_revision,
            "target_artifact_ids": list(self.target_artifact_ids),
            "update_supported": self.update_supported,
            "safety": {"digest": self.safety.digest},
        }

    def __post_init__(self) -> None:
        super(AdmissionDecision, self).__post_init__()
        object.__setattr__(self, "mutation_kind", MutationKind(self.mutation_kind))
        for name in ("candidate_fact_ids", "accepted_fact_ids", "filtered_fact_ids", "target_artifact_ids"):
            object.__setattr__(self, name, _tuple_strings(getattr(self, name), name))
        if set(self.accepted_fact_ids).difference(self.candidate_fact_ids):
            raise ValueError("accepted facts must be candidates")
        if set(self.filtered_fact_ids).difference(self.candidate_fact_ids):
            raise ValueError("filtered facts must be candidates")
        if set(self.accepted_fact_ids).intersection(self.filtered_fact_ids):
            raise ValueError("a fact cannot be both accepted and filtered")
        if self.mutation_kind in {MutationKind.UPDATE, MutationKind.DELETE}:
            if not self.update_supported:
                raise ValueError("backend does not support update/target mutation")
            if not self.target_artifact_ids or not self.backend_revision:
                raise ValueError("target admission requires target artifacts and backend revision")
        elif self.target_artifact_ids:
            raise ValueError("only update admission may carry target artifacts")
        if type(self.update_supported) is not bool:
            raise ValueError("update_supported must be bool")
        if not isinstance(self.safety, SafetyBoundary):
            raise ValueError("admission safety must be a SafetyBoundary")
        self.safety.require_safe()
        if self.action == DecisionAction.RUN and not self.backend_revision:
            raise ValueError("run admission requires current backend revision")
        if self.action != DecisionAction.RUN and (self.mutation_kind != MutationKind.NONE or self.accepted_fact_ids or self.target_artifact_ids):
            raise ValueError("skip/defer admission cannot schedule a mutation")


@dataclass(frozen=True, slots=True)
class CommitDecision(PolicyDecision):
    layer: ClassVar[PolicyLayer] = PolicyLayer.COMMIT
    commit_mode: CommitMode = CommitMode.IMMEDIATE
    mutation_ids: tuple[str, ...] = ()
    expected_revision: str | None = None
    execution_boundary: str | None = None
    final_receipt_id: str | None = None
    safety: SafetyBoundary = field(default_factory=SafetyBoundary, repr=False, compare=False)

    @property
    def extra_identity_payload(self) -> dict[str, object]:
        return {
            "commit_mode": self.commit_mode.value,
            "mutation_ids": list(self.mutation_ids),
            "expected_revision": self.expected_revision,
            "execution_boundary": self.execution_boundary,
            "final_receipt_id": self.final_receipt_id,
            "safety": {"digest": self.safety.digest},
        }

    def __post_init__(self) -> None:
        super(CommitDecision, self).__post_init__()
        object.__setattr__(self, "commit_mode", CommitMode(self.commit_mode))
        object.__setattr__(self, "mutation_ids", _tuple_strings(self.mutation_ids, "mutation IDs"))
        if self.action == DecisionAction.RUN and not self.mutation_ids:
            raise ValueError("run commit requires mutation IDs")
        if self.action == DecisionAction.RUN and not self.expected_revision:
            raise ValueError("run commit requires expected revision")
        if self.action != DecisionAction.RUN and self.mutation_ids:
            raise ValueError("skip/defer commit cannot carry mutation IDs")
        if self.commit_mode == CommitMode.DEFERRED and self.execution_boundary is None:
            raise ValueError("deferred commit requires execution boundary")
        for value, name in ((self.expected_revision, "expected revision"), (self.execution_boundary, "execution boundary"), (self.final_receipt_id, "final receipt ID")):
            if value is not None:
                _require_identifier(value, name)
        if not isinstance(self.safety, SafetyBoundary):
            raise ValueError("commit safety must be a SafetyBoundary")
        self.safety.require_safe()


@dataclass(frozen=True, slots=True)
class ExposureDecision(PolicyDecision):
    layer: ClassVar[PolicyLayer] = PolicyLayer.EXPOSURE
    exposure_mode: ExposureMode = ExposureMode.NOT_EXPOSED
    selected_artifact_ids: tuple[str, ...] = ()
    ordering: tuple[str, ...] = ()
    injection_position: str | None = None
    budget_tokens: int | None = None
    injection_receipt_id: str | None = None

    @property
    def extra_identity_payload(self) -> dict[str, object]:
        return {
            "exposure_mode": self.exposure_mode.value,
            "selected_artifact_ids": list(self.selected_artifact_ids),
            "ordering": list(self.ordering),
            "injection_position": self.injection_position,
            "budget_tokens": self.budget_tokens,
            "injection_receipt_id": self.injection_receipt_id,
        }

    def __post_init__(self) -> None:
        super(ExposureDecision, self).__post_init__()
        object.__setattr__(self, "exposure_mode", ExposureMode(self.exposure_mode))
        object.__setattr__(self, "selected_artifact_ids", _tuple_strings(self.selected_artifact_ids, "selected artifact IDs"))
        object.__setattr__(self, "ordering", _tuple_strings(self.ordering, "artifact ordering"))
        if set(self.ordering) != set(self.selected_artifact_ids):
            raise ValueError("artifact ordering must exactly match selected artifacts")
        if self.action == DecisionAction.RUN and self.exposure_mode == ExposureMode.NOT_EXPOSED:
            raise ValueError("run exposure must declare an exposure mode")
        if self.action != DecisionAction.RUN and self.selected_artifact_ids:
            raise ValueError("skip/defer exposure cannot select artifacts")
        if self.budget_tokens is not None and (type(self.budget_tokens) is not int or self.budget_tokens < 0):
            raise ValueError("exposure budget must be a non-negative integer")
        for value, name in ((self.injection_position, "injection position"), (self.injection_receipt_id, "injection receipt ID")):
            if value is not None:
                _require_identifier(value, name)


@dataclass(frozen=True, slots=True)
class PolicyLineage:
    """Stable join key for event -> decisions -> mutation/injection -> feedback."""

    lineage_id: str
    trigger_event_id: str
    decision_ids: tuple[str, ...] = ()
    mutation_receipt_ids: tuple[str, ...] = ()
    injection_receipt_ids: tuple[str, ...] = ()
    future_feedback_ids: tuple[str, ...] = ()

    @classmethod
    def create(cls, *, trigger_event_id: str, decision_ids: Sequence[str] = (), mutation_receipt_ids: Sequence[str] = (), injection_receipt_ids: Sequence[str] = (), future_feedback_ids: Sequence[str] = ()) -> "PolicyLineage":
        _require_identifier(trigger_event_id, "trigger event ID")
        identity = {
            "trigger_event_id": trigger_event_id,
            "decision_ids": list(decision_ids),
            "mutation_receipt_ids": list(mutation_receipt_ids),
            "injection_receipt_ids": list(injection_receipt_ids),
            "future_feedback_ids": list(future_feedback_ids),
        }
        return cls(
            lineage_id=f"lineage.{content_digest(identity)[:40]}",
            trigger_event_id=trigger_event_id,
            decision_ids=tuple(decision_ids),
            mutation_receipt_ids=tuple(mutation_receipt_ids),
            injection_receipt_ids=tuple(injection_receipt_ids),
            future_feedback_ids=tuple(future_feedback_ids),
        )

    def __post_init__(self) -> None:
        for value, name in ((self.lineage_id, "lineage ID"), (self.trigger_event_id, "trigger event ID")):
            _require_identifier(value, name)
        for name in ("decision_ids", "mutation_receipt_ids", "injection_receipt_ids", "future_feedback_ids"):
            object.__setattr__(self, name, _tuple_strings(getattr(self, name), name))

    @property
    def digest(self) -> str:
        return content_digest(self.payload())

    def payload(self) -> dict[str, object]:
        return {
            "lineage_id": self.lineage_id,
            "trigger_event_id": self.trigger_event_id,
            "decision_ids": list(self.decision_ids),
            "mutation_receipt_ids": list(self.mutation_receipt_ids),
            "injection_receipt_ids": list(self.injection_receipt_ids),
            "future_feedback_ids": list(self.future_feedback_ids),
        }

    @classmethod
    def from_decisions(
        cls,
        decisions: Sequence[PolicyDecision],
        *,
        mutation_receipt_ids: Sequence[str] = (),
        injection_receipt_ids: Sequence[str] = (),
        future_feedback_ids: Sequence[str] = (),
    ) -> "PolicyLineage":
        items = tuple(decisions)
        if not items:
            raise ValueError("lineage requires at least one decision")
        lineage_ids = {item.lineage_id for item in items}
        if len(lineage_ids) != 1:
            raise ValueError("decisions must share one lineage")
        trigger_ids = {item.trigger_event_id for item in items if item.trigger_event_id is not None}
        if len(trigger_ids) != 1:
            raise ValueError("decisions must share one trigger event")
        return cls(
            lineage_id=items[0].lineage_id,
            trigger_event_id=next(iter(trigger_ids)),
            decision_ids=tuple(item.decision_id for item in items),
            mutation_receipt_ids=tuple(mutation_receipt_ids),
            injection_receipt_ids=tuple(injection_receipt_ids),
            future_feedback_ids=tuple(future_feedback_ids),
        )


@dataclass(frozen=True, slots=True)
class PolicyArtifactIdentity:
    """Identity of fixed, single-layer adaptive, or joint policy artifacts."""

    artifact_id: str
    policy_version: str
    kind: PolicyArtifactKind
    layers: tuple[PolicyLayer, ...]
    parent_artifact_id: str | None = None

    def __post_init__(self) -> None:
        _require_identifier(self.artifact_id, "policy artifact ID")
        _require_identifier(self.policy_version, "policy artifact version")
        object.__setattr__(self, "kind", PolicyArtifactKind(self.kind))
        normalized = tuple(sorted((PolicyLayer(layer) for layer in self.layers), key=lambda item: item.value))
        if not normalized or len(normalized) != len(set(normalized)):
            raise ValueError("policy artifact must declare unique layers")
        if self.kind == PolicyArtifactKind.FIXED and len(normalized) != 1:
            raise ValueError("fixed artifact must declare exactly one layer")
        if self.kind == PolicyArtifactKind.SINGLE_LAYER_ADAPTIVE and len(normalized) != 1:
            raise ValueError("single-layer adaptive artifact must declare one layer")
        if self.kind == PolicyArtifactKind.JOINT and len(normalized) < 2:
            raise ValueError("joint artifact must declare at least two layers")
        object.__setattr__(self, "layers", normalized)
        if self.parent_artifact_id is not None:
            _require_identifier(self.parent_artifact_id, "parent policy artifact ID")

    @property
    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": MEMORY_POLICY_CONTRACT_SCHEMA_VERSION,
            "policy_version": self.policy_version,
            "kind": self.kind.value,
            "layers": [layer.value for layer in self.layers],
            "parent_artifact_id": self.parent_artifact_id,
        }

    @property
    def canonical_id(self) -> str:
        return f"policy-artifact.{content_digest(self.identity_payload)[:40]}"

    @property
    def is_canonical(self) -> bool:
        return self.artifact_id == self.canonical_id

    @classmethod
    def create(cls, *, policy_version: str, kind: PolicyArtifactKind | str, layers: Sequence[PolicyLayer | str], parent_artifact_id: str | None = None) -> "PolicyArtifactIdentity":
        normalized = tuple(sorted((PolicyLayer(layer) for layer in layers), key=lambda item: item.value))
        identity = {
            "schema_version": MEMORY_POLICY_CONTRACT_SCHEMA_VERSION,
            "policy_version": policy_version,
            "kind": PolicyArtifactKind(kind).value,
            "layers": [layer.value for layer in normalized],
            "parent_artifact_id": parent_artifact_id,
        }
        return cls(
            artifact_id=f"policy-artifact.{content_digest(identity)[:40]}",
            policy_version=policy_version,
            kind=kind,
            layers=normalized,
            parent_artifact_id=parent_artifact_id,
        )


@runtime_checkable
class MemoryFormationPolicy(Protocol):
    """Host-neutral formation policy interface."""

    @property
    def artifact_identity(self) -> PolicyArtifactIdentity: ...

    def decide_trigger(self, event: TriggerEvent) -> TriggerDecision: ...

    def select_source(self, event: TriggerEvent, safety: SafetyBoundary) -> SourceSelectionDecision: ...

    def decide_extraction(self, source: SourceSelectionDecision) -> ExtractionDecision: ...

    def decide_admission(self, extraction: ExtractionDecision, existing_revision: str | None = None) -> AdmissionDecision: ...

    def decide_commit(self, admission: AdmissionDecision, boundary: str) -> CommitDecision: ...


@runtime_checkable
class MemoryExposurePolicy(Protocol):
    """Host-neutral exposure policy interface."""

    @property
    def artifact_identity(self) -> PolicyArtifactIdentity: ...

    def decide_exposure(
        self,
        event: TriggerEvent,
        artifact_ids: Sequence[str],
        budget_tokens: int | None = None,
        safety: SafetyBoundary | None = None,
    ) -> ExposureDecision: ...


@dataclass(frozen=True, slots=True)
class PolicyAuditReport:
    ok: bool
    errors: tuple[str, ...] = ()
    lineage_id: str | None = None
    decision_ids: tuple[str, ...] = ()


def validate_policy_episode(
    decisions: Sequence[PolicyDecision],
    *,
    mutation_receipt_ids: Sequence[str] = (),
    injection_receipt_ids: Sequence[str] = (),
    future_feedback_ids: Sequence[str] = (),
    require_all_layers: bool = True,
) -> PolicyAuditReport:
    """Validate the content-free audit envelope for one episode.

    This validator intentionally does not inspect memory content.  It rejects
    missing decision metadata, inconsistent lineage, and missing execution
    receipts before any downstream learner or report can consume the episode.
    """

    errors: list[str] = []
    items = tuple(decisions)
    if not items:
        return PolicyAuditReport(False, ("missing policy decisions",))
    lineage_ids = {item.lineage_id for item in items}
    if len(lineage_ids) != 1:
        errors.append("decisions do not share one lineage")
    layers = {item.layer for item in items}
    if require_all_layers and not set(FORMATION_LAYERS).issubset(layers):
        missing = sorted(layer.value for layer in set(FORMATION_LAYERS).difference(layers))
        errors.append(f"missing decisions: {','.join(missing)}")
    for item in items:
        if not item.source_revision:
            errors.append(f"{item.decision_id}: missing source revision")
        if _DIGEST.fullmatch(item.input_digest) is None or _DIGEST.fullmatch(item.output_digest) is None:
            errors.append(f"{item.decision_id}: missing decision digest")
        if not item.policy_version:
            errors.append(f"{item.decision_id}: missing policy identity")
        if item.action == DecisionAction.RUN and item.execution_status == ExecutionStatus.EXECUTED and not item.execution_receipt_id:
            errors.append(f"{item.decision_id}: missing execution receipt")
        if item.action in {DecisionAction.SKIP, DecisionAction.DEFER} and item.execution_receipt_id:
            errors.append(f"{item.decision_id}: non-executing decision has execution receipt")
    for values, name in ((mutation_receipt_ids, "mutation receipt IDs"), (injection_receipt_ids, "injection receipt IDs"), (future_feedback_ids, "future feedback IDs")):
        try:
            _tuple_strings(values, name)
        except ValueError as exc:
            errors.append(str(exc))
    return PolicyAuditReport(not errors, tuple(errors), next(iter(lineage_ids), None), tuple(item.decision_id for item in items))


def audit_policy_episode(*args: object, **kwargs: object) -> PolicyAuditReport:
    report = validate_policy_episode(*args, **kwargs)  # type: ignore[arg-type]
    if not report.ok:
        raise ValueError("; ".join(report.errors))
    return report


def validate_policy_lineage(
    lineage: PolicyLineage,
    decisions: Sequence[PolicyDecision],
    *,
    mutation_receipt_ids: Sequence[str] = (),
    injection_receipt_ids: Sequence[str] = (),
    future_feedback_ids: Sequence[str] = (),
    require_all_layers: bool = True,
) -> PolicyAuditReport:
    """Verify that decisions and downstream receipts form one stable join."""

    report = validate_policy_episode(
        decisions,
        mutation_receipt_ids=mutation_receipt_ids,
        injection_receipt_ids=injection_receipt_ids,
        future_feedback_ids=future_feedback_ids,
        require_all_layers=require_all_layers,
    )
    errors = list(report.errors)
    if report.lineage_id != lineage.lineage_id:
        errors.append("lineage ID does not match decisions")
    decision_ids = tuple(item.decision_id for item in decisions)
    if lineage.decision_ids and set(lineage.decision_ids) != set(decision_ids):
        errors.append("lineage decision IDs do not match decisions")
    trigger_ids = {item.trigger_event_id for item in decisions if item.trigger_event_id is not None}
    if trigger_ids != {lineage.trigger_event_id}:
        errors.append("lineage trigger event does not match decisions")
    for values, expected, name in (
        (mutation_receipt_ids, lineage.mutation_receipt_ids, "mutation receipt IDs"),
        (injection_receipt_ids, lineage.injection_receipt_ids, "injection receipt IDs"),
        (future_feedback_ids, lineage.future_feedback_ids, "future feedback IDs"),
    ):
        if expected and set(expected) != set(values):
            errors.append(f"lineage {name} do not match evidence")
    return PolicyAuditReport(not errors, tuple(errors), lineage.lineage_id, decision_ids)


def audit_policy_lineage(*args: object, **kwargs: object) -> PolicyAuditReport:
    report = validate_policy_lineage(*args, **kwargs)  # type: ignore[arg-type]
    if not report.ok:
        raise ValueError("; ".join(report.errors))
    return report


__all__ = [
    "MEMORY_POLICY_CONTRACT_SCHEMA_VERSION",
    "MEMORY_POLICY_CONTRACT_SCHEMA",
    "canonical_json",
    "content_digest",
    "PolicyLayer",
    "FORMATION_LAYERS",
    "DecisionAction",
    "ExecutionStatus",
    "ProjectionMode",
    "MutationKind",
    "CommitMode",
    "ExposureMode",
    "PolicyArtifactKind",
    "TriggerEvent",
    "SafetyBoundary",
    "PolicyDecision",
    "TriggerDecision",
    "SourceSelectionDecision",
    "ExtractionDecision",
    "AdmissionDecision",
    "CommitDecision",
    "ExposureDecision",
    "PolicyLineage",
    "PolicyArtifactIdentity",
    "MemoryFormationPolicy",
    "MemoryExposurePolicy",
    "PolicyAuditReport",
    "validate_policy_episode",
    "audit_policy_episode",
    "validate_policy_lineage",
    "audit_policy_lineage",
]
