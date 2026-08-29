"""Extraction-owned delayed feedback contracts and deterministic labels."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Mapping, Protocol, runtime_checkable


EXTRACTION_FEEDBACK_SCHEMA_VERSION = 2
EXTRACTION_FEEDBACK_SCHEMA = "extraction-feedback-v2"
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$")
_SEMANTIC_KEY = re.compile(r"^[a-z][a-z0-9_.:-]{0,127}$")


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _stable_id(prefix: str, value: object) -> str:
    return f"{prefix}.{_digest(value)[:40]}"


def _require_id(value: str, name: str) -> None:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{name} must be a stable identifier")


def _require_digest(value: str, name: str) -> None:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{name} must be sha256")


def _require_key(value: str) -> None:
    if not isinstance(value, str) or _SEMANTIC_KEY.fullmatch(value) is None:
        raise ValueError("semantic key must be normalized and stable")


def _require_unique(values: tuple[str, ...], name: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must be unique")


def _normalized_recipient_id(value: str) -> str:
    """Normalize human-readable recipient names for deterministic contracts."""

    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


class ExtractionFeedbackLabel(StrEnum):
    USEFUL = "useful"
    HARMFUL = "harmful"
    MISSED = "missed"
    UNRESOLVED = "unresolved"
    CENSORED = "censored"


class ExtractionFeedbackLevel(StrEnum):
    SOURCE = "source"
    EXTRACTION_SET = "extraction_set"
    FACT = "fact"


class ExtractionSetStatus(StrEnum):
    NONEMPTY = "nonempty"
    EMPTY = "empty"
    FILTERED = "filtered"
    NONE = "none"
    MUTATION_FAILED = "mutation_failed"


class FactDisposition(StrEnum):
    PERSISTED = "persisted"
    FILTERED = "filtered"
    NONE = "none"
    MUTATION_FAILED = "mutation_failed"


class ExtractionQualityIssue(StrEnum):
    UNSUPPORTED = "unsupported"
    TRANSIENT = "transient"
    CONFLICTING = "conflicting"


class ExposureMode(StrEnum):
    EAGER_SYSTEM_PROMPT = "eager_system_prompt"
    SELECTIVE_RETRIEVAL = "selective_retrieval"
    NOT_EXPOSED = "not_exposed"


class AttributionConfidence(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


class DeploymentSurface(StrEnum):
    CURRENT_INPUT = "current_input"
    FINAL_RESPONSE = "final_response"
    TOOL_EVENT = "tool_event"
    TASK_COMPLETION = "task_completion"


@dataclass(frozen=True, slots=True)
class OpportunityContract:
    contract_id: str
    family_id: str
    eligible_stages: tuple[str, ...]
    memory_scope_keys: tuple[str, ...]
    allowed_surfaces: tuple[DeploymentSurface, ...]
    ambiguity_semantics: str = "set_level_unless_unique"
    schema_version: int = EXTRACTION_FEEDBACK_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_id(self.contract_id, "opportunity contract ID")
        _require_id(self.family_id, "opportunity family ID")
        if self.schema_version != EXTRACTION_FEEDBACK_SCHEMA_VERSION:
            raise ValueError("unsupported opportunity contract schema")
        object.__setattr__(
            self,
            "allowed_surfaces",
            tuple(DeploymentSurface(value) for value in self.allowed_surfaces),
        )
        _require_unique(self.eligible_stages, "eligible stages")
        _require_unique(self.memory_scope_keys, "memory scope keys")
        if not self.eligible_stages or not self.memory_scope_keys:
            raise ValueError("opportunity contract requires stages and scope keys")
        for value in self.memory_scope_keys:
            _require_key(value)
        if set(self.allowed_surfaces) != {
            DeploymentSurface.CURRENT_INPUT,
            DeploymentSurface.TASK_COMPLETION,
        }:
            raise ValueError("opportunity contract surfaces are not least privilege")
        if self.ambiguity_semantics != "set_level_unless_unique":
            raise ValueError("unsupported opportunity ambiguity semantics")

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "contract_id": self.contract_id,
            "family_id": self.family_id,
            "eligible_stages": list(self.eligible_stages),
            "memory_scope_keys": list(self.memory_scope_keys),
            "allowed_surfaces": [value.value for value in self.allowed_surfaces],
            "ambiguity_semantics": self.ambiguity_semantics,
        }


@dataclass(frozen=True, slots=True)
class UseContract:
    contract_id: str
    family_id: str
    parser_id: str
    allowed_surfaces: tuple[DeploymentSurface, ...]
    schema_version: int = EXTRACTION_FEEDBACK_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for value, name in (
            (self.contract_id, "use contract ID"),
            (self.family_id, "use contract family ID"),
            (self.parser_id, "use parser ID"),
        ):
            _require_id(value, name)
        if self.schema_version != EXTRACTION_FEEDBACK_SCHEMA_VERSION:
            raise ValueError("unsupported use contract schema")
        object.__setattr__(
            self,
            "allowed_surfaces",
            tuple(DeploymentSurface(value) for value in self.allowed_surfaces),
        )
        _require_unique(
            tuple(value.value for value in self.allowed_surfaces),
            "use contract surfaces",
        )
        if not self.allowed_surfaces or DeploymentSurface.TASK_COMPLETION in self.allowed_surfaces:
            raise ValueError("use contract surfaces are invalid")

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "contract_id": self.contract_id,
            "family_id": self.family_id,
            "parser_id": self.parser_id,
            "allowed_surfaces": [value.value for value in self.allowed_surfaces],
        }


@dataclass(frozen=True, slots=True)
class OutcomeContract:
    contract_id: str
    family_id: str
    parser_id: str
    allowed_surfaces: tuple[DeploymentSurface, ...]
    schema_version: int = EXTRACTION_FEEDBACK_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for value, name in (
            (self.contract_id, "outcome contract ID"),
            (self.family_id, "outcome contract family ID"),
            (self.parser_id, "outcome parser ID"),
        ):
            _require_id(value, name)
        if self.schema_version != EXTRACTION_FEEDBACK_SCHEMA_VERSION:
            raise ValueError("unsupported outcome contract schema")
        object.__setattr__(
            self,
            "allowed_surfaces",
            tuple(DeploymentSurface(value) for value in self.allowed_surfaces),
        )
        _require_unique(
            tuple(value.value for value in self.allowed_surfaces),
            "outcome contract surfaces",
        )
        if DeploymentSurface.TASK_COMPLETION not in self.allowed_surfaces:
            raise ValueError("outcome contract requires task completion evidence")

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "contract_id": self.contract_id,
            "family_id": self.family_id,
            "parser_id": self.parser_id,
            "allowed_surfaces": [value.value for value in self.allowed_surfaces],
        }


@dataclass(frozen=True, slots=True)
class FamilyFeedbackContract:
    opportunity: OpportunityContract
    use: UseContract
    outcome: OutcomeContract
    contract_digest: str
    contract_schema: str = EXTRACTION_FEEDBACK_SCHEMA
    schema_version: int = EXTRACTION_FEEDBACK_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            self.schema_version != EXTRACTION_FEEDBACK_SCHEMA_VERSION
            or self.contract_schema != EXTRACTION_FEEDBACK_SCHEMA
        ):
            raise ValueError("unsupported family feedback contract schema")
        families = {
            self.opportunity.family_id,
            self.use.family_id,
            self.outcome.family_id,
        }
        if len(families) != 1:
            raise ValueError("family feedback contract components disagree")
        if self.contract_digest != _digest(self.identity_payload()):
            raise ValueError("family feedback contract digest mismatch")

    @classmethod
    def create(
        cls,
        opportunity: OpportunityContract,
        use: UseContract,
        outcome: OutcomeContract,
    ) -> "FamilyFeedbackContract":
        core = {
            "schema_version": EXTRACTION_FEEDBACK_SCHEMA_VERSION,
            "contract_schema": EXTRACTION_FEEDBACK_SCHEMA,
            "opportunity": opportunity.payload(),
            "use": use.payload(),
            "outcome": outcome.payload(),
        }
        return cls(opportunity, use, outcome, _digest(core))

    @property
    def family_id(self) -> str:
        return self.opportunity.family_id

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "contract_schema": self.contract_schema,
            "opportunity": self.opportunity.payload(),
            "use": self.use.payload(),
            "outcome": self.outcome.payload(),
        }


@dataclass(frozen=True, slots=True)
class ObservableToolEvent:
    event_id: str
    tool_name: str
    success: bool
    subject_ids: tuple[str, ...] = ()
    recipient_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_id(self.event_id, "tool event ID")
        _require_id(self.tool_name, "tool name")
        if type(self.success) is not bool:
            raise TypeError("tool event success must be bool")
        _require_unique(self.subject_ids, "tool event subjects")
        _require_unique(self.recipient_ids, "tool event recipients")

    def payload(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "tool_name": self.tool_name,
            "success": self.success,
            "subject_ids": list(self.subject_ids),
            "recipient_ids": list(self.recipient_ids),
        }

    @classmethod
    def from_payload(cls, value: object) -> "ObservableToolEvent":
        fields = {
            "event_id",
            "tool_name",
            "success",
            "subject_ids",
            "recipient_ids",
        }
        if (
            not isinstance(value, Mapping)
            or set(value) != fields
            or not isinstance(value["subject_ids"], list)
            or not isinstance(value["recipient_ids"], list)
        ):
            raise ValueError("malformed observable tool event")
        try:
            return cls(
                event_id=value["event_id"],
                tool_name=value["tool_name"],
                success=value["success"],
                subject_ids=tuple(value["subject_ids"]),
                recipient_ids=tuple(value["recipient_ids"]),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("malformed observable tool event") from exc


@dataclass(frozen=True, slots=True)
class DeploymentObservation:
    observation_id: str
    family_id: str
    stage: str
    task_id: str
    current_input_projection_digest: str
    current_input_semantic_keys: tuple[str, ...]
    task_semantic_keys: tuple[str, ...]
    final_response: str
    tool_events: tuple[ObservableToolEvent, ...]
    completed: bool
    observation_complete: bool = True
    censor_reason: str | None = None

    def __post_init__(self) -> None:
        for value, name in (
            (self.observation_id, "deployment observation ID"),
            (self.family_id, "deployment family ID"),
            (self.stage, "deployment stage"),
            (self.task_id, "deployment task ID"),
        ):
            _require_id(value, name)
        _require_digest(
            self.current_input_projection_digest,
            "current input projection digest",
        )
        for values in (self.current_input_semantic_keys, self.task_semantic_keys):
            _require_unique(values, "deployment semantic keys")
            for value in values:
                _require_key(value)
        if type(self.completed) is not bool or type(self.observation_complete) is not bool:
            raise TypeError("deployment observation flags must be bool")
        if self.observation_complete == (self.censor_reason is not None):
            raise ValueError("deployment censor reason must match completeness")

    def payload(self) -> dict[str, object]:
        return {
            "observation_id": self.observation_id,
            "family_id": self.family_id,
            "stage": self.stage,
            "task_id": self.task_id,
            "current_input_projection_digest": (
                self.current_input_projection_digest
            ),
            "current_input_semantic_keys": list(
                self.current_input_semantic_keys
            ),
            "task_semantic_keys": list(self.task_semantic_keys),
            "final_response": self.final_response,
            "tool_events": [value.payload() for value in self.tool_events],
            "completed": self.completed,
            "observation_complete": self.observation_complete,
            "censor_reason": self.censor_reason,
        }

    @classmethod
    def from_payload(cls, value: object) -> "DeploymentObservation":
        fields = {
            "observation_id",
            "family_id",
            "stage",
            "task_id",
            "current_input_projection_digest",
            "current_input_semantic_keys",
            "task_semantic_keys",
            "final_response",
            "tool_events",
            "completed",
            "observation_complete",
            "censor_reason",
        }
        if (
            not isinstance(value, Mapping)
            or set(value) != fields
            or any(
                not isinstance(value[field], list)
                for field in (
                    "current_input_semantic_keys",
                    "task_semantic_keys",
                    "tool_events",
                )
            )
        ):
            raise ValueError("malformed deployment observation")
        try:
            return cls(
                observation_id=value["observation_id"],
                family_id=value["family_id"],
                stage=value["stage"],
                task_id=value["task_id"],
                current_input_projection_digest=value[
                    "current_input_projection_digest"
                ],
                current_input_semantic_keys=tuple(
                    value["current_input_semantic_keys"]
                ),
                task_semantic_keys=tuple(value["task_semantic_keys"]),
                final_response=value["final_response"],
                tool_events=tuple(
                    ObservableToolEvent.from_payload(item)
                    for item in value["tool_events"]
                ),
                completed=value["completed"],
                observation_complete=value["observation_complete"],
                censor_reason=value["censor_reason"],
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("malformed deployment observation") from exc


@dataclass(frozen=True, slots=True)
class ArtifactSemanticBinding:
    artifact_id: str
    semantic_keys: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_id(self.artifact_id, "memory artifact ID")
        values = (
            (self.semantic_keys,)
            if isinstance(self.semantic_keys, str)
            else tuple(self.semantic_keys)
        )
        object.__setattr__(self, "semantic_keys", values)
        _require_unique(values, "artifact semantic keys")
        if not values:
            raise ValueError("memory artifact binding requires semantic keys")
        for value in values:
            _require_key(value)

    @property
    def semantic_key(self) -> str | None:
        return self.semantic_keys[0] if len(self.semantic_keys) == 1 else None


@dataclass(frozen=True, slots=True)
class FutureMemoryEvidence:
    future_opportunity_id: str
    exposure_mode: ExposureMode
    artifact_bindings: tuple[ArtifactSemanticBinding, ...]
    opportunity_operation_id: str | None
    injection_operation_id: str | None

    def __post_init__(self) -> None:
        _require_id(self.future_opportunity_id, "future opportunity ID")
        object.__setattr__(self, "exposure_mode", ExposureMode(self.exposure_mode))
        artifact_ids = tuple(value.artifact_id for value in self.artifact_bindings)
        _require_unique(artifact_ids, "future memory artifact IDs")
        for value in (self.opportunity_operation_id, self.injection_operation_id):
            if value is not None:
                _require_id(value, "future operation ID")
        if self.exposure_mode == ExposureMode.NOT_EXPOSED:
            if self.injection_operation_id is not None:
                raise ValueError("not-exposed evidence cannot carry injection")
        elif self.injection_operation_id is None:
            raise ValueError("exposed memory evidence requires injection operation")


@dataclass(frozen=True, slots=True)
class ExtractedFactEvidence:
    fact_id: str
    semantic_keys: tuple[str, ...]
    disposition: FactDisposition
    artifact_id: str | None = None
    quality_issue: ExtractionQualityIssue | None = None

    def __post_init__(self) -> None:
        _require_id(self.fact_id, "extracted fact ID")
        values = (
            (self.semantic_keys,)
            if isinstance(self.semantic_keys, str)
            else tuple(self.semantic_keys)
        )
        object.__setattr__(self, "semantic_keys", values)
        _require_unique(values, "extracted fact semantic keys")
        for value in values:
            _require_key(value)
        object.__setattr__(self, "disposition", FactDisposition(self.disposition))
        if self.artifact_id is not None:
            _require_id(self.artifact_id, "extracted fact artifact ID")
        if self.quality_issue is not None:
            object.__setattr__(
                self,
                "quality_issue",
                ExtractionQualityIssue(self.quality_issue),
            )
        if (self.disposition == FactDisposition.PERSISTED) != (
            self.artifact_id is not None
        ):
            raise ValueError("persisted fact disposition must match artifact identity")

    @property
    def semantic_key(self) -> str | None:
        return self.semantic_keys[0] if len(self.semantic_keys) == 1 else None


@dataclass(frozen=True, slots=True)
class ExtractionSourceEvidence:
    source_id: str
    source_projection_digest: str
    extraction_set_id: str
    status: ExtractionSetStatus
    available_semantic_keys: tuple[str, ...]
    facts: tuple[ExtractedFactEvidence, ...]

    def __post_init__(self) -> None:
        _require_id(self.source_id, "extraction source ID")
        _require_digest(self.source_projection_digest, "source projection digest")
        _require_id(self.extraction_set_id, "extraction set ID")
        object.__setattr__(self, "status", ExtractionSetStatus(self.status))
        _require_unique(self.available_semantic_keys, "source semantic keys")
        for value in self.available_semantic_keys:
            _require_key(value)
        fact_ids = tuple(value.fact_id for value in self.facts)
        _require_unique(fact_ids, "extracted fact IDs")
        if self.status == ExtractionSetStatus.EMPTY and self.facts:
            raise ValueError("empty extraction set cannot carry facts")
        if self.status == ExtractionSetStatus.NONEMPTY and not self.facts:
            raise ValueError("nonempty extraction set requires facts")

    def payload(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "source_projection_digest": self.source_projection_digest,
            "extraction_set_id": self.extraction_set_id,
            "status": self.status.value,
            "available_semantic_keys": list(self.available_semantic_keys),
            "facts": [{
                "fact_id": fact.fact_id,
                "semantic_keys": list(fact.semantic_keys),
                "disposition": fact.disposition.value,
                "artifact_id": fact.artifact_id,
                "quality_issue": (
                    fact.quality_issue.value
                    if fact.quality_issue is not None
                    else None
                ),
            } for fact in self.facts],
        }

    @classmethod
    def from_payload(cls, value: object) -> "ExtractionSourceEvidence":
        fields = {
            "source_id",
            "source_projection_digest",
            "extraction_set_id",
            "status",
            "available_semantic_keys",
            "facts",
        }
        if not isinstance(value, dict) or set(value) != fields:
            raise ValueError("malformed extraction source evidence")
        raw_facts = value["facts"]
        available = value["available_semantic_keys"]
        if not isinstance(raw_facts, list) or not isinstance(available, list):
            raise ValueError("extraction source collections must be lists")
        if any(not isinstance(item, str) for item in available) or any(
            not isinstance(value[field], str)
            for field in (
                "source_id",
                "source_projection_digest",
                "extraction_set_id",
                "status",
            )
        ):
            raise ValueError("extraction source scalar fields are invalid")
        fact_fields = {
            "fact_id",
            "semantic_keys",
            "disposition",
            "artifact_id",
            "quality_issue",
        }
        facts = []
        for item in raw_facts:
            if not isinstance(item, dict) or set(item) != fact_fields:
                raise ValueError("malformed extracted fact evidence")
            semantic_keys = item["semantic_keys"]
            if (
                not isinstance(item["fact_id"], str)
                or not isinstance(item["disposition"], str)
                or not isinstance(semantic_keys, list)
                or any(not isinstance(key, str) for key in semantic_keys)
                or (
                    item["artifact_id"] is not None
                    and not isinstance(item["artifact_id"], str)
                )
                or (
                    item["quality_issue"] is not None
                    and not isinstance(item["quality_issue"], str)
                )
            ):
                raise ValueError("extracted fact semantic keys must be a list")
            facts.append(ExtractedFactEvidence(
                item["fact_id"],
                tuple(semantic_keys),
                FactDisposition(item["disposition"]),
                artifact_id=item["artifact_id"],
                quality_issue=(
                    ExtractionQualityIssue(item["quality_issue"])
                    if item["quality_issue"] is not None
                    else None
                ),
            ))
        return cls(
            value["source_id"],
            value["source_projection_digest"],
            value["extraction_set_id"],
            ExtractionSetStatus(value["status"]),
            tuple(available),
            tuple(facts),
        )


@dataclass(frozen=True, slots=True)
class MissedExtractionEvidence:
    missed_id: str
    semantic_key: str
    source_span_digest: str
    future_opportunity_id: str
    absence_outcome_operation_id: str
    deterministically_attributed: bool

    def __post_init__(self) -> None:
        _require_id(self.missed_id, "missed extraction ID")
        _require_key(self.semantic_key)
        _require_digest(self.source_span_digest, "missed source span digest")
        _require_id(self.future_opportunity_id, "missed future opportunity ID")
        _require_id(
            self.absence_outcome_operation_id,
            "missed absence outcome operation ID",
        )
        if type(self.deterministically_attributed) is not bool:
            raise TypeError("missed attribution flag must be bool")

    @classmethod
    def create(
        cls,
        *,
        semantic_key: str,
        source_span_digest: str,
        future_opportunity_id: str,
        absence_outcome_operation_id: str,
    ) -> "MissedExtractionEvidence":
        identity = {
            "semantic_key": semantic_key,
            "source_span_digest": source_span_digest,
            "future_opportunity_id": future_opportunity_id,
            "absence_outcome_operation_id": absence_outcome_operation_id,
            "deterministically_attributed": True,
        }
        return cls(
            _stable_id("missed-extraction", identity),
            semantic_key,
            source_span_digest,
            future_opportunity_id,
            absence_outcome_operation_id,
            True,
        )


@dataclass(frozen=True, slots=True)
class ContractResolution:
    opportunity_observed: bool
    current_input_confounded: bool
    explicit_use: bool
    successful_outcome: bool | None
    harmful_outcome: bool
    used_artifact_ids: tuple[str, ...]
    opportunity_operation_id: str | None
    use_operation_id: str | None
    outcome_operation_id: str | None
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for value in (
            self.opportunity_observed,
            self.current_input_confounded,
            self.explicit_use,
            self.harmful_outcome,
        ):
            if type(value) is not bool:
                raise TypeError("feedback resolution flags must be bool")
        if self.successful_outcome is not None and type(self.successful_outcome) is not bool:
            raise TypeError("feedback outcome must be bool or unknown")
        _require_unique(self.used_artifact_ids, "used artifact IDs")
        for value in (
            self.opportunity_operation_id,
            self.use_operation_id,
            self.outcome_operation_id,
        ):
            if value is not None:
                _require_id(value, "feedback operation ID")
        _require_unique(self.reason_codes, "feedback reason codes")
        if self.explicit_use and self.use_operation_id is None:
            raise ValueError("explicit use requires an operation identity")
        if self.successful_outcome is not None and self.outcome_operation_id is None:
            raise ValueError("known outcome requires an operation identity")


@dataclass(frozen=True, slots=True)
class FeedbackOperationJoin:
    opportunity_operation_id: str
    use_operation_id: str
    outcome_operation_id: str

    def __post_init__(self) -> None:
        for value in (
            self.opportunity_operation_id,
            self.use_operation_id,
            self.outcome_operation_id,
        ):
            _require_id(value, "feedback operation join ID")


@runtime_checkable
class FamilyFeedbackResolver(Protocol):
    @property
    def contract(self) -> FamilyFeedbackContract: ...

    def resolve(
        self,
        observation: DeploymentObservation,
        future: FutureMemoryEvidence,
    ) -> ContractResolution: ...


class FeedbackContractRegistry:
    def __init__(self) -> None:
        self._resolvers: dict[str, FamilyFeedbackResolver] = {}

    def register(self, resolver: FamilyFeedbackResolver) -> None:
        family_id = resolver.contract.family_id
        if family_id in self._resolvers:
            raise ValueError(f"feedback family already registered: {family_id}")
        self._resolvers[family_id] = resolver

    def resolver(self, family_id: str) -> FamilyFeedbackResolver:
        try:
            return self._resolvers[family_id]
        except KeyError as exc:
            raise KeyError(f"unregistered feedback family: {family_id}") from exc

    def resolve(
        self,
        observation: DeploymentObservation,
        future: FutureMemoryEvidence,
    ) -> tuple[FamilyFeedbackContract, ContractResolution]:
        resolver = self.resolver(observation.family_id)
        return resolver.contract, resolver.resolve(observation, future)


@dataclass(frozen=True, slots=True)
class ExtractionFeedbackExample:
    example_id: str
    primary_unit_id: str
    level: ExtractionFeedbackLevel
    primary: bool
    label: ExtractionFeedbackLabel
    source_id: str
    extraction_set_id: str
    future_opportunity_id: str
    fact_id: str | None
    semantic_key: str | None
    artifact_ids: tuple[str, ...]
    exposure_mode: ExposureMode
    opportunity_operation_id: str | None
    use_operation_id: str | None
    outcome_operation_id: str | None
    attribution_confidence: AttributionConfidence
    reason_codes: tuple[str, ...]
    contract_digest: str
    schema_version: int = EXTRACTION_FEEDBACK_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for value in (self.example_id, self.primary_unit_id, self.source_id, self.extraction_set_id):
            _require_id(value, "extraction feedback identity")
        _require_id(self.future_opportunity_id, "future opportunity ID")
        object.__setattr__(self, "level", ExtractionFeedbackLevel(self.level))
        object.__setattr__(self, "label", ExtractionFeedbackLabel(self.label))
        object.__setattr__(self, "exposure_mode", ExposureMode(self.exposure_mode))
        object.__setattr__(
            self,
            "attribution_confidence",
            AttributionConfidence(self.attribution_confidence),
        )
        if type(self.primary) is not bool:
            raise TypeError("feedback primary flag must be bool")
        if self.primary != (self.level == ExtractionFeedbackLevel.EXTRACTION_SET):
            raise ValueError("only extraction-set feedback is primary")
        if (self.level == ExtractionFeedbackLevel.FACT) != (self.fact_id is not None):
            raise ValueError("fact-level feedback requires a fact identity")
        if self.fact_id is not None:
            _require_id(self.fact_id, "feedback fact ID")
        if self.semantic_key is not None:
            _require_key(self.semantic_key)
        _require_unique(self.artifact_ids, "feedback artifact IDs")
        _require_unique(self.reason_codes, "feedback reason codes")
        _require_digest(self.contract_digest, "feedback contract digest")
        expected_primary = _stable_id("feedback-unit", {
            "source_id": self.source_id,
            "extraction_set_id": self.extraction_set_id,
            "future_opportunity_id": self.future_opportunity_id,
        })
        if self.primary_unit_id != expected_primary:
            raise ValueError("feedback primary unit ID mismatch")
        expected_example = _stable_id("feedback-example", {
            "primary_unit_id": self.primary_unit_id,
            "level": self.level.value,
            "fact_id": self.fact_id,
            "label": self.label.value,
            "contract_digest": self.contract_digest,
        })
        if self.example_id != expected_example:
            raise ValueError("feedback example ID mismatch")


@dataclass(frozen=True, slots=True)
class ExtractionFeedbackDataset:
    dataset_id: str
    source_projection_digest: str
    contract_digest: str
    examples: tuple[ExtractionFeedbackExample, ...]
    schema_version: int = EXTRACTION_FEEDBACK_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_id(self.dataset_id, "extraction feedback dataset ID")
        _require_digest(self.source_projection_digest, "feedback source digest")
        _require_digest(self.contract_digest, "feedback contract digest")
        ids = tuple(value.example_id for value in self.examples)
        _require_unique(ids, "feedback example IDs")
        if sum(value.primary for value in self.examples) != 1:
            raise ValueError("feedback dataset requires exactly one primary unit")
        expected = _stable_id("extraction-feedback", {
            "source_projection_digest": self.source_projection_digest,
            "contract_digest": self.contract_digest,
            "examples": list(ids),
        })
        if self.dataset_id != expected:
            raise ValueError("extraction feedback dataset ID mismatch")

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "source_projection_digest": self.source_projection_digest,
            "contract_digest": self.contract_digest,
            "examples": [{
                "example_id": example.example_id,
                "primary_unit_id": example.primary_unit_id,
                "level": example.level.value,
                "primary": example.primary,
                "label": example.label.value,
                "source_id": example.source_id,
                "extraction_set_id": example.extraction_set_id,
                "future_opportunity_id": example.future_opportunity_id,
                "fact_id": example.fact_id,
                "semantic_key": example.semantic_key,
                "artifact_ids": list(example.artifact_ids),
                "exposure_mode": example.exposure_mode.value,
                "opportunity_operation_id": example.opportunity_operation_id,
                "use_operation_id": example.use_operation_id,
                "outcome_operation_id": example.outcome_operation_id,
                "attribution_confidence": example.attribution_confidence.value,
                "reason_codes": list(example.reason_codes),
                "contract_digest": example.contract_digest,
            } for example in self.examples],
        }

    @classmethod
    def from_payload(cls, value: object) -> "ExtractionFeedbackDataset":
        fields = {
            "schema_version",
            "dataset_id",
            "source_projection_digest",
            "contract_digest",
            "examples",
        }
        example_fields = {
            "example_id",
            "primary_unit_id",
            "level",
            "primary",
            "label",
            "source_id",
            "extraction_set_id",
            "future_opportunity_id",
            "fact_id",
            "semantic_key",
            "artifact_ids",
            "exposure_mode",
            "opportunity_operation_id",
            "use_operation_id",
            "outcome_operation_id",
            "attribution_confidence",
            "reason_codes",
            "contract_digest",
        }
        if (
            not isinstance(value, dict)
            or set(value) != fields
            or type(value["schema_version"]) is not int
            or not isinstance(value["examples"], list)
        ):
            raise ValueError("malformed extraction feedback dataset")
        examples = []
        for item in value["examples"]:
            if not isinstance(item, dict) or set(item) != example_fields:
                raise ValueError("malformed extraction feedback example")
            artifact_ids = item["artifact_ids"]
            reason_codes = item["reason_codes"]
            if (
                not isinstance(artifact_ids, list)
                or any(not isinstance(entry, str) for entry in artifact_ids)
                or not isinstance(reason_codes, list)
                or any(not isinstance(entry, str) for entry in reason_codes)
            ):
                raise ValueError("malformed extraction feedback collections")
            try:
                examples.append(ExtractionFeedbackExample(
                    example_id=item["example_id"],
                    primary_unit_id=item["primary_unit_id"],
                    level=item["level"],
                    primary=item["primary"],
                    label=item["label"],
                    source_id=item["source_id"],
                    extraction_set_id=item["extraction_set_id"],
                    future_opportunity_id=item["future_opportunity_id"],
                    fact_id=item["fact_id"],
                    semantic_key=item["semantic_key"],
                    artifact_ids=tuple(artifact_ids),
                    exposure_mode=item["exposure_mode"],
                    opportunity_operation_id=item["opportunity_operation_id"],
                    use_operation_id=item["use_operation_id"],
                    outcome_operation_id=item["outcome_operation_id"],
                    attribution_confidence=item["attribution_confidence"],
                    reason_codes=tuple(reason_codes),
                    contract_digest=item["contract_digest"],
                    schema_version=value["schema_version"],
                ))
            except (TypeError, ValueError) as exc:
                raise ValueError("malformed extraction feedback example") from exc
        try:
            return cls(
                value["dataset_id"],
                value["source_projection_digest"],
                value["contract_digest"],
                tuple(examples),
                schema_version=value["schema_version"],
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("malformed extraction feedback dataset") from exc


class ExtractionFeedbackBuilder:
    def __init__(self, registry: FeedbackContractRegistry) -> None:
        self.registry = registry

    def derive_missed(
        self,
        source: ExtractionSourceEvidence,
        observation: DeploymentObservation,
        future: FutureMemoryEvidence,
        *,
        operation_join: FeedbackOperationJoin,
    ) -> tuple[MissedExtractionEvidence, ...]:
        """Derive only exact family-contract absence attribution."""

        contract, resolution = self.registry.resolve(observation, future)
        if observation.stage not in contract.opportunity.eligible_stages:
            raise ValueError("feedback observation stage is not contract-eligible")
        if future.opportunity_operation_id != operation_join.opportunity_operation_id:
            raise ValueError("missed opportunity operation join mismatch")
        resolution = replace(
            resolution,
            opportunity_operation_id=operation_join.opportunity_operation_id,
            use_operation_id=operation_join.use_operation_id,
            outcome_operation_id=operation_join.outcome_operation_id,
        )
        extracted_keys = {
            semantic_key
            for fact in source.facts
            for semantic_key in fact.semantic_keys
        }
        # A non-empty unclassified fact may be a paraphrase or one part of a
        # set-level expression of the target rule.  Without a deterministic
        # equivalence proof, treating the absent key as an extraction miss
        # would incorrectly move retrieval/exposure uncertainty upstream.
        if any(not fact.semantic_keys for fact in source.facts):
            return ()
        bound_keys = {
            semantic_key
            for binding in future.artifact_bindings
            for semantic_key in binding.semantic_keys
        }
        missing_keys = tuple(
            semantic_key
            for semantic_key in source.available_semantic_keys
            if semantic_key in contract.opportunity.memory_scope_keys
            and semantic_key not in extracted_keys
            and semantic_key not in bound_keys
        )
        if (
            not missing_keys
            or not observation.observation_complete
            or resolution.current_input_confounded
            or not resolution.opportunity_observed
            or resolution.explicit_use
            or resolution.successful_outcome is not False
        ):
            return ()
        return tuple(
            MissedExtractionEvidence.create(
                semantic_key=semantic_key,
                source_span_digest=source.source_projection_digest,
                future_opportunity_id=future.future_opportunity_id,
                absence_outcome_operation_id=operation_join.outcome_operation_id,
            )
            for semantic_key in missing_keys
        )

    def build(
        self,
        source: ExtractionSourceEvidence,
        observation: DeploymentObservation,
        future: FutureMemoryEvidence,
        *,
        missed: tuple[MissedExtractionEvidence, ...] = (),
        operation_join: FeedbackOperationJoin | None = None,
    ) -> ExtractionFeedbackDataset:
        contract, resolution = self.registry.resolve(observation, future)
        if observation.family_id != contract.family_id:
            raise ValueError("feedback observation family differs from contract")
        if observation.stage not in contract.opportunity.eligible_stages:
            raise ValueError("feedback observation stage is not contract-eligible")
        bound_keys = {
            semantic_key
            for binding in future.artifact_bindings
            for semantic_key in binding.semantic_keys
        }
        if bound_keys - set(contract.opportunity.memory_scope_keys):
            raise ValueError("future memory evidence escapes contract scope")
        source_keys_by_artifact: dict[str, set[str]] = {}
        for fact in source.facts:
            if fact.artifact_id is not None:
                source_keys_by_artifact.setdefault(fact.artifact_id, set()).update(
                    fact.semantic_keys
                )
        for binding in future.artifact_bindings:
            source_keys = source_keys_by_artifact.get(binding.artifact_id)
            if source_keys is not None and not set(binding.semantic_keys).issubset(
                source_keys
            ):
                raise ValueError(
                    "future memory binding disagrees with source artifact semantics"
                )
        source_used_artifact_ids = tuple(
            artifact_id
            for artifact_id in resolution.used_artifact_ids
            if artifact_id in source_keys_by_artifact
        )
        if source_used_artifact_ids != resolution.used_artifact_ids:
            resolution = replace(
                resolution,
                used_artifact_ids=source_used_artifact_ids,
            )
        if operation_join is not None:
            if (
                future.opportunity_operation_id
                != operation_join.opportunity_operation_id
            ):
                raise ValueError("feedback opportunity operation join mismatch")
            resolution = replace(
                resolution,
                opportunity_operation_id=operation_join.opportunity_operation_id,
                use_operation_id=operation_join.use_operation_id,
                outcome_operation_id=operation_join.outcome_operation_id,
            )
        primary_id = _stable_id("feedback-unit", {
            "source_id": source.source_id,
            "extraction_set_id": source.extraction_set_id,
            "future_opportunity_id": future.future_opportunity_id,
        })
        label, confidence, reasons = self._set_label(
            source,
            observation,
            future,
            resolution,
            missed,
        )
        common = {
            "primary_unit_id": primary_id,
            "label": label,
            "source_id": source.source_id,
            "extraction_set_id": source.extraction_set_id,
            "future_opportunity_id": future.future_opportunity_id,
            "artifact_ids": tuple(
                binding.artifact_id for binding in future.artifact_bindings
            ),
            "exposure_mode": future.exposure_mode,
            "opportunity_operation_id": resolution.opportunity_operation_id,
            "use_operation_id": resolution.use_operation_id,
            "outcome_operation_id": resolution.outcome_operation_id,
            "attribution_confidence": confidence,
            "reason_codes": reasons,
            "contract_digest": contract.contract_digest,
        }
        examples = [
            self._example(ExtractionFeedbackLevel.SOURCE, False, None, None, **common),
            self._example(
                ExtractionFeedbackLevel.EXTRACTION_SET,
                True,
                None,
                None,
                **common,
            ),
        ]
        facts_by_artifact: dict[str, list[ExtractedFactEvidence]] = {}
        for fact in source.facts:
            if fact.artifact_id is not None:
                facts_by_artifact.setdefault(fact.artifact_id, []).append(fact)
        used_facts = (
            facts_by_artifact.get(resolution.used_artifact_ids[0], [])
            if len(resolution.used_artifact_ids) == 1
            else []
        )
        uniquely_used = (
            used_facts[0]
            if len(used_facts) == 1
            else None
        )
        for fact in source.facts:
            fact_label = ExtractionFeedbackLabel.UNRESOLVED
            fact_confidence = AttributionConfidence.NONE
            fact_reasons = ("fact_contribution_ambiguous",)
            if fact.quality_issue is not None:
                fact_label = ExtractionFeedbackLabel.HARMFUL
                fact_confidence = AttributionConfidence.HIGH
                fact_reasons = (f"extraction_{fact.quality_issue.value}",)
            elif uniquely_used is fact and label in {
                ExtractionFeedbackLabel.USEFUL,
                ExtractionFeedbackLabel.HARMFUL,
            }:
                fact_label = label
                fact_confidence = AttributionConfidence.HIGH
                fact_reasons = reasons
            examples.append(self._example(
                ExtractionFeedbackLevel.FACT,
                False,
                fact.fact_id,
                fact.semantic_key,
                **{
                    **common,
                    "label": fact_label,
                    "artifact_ids": (
                        (fact.artifact_id,) if fact.artifact_id is not None else ()
                    ),
                    "attribution_confidence": fact_confidence,
                    "reason_codes": fact_reasons,
                },
            ))
        dataset_id = _stable_id("extraction-feedback", {
            "source_projection_digest": source.source_projection_digest,
            "contract_digest": contract.contract_digest,
            "examples": [value.example_id for value in examples],
        })
        return ExtractionFeedbackDataset(
            dataset_id,
            source.source_projection_digest,
            contract.contract_digest,
            tuple(examples),
        )

    @staticmethod
    def _example(
        level: ExtractionFeedbackLevel,
        primary: bool,
        fact_id: str | None,
        semantic_key: str | None,
        **values: object,
    ) -> ExtractionFeedbackExample:
        identity = {
            "primary_unit_id": values["primary_unit_id"],
            "level": level.value,
            "fact_id": fact_id,
            "label": ExtractionFeedbackLabel(values["label"]).value,
            "contract_digest": values["contract_digest"],
        }
        return ExtractionFeedbackExample(
            example_id=_stable_id("feedback-example", identity),
            level=level,
            primary=primary,
            fact_id=fact_id,
            semantic_key=semantic_key,
            **values,
        )

    @staticmethod
    def _set_label(
        source: ExtractionSourceEvidence,
        observation: DeploymentObservation,
        future: FutureMemoryEvidence,
        resolution: ContractResolution,
        missed: tuple[MissedExtractionEvidence, ...],
    ) -> tuple[ExtractionFeedbackLabel, AttributionConfidence, tuple[str, ...]]:
        if not observation.observation_complete:
            return (
                ExtractionFeedbackLabel.CENSORED,
                AttributionConfidence.NONE,
                (observation.censor_reason or "observation_censored",),
            )
        quality_issues = tuple(
            fact.quality_issue for fact in source.facts if fact.quality_issue is not None
        )
        if quality_issues:
            return (
                ExtractionFeedbackLabel.HARMFUL,
                AttributionConfidence.HIGH,
                tuple(sorted({f"extraction_{value.value}" for value in quality_issues})),
            )
        extracted_keys = {
            semantic_key
            for fact in source.facts
            for semantic_key in fact.semantic_keys
        }
        has_unclassified_facts = any(
            not fact.semantic_keys for fact in source.facts
        )
        valid_missed = tuple(
            value for value in missed
            if not has_unclassified_facts
            and value.semantic_key in source.available_semantic_keys
            and value.semantic_key not in extracted_keys
            and value.future_opportunity_id == future.future_opportunity_id
            and value.deterministically_attributed
            and resolution.opportunity_observed
            and resolution.successful_outcome is False
            and value.absence_outcome_operation_id == resolution.outcome_operation_id
        )
        if valid_missed:
            return (
                ExtractionFeedbackLabel.MISSED,
                AttributionConfidence.HIGH,
                ("high_confidence_missed_extraction",),
            )
        if resolution.current_input_confounded:
            return (
                ExtractionFeedbackLabel.UNRESOLVED,
                AttributionConfidence.NONE,
                ("current_input_confounded",),
            )
        if not resolution.opportunity_observed:
            return (
                ExtractionFeedbackLabel.UNRESOLVED,
                AttributionConfidence.NONE,
                ("opportunity_not_observed",),
            )
        if not resolution.explicit_use:
            return (
                ExtractionFeedbackLabel.UNRESOLVED,
                AttributionConfidence.NONE,
                (
                    "injected_not_used"
                    if future.exposure_mode != ExposureMode.NOT_EXPOSED
                    else "not_exposed"
                ,),
            )
        if not resolution.used_artifact_ids:
            return (
                ExtractionFeedbackLabel.UNRESOLVED,
                AttributionConfidence.NONE,
                ("use_not_bound_to_memory",),
            )
        if any(value is None for value in (
            resolution.opportunity_operation_id,
            resolution.use_operation_id,
            resolution.outcome_operation_id,
        )):
            return (
                ExtractionFeedbackLabel.UNRESOLVED,
                AttributionConfidence.NONE,
                ("incomplete_opportunity_use_outcome_chain",),
            )
        if resolution.successful_outcome is True:
            return (
                ExtractionFeedbackLabel.USEFUL,
                AttributionConfidence.HIGH,
                ("opportunity_use_outcome_success",),
            )
        if resolution.harmful_outcome and resolution.successful_outcome is False:
            return (
                ExtractionFeedbackLabel.HARMFUL,
                AttributionConfidence.HIGH,
                ("memory_use_harmfully_attributed",),
            )
        return (
            ExtractionFeedbackLabel.UNRESOLVED,
            AttributionConfidence.LOW,
            resolution.reason_codes or ("outcome_not_attributable",),
        )


class _NotesFamilyResolver:
    def __init__(self, contract: FamilyFeedbackContract, parser: str) -> None:
        self._contract = contract
        self.parser = parser

    @property
    def contract(self) -> FamilyFeedbackContract:
        return self._contract

    @staticmethod
    def _tsv_rows(value: str) -> tuple[tuple[str, ...], ...]:
        rows = []
        for line in value.splitlines():
            fields = tuple(field.strip() for field in line.strip().strip("`").split("\t"))
            if len(fields) == 4:
                rows.append(fields)
        return tuple(rows)

    @classmethod
    def _valid_tsv(cls, value: str, *, normalized: bool) -> bool:
        rows = cls._tsv_rows(value)
        header = next((index for index, row in enumerate(rows) if tuple(
            field.casefold() for field in row
        ) == ("owner", "priority", "task", "due_date")), None)
        data = rows[header + 1:] if header is not None else ()
        if not data or not all(all(field for field in row) for row in data):
            return False
        if not normalized:
            return True
        priorities = {"low", "medium", "high", "critical"}
        return all(
            row[1].casefold() in priorities
            and re.fullmatch(r"\d{4}/\d{2}/\d{2}", row[3]) is not None
            for row in data
        )

    @staticmethod
    def _successful_share(observation: DeploymentObservation) -> bool:
        return any(
            event.tool_name == "notes_share" and event.success
            for event in observation.tool_events
        )

    def resolve(
        self,
        observation: DeploymentObservation,
        future: FutureMemoryEvidence,
    ) -> ContractResolution:
        if observation.family_id != self.contract.family_id:
            raise ValueError("deployment observation family differs from resolver")
        scope = set(self.contract.opportunity.memory_scope_keys)
        relevant_task = bool(scope & set(observation.task_semantic_keys))
        confounded = bool(scope & set(observation.current_input_semantic_keys))
        artifact_ids = tuple(
            binding.artifact_id
            for binding in future.artifact_bindings
            if set(binding.semantic_keys) & scope
        )
        exposed = (
            future.exposure_mode != ExposureMode.NOT_EXPOSED
            and future.injection_operation_id is not None
        )
        explicit_use = False
        harmful = False
        if self.parser == "sm01_tsv_use_v1":
            explicit_use = self._valid_tsv(observation.final_response, normalized=False)
        elif self.parser == "sm05_normalized_tsv_use_v1":
            explicit_use = self._valid_tsv(observation.final_response, normalized=True)
        elif self.parser == "sm02_boundary_use_v1":
            shares = tuple(
                event for event in observation.tool_events
                if event.tool_name == "notes_share" and event.success
            )
            explicit_use = bool(shares) and all(
                "ava_chen" not in {
                    _normalized_recipient_id(value)
                    for value in event.recipient_ids
                }
                for event in shares
            )
            harmful = any(
                "ava_chen" in {
                    _normalized_recipient_id(value)
                    for value in event.recipient_ids
                }
                for event in shares
            )
        elif self.parser == "sm03_fact_correction_use_v1":
            response = observation.final_response.casefold()
            corrected = "2026-06-16" in response
            stale = "2026-06-14" in response
            # Quoting the stale value while explicitly describing the
            # correction is not harmful evidence.  A stale-only response is
            # harmful only when it is not framed as historical context.
            correction_context = (
                any(token in response for token in ("obsolete", "superseded"))
                or (
                    "replaced" in response
                    and any(token in response for token in (" by ", " with "))
                )
            )
            explicit_use = corrected
            harmful = stale and not corrected and not correction_context
        else:
            raise ValueError("unknown family feedback parser")
        opportunity_id = future.opportunity_operation_id if relevant_task else None
        use_id = (
            _stable_id("use", {
                "observation_id": observation.observation_id,
                "parser": self.parser,
            })
            if explicit_use or harmful
            else None
        )
        successful_share = self._successful_share(observation)
        known_outcome = relevant_task and observation.observation_complete
        success = (
            observation.completed and successful_share and explicit_use
            if known_outcome
            else None
        )
        harmful_outcome = bool(
            known_outcome and observation.completed and successful_share and harmful
        )
        outcome_id = (
            _stable_id("outcome", {
                "observation_id": observation.observation_id,
                "completed": observation.completed,
                "share": successful_share,
            })
            if known_outcome
            else None
        )
        return ContractResolution(
            opportunity_observed=relevant_task,
            current_input_confounded=confounded,
            explicit_use=explicit_use or harmful,
            successful_outcome=(False if harmful_outcome else success),
            harmful_outcome=harmful_outcome,
            used_artifact_ids=(
                artifact_ids
                if (explicit_use or harmful) and not confounded and exposed
                else ()
            ),
            opportunity_operation_id=opportunity_id,
            use_operation_id=use_id,
            outcome_operation_id=outcome_id,
            reason_codes=(),
        )


def _notes_contract(
    family_id: str,
    semantic_keys: tuple[str, ...],
    parser_id: str,
) -> FamilyFeedbackContract:
    return FamilyFeedbackContract.create(
        OpportunityContract(
            f"{family_id}.opportunity-v1",
            family_id,
            ("eval_far", "eval_near"),
            semantic_keys,
            (
                DeploymentSurface.CURRENT_INPUT,
                DeploymentSurface.TASK_COMPLETION,
            ),
        ),
        UseContract(
            f"{family_id}.use-v1",
            family_id,
            parser_id,
            (DeploymentSurface.FINAL_RESPONSE, DeploymentSurface.TOOL_EVENT),
        ),
        OutcomeContract(
            f"{family_id}.outcome-v1",
            family_id,
            "notes_share_completion-v1",
            (DeploymentSurface.TOOL_EVENT, DeploymentSurface.TASK_COMPLETION),
        ),
    )


def default_feedback_contract_registry() -> FeedbackContractRegistry:
    registry = FeedbackContractRegistry()
    for family_id, keys, parser in (
        (
            "SM01_preference_adoption",
            ("preference.summary.tsv",),
            "sm01_tsv_use_v1",
        ),
        (
            "SM02_constraint_retention",
            ("constraint.share.exclude_ava_chen",),
            "sm02_boundary_use_v1",
        ),
        (
            "SM03_fact_correction",
            ("fact.phoenix.release_freeze_date",),
            "sm03_fact_correction_use_v1",
        ),
        (
            "SM05_weak_trigger_preference_adoption",
            (
                "preference.summary.tsv",
                "preference.priority.normalized",
                "preference.date.yyyy_mm_dd",
            ),
            "sm05_normalized_tsv_use_v1",
        ),
    ):
        contract = _notes_contract(family_id, keys, parser)
        registry.register(_NotesFamilyResolver(contract, parser))
    return registry


def detect_current_input_semantic_keys(
    family_id: str,
    current_input: str,
) -> tuple[str, ...]:
    """Conservatively detect a complete local restatement of registered rules."""

    value = current_input.casefold()
    keys = []
    if family_id in {"SM01_preference_adoption", "SM05_weak_trigger_preference_adoption"}:
        if (
            ("tsv" in value or "tab-separated" in value)
            and all(field in value for field in ("owner", "priority", "task", "due_date"))
        ):
            keys.append("preference.summary.tsv")
    if family_id == "SM05_weak_trigger_preference_adoption":
        if "priorit" in value and any(word in value for word in ("normalize", "normalise")):
            keys.append("preference.priority.normalized")
        if "yyyy/mm/dd" in value or "yyyy-mm-dd" in value:
            keys.append("preference.date.yyyy_mm_dd")
    if family_id == "SM02_constraint_retention" and (
        "ava chen" in value or "ava_chen" in value
    ) and any(phrase in value for phrase in (
        "do not share",
        "don't share",
        "exclude",
        "never share",
        "never be shared",
        "must never be shared",
    )):
        keys.append("constraint.share.exclude_ava_chen")
    if family_id == "SM03_fact_correction" and (
        "phoenix" in value
        and "freeze date" in value
        and "2026-06-16" in value
    ):
        keys.append("fact.phoenix.release_freeze_date")
    return tuple(keys)


def detect_extracted_fact_semantic_keys(
    family_id: str,
    fact_content: str,
) -> tuple[str, ...]:
    """Project fact content to registered keys without retaining the content."""

    value = fact_content.casefold()
    keys = []
    tsv_rule = (
        "tsv" in value
        or "tab-separated" in value
        or all(field in value for field in ("owner", "priority", "task", "due_date"))
    )
    if family_id in {
        "SM01_preference_adoption",
        "SM05_weak_trigger_preference_adoption",
    } and tsv_rule:
        keys.append("preference.summary.tsv")
    if family_id == "SM05_weak_trigger_preference_adoption":
        if "priorit" in value and any(token in value for token in (
            "normaliz",
            "low",
            "medium",
            "high",
            "critical",
        )):
            keys.append("preference.priority.normalized")
        if "yyyy/mm/dd" in value or "yyyy-mm-dd" in value:
            keys.append("preference.date.yyyy_mm_dd")
    if family_id == "SM02_constraint_retention" and (
        "ava chen" in value or "ava_chen" in value
    ) and any(phrase in value for phrase in (
        "do not share",
        "don't share",
        "exclude",
        "never share",
        "never be shared",
        "must not share",
        "must never be shared",
    )):
        keys.append("constraint.share.exclude_ava_chen")
    if family_id == "SM03_fact_correction" and (
        "phoenix" in value
        and "freeze" in value
        and "2026-06-16" in value
    ):
        keys.append("fact.phoenix.release_freeze_date")
    return tuple(dict.fromkeys(keys))


def detect_source_semantic_keys(
    family_id: str,
    source_contents: tuple[str, ...],
) -> tuple[str, ...]:
    """Detect durable family rules in the bounded extraction source projection."""

    value = "\n".join(source_contents).casefold()
    keys = []
    if family_id in {
        "SM01_preference_adoption",
        "SM05_weak_trigger_preference_adoption",
    } and (
        "tsv" in value or "tab-separated" in value
    ) and any(token in value for token in (
        "always",
        "default",
        "prefer",
        "preference",
        "use",
    )):
        keys.append("preference.summary.tsv")
    if family_id == "SM05_weak_trigger_preference_adoption":
        if "priorit" in value and any(token in value for token in (
            "normaliz",
            "low",
            "medium",
            "high",
            "critical",
        )):
            keys.append("preference.priority.normalized")
        if "yyyy/mm/dd" in value or "yyyy-mm-dd" in value:
            keys.append("preference.date.yyyy_mm_dd")
    if family_id == "SM02_constraint_retention" and (
        "ava chen" in value or "ava_chen" in value
    ) and any(phrase in value for phrase in (
        "do not share",
        "don't share",
        "exclude",
        "never share",
        "never be shared",
        "must not share",
        "must never be shared",
    )):
        keys.append("constraint.share.exclude_ava_chen")
    if family_id == "SM03_fact_correction" and (
        "phoenix" in value
        and "freeze" in value
        and "2026-06-16" in value
        and any(token in value for token in (
            "official",
            "authoritative",
            "correct",
            "replace",
            "going forward",
        ))
    ):
        keys.append("fact.phoenix.release_freeze_date")
    return tuple(dict.fromkeys(keys))


def detect_user_source_semantic_keys(
    family_id: str,
    source_messages: tuple[tuple[str, str], ...],
) -> tuple[str, ...]:
    """Detect durable source keys without learning from agent/tool output."""

    if any(
        not isinstance(role, str) or not isinstance(content, str)
        for role, content in source_messages
    ):
        raise TypeError("source message roles and contents must be strings")
    return detect_source_semantic_keys(
        family_id,
        tuple(content for role, content in source_messages if role == "user"),
    )
