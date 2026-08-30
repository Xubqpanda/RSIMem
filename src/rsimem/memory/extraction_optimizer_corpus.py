"""Owner-controlled content-bearing corpus contracts for prompt optimization."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Mapping

from .extraction_feedback import (
    AttributionConfidence,
    ExposureMode,
    ExtractionFeedbackLabel,
    ExtractionFeedbackLevel,
    FactDisposition,
)
from .optimizer_content_boundary import OptimizerUntrustedText
from .evidence_planes import EvidencePlane, require_optimizer_plane
from .prompt_components import content_digest


EXTRACTION_OPTIMIZER_CORPUS_SCHEMA_VERSION = 6
EXTRACTION_OPTIMIZER_CORPUS_SCHEMA = "extraction-optimizer-corpus-v6"
PROCESS_SIGNAL_GATE_NOT_BOUND = "not_bound"
PROCESS_SIGNAL_GATE_NO_SIGNAL = "no_signal"
PROCESS_SIGNAL_GATE_READY = "ready"
_PROCESS_SIGNAL_GATES = frozenset({
    PROCESS_SIGNAL_GATE_NOT_BOUND,
    PROCESS_SIGNAL_GATE_NO_SIGNAL,
    PROCESS_SIGNAL_GATE_READY,
})
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_ISO_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")


def _require_id(value: object, name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{name} must be a stable identifier")
    return value


def _require_digest(value: object, name: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{name} must be sha256")
    return value


def _require_utc(value: object, name: str) -> str:
    if not isinstance(value, str) or _ISO_UTC.fullmatch(value) is None:
        raise ValueError(f"{name} must be an ISO UTC timestamp")
    try:
        datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO UTC timestamp") from exc
    return value


def _parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.removesuffix("Z") + "+00:00")


def _require_unique(values: tuple[str, ...], name: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must be unique")


def _strict(value: object, fields: set[str], name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"malformed {name}")
    return value


class OptimizerCorpusSplit(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    FUTURE_TEST = "future_test"


class OptimizerCorpusRetention(StrEnum):
    DELETE_AFTER_POLICY_DECISION = "delete_after_policy_decision"
    DELETE_AFTER_EXPERIMENT = "delete_after_experiment"


class OptimizerComponentOwnership(StrEnum):
    EXTRACTION = "extraction"
    RETRIEVAL = "retrieval"
    APPLICATION = "application"
    OUTCOME = "outcome"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True, slots=True)
class OptimizerSourceMessage:
    segment_id: str
    source_message_id: str
    role: str
    segment_kind: str
    tool_call_id: str | None
    content_truncated: bool
    content: OptimizerUntrustedText

    def __post_init__(self) -> None:
        for value, name in (
            (self.segment_id, "optimizer source segment ID"),
            (self.source_message_id, "optimizer source message ID"),
            (self.role, "optimizer source role"),
            (self.segment_kind, "optimizer source segment kind"),
        ):
            _require_id(value, name)
        if self.tool_call_id is not None:
            _require_id(self.tool_call_id, "optimizer source tool call ID")
        if type(self.content_truncated) is not bool:
            raise TypeError("optimizer source truncation flag must be bool")
        if not isinstance(self.content, OptimizerUntrustedText):
            raise TypeError("optimizer source content has the wrong type")

    def payload(self) -> dict[str, object]:
        return {
            "segment_id": self.segment_id,
            "source_message_id": self.source_message_id,
            "role": self.role,
            "segment_kind": self.segment_kind,
            "tool_call_id": self.tool_call_id,
            "content_truncated": self.content_truncated,
            "content": self.content.payload(),
        }

    @classmethod
    def from_payload(cls, value: object) -> "OptimizerSourceMessage":
        payload = _strict(value, {
            "segment_id", "source_message_id", "role", "segment_kind",
            "tool_call_id", "content_truncated", "content",
        }, "optimizer source message")
        try:
            return cls(
                payload["segment_id"],
                payload["source_message_id"],
                payload["role"],
                payload["segment_kind"],
                payload["tool_call_id"],
                payload["content_truncated"],
                OptimizerUntrustedText.from_payload(payload["content"]),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("malformed optimizer source message") from exc


@dataclass(frozen=True, slots=True)
class OptimizerExtractedFact:
    fact_id: str
    content: OptimizerUntrustedText
    content_digest: str
    accepted: bool
    reason_code: str | None
    semantic_keys: tuple[str, ...]
    disposition: FactDisposition
    persisted_artifact_id: str | None

    def __post_init__(self) -> None:
        _require_id(self.fact_id, "optimizer fact ID")
        _require_digest(self.content_digest, "optimizer fact content digest")
        if self.content.source_digest != self.content_digest:
            raise ValueError("optimizer fact content differs from extraction trace")
        if type(self.accepted) is not bool:
            raise TypeError("optimizer fact accepted flag must be bool")
        if self.reason_code is not None:
            _require_id(self.reason_code, "optimizer fact reason code")
        _require_unique(self.semantic_keys, "optimizer fact semantic keys")
        for value in self.semantic_keys:
            _require_id(value, "optimizer fact semantic key")
        object.__setattr__(self, "disposition", FactDisposition(self.disposition))
        if self.persisted_artifact_id is not None:
            _require_id(self.persisted_artifact_id, "optimizer persisted artifact ID")
        if (self.disposition == FactDisposition.PERSISTED) != (
            self.persisted_artifact_id is not None
        ):
            raise ValueError("optimizer fact persistence lineage is inconsistent")

    def trace_payload(self) -> dict[str, object]:
        return {
            "fact_id": self.fact_id,
            "content_digest": self.content_digest,
            "accepted": self.accepted,
            "reason_code": self.reason_code,
        }

    def payload(self) -> dict[str, object]:
        return {
            **self.trace_payload(),
            "content": self.content.payload(),
            "semantic_keys": list(self.semantic_keys),
            "disposition": self.disposition.value,
            "persisted_artifact_id": self.persisted_artifact_id,
        }

    @classmethod
    def from_payload(cls, value: object) -> "OptimizerExtractedFact":
        payload = _strict(value, {
            "fact_id", "content", "content_digest", "accepted", "reason_code",
            "semantic_keys", "disposition", "persisted_artifact_id",
        }, "optimizer extracted fact")
        keys = payload["semantic_keys"]
        if not isinstance(keys, list):
            raise ValueError("malformed optimizer fact semantic keys")
        try:
            return cls(
                payload["fact_id"],
                OptimizerUntrustedText.from_payload(payload["content"]),
                payload["content_digest"],
                payload["accepted"],
                payload["reason_code"],
                tuple(keys),
                FactDisposition(payload["disposition"]),
                payload["persisted_artifact_id"],
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("malformed optimizer extracted fact") from exc


@dataclass(frozen=True, slots=True)
class OptimizerArtifactLineage:
    artifact_id: str
    content_digest: str
    operation_ids: tuple[str, ...]
    mutation_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_id(self.artifact_id, "optimizer lineage artifact ID")
        _require_digest(self.content_digest, "optimizer lineage artifact digest")
        for values, name in (
            (self.operation_ids, "optimizer lineage operation IDs"),
            (self.mutation_ids, "optimizer lineage mutation IDs"),
        ):
            _require_unique(values, name)
            for value in values:
                _require_id(value, name)

    def payload(self) -> dict[str, object]:
        return {
            "artifact_id": self.artifact_id,
            "content_digest": self.content_digest,
            "operation_ids": list(self.operation_ids),
            "mutation_ids": list(self.mutation_ids),
        }

    @classmethod
    def from_payload(cls, value: object) -> "OptimizerArtifactLineage":
        payload = _strict(value, {
            "artifact_id", "content_digest", "operation_ids", "mutation_ids",
        }, "optimizer artifact lineage")
        if not isinstance(payload["operation_ids"], list) or not isinstance(
            payload["mutation_ids"], list
        ):
            raise ValueError("malformed optimizer artifact lineage")
        try:
            return cls(
                payload["artifact_id"],
                payload["content_digest"],
                tuple(payload["operation_ids"]),
                tuple(payload["mutation_ids"]),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("malformed optimizer artifact lineage") from exc


@dataclass(frozen=True, slots=True)
class OptimizerDelayedEvidence:
    observation_id: str
    source_completed_at: str
    observed_at: str
    future_opportunity_id: str
    opportunity_operation_id: str | None
    use_operation_id: str | None
    outcome_operation_id: str | None
    opportunity: OptimizerUntrustedText
    use: OptimizerUntrustedText
    outcome: OptimizerUntrustedText

    def __post_init__(self) -> None:
        _require_id(self.observation_id, "optimizer observation ID")
        _require_id(self.future_opportunity_id, "optimizer future opportunity ID")
        for value, name in (
            (self.source_completed_at, "optimizer source completion time"),
            (self.observed_at, "optimizer observation time"),
        ):
            _require_utc(value, name)
        if _parse_utc(self.observed_at) < _parse_utc(self.source_completed_at):
            raise ValueError("optimizer observation predates its source")
        for value in (
            self.opportunity_operation_id,
            self.use_operation_id,
            self.outcome_operation_id,
        ):
            if value is not None:
                _require_id(value, "optimizer delayed operation ID")
        for value in (self.opportunity, self.use, self.outcome):
            if not isinstance(value, OptimizerUntrustedText):
                raise TypeError("optimizer delayed content has the wrong type")

    def payload(self) -> dict[str, object]:
        return {
            "observation_id": self.observation_id,
            "source_completed_at": self.source_completed_at,
            "observed_at": self.observed_at,
            "future_opportunity_id": self.future_opportunity_id,
            "opportunity_operation_id": self.opportunity_operation_id,
            "use_operation_id": self.use_operation_id,
            "outcome_operation_id": self.outcome_operation_id,
            "opportunity": self.opportunity.payload(),
            "use": self.use.payload(),
            "outcome": self.outcome.payload(),
        }

    @classmethod
    def from_payload(cls, value: object) -> "OptimizerDelayedEvidence":
        payload = _strict(value, {
            "observation_id", "source_completed_at", "observed_at",
            "future_opportunity_id", "opportunity_operation_id",
            "use_operation_id", "outcome_operation_id", "opportunity", "use",
            "outcome",
        }, "optimizer delayed evidence")
        try:
            return cls(
                payload["observation_id"],
                payload["source_completed_at"],
                payload["observed_at"],
                payload["future_opportunity_id"],
                payload["opportunity_operation_id"],
                payload["use_operation_id"],
                payload["outcome_operation_id"],
                OptimizerUntrustedText.from_payload(payload["opportunity"]),
                OptimizerUntrustedText.from_payload(payload["use"]),
                OptimizerUntrustedText.from_payload(payload["outcome"]),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("malformed optimizer delayed evidence") from exc


@dataclass(frozen=True, slots=True)
class OptimizerAuditJoin:
    family_id: str
    source_record_id: str
    source_record_digest: str
    source_stage: str
    source_run_id: str
    source_episode_id: str
    source_session_id: str
    source_task_id: str
    source_projection_id: str
    source_projection_digest: str
    feedback_record_id: str
    feedback_dataset_id: str
    feedback_example_id: str
    feedback_stage: str
    feedback_run_id: str
    feedback_trace_id: str
    feedback_episode_id: str
    feedback_session_id: str
    feedback_task_id: str
    extraction_artifact_id: str
    extraction_artifact_digest: str
    extraction_output_digest: str
    operation_ids: tuple[str, ...]
    artifacts: tuple[OptimizerArtifactLineage, ...]
    # Frozen observation-window identity from the feedback dataset.  The
    # default keeps older in-memory fixtures constructible; formal corpus
    # builders always bind the dataset's window version explicitly.
    observation_window: str = "window.unbound"

    def __post_init__(self) -> None:
        for value, name in (
            (self.family_id, "optimizer family ID"),
            (self.source_record_id, "optimizer source record ID"),
            (self.source_stage, "optimizer source stage"),
            (self.source_run_id, "optimizer source run ID"),
            (self.source_episode_id, "optimizer source episode ID"),
            (self.source_session_id, "optimizer source session ID"),
            (self.source_task_id, "optimizer source task ID"),
            (self.source_projection_id, "optimizer source projection ID"),
            (self.feedback_record_id, "optimizer feedback record ID"),
            (self.feedback_dataset_id, "optimizer feedback dataset ID"),
            (self.feedback_example_id, "optimizer feedback example ID"),
            (self.feedback_stage, "optimizer feedback stage"),
            (self.feedback_run_id, "optimizer feedback run ID"),
            (self.feedback_trace_id, "optimizer feedback trace ID"),
            (self.feedback_episode_id, "optimizer feedback episode ID"),
            (self.feedback_session_id, "optimizer feedback session ID"),
            (self.feedback_task_id, "optimizer feedback task ID"),
            (self.extraction_artifact_id, "optimizer extraction artifact ID"),
            (self.observation_window, "optimizer observation window"),
        ):
            _require_id(value, name)
        for value, name in (
            (self.source_record_digest, "optimizer source record digest"),
            (self.source_projection_digest, "optimizer source projection digest"),
            (self.extraction_artifact_digest, "optimizer extraction artifact digest"),
            (self.extraction_output_digest, "optimizer extraction output digest"),
        ):
            _require_digest(value, name)
        _require_unique(self.operation_ids, "optimizer audit operation IDs")
        for value in self.operation_ids:
            _require_id(value, "optimizer audit operation ID")
        artifact_ids = tuple(value.artifact_id for value in self.artifacts)
        _require_unique(artifact_ids, "optimizer audit artifact IDs")

    def payload(self) -> dict[str, object]:
        return {
            "family_id": self.family_id,
            "source_record_id": self.source_record_id,
            "source_record_digest": self.source_record_digest,
            "source_stage": self.source_stage,
            "source_run_id": self.source_run_id,
            "source_episode_id": self.source_episode_id,
            "source_session_id": self.source_session_id,
            "source_task_id": self.source_task_id,
            "source_projection_id": self.source_projection_id,
            "source_projection_digest": self.source_projection_digest,
            "feedback_record_id": self.feedback_record_id,
            "feedback_dataset_id": self.feedback_dataset_id,
            "feedback_example_id": self.feedback_example_id,
            "feedback_stage": self.feedback_stage,
            "feedback_run_id": self.feedback_run_id,
            "feedback_trace_id": self.feedback_trace_id,
            "feedback_episode_id": self.feedback_episode_id,
            "feedback_session_id": self.feedback_session_id,
            "feedback_task_id": self.feedback_task_id,
            "extraction_artifact_id": self.extraction_artifact_id,
            "extraction_artifact_digest": self.extraction_artifact_digest,
            "extraction_output_digest": self.extraction_output_digest,
            "observation_window": self.observation_window,
            "operation_ids": list(self.operation_ids),
            "artifacts": [value.payload() for value in self.artifacts],
        }

    @classmethod
    def from_payload(cls, value: object) -> "OptimizerAuditJoin":
        payload = _strict(value, {
            "family_id", "source_record_id", "source_record_digest",
            "source_stage", "source_run_id", "source_episode_id",
            "source_session_id", "source_task_id", "source_projection_id",
            "source_projection_digest", "feedback_record_id",
            "feedback_dataset_id", "feedback_example_id",
            "feedback_stage", "feedback_run_id", "feedback_trace_id",
            "feedback_episode_id", "feedback_session_id", "feedback_task_id",
            "extraction_artifact_id", "extraction_artifact_digest",
            "extraction_output_digest", "observation_window", "operation_ids", "artifacts",
        }, "optimizer audit join")
        if not isinstance(payload["operation_ids"], list) or not isinstance(
            payload["artifacts"], list
        ):
            raise ValueError("malformed optimizer audit join")
        try:
            return cls(
                family_id=payload["family_id"],
                source_record_id=payload["source_record_id"],
                source_record_digest=payload["source_record_digest"],
                source_stage=payload["source_stage"],
                source_run_id=payload["source_run_id"],
                source_episode_id=payload["source_episode_id"],
                source_session_id=payload["source_session_id"],
                source_task_id=payload["source_task_id"],
                source_projection_id=payload["source_projection_id"],
                source_projection_digest=payload["source_projection_digest"],
                feedback_record_id=payload["feedback_record_id"],
                feedback_dataset_id=payload["feedback_dataset_id"],
                feedback_example_id=payload["feedback_example_id"],
                feedback_stage=payload["feedback_stage"],
                feedback_run_id=payload["feedback_run_id"],
                feedback_trace_id=payload["feedback_trace_id"],
                feedback_episode_id=payload["feedback_episode_id"],
                feedback_session_id=payload["feedback_session_id"],
                feedback_task_id=payload["feedback_task_id"],
                extraction_artifact_id=payload["extraction_artifact_id"],
                extraction_artifact_digest=payload["extraction_artifact_digest"],
                extraction_output_digest=payload["extraction_output_digest"],
                operation_ids=tuple(payload["operation_ids"]),
                artifacts=tuple(
                    OptimizerArtifactLineage.from_payload(item)
                    for item in payload["artifacts"]
                ),
                observation_window=payload["observation_window"],
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("malformed optimizer audit join") from exc


@dataclass(frozen=True, slots=True)
class ExtractionOptimizerCorpusExample:
    example_id: str
    example_digest: str
    primary_unit_id: str
    level: ExtractionFeedbackLevel
    primary: bool
    feedback_fact_id: str | None
    feedback_semantic_key: str | None
    feedback_artifact_ids: tuple[str, ...]
    exposure_mode: ExposureMode
    label: ExtractionFeedbackLabel
    attribution_confidence: AttributionConfidence
    reason_codes: tuple[str, ...]
    component_ownership: OptimizerComponentOwnership
    audit_join: OptimizerAuditJoin
    source_messages: tuple[OptimizerSourceMessage, ...]
    extracted_facts: tuple[OptimizerExtractedFact, ...]
    delayed_evidence: OptimizerDelayedEvidence
    evidence_plane: EvidencePlane = EvidencePlane.PURE_PROCESS
    schema_version: int = EXTRACTION_OPTIMIZER_CORPUS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != EXTRACTION_OPTIMIZER_CORPUS_SCHEMA_VERSION:
            raise ValueError("unsupported optimizer corpus example schema")
        _require_id(self.example_id, "optimizer corpus example ID")
        _require_digest(self.example_digest, "optimizer corpus example digest")
        _require_id(self.primary_unit_id, "optimizer primary unit ID")
        object.__setattr__(self, "level", ExtractionFeedbackLevel(self.level))
        object.__setattr__(self, "exposure_mode", ExposureMode(self.exposure_mode))
        object.__setattr__(self, "label", ExtractionFeedbackLabel(self.label))
        object.__setattr__(
            self,
            "attribution_confidence",
            AttributionConfidence(self.attribution_confidence),
        )
        object.__setattr__(
            self,
            "component_ownership",
            OptimizerComponentOwnership(self.component_ownership),
        )
        object.__setattr__(self, "evidence_plane", EvidencePlane(self.evidence_plane))
        if type(self.primary) is not bool or self.primary != (
            self.level == ExtractionFeedbackLevel.EXTRACTION_SET
        ):
            raise ValueError("optimizer primary unit must be extraction-set level")
        if (self.level == ExtractionFeedbackLevel.FACT) != (
            self.feedback_fact_id is not None
        ):
            raise ValueError("optimizer fact-level feedback requires a fact ID")
        if self.feedback_fact_id is not None:
            _require_id(self.feedback_fact_id, "optimizer feedback fact ID")
        if self.feedback_semantic_key is not None:
            _require_id(self.feedback_semantic_key, "optimizer feedback semantic key")
        _require_unique(self.feedback_artifact_ids, "optimizer feedback artifact IDs")
        for value in self.feedback_artifact_ids:
            _require_id(value, "optimizer feedback artifact ID")
        _require_unique(self.reason_codes, "optimizer reason codes")
        for value in self.reason_codes:
            _require_id(value, "optimizer reason code")
        if not self.source_messages:
            raise ValueError("optimizer corpus example requires bounded source messages")
        fact_ids = tuple(value.fact_id for value in self.extracted_facts)
        _require_unique(fact_ids, "optimizer extracted fact IDs")
        if self.feedback_fact_id is not None and self.feedback_fact_id not in fact_ids:
            raise ValueError("optimizer feedback fact is absent from extracted facts")
        if self.label in {
            ExtractionFeedbackLabel.USEFUL,
            ExtractionFeedbackLabel.HARMFUL,
            ExtractionFeedbackLabel.MISSED,
        } and self.component_ownership != OptimizerComponentOwnership.EXTRACTION:
            raise ValueError("resolved extraction labels must remain extraction-owned")
        delayed_operation_ids = {
            value
            for value in (
                self.delayed_evidence.opportunity_operation_id,
                self.delayed_evidence.use_operation_id,
                self.delayed_evidence.outcome_operation_id,
            )
            if value is not None
        }
        if not delayed_operation_ids.issubset(self.audit_join.operation_ids):
            raise ValueError("optimizer delayed operations escape the audit join")
        persisted_artifact_ids = {
            value.persisted_artifact_id
            for value in self.extracted_facts
            if value.persisted_artifact_id is not None
        }
        lineage_artifact_ids = {
            value.artifact_id for value in self.audit_join.artifacts
        }
        if not persisted_artifact_ids.issubset(lineage_artifact_ids):
            raise ValueError("optimizer persisted facts lack artifact lineage")
        if self.label == ExtractionFeedbackLabel.USEFUL and (
            self.delayed_evidence.opportunity_operation_id is None
            or self.delayed_evidence.use_operation_id is None
            or self.delayed_evidence.outcome_operation_id is None
            or not self.delayed_evidence.opportunity.text
            or not self.delayed_evidence.use.text
            or not self.delayed_evidence.outcome.text
        ):
            raise ValueError("useful optimizer example lacks three-stage evidence")
        if self.label in {
            ExtractionFeedbackLabel.HARMFUL,
            ExtractionFeedbackLabel.MISSED,
        } and (
            self.delayed_evidence.opportunity_operation_id is None
            or self.delayed_evidence.outcome_operation_id is None
            or not self.delayed_evidence.opportunity.text
            or not self.delayed_evidence.outcome.text
        ):
            raise ValueError("negative optimizer example lacks attribution evidence")
        digest = content_digest(self.identity_payload())
        if self.example_digest != digest:
            # A caller mutating a pure-process example into a benchmark/final
            # plane without rebuilding its canonical digest is attempting to
            # smuggle diagnostic evidence through an optimizer contract.  Keep
            # this failure explicit; valid diagnostic examples are accepted as
            # corpus artifacts and are rejected later at request construction.
            if self.evidence_plane != EvidencePlane.PURE_PROCESS:
                raise ValueError(
                    "optimizer requires pure_process evidence; benchmark/final "
                    "evidence is diagnostic-only"
                )
            raise ValueError("optimizer corpus example digest mismatch")
        if self.example_id != f"optimizer-example.{digest[:40]}":
            raise ValueError("optimizer corpus example ID mismatch")

    @classmethod
    def create(
        cls,
        *,
        primary_unit_id: str,
        level: ExtractionFeedbackLevel,
        primary: bool,
        feedback_fact_id: str | None,
        feedback_semantic_key: str | None,
        feedback_artifact_ids: tuple[str, ...],
        exposure_mode: ExposureMode,
        label: ExtractionFeedbackLabel,
        attribution_confidence: AttributionConfidence,
        reason_codes: tuple[str, ...],
        component_ownership: OptimizerComponentOwnership,
        audit_join: OptimizerAuditJoin,
        source_messages: tuple[OptimizerSourceMessage, ...],
        extracted_facts: tuple[OptimizerExtractedFact, ...],
        delayed_evidence: OptimizerDelayedEvidence,
        evidence_plane: EvidencePlane = EvidencePlane.PURE_PROCESS,
    ) -> "ExtractionOptimizerCorpusExample":
        values = {
            "primary_unit_id": primary_unit_id,
            "level": ExtractionFeedbackLevel(level),
            "primary": primary,
            "feedback_fact_id": feedback_fact_id,
            "feedback_semantic_key": feedback_semantic_key,
            "feedback_artifact_ids": feedback_artifact_ids,
            "exposure_mode": ExposureMode(exposure_mode),
            "label": ExtractionFeedbackLabel(label),
            "attribution_confidence": AttributionConfidence(attribution_confidence),
            "reason_codes": reason_codes,
            "component_ownership": OptimizerComponentOwnership(component_ownership),
            "audit_join": audit_join,
            "source_messages": source_messages,
            "extracted_facts": extracted_facts,
            "delayed_evidence": delayed_evidence,
            # Corpus builders may materialize benchmark-audit examples for
            # diagnostics.  The optimizer request boundary performs the
            # pure-process gate; retaining the plane here preserves provenance.
            "evidence_plane": EvidencePlane(evidence_plane),
            "schema_version": EXTRACTION_OPTIMIZER_CORPUS_SCHEMA_VERSION,
        }
        identity = cls._identity(values)
        digest = content_digest(identity)
        return cls(
            example_id=f"optimizer-example.{digest[:40]}",
            example_digest=digest,
            **values,
        )

    @staticmethod
    def _identity(values: dict[str, object]) -> dict[str, object]:
        return {
            "schema_version": values["schema_version"],
            "primary_unit_id": values["primary_unit_id"],
            "level": values["level"].value,
            "primary": values["primary"],
            "feedback_fact_id": values["feedback_fact_id"],
            "feedback_semantic_key": values["feedback_semantic_key"],
            "feedback_artifact_ids": list(values["feedback_artifact_ids"]),
            "exposure_mode": values["exposure_mode"].value,
            "label": values["label"].value,
            "attribution_confidence": values["attribution_confidence"].value,
            "reason_codes": list(values["reason_codes"]),
            "component_ownership": values["component_ownership"].value,
            "audit_join": values["audit_join"].payload(),
            "source_messages": [value.payload() for value in values["source_messages"]],
            "extracted_facts": [value.payload() for value in values["extracted_facts"]],
            "delayed_evidence": values["delayed_evidence"].payload(),
            "evidence_plane": values["evidence_plane"].value,
        }

    def identity_payload(self) -> dict[str, object]:
        return self._identity({
            "schema_version": self.schema_version,
            "primary_unit_id": self.primary_unit_id,
            "level": self.level,
            "primary": self.primary,
            "feedback_fact_id": self.feedback_fact_id,
            "feedback_semantic_key": self.feedback_semantic_key,
            "feedback_artifact_ids": self.feedback_artifact_ids,
            "exposure_mode": self.exposure_mode,
            "label": self.label,
            "attribution_confidence": self.attribution_confidence,
            "reason_codes": self.reason_codes,
            "component_ownership": self.component_ownership,
            "audit_join": self.audit_join,
            "source_messages": self.source_messages,
            "extracted_facts": self.extracted_facts,
            "delayed_evidence": self.delayed_evidence,
            "evidence_plane": self.evidence_plane,
        })

    def payload(self) -> dict[str, object]:
        return {
            **self.identity_payload(),
            "example_id": self.example_id,
            "example_digest": self.example_digest,
        }

    @classmethod
    def from_payload(cls, value: object) -> "ExtractionOptimizerCorpusExample":
        payload = _strict(value, {
            "schema_version", "example_id", "example_digest", "primary_unit_id",
            "level", "primary", "feedback_fact_id", "feedback_semantic_key",
            "feedback_artifact_ids", "exposure_mode", "label", "attribution_confidence",
            "reason_codes", "component_ownership", "audit_join",
            "source_messages", "extracted_facts", "delayed_evidence",
            "evidence_plane",
        }, "optimizer corpus example")
        collections = (
            payload["reason_codes"],
            payload["feedback_artifact_ids"],
            payload["source_messages"],
            payload["extracted_facts"],
        )
        if any(not isinstance(item, list) for item in collections):
            raise ValueError("malformed optimizer corpus example collections")
        try:
            return cls(
                example_id=payload["example_id"],
                example_digest=payload["example_digest"],
                primary_unit_id=payload["primary_unit_id"],
                level=ExtractionFeedbackLevel(payload["level"]),
                primary=payload["primary"],
                feedback_fact_id=payload["feedback_fact_id"],
                feedback_semantic_key=payload["feedback_semantic_key"],
                feedback_artifact_ids=tuple(payload["feedback_artifact_ids"]),
                exposure_mode=ExposureMode(payload["exposure_mode"]),
                label=ExtractionFeedbackLabel(payload["label"]),
                attribution_confidence=AttributionConfidence(
                    payload["attribution_confidence"]
                ),
                reason_codes=tuple(payload["reason_codes"]),
                component_ownership=OptimizerComponentOwnership(
                    payload["component_ownership"]
                ),
                audit_join=OptimizerAuditJoin.from_payload(payload["audit_join"]),
                source_messages=tuple(
                    OptimizerSourceMessage.from_payload(item)
                    for item in payload["source_messages"]
                ),
                extracted_facts=tuple(
                    OptimizerExtractedFact.from_payload(item)
                    for item in payload["extracted_facts"]
                ),
                delayed_evidence=OptimizerDelayedEvidence.from_payload(
                    payload["delayed_evidence"]
                ),
                evidence_plane=EvidencePlane(payload["evidence_plane"]),
                schema_version=payload["schema_version"],
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("malformed optimizer corpus example") from exc


@dataclass(frozen=True, slots=True)
class ExtractionOptimizerCorpus:
    corpus_id: str
    corpus_digest: str
    batch_id: str
    attempt_id: str
    split: OptimizerCorpusSplit
    observation_cutoff: str
    retention: OptimizerCorpusRetention
    activation_artifact_id: str | None
    examples: tuple[ExtractionOptimizerCorpusExample, ...]
    process_signal_gate: str = PROCESS_SIGNAL_GATE_NOT_BOUND
    process_signal_protocol_id: str | None = None
    process_signal_case_digest: str | None = None
    process_signal_case_count: int = 0
    process_signal_optimization_count: int = 0
    process_signal_hypothesis_digest: str | None = None
    corpus_schema: str = EXTRACTION_OPTIMIZER_CORPUS_SCHEMA
    schema_version: int = EXTRACTION_OPTIMIZER_CORPUS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            self.schema_version != EXTRACTION_OPTIMIZER_CORPUS_SCHEMA_VERSION
            or self.corpus_schema != EXTRACTION_OPTIMIZER_CORPUS_SCHEMA
        ):
            raise ValueError("unsupported extraction optimizer corpus schema")
        for value, name in (
            (self.corpus_id, "optimizer corpus ID"),
            (self.batch_id, "optimizer corpus batch ID"),
            (self.attempt_id, "optimizer corpus attempt ID"),
        ):
            _require_id(value, name)
        _require_digest(self.corpus_digest, "optimizer corpus digest")
        _require_utc(self.observation_cutoff, "optimizer corpus cutoff")
        object.__setattr__(self, "split", OptimizerCorpusSplit(self.split))
        object.__setattr__(self, "retention", OptimizerCorpusRetention(self.retention))
        if self.process_signal_gate not in _PROCESS_SIGNAL_GATES:
            raise ValueError("optimizer corpus process-signal gate is invalid")
        if self.process_signal_protocol_id is not None:
            _require_id(self.process_signal_protocol_id, "process signal protocol ID")
        if self.process_signal_case_digest is not None:
            _require_digest(self.process_signal_case_digest, "process signal case digest")
        if self.process_signal_hypothesis_digest is not None:
            _require_digest(
                self.process_signal_hypothesis_digest,
                "process signal hypothesis digest",
            )
        for value, name in (
            (self.process_signal_case_count, "process signal case count"),
            (self.process_signal_optimization_count, "process signal optimization count"),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.process_signal_optimization_count > self.process_signal_case_count:
            raise ValueError("process signal optimization count exceeds case count")
        if self.process_signal_case_count == 0:
            if self.process_signal_case_digest is not None:
                raise ValueError("empty process signal gate cannot carry case digest")
        elif self.process_signal_case_digest is None:
            raise ValueError("process signal cases require a case digest")
        if self.process_signal_gate == PROCESS_SIGNAL_GATE_NOT_BOUND:
            if any((
                self.process_signal_protocol_id is not None,
                self.process_signal_case_digest is not None,
                self.process_signal_case_count,
                self.process_signal_optimization_count,
                self.process_signal_hypothesis_digest is not None,
            )):
                raise ValueError("unbound process signal gate cannot carry evidence")
        elif self.process_signal_case_count and self.process_signal_protocol_id is None:
            raise ValueError("bound process signal gate requires protocol identity")
        if self.process_signal_gate == PROCESS_SIGNAL_GATE_READY:
            if self.process_signal_optimization_count < 2:
                raise ValueError("ready process signal gate requires replicated cases")
            if self.process_signal_hypothesis_digest is None:
                raise ValueError("ready process signal gate requires hypothesis identity")
        elif self.process_signal_hypothesis_digest is not None and self.process_signal_optimization_count == 0:
            raise ValueError("process signal hypothesis requires optimization cases")
        if not self.examples:
            raise ValueError("optimizer corpus requires examples")
        observation_windows = {
            example.audit_join.observation_window
            for example in self.examples
            if example.primary
        }
        if self.process_signal_gate == PROCESS_SIGNAL_GATE_READY and (
            len(observation_windows) != 1
            or observation_windows == {"window.unbound"}
        ):
            raise ValueError(
                "ready process signal gate requires one bound observation window"
            )
        if self.examples != tuple(sorted(self.examples, key=_example_sort_key)):
            raise ValueError("optimizer corpus examples must be canonically ordered")
        example_ids = tuple(value.example_id for value in self.examples)
        _require_unique(example_ids, "optimizer corpus example IDs")
        cutoff = _parse_utc(self.observation_cutoff)
        if any(
            _parse_utc(value.delayed_evidence.observed_at) > cutoff
            for value in self.examples
        ):
            raise ValueError("optimizer corpus contains future-dated evidence")
        if self.split == OptimizerCorpusSplit.FUTURE_TEST:
            if self.activation_artifact_id is None:
                raise ValueError("future-test corpus requires activation identity")
            _require_id(self.activation_artifact_id, "future-test activation artifact ID")
        elif self.activation_artifact_id is not None:
            raise ValueError("non-future corpus cannot carry activation identity")
        digest = content_digest(self.identity_payload())
        if self.corpus_digest != digest:
            raise ValueError("optimizer corpus digest mismatch")
        if self.corpus_id != f"optimizer-corpus.{digest[:40]}":
            raise ValueError("optimizer corpus ID mismatch")

    @classmethod
    def create(
        cls,
        *,
        batch_id: str,
        attempt_id: str,
        split: OptimizerCorpusSplit,
        observation_cutoff: str,
        retention: OptimizerCorpusRetention,
        examples: tuple[ExtractionOptimizerCorpusExample, ...],
        activation_artifact_id: str | None = None,
        process_signal_gate: str = PROCESS_SIGNAL_GATE_NOT_BOUND,
        process_signal_protocol_id: str | None = None,
        process_signal_case_digest: str | None = None,
        process_signal_case_count: int = 0,
        process_signal_optimization_count: int = 0,
        process_signal_hypothesis_digest: str | None = None,
    ) -> "ExtractionOptimizerCorpus":
        values = {
            "batch_id": batch_id,
            "attempt_id": attempt_id,
            "split": OptimizerCorpusSplit(split),
            "observation_cutoff": observation_cutoff,
            "retention": OptimizerCorpusRetention(retention),
            "activation_artifact_id": activation_artifact_id,
            "examples": tuple(sorted(examples, key=_example_sort_key)),
            "process_signal_gate": process_signal_gate,
            "process_signal_protocol_id": process_signal_protocol_id,
            "process_signal_case_digest": process_signal_case_digest,
            "process_signal_case_count": process_signal_case_count,
            "process_signal_optimization_count": process_signal_optimization_count,
            "process_signal_hypothesis_digest": process_signal_hypothesis_digest,
            "corpus_schema": EXTRACTION_OPTIMIZER_CORPUS_SCHEMA,
            "schema_version": EXTRACTION_OPTIMIZER_CORPUS_SCHEMA_VERSION,
        }
        identity = cls._identity(values)
        digest = content_digest(identity)
        return cls(
            corpus_id=f"optimizer-corpus.{digest[:40]}",
            corpus_digest=digest,
            **values,
        )

    @staticmethod
    def _identity(values: dict[str, object]) -> dict[str, object]:
        return {
            "schema_version": values["schema_version"],
            "corpus_schema": values["corpus_schema"],
            "batch_id": values["batch_id"],
            "attempt_id": values["attempt_id"],
            "split": values["split"].value,
            "observation_cutoff": values["observation_cutoff"],
            "retention": values["retention"].value,
            "activation_artifact_id": values["activation_artifact_id"],
            "examples": [value.payload() for value in values["examples"]],
            "process_signal_gate": values["process_signal_gate"],
            "process_signal_protocol_id": values["process_signal_protocol_id"],
            "process_signal_case_digest": values["process_signal_case_digest"],
            "process_signal_case_count": values["process_signal_case_count"],
            "process_signal_optimization_count": values["process_signal_optimization_count"],
            "process_signal_hypothesis_digest": values["process_signal_hypothesis_digest"],
        }

    def identity_payload(self) -> dict[str, object]:
        return self._identity({
            "schema_version": self.schema_version,
            "corpus_schema": self.corpus_schema,
            "batch_id": self.batch_id,
            "attempt_id": self.attempt_id,
            "split": self.split,
            "observation_cutoff": self.observation_cutoff,
            "retention": self.retention,
            "activation_artifact_id": self.activation_artifact_id,
            "examples": self.examples,
            "process_signal_gate": self.process_signal_gate,
            "process_signal_protocol_id": self.process_signal_protocol_id,
            "process_signal_case_digest": self.process_signal_case_digest,
            "process_signal_case_count": self.process_signal_case_count,
            "process_signal_optimization_count": self.process_signal_optimization_count,
            "process_signal_hypothesis_digest": self.process_signal_hypothesis_digest,
        })

    def payload(self) -> dict[str, object]:
        return {
            **self.identity_payload(),
            "corpus_id": self.corpus_id,
            "corpus_digest": self.corpus_digest,
        }

    @classmethod
    def from_payload(cls, value: object) -> "ExtractionOptimizerCorpus":
        payload = _strict(value, {
            "schema_version", "corpus_schema", "corpus_id", "corpus_digest",
            "batch_id", "attempt_id", "split", "observation_cutoff",
            "retention", "activation_artifact_id", "examples", "process_signal_gate",
            "process_signal_protocol_id", "process_signal_case_digest",
            "process_signal_case_count", "process_signal_optimization_count",
            "process_signal_hypothesis_digest",
        }, "extraction optimizer corpus")
        if not isinstance(payload["examples"], list):
            raise ValueError("malformed optimizer corpus examples")
        try:
            return cls(
                corpus_id=payload["corpus_id"],
                corpus_digest=payload["corpus_digest"],
                batch_id=payload["batch_id"],
                attempt_id=payload["attempt_id"],
                split=OptimizerCorpusSplit(payload["split"]),
                observation_cutoff=payload["observation_cutoff"],
                retention=OptimizerCorpusRetention(payload["retention"]),
                activation_artifact_id=payload["activation_artifact_id"],
                examples=tuple(
                    ExtractionOptimizerCorpusExample.from_payload(item)
                    for item in payload["examples"]
                ),
                process_signal_gate=payload["process_signal_gate"],
                process_signal_protocol_id=payload["process_signal_protocol_id"],
                process_signal_case_digest=payload["process_signal_case_digest"],
                process_signal_case_count=payload["process_signal_case_count"],
                process_signal_optimization_count=payload["process_signal_optimization_count"],
                process_signal_hypothesis_digest=payload["process_signal_hypothesis_digest"],
                corpus_schema=payload["corpus_schema"],
                schema_version=payload["schema_version"],
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("malformed extraction optimizer corpus") from exc


def _example_sort_key(
    value: ExtractionOptimizerCorpusExample,
) -> tuple[str, str, str]:
    return (
        value.audit_join.source_record_id,
        value.audit_join.feedback_record_id,
        value.audit_join.feedback_example_id,
    )
