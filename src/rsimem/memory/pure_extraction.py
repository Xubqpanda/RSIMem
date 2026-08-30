"""Pure-process extraction evidence contracts.

The family-bound extraction records live in :mod:`extraction_projection` and
are intentionally benchmark-audit evidence.  This module provides the
deployment-visible counterpart used by a generic optimizer.  It deliberately
does not contain family IDs, stages, grader labels, or answer-key material.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Iterator, Mapping

from .artifact_set import ArtifactSetSemanticBinding
from .evidence_planes import (
    EvidencePlane,
    EvidenceSourceKind,
    validate_plane_source,
    validate_pure_process_payload,
)
from .extraction_feedback import ExtractionSourceEvidence
from .opportunity import OpportunityEvidence
from .prompt_components import content_digest
from .tool_exact_join import ToolCallResultJoin
from .use_attribution import MemoryUseEvidence


PURE_EXTRACTION_SOURCE_SCHEMA_VERSION = 1
PURE_EXTRACTION_SOURCE_SCHEMA = "rsimem-pure-extraction-source-v1"
PURE_EXTRACTION_FEEDBACK_SCHEMA_VERSION = 1
PURE_EXTRACTION_FEEDBACK_SCHEMA = "rsimem-pure-extraction-feedback-v1"
PURE_EXTRACTION_ATTRIBUTION_SCHEMA_VERSION = 1
PURE_EXTRACTION_OPTIMIZER_SCHEMA_VERSION = 1
PURE_EXTRACTION_OPTIMIZER_SCHEMA = "rsimem-pure-extraction-optimizer-v1"
PURE_EXTRACTION_CORPUS_SCHEMA_VERSION = 1
PURE_EXTRACTION_CORPUS_SCHEMA = "rsimem-pure-extraction-corpus-v1"
_IDENTIFIER = r"^[A-Za-z0-9][A-Za-z0-9_.:+-]{0,255}$"


def _id(value: object, name: str) -> str:
    import re

    if not isinstance(value, str) or re.fullmatch(_IDENTIFIER, value) is None:
        raise ValueError(f"{name} must be a stable identifier")
    return value


def _digest(value: object, name: str) -> str:
    import re

    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be sha256")
    return value


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


class PureExtractionAttribution(StrEnum):
    """Generic process attribution, without benchmark quality semantics."""

    ATTRIBUTABLE_SUCCESS = "attributable_success"
    ATTRIBUTABLE_FAILURE = "attributable_failure"
    UNRESOLVED = "unresolved"
    CENSORED = "censored"


@dataclass(frozen=True, slots=True)
class PureExtractionSourceRecord:
    """Content-free identity for one deployment extraction result."""

    record_id: str
    source_projection_id: str
    source_projection_digest: str
    context_revision: str
    extraction_set_id: str
    extraction_artifact_id: str
    extraction_artifact_digest: str
    extraction_output_digest: str
    source: ExtractionSourceEvidence
    activation: object
    provenance_id: str
    schema_version: int = PURE_EXTRACTION_SOURCE_SCHEMA_VERSION
    evidence_plane: EvidencePlane = EvidencePlane.PURE_PROCESS
    evidence_source: EvidenceSourceKind = EvidenceSourceKind.RUNTIME_OBSERVATION

    def __post_init__(self) -> None:
        if self.schema_version != PURE_EXTRACTION_SOURCE_SCHEMA_VERSION:
            raise ValueError("unsupported pure extraction source schema")
        plane, source = validate_plane_source(self.evidence_plane, self.evidence_source)
        if plane is not EvidencePlane.PURE_PROCESS or source is not EvidenceSourceKind.RUNTIME_OBSERVATION:
            raise ValueError("pure extraction source must be runtime pure-process evidence")
        object.__setattr__(self, "evidence_plane", plane)
        object.__setattr__(self, "evidence_source", source)
        for value, name in (
            (self.record_id, "pure extraction source record ID"),
            (self.source_projection_id, "source projection ID"),
            (self.context_revision, "context revision"),
            (self.extraction_set_id, "extraction set ID"),
            (self.extraction_artifact_id, "extraction artifact ID"),
            (self.provenance_id, "extraction provenance ID"),
        ):
            _id(value, name)
        for value, name in (
            (self.source_projection_digest, "source projection digest"),
            (self.extraction_artifact_digest, "extraction artifact digest"),
            (self.extraction_output_digest, "extraction output digest"),
        ):
            _digest(value, name)
        if not isinstance(self.source, ExtractionSourceEvidence):
            raise TypeError("pure extraction source evidence has the wrong type")
        # Import lazily: ``memory`` is imported while the lifecycle package is
        # still initializing, and extraction_projection depends on lifecycle.
        from .extraction_projection import ExtractionActivationFingerprint

        if not isinstance(self.activation, ExtractionActivationFingerprint):
            raise TypeError("pure extraction activation has the wrong type")
        if self.source.source_projection_digest != self.source_projection_digest:
            raise ValueError("pure extraction source projection digest mismatch")
        if self.source.extraction_set_id != self.extraction_set_id:
            raise ValueError("pure extraction set identity mismatch")
        if self.activation.extraction_operation_id != self.extraction_set_id:
            raise ValueError("pure extraction activation set identity mismatch")
        if self.activation.runtime_binding.component_artifact_id != self.extraction_artifact_id:
            raise ValueError("pure extraction activation artifact identity mismatch")
        if self.activation.runtime_binding.component_body_digest != self.extraction_artifact_digest:
            raise ValueError("pure extraction activation artifact digest mismatch")
        validate_pure_process_payload(self.payload())
        expected = f"pure-extraction-source.{content_digest(self.identity_payload())[:40]}"
        if self.record_id != expected:
            raise ValueError("pure extraction source record ID mismatch")

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "source_projection_id": self.source_projection_id,
            "source_projection_digest": self.source_projection_digest,
            "context_revision": self.context_revision,
            "extraction_set_id": self.extraction_set_id,
            "extraction_artifact_id": self.extraction_artifact_id,
            "extraction_artifact_digest": self.extraction_artifact_digest,
            "extraction_output_digest": self.extraction_output_digest,
            "source": self.source.payload(),
            "activation": self.activation.payload(),
            "provenance_id": self.provenance_id,
            "evidence_plane": self.evidence_plane.value,
            "evidence_source": self.evidence_source.value,
        }

    def payload(self) -> dict[str, object]:
        return {
            "schema": PURE_EXTRACTION_SOURCE_SCHEMA,
            "record_id": self.record_id,
            **self.identity_payload(),
        }

    @classmethod
    def create(
        cls,
        *,
        source_projection_id: str,
        source_projection_digest: str,
        context_revision: str,
        extraction_set_id: str,
        extraction_artifact_id: str,
        extraction_artifact_digest: str,
        extraction_output_digest: str,
        source: ExtractionSourceEvidence,
        activation: ExtractionActivationFingerprint,
        provenance_id: str,
    ) -> "PureExtractionSourceRecord":
        if not isinstance(source, ExtractionSourceEvidence):
            raise TypeError("pure extraction source evidence has the wrong type")
        from .extraction_projection import ExtractionActivationFingerprint

        if not isinstance(activation, ExtractionActivationFingerprint):
            raise TypeError("pure extraction activation has the wrong type")
        values = {
            "source_projection_id": source_projection_id,
            "source_projection_digest": source_projection_digest,
            "context_revision": context_revision,
            "extraction_set_id": extraction_set_id,
            "extraction_artifact_id": extraction_artifact_id,
            "extraction_artifact_digest": extraction_artifact_digest,
            "extraction_output_digest": extraction_output_digest,
            "source": source,
            "activation": activation,
            "provenance_id": provenance_id,
            "schema_version": PURE_EXTRACTION_SOURCE_SCHEMA_VERSION,
            "evidence_plane": EvidencePlane.PURE_PROCESS,
            "evidence_source": EvidenceSourceKind.RUNTIME_OBSERVATION,
        }
        identity = {
            "schema_version": PURE_EXTRACTION_SOURCE_SCHEMA_VERSION,
            "source_projection_id": source_projection_id,
            "source_projection_digest": source_projection_digest,
            "context_revision": context_revision,
            "extraction_set_id": extraction_set_id,
            "extraction_artifact_id": extraction_artifact_id,
            "extraction_artifact_digest": extraction_artifact_digest,
            "extraction_output_digest": extraction_output_digest,
            "source": source.payload(),
            "activation": activation.payload(),
            "provenance_id": provenance_id,
            "evidence_plane": EvidencePlane.PURE_PROCESS.value,
            "evidence_source": EvidenceSourceKind.RUNTIME_OBSERVATION.value,
        }
        return cls(
            record_id=f"pure-extraction-source.{content_digest(identity)[:40]}",
            **values,
        )

    @classmethod
    def from_family_record(
        cls,
        record: object,
        *,
        source_projection_id: str | None = None,
        context_revision: str | None = None,
        provenance_id: str | None = None,
        visible_semantic_keys: tuple[str, ...] = (),
    ) -> "PureExtractionSourceRecord":
        """Project a family-bound audit record without copying its labels.

        ``family_id`` and ``stage`` are intentionally ignored.  This helper
        is useful when replaying an existing capture into the deployment-only
        plane; callers must provide the source projection identity because
        the legacy family record stores only its digest.
        """

        from .extraction_projection import ExtractionSourceRecord

        if not isinstance(record, ExtractionSourceRecord):
            raise TypeError("pure extraction projection requires ExtractionSourceRecord")
        projection_id = source_projection_id or (
            f"extraction-source.{record.source.source_projection_digest[:40]}"
        )
        provenance = provenance_id or (
            f"extraction-provenance.{record.activation.fingerprint_digest[:40]}"
        )
        # ``available_semantic_keys`` on the legacy record may have been
        # populated by a family contract parser.  Only an explicit caller
        # projection is trusted for the deployment-only plane.
        if not isinstance(visible_semantic_keys, tuple):
            raise TypeError("visible semantic keys must be a tuple")
        source_evidence = record.source
        sanitized_source = ExtractionSourceEvidence(
            source_evidence.source_id,
            source_evidence.source_projection_digest,
            source_evidence.extraction_set_id,
            source_evidence.status,
            visible_semantic_keys,
            source_evidence.facts,
        )
        return cls.create(
            source_projection_id=projection_id,
            source_projection_digest=record.source.source_projection_digest,
            context_revision=context_revision
            or f"revision.{record.activation.fingerprint_digest[:40]}",
            extraction_set_id=record.source.extraction_set_id,
            extraction_artifact_id=record.extraction_artifact_id,
            extraction_artifact_digest=record.extraction_artifact_digest,
            extraction_output_digest=record.extraction_output_digest,
            source=sanitized_source,
            activation=record.activation,
            provenance_id=provenance,
        )

    @classmethod
    def from_payload(cls, value: object) -> "PureExtractionSourceRecord":
        fields = {
            "schema", "record_id", "schema_version", "source_projection_id",
            "source_projection_digest", "context_revision", "extraction_set_id",
            "extraction_artifact_id", "extraction_artifact_digest",
            "extraction_output_digest", "source", "activation", "provenance_id",
            "evidence_plane", "evidence_source",
        }
        if not isinstance(value, Mapping) or set(value) != fields or value.get("schema") != PURE_EXTRACTION_SOURCE_SCHEMA:
            raise ValueError("malformed pure extraction source record")
        try:
            result = cls(
                record_id=value["record_id"],
                source_projection_id=value["source_projection_id"],
                source_projection_digest=value["source_projection_digest"],
                context_revision=value["context_revision"],
                extraction_set_id=value["extraction_set_id"],
                extraction_artifact_id=value["extraction_artifact_id"],
                extraction_artifact_digest=value["extraction_artifact_digest"],
                extraction_output_digest=value["extraction_output_digest"],
                source=ExtractionSourceEvidence.from_payload(value["source"]),
                activation=__import__(
                    "rsimem.memory.extraction_projection",
                    fromlist=["ExtractionActivationFingerprint"],
                ).ExtractionActivationFingerprint.from_payload(value["activation"]),
                provenance_id=value["provenance_id"],
                schema_version=value["schema_version"],
                evidence_plane=EvidencePlane(value["evidence_plane"]),
                evidence_source=EvidenceSourceKind(value["evidence_source"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("malformed pure extraction source record") from exc
        if result.payload() != dict(value):
            raise ValueError("non-canonical pure extraction source record")
        return result


@dataclass(frozen=True, slots=True)
class PureExtractionFeedbackRecord:
    """Exact pure-process joins for delayed extraction attribution."""

    record_id: str
    source_record_id: str
    source_projection_digest: str
    extraction_set_id: str
    opportunity: OpportunityEvidence | None
    memory_use: MemoryUseEvidence | None
    artifact_set_binding: ArtifactSetSemanticBinding | None
    tool_joins: tuple[ToolCallResultJoin, ...]
    observation_window: str
    provenance_id: str
    attribution: PureExtractionAttribution = PureExtractionAttribution.UNRESOLVED
    reason_codes: tuple[str, ...] = ()
    observation_complete: bool = True
    schema_version: int = PURE_EXTRACTION_FEEDBACK_SCHEMA_VERSION
    attribution_schema_version: int = PURE_EXTRACTION_ATTRIBUTION_SCHEMA_VERSION
    evidence_plane: EvidencePlane = EvidencePlane.PURE_PROCESS
    evidence_source: EvidenceSourceKind = EvidenceSourceKind.RUNTIME_OBSERVATION

    def __post_init__(self) -> None:
        if self.schema_version != PURE_EXTRACTION_FEEDBACK_SCHEMA_VERSION:
            raise ValueError("unsupported pure extraction feedback schema")
        if self.attribution_schema_version != PURE_EXTRACTION_ATTRIBUTION_SCHEMA_VERSION:
            raise ValueError("unsupported pure extraction attribution schema")
        plane, source = validate_plane_source(self.evidence_plane, self.evidence_source)
        if plane is not EvidencePlane.PURE_PROCESS or source is not EvidenceSourceKind.RUNTIME_OBSERVATION:
            raise ValueError("pure extraction feedback must be runtime pure-process evidence")
        object.__setattr__(self, "evidence_plane", plane)
        object.__setattr__(self, "evidence_source", source)
        for value, name in (
            (self.record_id, "pure extraction feedback record ID"),
            (self.source_record_id, "pure extraction source record ID"),
            (self.extraction_set_id, "extraction set ID"),
            (self.observation_window, "observation window"),
            (self.provenance_id, "feedback provenance ID"),
        ):
            _id(value, name)
        _digest(self.source_projection_digest, "feedback source projection digest")
        if self.opportunity is not None and not isinstance(self.opportunity, OpportunityEvidence):
            raise TypeError("pure extraction opportunity has the wrong type")
        if self.memory_use is not None and not isinstance(self.memory_use, MemoryUseEvidence):
            raise TypeError("pure extraction memory-use evidence has the wrong type")
        if self.artifact_set_binding is not None and not isinstance(self.artifact_set_binding, ArtifactSetSemanticBinding):
            raise TypeError("pure extraction artifact-set binding has the wrong type")
        object.__setattr__(self, "attribution", PureExtractionAttribution(self.attribution))
        if self.memory_use is not None:
            if self.memory_use.provenance_id != self.provenance_id:
                raise ValueError("memory-use provenance does not match extraction feedback")
            if self.artifact_set_binding is not None and self.memory_use.artifact_set_id != self.artifact_set_binding.binding_id:
                raise ValueError("artifact-set binding does not match memory-use evidence")
            if self.memory_use.artifact_set_id is not None and self.artifact_set_binding is None:
                raise ValueError("memory-use artifact set requires a trusted binding")
        if self.attribution in {
            PureExtractionAttribution.ATTRIBUTABLE_SUCCESS,
            PureExtractionAttribution.ATTRIBUTABLE_FAILURE,
        }:
            if self.opportunity is None or self.memory_use is None:
                raise ValueError("attributable extraction feedback requires opportunity and use evidence")
            if not self.memory_use.used_artifact_ids:
                raise ValueError("attributable extraction feedback requires used artifacts")
            expected_outcome = self.attribution is PureExtractionAttribution.ATTRIBUTABLE_SUCCESS
            if self.memory_use.outcome_success is not expected_outcome:
                raise ValueError("attributable extraction feedback outcome is inconsistent")
        join_ids = tuple(join.join_id for join in self.tool_joins)
        if len(join_ids) != len(set(join_ids)):
            raise ValueError("pure extraction tool joins must be unique")
        for join in self.tool_joins:
            if not isinstance(join, ToolCallResultJoin):
                raise TypeError("pure extraction tool join has the wrong type")
            if (
                join.memory_use_operation_id is not None
                and self.memory_use is not None
                and join.memory_use_operation_id
                not in {
                    self.memory_use.retrieval_operation_id,
                    self.memory_use.injection_operation_id,
                    self.memory_use.downstream_operation_id,
                    self.memory_use.outcome_operation_id,
                }
            ):
                raise ValueError("tool join operation does not match memory-use evidence")
        if type(self.observation_complete) is not bool:
            raise TypeError("pure extraction observation completeness must be bool")
        if not self.observation_complete and self.attribution is not PureExtractionAttribution.CENSORED:
            raise ValueError("incomplete pure extraction observation must be censored")
        if self.attribution is PureExtractionAttribution.CENSORED and self.observation_complete:
            raise ValueError("censored pure extraction feedback must be incomplete")
        if len(self.reason_codes) != len(set(self.reason_codes)):
            raise ValueError("pure extraction reason codes must be unique")
        for value in self.reason_codes:
            _id(value, "pure extraction reason code")
        validate_pure_process_payload(self.payload())
        expected = f"pure-extraction-feedback.{content_digest(self.identity_payload())[:40]}"
        if self.record_id != expected:
            raise ValueError("pure extraction feedback record ID mismatch")

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "attribution_schema_version": self.attribution_schema_version,
            "source_record_id": self.source_record_id,
            "source_projection_digest": self.source_projection_digest,
            "extraction_set_id": self.extraction_set_id,
            "opportunity": self.opportunity.payload() if self.opportunity else None,
            "memory_use": self.memory_use.payload() if self.memory_use else None,
            "artifact_set_binding": self.artifact_set_binding.payload() if self.artifact_set_binding else None,
            "tool_joins": [value.payload() for value in self.tool_joins],
            "observation_window": self.observation_window,
            "provenance_id": self.provenance_id,
            "attribution": self.attribution.value,
            "reason_codes": list(self.reason_codes),
            "observation_complete": self.observation_complete,
            "evidence_plane": self.evidence_plane.value,
            "evidence_source": self.evidence_source.value,
        }

    def payload(self) -> dict[str, object]:
        return {
            "schema": PURE_EXTRACTION_FEEDBACK_SCHEMA,
            "record_id": self.record_id,
            **self.identity_payload(),
        }

    @classmethod
    def create(
        cls,
        *,
        source_record_id: str,
        source_projection_digest: str,
        extraction_set_id: str,
        opportunity: OpportunityEvidence | None,
        memory_use: MemoryUseEvidence | None,
        artifact_set_binding: ArtifactSetSemanticBinding | None = None,
        tool_joins: tuple[ToolCallResultJoin, ...] = (),
        observation_window: str,
        provenance_id: str,
        attribution: PureExtractionAttribution = PureExtractionAttribution.UNRESOLVED,
        reason_codes: tuple[str, ...] = (),
        observation_complete: bool = True,
    ) -> "PureExtractionFeedbackRecord":
        values = {
            "source_record_id": source_record_id,
            "source_projection_digest": source_projection_digest,
            "extraction_set_id": extraction_set_id,
            "opportunity": opportunity,
            "memory_use": memory_use,
            "artifact_set_binding": artifact_set_binding,
            "tool_joins": tuple(tool_joins),
            "observation_window": observation_window,
            "provenance_id": provenance_id,
            "attribution": PureExtractionAttribution(attribution),
            "reason_codes": tuple(reason_codes),
            "observation_complete": observation_complete,
            "schema_version": PURE_EXTRACTION_FEEDBACK_SCHEMA_VERSION,
            "attribution_schema_version": PURE_EXTRACTION_ATTRIBUTION_SCHEMA_VERSION,
            "evidence_plane": EvidencePlane.PURE_PROCESS,
            "evidence_source": EvidenceSourceKind.RUNTIME_OBSERVATION,
        }
        identity = {
            **values,
            "opportunity": opportunity.payload() if opportunity else None,
            "memory_use": memory_use.payload() if memory_use else None,
            "artifact_set_binding": artifact_set_binding.payload() if artifact_set_binding else None,
            "tool_joins": [value.payload() for value in values["tool_joins"]],
            "attribution": values["attribution"].value,
            "reason_codes": list(values["reason_codes"]),
            "evidence_plane": EvidencePlane.PURE_PROCESS.value,
            "evidence_source": EvidenceSourceKind.RUNTIME_OBSERVATION.value,
        }
        return cls(
            record_id=f"pure-extraction-feedback.{content_digest(identity)[:40]}",
            **values,
        )

    @classmethod
    def derive_from_evidence(
        cls,
        *,
        source: PureExtractionSourceRecord,
        opportunity: OpportunityEvidence | None,
        memory_use: MemoryUseEvidence | None,
        observation_window: str,
        provenance_id: str,
        artifact_set_binding: ArtifactSetSemanticBinding | None = None,
        tool_joins: tuple[ToolCallResultJoin, ...] = (),
        current_input_requirements: tuple[str, ...] = (),
        operation_graph: object | None = None,
        observation_complete: bool = True,
    ) -> "PureExtractionFeedbackRecord":
        """Derive attribution without trusting a caller-provided label.

        Opportunity/use resolvers are deterministic and fail closed.  A
        missing or partial join therefore remains ``UNRESOLVED`` rather than
        becoming extraction credit or blame.
        """

        if not isinstance(source, PureExtractionSourceRecord):
            raise TypeError("pure extraction feedback source has the wrong type")
        from .opportunity import OpportunityResolutionStatus, resolve_opportunity
        from .use_attribution import MemoryUseResolutionStatus, resolve_memory_use

        opportunity_resolution = resolve_opportunity(
            opportunity,
            current_input_requirements=current_input_requirements,
            observation_complete=observation_complete,
        )
        memory_resolution = None
        if memory_use is not None:
            memory_resolution = resolve_memory_use(
                memory_use,
                artifact_set_binding=artifact_set_binding,
                operation_graph=operation_graph,
            )
        attribution = PureExtractionAttribution.UNRESOLVED
        reasons: list[str] = []
        if not observation_complete:
            attribution = PureExtractionAttribution.CENSORED
            reasons.append("observation_censored")
        elif opportunity_resolution.status is not OpportunityResolutionStatus.OBSERVED:
            reasons.append(opportunity_resolution.reason_code)
        elif memory_resolution is None:
            reasons.append("memory_use_missing")
        elif memory_resolution.status is not MemoryUseResolutionStatus.ATTRIBUTABLE_USE:
            reasons.append(memory_resolution.reason_code)
        elif memory_use is None or memory_use.outcome_success is None:
            reasons.append("outcome_unknown")
        elif memory_use.outcome_success:
            attribution = PureExtractionAttribution.ATTRIBUTABLE_SUCCESS
            reasons.append("attributable_success")
        else:
            attribution = PureExtractionAttribution.ATTRIBUTABLE_FAILURE
            reasons.append("attributable_failure")
        return cls.create(
            source_record_id=source.record_id,
            source_projection_digest=source.source_projection_digest,
            extraction_set_id=source.extraction_set_id,
            opportunity=opportunity,
            memory_use=memory_use,
            artifact_set_binding=artifact_set_binding,
            tool_joins=tool_joins,
            observation_window=observation_window,
            provenance_id=provenance_id,
            attribution=attribution,
            reason_codes=tuple(dict.fromkeys(reasons)),
            observation_complete=observation_complete,
        )

    @classmethod
    def from_payload(cls, value: object) -> "PureExtractionFeedbackRecord":
        fields = {
            "schema", "record_id", "schema_version", "attribution_schema_version",
            "source_record_id", "source_projection_digest", "extraction_set_id",
            "opportunity", "memory_use", "artifact_set_binding", "tool_joins",
            "observation_window", "provenance_id", "attribution", "reason_codes",
            "observation_complete", "evidence_plane", "evidence_source",
        }
        if not isinstance(value, Mapping) or set(value) != fields or value.get("schema") != PURE_EXTRACTION_FEEDBACK_SCHEMA:
            raise ValueError("malformed pure extraction feedback record")
        if not isinstance(value["tool_joins"], list) or not isinstance(value["reason_codes"], list):
            raise ValueError("malformed pure extraction feedback collections")
        try:
            result = cls(
                record_id=value["record_id"],
                source_record_id=value["source_record_id"],
                source_projection_digest=value["source_projection_digest"],
                extraction_set_id=value["extraction_set_id"],
                opportunity=OpportunityEvidence.from_payload(value["opportunity"]) if value["opportunity"] is not None else None,
                memory_use=MemoryUseEvidence.from_payload(value["memory_use"]) if value["memory_use"] is not None else None,
                artifact_set_binding=ArtifactSetSemanticBinding.from_payload(value["artifact_set_binding"]) if value["artifact_set_binding"] is not None else None,
                tool_joins=tuple(ToolCallResultJoin.from_payload(item) for item in value["tool_joins"]),
                observation_window=value["observation_window"],
                provenance_id=value["provenance_id"],
                attribution=value["attribution"],
                reason_codes=tuple(value["reason_codes"]),
                observation_complete=value["observation_complete"],
                schema_version=value["schema_version"],
                attribution_schema_version=value["attribution_schema_version"],
                evidence_plane=EvidencePlane(value["evidence_plane"]),
                evidence_source=EvidenceSourceKind(value["evidence_source"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("malformed pure extraction feedback record") from exc
        if result.payload() != dict(value):
            raise ValueError("non-canonical pure extraction feedback record")
        return result


@dataclass(frozen=True, slots=True)
class PureExtractionOptimizerExample:
    """A content-free optimizer unit built only from pure-process joins.

    Source text and extracted fact text remain in owner-controlled capture
    storage.  This identity is the safe join presented to the optimizer
    builder; it has no benchmark family, stage, score, or answer metadata.
    """

    example_id: str
    source_record_id: str
    source_projection_id: str
    source_projection_digest: str
    extraction_set_id: str
    extraction_artifact_id: str
    extraction_artifact_digest: str
    extraction_output_digest: str
    fact_ids: tuple[str, ...]
    semantic_keys: tuple[str, ...]
    opportunity_evidence_id: str | None
    opportunity_operation_id: str | None
    memory_use_evidence_id: str | None
    memory_use_operation_id: str | None
    outcome_operation_id: str | None
    artifact_set_binding_id: str | None
    tool_join_ids: tuple[str, ...]
    observation_window: str
    provenance_id: str
    attribution: PureExtractionAttribution
    reason_codes: tuple[str, ...]
    observation_complete: bool
    schema_version: int = PURE_EXTRACTION_OPTIMIZER_SCHEMA_VERSION
    evidence_plane: EvidencePlane = EvidencePlane.PURE_PROCESS
    evidence_source: EvidenceSourceKind = EvidenceSourceKind.RUNTIME_OBSERVATION

    def __post_init__(self) -> None:
        if self.schema_version != PURE_EXTRACTION_OPTIMIZER_SCHEMA_VERSION:
            raise ValueError("unsupported pure extraction optimizer schema")
        plane, source = validate_plane_source(self.evidence_plane, self.evidence_source)
        if plane is not EvidencePlane.PURE_PROCESS or source is not EvidenceSourceKind.RUNTIME_OBSERVATION:
            raise ValueError("pure extraction optimizer example must be runtime pure-process evidence")
        object.__setattr__(self, "evidence_plane", plane)
        object.__setattr__(self, "evidence_source", source)
        for value, name in (
            (self.example_id, "pure extraction optimizer example ID"),
            (self.source_record_id, "pure extraction source record ID"),
            (self.source_projection_id, "source projection ID"),
            (self.extraction_set_id, "extraction set ID"),
            (self.extraction_artifact_id, "extraction artifact ID"),
            (self.observation_window, "observation window"),
            (self.provenance_id, "optimizer provenance ID"),
        ):
            _id(value, name)
        for value, name in (
            (self.source_projection_digest, "source projection digest"),
            (self.extraction_artifact_digest, "extraction artifact digest"),
            (self.extraction_output_digest, "extraction output digest"),
        ):
            _digest(value, name)
        for values, name in (
            (self.fact_ids, "optimizer fact IDs"),
            (self.semantic_keys, "optimizer semantic keys"),
            (self.tool_join_ids, "optimizer tool join IDs"),
            (self.reason_codes, "optimizer reason codes"),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{name} must be unique")
            for value in values:
                _id(value, name)
        for value, name in (
            (self.opportunity_evidence_id, "opportunity evidence ID"),
            (self.opportunity_operation_id, "opportunity operation ID"),
            (self.memory_use_evidence_id, "memory-use evidence ID"),
            (self.memory_use_operation_id, "memory-use operation ID"),
            (self.outcome_operation_id, "outcome operation ID"),
            (self.artifact_set_binding_id, "artifact-set binding ID"),
        ):
            if value is not None:
                _id(value, name)
        object.__setattr__(self, "attribution", PureExtractionAttribution(self.attribution))
        if type(self.observation_complete) is not bool:
            raise TypeError("optimizer observation completeness must be bool")
        if not self.observation_complete and self.attribution is not PureExtractionAttribution.CENSORED:
            raise ValueError("incomplete optimizer observation must be censored")
        if self.attribution is PureExtractionAttribution.CENSORED and self.observation_complete:
            raise ValueError("censored optimizer observation must be incomplete")
        if self.memory_use_evidence_id is None and any(
            value is not None
            for value in (self.memory_use_operation_id, self.outcome_operation_id)
        ):
            raise ValueError("memory-use operations require memory-use evidence")
        if self.opportunity_evidence_id is None and self.opportunity_operation_id is not None:
            raise ValueError("opportunity operation requires opportunity evidence")
        validate_pure_process_payload(self.payload())
        if self.example_id != f"pure-extraction-example.{content_digest(self.identity_payload())[:40]}":
            raise ValueError("pure extraction optimizer example ID mismatch")

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "source_record_id": self.source_record_id,
            "source_projection_id": self.source_projection_id,
            "source_projection_digest": self.source_projection_digest,
            "extraction_set_id": self.extraction_set_id,
            "extraction_artifact_id": self.extraction_artifact_id,
            "extraction_artifact_digest": self.extraction_artifact_digest,
            "extraction_output_digest": self.extraction_output_digest,
            "fact_ids": list(self.fact_ids),
            "semantic_keys": list(self.semantic_keys),
            "opportunity_evidence_id": self.opportunity_evidence_id,
            "opportunity_operation_id": self.opportunity_operation_id,
            "memory_use_evidence_id": self.memory_use_evidence_id,
            "memory_use_operation_id": self.memory_use_operation_id,
            "outcome_operation_id": self.outcome_operation_id,
            "artifact_set_binding_id": self.artifact_set_binding_id,
            "tool_join_ids": list(self.tool_join_ids),
            "observation_window": self.observation_window,
            "provenance_id": self.provenance_id,
            "attribution": self.attribution.value,
            "reason_codes": list(self.reason_codes),
            "observation_complete": self.observation_complete,
            "evidence_plane": self.evidence_plane.value,
            "evidence_source": self.evidence_source.value,
        }

    def payload(self) -> dict[str, object]:
        return {
            "schema": PURE_EXTRACTION_OPTIMIZER_SCHEMA,
            **self.identity_payload(),
            "example_id": self.example_id,
        }

    @classmethod
    def from_records(
        cls,
        source: PureExtractionSourceRecord,
        feedback: PureExtractionFeedbackRecord,
    ) -> "PureExtractionOptimizerExample":
        if not isinstance(source, PureExtractionSourceRecord):
            raise TypeError("pure optimizer source has the wrong type")
        if not isinstance(feedback, PureExtractionFeedbackRecord):
            raise TypeError("pure optimizer feedback has the wrong type")
        if feedback.source_record_id != source.record_id:
            raise ValueError("pure optimizer source/feedback record join mismatch")
        if feedback.source_projection_digest != source.source_projection_digest:
            raise ValueError("pure optimizer source projection join mismatch")
        if feedback.extraction_set_id != source.extraction_set_id:
            raise ValueError("pure optimizer extraction set join mismatch")
        if feedback.observation_window == "window.unbound":
            raise ValueError("pure optimizer observation window is unbound")
        if (
            feedback.attribution
            in {
                PureExtractionAttribution.ATTRIBUTABLE_SUCCESS,
                PureExtractionAttribution.ATTRIBUTABLE_FAILURE,
            }
            and feedback.opportunity is not None
            and feedback.opportunity.semantic_requirement
            not in source.source.available_semantic_keys
        ):
            raise ValueError("pure optimizer opportunity is not bound to source evidence")
        if feedback.attribution in {
            PureExtractionAttribution.ATTRIBUTABLE_SUCCESS,
            PureExtractionAttribution.ATTRIBUTABLE_FAILURE,
        }:
            source_artifacts = {
                value.artifact_id
                for value in source.source.facts
                if value.artifact_id is not None
            }
            used = set(feedback.memory_use.used_artifact_ids) if feedback.memory_use else set()
            if not used or not used.issubset(source_artifacts):
                raise ValueError("pure optimizer attribution escapes source artifacts")
        opportunity_id = feedback.opportunity.evidence_id if feedback.opportunity else None
        opportunity_operation_id = feedback.opportunity.operation_id if feedback.opportunity else None
        memory_use_id = feedback.memory_use.evidence_id if feedback.memory_use else None
        memory_use_operation_id = (
            feedback.memory_use.downstream_operation_id if feedback.memory_use else None
        )
        outcome_operation_id = feedback.memory_use.outcome_operation_id if feedback.memory_use else None
        fact_ids = tuple(value.fact_id for value in source.source.facts)
        semantic_keys = tuple(dict.fromkeys(
            key for value in source.source.facts for key in value.semantic_keys
        ))
        values = {
            "source_record_id": source.record_id,
            "source_projection_id": source.source_projection_id,
            "source_projection_digest": source.source_projection_digest,
            "extraction_set_id": source.extraction_set_id,
            "extraction_artifact_id": source.extraction_artifact_id,
            "extraction_artifact_digest": source.extraction_artifact_digest,
            "extraction_output_digest": source.extraction_output_digest,
            "fact_ids": fact_ids,
            "semantic_keys": semantic_keys,
            "opportunity_evidence_id": opportunity_id,
            "opportunity_operation_id": opportunity_operation_id,
            "memory_use_evidence_id": memory_use_id,
            "memory_use_operation_id": memory_use_operation_id,
            "outcome_operation_id": outcome_operation_id,
            "artifact_set_binding_id": (
                feedback.artifact_set_binding.binding_id
                if feedback.artifact_set_binding else None
            ),
            "tool_join_ids": tuple(value.join_id for value in feedback.tool_joins),
            "observation_window": feedback.observation_window,
            "provenance_id": feedback.provenance_id,
            "attribution": feedback.attribution,
            "reason_codes": feedback.reason_codes,
            "observation_complete": feedback.observation_complete,
            "schema_version": PURE_EXTRACTION_OPTIMIZER_SCHEMA_VERSION,
            "evidence_plane": EvidencePlane.PURE_PROCESS,
            "evidence_source": EvidenceSourceKind.RUNTIME_OBSERVATION,
        }
        identity = {
            **values,
            "fact_ids": list(fact_ids),
            "semantic_keys": list(semantic_keys),
            "tool_join_ids": list(values["tool_join_ids"]),
            "reason_codes": list(feedback.reason_codes),
            "attribution": feedback.attribution.value,
            "evidence_plane": EvidencePlane.PURE_PROCESS.value,
            "evidence_source": EvidenceSourceKind.RUNTIME_OBSERVATION.value,
        }
        return cls(
            example_id=f"pure-extraction-example.{content_digest(identity)[:40]}",
            **values,
        )

    @classmethod
    def from_payload(cls, value: object) -> "PureExtractionOptimizerExample":
        fields = {
            "schema", "example_id", "schema_version", "source_record_id",
            "source_projection_id", "source_projection_digest", "extraction_set_id",
            "extraction_artifact_id", "extraction_artifact_digest",
            "extraction_output_digest", "fact_ids", "semantic_keys",
            "opportunity_evidence_id", "opportunity_operation_id",
            "memory_use_evidence_id", "memory_use_operation_id", "outcome_operation_id",
            "artifact_set_binding_id", "tool_join_ids", "observation_window",
            "provenance_id", "attribution", "reason_codes", "observation_complete",
            "evidence_plane", "evidence_source",
        }
        if not isinstance(value, Mapping) or set(value) != fields or value.get("schema") != PURE_EXTRACTION_OPTIMIZER_SCHEMA:
            raise ValueError("malformed pure extraction optimizer example")
        for name in ("fact_ids", "semantic_keys", "tool_join_ids", "reason_codes"):
            if not isinstance(value[name], list):
                raise ValueError("malformed pure extraction optimizer collections")
        try:
            result = cls(
                example_id=value["example_id"],
                source_record_id=value["source_record_id"],
                source_projection_id=value["source_projection_id"],
                source_projection_digest=value["source_projection_digest"],
                extraction_set_id=value["extraction_set_id"],
                extraction_artifact_id=value["extraction_artifact_id"],
                extraction_artifact_digest=value["extraction_artifact_digest"],
                extraction_output_digest=value["extraction_output_digest"],
                fact_ids=tuple(value["fact_ids"]),
                semantic_keys=tuple(value["semantic_keys"]),
                opportunity_evidence_id=value["opportunity_evidence_id"],
                opportunity_operation_id=value["opportunity_operation_id"],
                memory_use_evidence_id=value["memory_use_evidence_id"],
                memory_use_operation_id=value["memory_use_operation_id"],
                outcome_operation_id=value["outcome_operation_id"],
                artifact_set_binding_id=value["artifact_set_binding_id"],
                tool_join_ids=tuple(value["tool_join_ids"]),
                observation_window=value["observation_window"],
                provenance_id=value["provenance_id"],
                attribution=value["attribution"],
                reason_codes=tuple(value["reason_codes"]),
                observation_complete=value["observation_complete"],
                schema_version=value["schema_version"],
                evidence_plane=EvidencePlane(value["evidence_plane"]),
                evidence_source=EvidenceSourceKind(value["evidence_source"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("malformed pure extraction optimizer example") from exc
        if result.payload() != dict(value):
            raise ValueError("non-canonical pure extraction optimizer example")
        return result


@dataclass(frozen=True, slots=True)
class PureExtractionOptimizerCorpus:
    """Replay-stable collection of pure extraction optimizer identities."""

    corpus_id: str
    split: str
    observation_cutoff: str
    examples: tuple[PureExtractionOptimizerExample, ...]
    schema_version: int = PURE_EXTRACTION_CORPUS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PURE_EXTRACTION_CORPUS_SCHEMA_VERSION:
            raise ValueError("unsupported pure extraction corpus schema")
        _id(self.corpus_id, "pure extraction corpus ID")
        if self.split not in {"train", "validation", "future_test"}:
            raise ValueError("pure extraction corpus split is invalid")
        if not isinstance(self.observation_cutoff, str) or not self.observation_cutoff.endswith("Z"):
            raise ValueError("pure extraction corpus cutoff is invalid")
        try:
            datetime.fromisoformat(self.observation_cutoff.removesuffix("Z") + "+00:00")
        except ValueError as exc:
            raise ValueError("pure extraction corpus cutoff is invalid") from exc
        if not self.examples:
            raise ValueError("pure extraction corpus requires examples")
        if any(not isinstance(value, PureExtractionOptimizerExample) for value in self.examples):
            raise TypeError("pure extraction corpus example has the wrong type")
        ids = tuple(value.example_id for value in self.examples)
        if len(ids) != len(set(ids)):
            raise ValueError("pure extraction corpus examples must be unique")
        if ids != tuple(sorted(ids)):
            raise ValueError("pure extraction corpus examples must be canonically ordered")
        expected = f"pure-extraction-corpus.{content_digest(self.identity_payload())[:40]}"
        if self.corpus_id != expected:
            raise ValueError("pure extraction corpus ID mismatch")

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "split": self.split,
            "observation_cutoff": self.observation_cutoff,
            "example_ids": [value.example_id for value in self.examples],
        }

    def payload(self) -> dict[str, object]:
        return {
            "schema": PURE_EXTRACTION_CORPUS_SCHEMA,
            "corpus_id": self.corpus_id,
            **self.identity_payload(),
            "examples": [value.payload() for value in self.examples],
        }

    @classmethod
    def create(
        cls,
        *,
        split: str,
        observation_cutoff: str,
        examples: tuple[PureExtractionOptimizerExample, ...],
    ) -> "PureExtractionOptimizerCorpus":
        ordered = tuple(sorted(examples, key=lambda value: value.example_id))
        identity = {
            "schema_version": PURE_EXTRACTION_CORPUS_SCHEMA_VERSION,
            "split": split,
            "observation_cutoff": observation_cutoff,
            "example_ids": [value.example_id for value in ordered],
        }
        return cls(
            corpus_id=f"pure-extraction-corpus.{content_digest(identity)[:40]}",
            split=split,
            observation_cutoff=observation_cutoff,
            examples=ordered,
        )

    @classmethod
    def from_payload(cls, value: object) -> "PureExtractionOptimizerCorpus":
        fields = {
            "schema", "corpus_id", "schema_version", "split", "observation_cutoff",
            "example_ids", "examples",
        }
        if not isinstance(value, Mapping) or set(value) != fields or value.get("schema") != PURE_EXTRACTION_CORPUS_SCHEMA:
            raise ValueError("malformed pure extraction optimizer corpus")
        if not isinstance(value["example_ids"], list) or not isinstance(value["examples"], list):
            raise ValueError("malformed pure extraction corpus collections")
        try:
            examples = tuple(PureExtractionOptimizerExample.from_payload(item) for item in value["examples"])
            result = cls(
                corpus_id=value["corpus_id"],
                split=value["split"],
                observation_cutoff=value["observation_cutoff"],
                examples=examples,
                schema_version=value["schema_version"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("malformed pure extraction optimizer corpus") from exc
        if list(item.example_id for item in result.examples) != value["example_ids"] or result.payload() != dict(value):
            raise ValueError("non-canonical pure extraction optimizer corpus")
        return result


class PureExtractionSourceProjector:
    """Project a live extraction trace without consulting a family contract.

    The projector intentionally leaves fact semantic keys empty unless the
    caller supplies an explicit deployment-visible mapping.  This prevents a
    benchmark adapter's parser from becoming an implicit opportunity source.
    """

    def project_record(
        self,
        boundary: object,
        policy: object,
        runtime_binding: object,
        *,
        source_projection_id: str,
        context_revision: str,
        provenance_id: str,
        visible_semantic_keys: tuple[str, ...] = (),
        fact_semantic_keys: Mapping[str, tuple[str, ...]] | None = None,
    ) -> PureExtractionSourceRecord:
        from .extraction_projection import ExtractionActivationFingerprint
        from .executor import MutationExecutionStatus
        from .ingestion import InternalMemoryAction, MemoryIngestStatus
        from .live_writeback import ExtractionRuntimeBinding, StaticSemanticBoundaryResult
        from ..memory_systems.mem0_flat.policy import Mem0FlatSemanticPolicy

        if not isinstance(boundary, StaticSemanticBoundaryResult):
            raise TypeError("pure extraction projector boundary has the wrong type")
        if not isinstance(policy, Mem0FlatSemanticPolicy):
            raise TypeError("pure extraction projector policy has the wrong type")
        if not isinstance(runtime_binding, ExtractionRuntimeBinding):
            raise TypeError("pure extraction projector runtime binding has the wrong type")
        if boundary.duplicate or boundary.writeback is None:
            raise ValueError("pure extraction projection requires an original writeback result")
        ingestion = boundary.writeback.ingestion
        if ingestion is None:
            raise ValueError("pure extraction projection requires an ingestion result")
        trace = policy.operation_trace(ingestion.idempotency_key)
        if trace is None:
            raise ValueError("pure extraction projection requires a Mem0-flat operation trace")
        if not isinstance(visible_semantic_keys, tuple) or len(visible_semantic_keys) != len(set(visible_semantic_keys)):
            raise ValueError("visible extraction semantic keys must be a unique tuple")
        fact_semantic_keys = fact_semantic_keys or {}
        operations = ingestion.operations
        executions = boundary.writeback.executions
        accepted_index = 0
        facts: list[object] = []
        for extraction in trace.fact_extractions:
            fact = policy.fact_for_digest(extraction.content_digest)
            if fact is None or fact.fact_id != extraction.fact_id:
                raise ValueError("pure extraction fact owner disagrees with trace")
            keys = tuple(fact_semantic_keys.get(extraction.fact_id, ()))
            quality_issue = None
            artifact_id = None
            if not extraction.accepted:
                disposition = FactDisposition.FILTERED
            else:
                operation = operations[accepted_index] if accepted_index < len(operations) else None
                execution = executions[accepted_index] if accepted_index < len(executions) else None
                accepted_index += 1
                if operation is None or ingestion.status is not MemoryIngestStatus.SUCCESS:
                    disposition = FactDisposition.MUTATION_FAILED
                elif operation.action in {InternalMemoryAction.NONE, InternalMemoryAction.DELETE}:
                    disposition = FactDisposition.NONE
                elif (
                    execution is not None
                    and execution.status in {MutationExecutionStatus.COMMITTED, MutationExecutionStatus.DUPLICATE}
                    and execution.artifact_id is not None
                ):
                    disposition = FactDisposition.PERSISTED
                    artifact_id = execution.artifact_id
                else:
                    disposition = FactDisposition.MUTATION_FAILED
            facts.append(ExtractedFactEvidence(
                extraction.fact_id,
                keys,
                disposition,
                artifact_id=artifact_id,
                quality_issue=quality_issue,
            ))
        dispositions = {fact.disposition for fact in facts}
        if not facts:
            status = (
                ExtractionSetStatus.EMPTY
                if ingestion.status is MemoryIngestStatus.SUCCESS
                and any(operation.action is InternalMemoryAction.NONE for operation in operations)
                else ExtractionSetStatus.NONE
            )
        elif FactDisposition.MUTATION_FAILED in dispositions:
            status = ExtractionSetStatus.MUTATION_FAILED
        elif FactDisposition.PERSISTED in dispositions:
            status = ExtractionSetStatus.NONEMPTY
        elif dispositions == {FactDisposition.FILTERED}:
            status = ExtractionSetStatus.FILTERED
        else:
            status = ExtractionSetStatus.NONE
        source = ExtractionSourceEvidence(
            trace.source_artifact_id,
            ingestion.source_digest,
            trace.extraction_operation_id,
            status,
            visible_semantic_keys,
            tuple(facts),
        )
        invocation = policy.extraction_invocation(ingestion.idempotency_key)
        if invocation is None:
            raise ValueError("pure extraction projection requires invocation fingerprint")
        extraction_output_digest = content_digest([
            {
                "fact_id": fact.fact_id,
                "content_digest": fact.content_digest,
                "accepted": fact.accepted,
                "reason_code": fact.reason_code,
            }
            for fact in trace.fact_extractions
        ])
        activation = ExtractionActivationFingerprint.create(
            compilation_id=boundary.compilation_id,
            extraction_operation_id=trace.extraction_operation_id,
            runtime_binding=runtime_binding,
            semantic_policy=policy.semantic_manifest,
            invocation=invocation,
            parsed_output_digest=extraction_output_digest,
            mutation_ids=tuple(dict.fromkeys(execution.mutation_id for execution in executions)),
            persisted_artifact_ids=tuple(dict.fromkeys(
                execution.artifact_id
                for execution in executions
                if execution.artifact_id is not None
                and execution.status in {MutationExecutionStatus.COMMITTED, MutationExecutionStatus.DUPLICATE}
            )),
        )
        return PureExtractionSourceRecord.create(
            source_projection_id=source_projection_id,
            source_projection_digest=ingestion.source_digest,
            context_revision=context_revision,
            extraction_set_id=trace.extraction_operation_id,
            extraction_artifact_id=policy.semantic_manifest.extraction_component_id,
            extraction_artifact_digest=policy.semantic_manifest.extraction_component_digest,
            extraction_output_digest=extraction_output_digest,
            source=source,
            activation=activation,
            provenance_id=provenance_id,
        )


class _JsonPureExtractionStore:
    record_type = PureExtractionSourceRecord

    def __init__(self, path: Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.lock_path = self.path.with_name(self.path.name + ".lock")

    @contextmanager
    def _lock(self, operation: int) -> Iterator[None]:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), operation)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _read_unlocked(self) -> tuple[object, ...]:
        if not self.path.exists():
            return ()
        records: list[object] = []
        canonical_by_id: dict[str, str] = {}
        for line_number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                raise ValueError(f"malformed pure extraction store at line {line_number}")
            try:
                record = self.record_type.from_payload(json.loads(line))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(f"malformed pure extraction store at line {line_number}") from exc
            canonical = _canonical(record.payload())
            previous = canonical_by_id.get(record.record_id)
            if previous is not None and previous != canonical:
                raise ValueError("conflicting pure extraction record")
            if previous is None:
                canonical_by_id[record.record_id] = canonical
                records.append(record)
        return tuple(records)

    def records(self) -> tuple[object, ...]:
        with self._lock(fcntl.LOCK_SH):
            return self._read_unlocked()

    def append(self, record: object) -> bool:
        if not isinstance(record, self.record_type):
            raise TypeError("pure extraction store received the wrong record type")
        serialized = _canonical(record.payload())
        with self._lock(fcntl.LOCK_EX):
            existing = self._read_unlocked()
            prior = next((value for value in existing if value.record_id == record.record_id), None)
            if prior is not None:
                if _canonical(prior.payload()) != serialized:
                    raise ValueError("conflicting pure extraction record")
                return False
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(serialized + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        return True


class JsonPureExtractionOptimizerCorpusStore:
    """Crash-safe persistence for the pure-process optimizer projection."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.lock_path = self.path.with_name(self.path.name + ".lock")

    @contextmanager
    def _lock(self, operation: int) -> Iterator[None]:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), operation)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def write(self, corpus: PureExtractionOptimizerCorpus) -> bool:
        if not isinstance(corpus, PureExtractionOptimizerCorpus):
            raise TypeError("pure extraction corpus store received the wrong type")
        serialized = _canonical(corpus.payload()) + "\n"
        with self._lock(fcntl.LOCK_EX):
            if self.path.exists():
                if self.path.is_symlink():
                    raise ValueError("pure extraction optimizer corpus cannot be a symlink")
                existing = self._read_unlocked()
                if existing != corpus:
                    raise ValueError("conflicting pure extraction optimizer corpus")
                return False
            self.path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(
                prefix=self.path.name + ".", dir=self.path.parent
            )
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    handle.write(serialized)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, self.path)
                directory = os.open(self.path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        return True

    def read(self) -> PureExtractionOptimizerCorpus:
        with self._lock(fcntl.LOCK_SH):
            result = self._read_unlocked()
        if result is None:
            raise FileNotFoundError("pure extraction optimizer corpus has not been persisted")
        return result

    def read_for_optimizer(self) -> PureExtractionOptimizerCorpus:
        result = self.read()
        if result.split != "train":
            raise PermissionError("pure optimizer can read only the training corpus")
        for example in result.examples:
            if example.evidence_plane is not EvidencePlane.PURE_PROCESS:
                raise ValueError("pure optimizer corpus contains non-pure evidence")
        return result

    def _read_unlocked(self) -> PureExtractionOptimizerCorpus | None:
        if not self.path.exists():
            return None
        if self.path.is_symlink():
            raise ValueError("pure extraction optimizer corpus cannot be a symlink")
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("malformed pure extraction optimizer corpus store") from exc
        return PureExtractionOptimizerCorpus.from_payload(value)


class JsonPureExtractionSourceRecordStore(_JsonPureExtractionStore):
    record_type = PureExtractionSourceRecord


class JsonPureExtractionFeedbackRecordStore(_JsonPureExtractionStore):
    record_type = PureExtractionFeedbackRecord


__all__ = [
    "PURE_EXTRACTION_ATTRIBUTION_SCHEMA_VERSION",
    "PURE_EXTRACTION_FEEDBACK_SCHEMA",
    "PURE_EXTRACTION_FEEDBACK_SCHEMA_VERSION",
    "PURE_EXTRACTION_CORPUS_SCHEMA",
    "PURE_EXTRACTION_CORPUS_SCHEMA_VERSION",
    "PURE_EXTRACTION_OPTIMIZER_SCHEMA",
    "PURE_EXTRACTION_OPTIMIZER_SCHEMA_VERSION",
    "PURE_EXTRACTION_SOURCE_SCHEMA",
    "PURE_EXTRACTION_SOURCE_SCHEMA_VERSION",
    "JsonPureExtractionFeedbackRecordStore",
    "JsonPureExtractionOptimizerCorpusStore",
    "JsonPureExtractionSourceRecordStore",
    "PureExtractionAttribution",
    "PureExtractionFeedbackRecord",
    "PureExtractionOptimizerExample",
    "PureExtractionOptimizerCorpus",
    "PureExtractionSourceRecord",
    "PureExtractionSourceProjector",
]
