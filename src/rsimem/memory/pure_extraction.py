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
from contextlib import contextmanager
from dataclasses import dataclass
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
        provenance_id: str | None = None,
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
        return cls.create(
            source_projection_id=projection_id,
            source_projection_digest=record.source.source_projection_digest,
            context_revision=f"revision.{record.activation.fingerprint_digest[:40]}",
            extraction_set_id=record.source.extraction_set_id,
            extraction_artifact_id=record.extraction_artifact_id,
            extraction_artifact_digest=record.extraction_artifact_digest,
            extraction_output_digest=record.extraction_output_digest,
            source=record.source,
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
        if self.memory_use is not None:
            if self.memory_use.provenance_id != self.provenance_id:
                raise ValueError("memory-use provenance does not match extraction feedback")
            if self.artifact_set_binding is not None and self.memory_use.artifact_set_id != self.artifact_set_binding.binding_id:
                raise ValueError("artifact-set binding does not match memory-use evidence")
            if self.memory_use.artifact_set_id is not None and self.artifact_set_binding is None:
                raise ValueError("memory-use artifact set requires a trusted binding")
        join_ids = tuple(join.join_id for join in self.tool_joins)
        if len(join_ids) != len(set(join_ids)):
            raise ValueError("pure extraction tool joins must be unique")
        for join in self.tool_joins:
            if not isinstance(join, ToolCallResultJoin):
                raise TypeError("pure extraction tool join has the wrong type")
            if join.policy_lineage_id is not None and join.policy_lineage_id != self.provenance_id:
                raise ValueError("tool join provenance does not match extraction feedback")
        if type(self.observation_complete) is not bool:
            raise TypeError("pure extraction observation completeness must be bool")
        object.__setattr__(self, "attribution", PureExtractionAttribution(self.attribution))
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


class JsonPureExtractionSourceRecordStore(_JsonPureExtractionStore):
    record_type = PureExtractionSourceRecord


class JsonPureExtractionFeedbackRecordStore(_JsonPureExtractionStore):
    record_type = PureExtractionFeedbackRecord


__all__ = [
    "PURE_EXTRACTION_ATTRIBUTION_SCHEMA_VERSION",
    "PURE_EXTRACTION_FEEDBACK_SCHEMA",
    "PURE_EXTRACTION_FEEDBACK_SCHEMA_VERSION",
    "PURE_EXTRACTION_SOURCE_SCHEMA",
    "PURE_EXTRACTION_SOURCE_SCHEMA_VERSION",
    "JsonPureExtractionFeedbackRecordStore",
    "JsonPureExtractionSourceRecordStore",
    "PureExtractionAttribution",
    "PureExtractionFeedbackRecord",
    "PureExtractionSourceRecord",
]
