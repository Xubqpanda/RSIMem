"""Project real Mem0-flat compilation results into extraction feedback evidence."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .executor import MutationExecutionStatus
from .extraction_feedback import (
    EXTRACTION_FEEDBACK_SCHEMA_VERSION,
    ExtractionFeedbackDataset,
    ExtractedFactEvidence,
    ExtractionQualityIssue,
    ExtractionSetStatus,
    ExtractionSourceEvidence,
    FactDisposition,
    FeedbackContractRegistry,
    default_feedback_contract_registry,
    detect_extracted_fact_semantic_keys,
)
from .evidence_planes import (
    EvidencePlane,
    EvidenceSourceKind,
    validate_plane_source,
)
from .ingestion import InternalMemoryAction, MemoryIngestStatus
from .live_writeback import ExtractionRuntimeBinding, StaticSemanticBoundaryResult
from .prompt_components import SemanticPolicyManifest
from ..memory_systems.mem0_flat.policy import (
    ExtractionInvocationFingerprint,
    Mem0FlatSemanticPolicy,
)


EXTRACTION_SOURCE_RECORD_SCHEMA_VERSION = 4
EXTRACTION_ACTIVATION_FINGERPRINT_SCHEMA_VERSION = 1
LIVE_EXTRACTION_FEEDBACK_SCHEMA_VERSION = 2


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _validate_feedback_dataset_payload(value: object) -> dict[str, object]:
    fields = {
        "schema_version",
        "dataset_id",
        "source_projection_digest",
        "contract_digest",
        "evidence_plane",
        "evidence_source",
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
        or value["schema_version"] != EXTRACTION_FEEDBACK_SCHEMA_VERSION
        or not isinstance(value["evidence_plane"], str)
        or not isinstance(value["evidence_source"], str)
        or not all(isinstance(value[field], str) for field in (
            "dataset_id",
            "source_projection_digest",
            "contract_digest",
        ))
        or not isinstance(value["examples"], list)
        or not value["examples"]
    ):
        raise ValueError("malformed extraction feedback dataset log")
    try:
        validate_plane_source(value["evidence_plane"], value["evidence_source"])
    except (TypeError, ValueError) as exc:
        raise ValueError("malformed extraction feedback dataset log") from exc
    for example in value["examples"]:
        if (
            not isinstance(example, dict)
            or set(example) != example_fields
            or type(example["primary"]) is not bool
            or not isinstance(example["artifact_ids"], list)
            or not isinstance(example["reason_codes"], list)
        ):
            raise ValueError("malformed extraction feedback dataset log")
    return value


@dataclass(frozen=True, slots=True)
class ExtractionActivationFingerprint:
    compilation_id: str
    extraction_operation_id: str
    runtime_binding: ExtractionRuntimeBinding
    semantic_policy: SemanticPolicyManifest
    invocation: ExtractionInvocationFingerprint
    parsed_output_digest: str
    mutation_ids: tuple[str, ...]
    persisted_artifact_ids: tuple[str, ...]
    fingerprint_digest: str
    schema_version: int = EXTRACTION_ACTIVATION_FINGERPRINT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != EXTRACTION_ACTIVATION_FINGERPRINT_SCHEMA_VERSION:
            raise ValueError("unsupported extraction activation fingerprint schema")
        if any(not isinstance(value, str) or not value.strip() for value in (
            self.compilation_id,
            self.extraction_operation_id,
        )):
            raise ValueError("extraction activation identity is incomplete")
        if not isinstance(self.runtime_binding, ExtractionRuntimeBinding):
            raise TypeError("extraction activation runtime binding has the wrong type")
        if not isinstance(self.semantic_policy, SemanticPolicyManifest):
            raise TypeError("extraction activation semantic policy has the wrong type")
        if not isinstance(self.invocation, ExtractionInvocationFingerprint):
            raise TypeError("extraction activation invocation has the wrong type")
        if (
            self.semantic_policy.extraction_component_id
            != self.runtime_binding.component_artifact_id
            or self.semantic_policy.extraction_component_digest
            != self.runtime_binding.component_body_digest
            or self.semantic_policy.model_profile != self.runtime_binding.model_profile
            or self.invocation.binding_id != self.runtime_binding.binding_id
            or self.invocation.rendered_template_digest
            != self.runtime_binding.rendered_template_digest
        ):
            raise ValueError("extraction activation binding and policy differ")
        if len(self.parsed_output_digest) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.parsed_output_digest
        ):
            raise ValueError("extraction activation parsed output digest is invalid")
        for values, name in (
            (self.mutation_ids, "mutation IDs"),
            (self.persisted_artifact_ids, "persisted artifact IDs"),
        ):
            if (
                not isinstance(values, tuple)
                or len(values) != len(set(values))
                or any(not isinstance(value, str) or not value.strip() for value in values)
            ):
                raise ValueError(f"extraction activation {name} are invalid")
        expected = hashlib.sha256(
            _canonical(self.identity_payload()).encode("utf-8")
        ).hexdigest()
        if self.fingerprint_digest != expected:
            raise ValueError("extraction activation fingerprint digest mismatch")

    @property
    def fingerprint_id(self) -> str:
        return f"extraction-activation.{self.fingerprint_digest[:40]}"

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "compilation_id": self.compilation_id,
            "extraction_operation_id": self.extraction_operation_id,
            "runtime_binding": self.runtime_binding.payload(),
            "semantic_policy": self.semantic_policy.payload(),
            "invocation": self.invocation.payload(),
            "parsed_output_digest": self.parsed_output_digest,
            "mutation_ids": list(self.mutation_ids),
            "persisted_artifact_ids": list(self.persisted_artifact_ids),
        }

    def payload(self) -> dict[str, object]:
        return {
            "fingerprint_id": self.fingerprint_id,
            **self.identity_payload(),
            "fingerprint_digest": self.fingerprint_digest,
        }

    @classmethod
    def create(
        cls,
        *,
        compilation_id: str,
        extraction_operation_id: str,
        runtime_binding: ExtractionRuntimeBinding,
        semantic_policy: SemanticPolicyManifest,
        invocation: ExtractionInvocationFingerprint,
        parsed_output_digest: str,
        mutation_ids: tuple[str, ...],
        persisted_artifact_ids: tuple[str, ...],
    ) -> "ExtractionActivationFingerprint":
        identity = {
            "schema_version": EXTRACTION_ACTIVATION_FINGERPRINT_SCHEMA_VERSION,
            "compilation_id": compilation_id,
            "extraction_operation_id": extraction_operation_id,
            "runtime_binding": runtime_binding.payload(),
            "semantic_policy": semantic_policy.payload(),
            "invocation": invocation.payload(),
            "parsed_output_digest": parsed_output_digest,
            "mutation_ids": list(mutation_ids),
            "persisted_artifact_ids": list(persisted_artifact_ids),
        }
        return cls(
            compilation_id,
            extraction_operation_id,
            runtime_binding,
            semantic_policy,
            invocation,
            parsed_output_digest,
            mutation_ids,
            persisted_artifact_ids,
            hashlib.sha256(_canonical(identity).encode("utf-8")).hexdigest(),
        )

    @classmethod
    def from_payload(cls, value: object) -> "ExtractionActivationFingerprint":
        fields = {
            "fingerprint_id",
            "schema_version",
            "compilation_id",
            "extraction_operation_id",
            "runtime_binding",
            "semantic_policy",
            "invocation",
            "parsed_output_digest",
            "mutation_ids",
            "persisted_artifact_ids",
            "fingerprint_digest",
        }
        if not isinstance(value, dict) or set(value) != fields:
            raise ValueError("malformed extraction activation fingerprint")
        invocation = value["invocation"]
        if (
            not isinstance(invocation, dict)
            or set(invocation) != {
                "render_id",
                "render_input_digest",
                "rendered_template_digest",
                "model_output_digest",
                "binding_id",
            }
            or not isinstance(value["mutation_ids"], list)
            or not isinstance(value["persisted_artifact_ids"], list)
        ):
            raise ValueError("malformed extraction activation invocation")
        try:
            result = cls(
                value["compilation_id"],
                value["extraction_operation_id"],
                ExtractionRuntimeBinding.from_payload(value["runtime_binding"]),
                SemanticPolicyManifest.from_payload(value["semantic_policy"]),
                ExtractionInvocationFingerprint(**invocation),
                value["parsed_output_digest"],
                tuple(value["mutation_ids"]),
                tuple(value["persisted_artifact_ids"]),
                value["fingerprint_digest"],
                schema_version=value["schema_version"],
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("malformed extraction activation fingerprint") from exc
        if value["fingerprint_id"] != result.fingerprint_id:
            raise ValueError("extraction activation fingerprint ID mismatch")
        return result


@dataclass(frozen=True, slots=True)
class ExtractionSourceRecord:
    family_id: str
    stage: str
    run_id: str
    episode_id: str
    session_id: str
    task_id: str
    compilation_id: str
    extraction_artifact_id: str
    extraction_artifact_digest: str
    extraction_output_digest: str
    source: ExtractionSourceEvidence
    activation: ExtractionActivationFingerprint
    content_digest: str
    evidence_plane: EvidencePlane = EvidencePlane.BENCHMARK_AUDIT
    evidence_source: EvidenceSourceKind = EvidenceSourceKind.BENCHMARK_CONTRACT
    schema_version: int = EXTRACTION_SOURCE_RECORD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != EXTRACTION_SOURCE_RECORD_SCHEMA_VERSION:
            raise ValueError("unsupported extraction source record schema")
        plane, source = validate_plane_source(self.evidence_plane, self.evidence_source)
        if plane is not EvidencePlane.BENCHMARK_AUDIT or source is not EvidenceSourceKind.BENCHMARK_CONTRACT:
            raise ValueError(
                "family-bound extraction source must be benchmark_audit evidence"
            )
        object.__setattr__(self, "evidence_plane", plane)
        object.__setattr__(self, "evidence_source", source)
        if any(not isinstance(value, str) or not value.strip() for value in (
            self.family_id,
            self.stage,
            self.run_id,
            self.episode_id,
            self.session_id,
            self.task_id,
            self.compilation_id,
            self.extraction_artifact_id,
        )):
            raise ValueError("extraction source record identity is incomplete")
        for value in (
            self.extraction_artifact_digest,
            self.extraction_output_digest,
            self.content_digest,
        ):
            if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError("extraction source fingerprint must be sha256")
        if (
            not isinstance(self.activation, ExtractionActivationFingerprint)
            or self.activation.compilation_id != self.compilation_id
            or self.activation.extraction_operation_id
            != self.source.extraction_set_id
            or self.activation.parsed_output_digest != self.extraction_output_digest
            or self.activation.runtime_binding.component_artifact_id
            != self.extraction_artifact_id
            or self.activation.runtime_binding.component_body_digest
            != self.extraction_artifact_digest
        ):
            raise ValueError("extraction source activation fingerprint differs")
        if self.content_digest != hashlib.sha256(
            _canonical(self.identity_payload()).encode("utf-8")
        ).hexdigest():
            raise ValueError("extraction source record digest mismatch")

    @property
    def record_id(self) -> str:
        return self.compilation_id

    @property
    def artifact_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(
            fact.artifact_id
            for fact in self.source.facts
            if fact.artifact_id is not None
        ))

    def payload(self) -> dict[str, object]:
        return {
            "record_id": self.record_id,
            **self.identity_payload(),
            "content_digest": self.content_digest,
        }

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "family_id": self.family_id,
            "stage": self.stage,
            "run_id": self.run_id,
            "episode_id": self.episode_id,
            "session_id": self.session_id,
            "task_id": self.task_id,
            "compilation_id": self.compilation_id,
            "extraction_artifact_id": self.extraction_artifact_id,
            "extraction_artifact_digest": self.extraction_artifact_digest,
            "extraction_output_digest": self.extraction_output_digest,
            "source": self.source.payload(),
            "activation": self.activation.payload(),
            "evidence_plane": self.evidence_plane.value,
            "evidence_source": self.evidence_source.value,
        }

    @classmethod
    def create(
        cls,
        *,
        family_id: str,
        stage: str,
        run_id: str,
        episode_id: str,
        session_id: str,
        task_id: str,
        compilation_id: str,
        extraction_artifact_id: str,
        extraction_artifact_digest: str,
        extraction_output_digest: str,
        source: ExtractionSourceEvidence,
        activation: ExtractionActivationFingerprint,
    ) -> "ExtractionSourceRecord":
        identity = {
            "schema_version": EXTRACTION_SOURCE_RECORD_SCHEMA_VERSION,
            "family_id": family_id,
            "stage": stage,
            "run_id": run_id,
            "episode_id": episode_id,
            "session_id": session_id,
            "task_id": task_id,
            "compilation_id": compilation_id,
            "extraction_artifact_id": extraction_artifact_id,
            "extraction_artifact_digest": extraction_artifact_digest,
            "extraction_output_digest": extraction_output_digest,
            "source": source.payload(),
            "activation": activation.payload(),
            "evidence_plane": EvidencePlane.BENCHMARK_AUDIT.value,
            "evidence_source": EvidenceSourceKind.BENCHMARK_CONTRACT.value,
        }
        return cls(
            family_id,
            stage,
            run_id,
            episode_id,
            session_id,
            task_id,
            compilation_id,
            extraction_artifact_id,
            extraction_artifact_digest,
            extraction_output_digest,
            source,
            activation,
            hashlib.sha256(_canonical(identity).encode("utf-8")).hexdigest(),
            evidence_plane=EvidencePlane.BENCHMARK_AUDIT,
            evidence_source=EvidenceSourceKind.BENCHMARK_CONTRACT,
        )

    @classmethod
    def from_payload(cls, value: object) -> "ExtractionSourceRecord":
        fields = {
            "schema_version",
            "record_id",
            "family_id",
            "stage",
            "run_id",
            "episode_id",
            "session_id",
            "task_id",
            "compilation_id",
            "extraction_artifact_id",
            "extraction_artifact_digest",
            "extraction_output_digest",
            "source",
            "activation",
            "content_digest",
            "evidence_plane",
            "evidence_source",
        }
        if not isinstance(value, dict) or set(value) != fields:
            raise ValueError("malformed extraction source record")
        scalar_fields = fields - {"schema_version", "source", "activation"}
        if (
            type(value["schema_version"]) is not int
            or any(not isinstance(value[field], str) for field in scalar_fields)
        ):
            raise ValueError("extraction source record scalar fields are invalid")
        if value["record_id"] != value["compilation_id"]:
            raise ValueError("extraction source record identity mismatch")
        return cls(
            value["family_id"],
            value["stage"],
            value["run_id"],
            value["episode_id"],
            value["session_id"],
            value["task_id"],
            value["compilation_id"],
            value["extraction_artifact_id"],
            value["extraction_artifact_digest"],
            value["extraction_output_digest"],
            ExtractionSourceEvidence.from_payload(value["source"]),
            ExtractionActivationFingerprint.from_payload(value["activation"]),
            value["content_digest"],
            EvidencePlane(value["evidence_plane"]),
            EvidenceSourceKind(value["evidence_source"]),
            schema_version=value["schema_version"],
        )


class JsonExtractionSourceRecordStore:
    """Append-only, restart-safe content-free extraction source records."""

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().resolve()
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")

    @contextmanager
    def _lock(self, operation: int) -> Iterator[None]:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), operation)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _read_unlocked(self) -> tuple[ExtractionSourceRecord, ...]:
        if not self.path.exists():
            return ()
        records = []
        identities: dict[str, str] = {}
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = ExtractionSourceRecord.from_payload(json.loads(line))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError("malformed extraction source record store") from exc
            canonical = _canonical(record.payload())
            previous = identities.get(record.record_id)
            if previous is not None:
                if previous != canonical:
                    raise ValueError("conflicting extraction source record")
                continue
            identities[record.record_id] = canonical
            records.append(record)
        return tuple(records)

    def records(self) -> tuple[ExtractionSourceRecord, ...]:
        with self._lock(fcntl.LOCK_SH):
            return self._read_unlocked()

    def append(self, record: ExtractionSourceRecord) -> bool:
        serialized = _canonical(record.payload())
        with self._lock(fcntl.LOCK_EX):
            records = self._read_unlocked()
            existing = next(
                (value for value in records if value.record_id == record.record_id),
                None,
            )
            if existing is not None:
                if _canonical(existing.payload()) != serialized:
                    raise ValueError("conflicting extraction source record")
                return False
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(serialized + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        return True

    def candidates(
        self,
        *,
        family_id: str,
        artifact_ids: tuple[str, ...],
        opportunity_semantic_keys: tuple[str, ...],
    ) -> tuple[ExtractionSourceRecord, ...]:
        artifacts = set(artifact_ids)
        keys = set(opportunity_semantic_keys)
        return tuple(
            record
            for record in self.records()
            if record.family_id == family_id
            and (
                bool(set(record.artifact_ids) & artifacts)
                if record.artifact_ids
                else bool(set(record.source.available_semantic_keys) & keys)
            )
        )


class JsonExtractionFeedbackDatasetLog:
    """Append content-free feedback datasets with conflict detection."""

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().resolve()
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")

    def append(self, dataset: ExtractionFeedbackDataset) -> bool:
        payload = dataset.payload()
        serialized = _canonical(payload)
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                existing: dict[str, str] = {}
                if self.path.exists():
                    for line in self.path.read_text(encoding="utf-8").splitlines():
                        if not line.strip():
                            continue
                        try:
                            value = _validate_feedback_dataset_payload(
                                json.loads(line)
                            )
                        except (json.JSONDecodeError, TypeError, ValueError) as exc:
                            raise ValueError(
                                "malformed extraction feedback dataset log"
                            ) from exc
                        canonical = _canonical(value)
                        prior = existing.get(value["dataset_id"])
                        if prior is not None and prior != canonical:
                            raise ValueError("conflicting extraction feedback dataset")
                        existing[value["dataset_id"]] = canonical
                prior = existing.get(dataset.dataset_id)
                if prior is not None:
                    if prior != serialized:
                        raise ValueError("conflicting extraction feedback dataset")
                    return False
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(serialized + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                return True
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


@dataclass(frozen=True, slots=True)
class LiveExtractionFeedbackRecord:
    record_id: str
    family_id: str
    stage: str
    run_id: str
    trace_id: str
    episode_id: str
    session_id: str
    task_id: str
    deployment_observation_id: str
    source_record_id: str
    opportunity_operation_id: str
    use_operation_id: str
    outcome_operation_id: str
    dataset: ExtractionFeedbackDataset
    evidence_plane: EvidencePlane = EvidencePlane.BENCHMARK_AUDIT
    evidence_source: EvidenceSourceKind = EvidenceSourceKind.BENCHMARK_CONTRACT
    schema_version: int = LIVE_EXTRACTION_FEEDBACK_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != LIVE_EXTRACTION_FEEDBACK_SCHEMA_VERSION:
            raise ValueError("unsupported live extraction feedback schema")
        plane, source = validate_plane_source(self.evidence_plane, self.evidence_source)
        if plane is not EvidencePlane.BENCHMARK_AUDIT or source is not EvidenceSourceKind.BENCHMARK_CONTRACT:
            raise ValueError(
                "family-bound live feedback must be benchmark_audit evidence"
            )
        if (
            self.dataset.evidence_plane is not plane
            or self.dataset.evidence_source is not source
        ):
            raise ValueError("live feedback dataset plane/source differs")
        object.__setattr__(self, "evidence_plane", plane)
        object.__setattr__(self, "evidence_source", source)
        for value in (
            self.record_id,
            self.family_id,
            self.stage,
            self.run_id,
            self.trace_id,
            self.episode_id,
            self.session_id,
            self.task_id,
            self.deployment_observation_id,
            self.source_record_id,
            self.opportunity_operation_id,
            self.use_operation_id,
            self.outcome_operation_id,
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError("live extraction feedback identity is incomplete")
        primary = next(
            (example for example in self.dataset.examples if example.primary),
            None,
        )
        if primary is None or (
            primary.opportunity_operation_id != self.opportunity_operation_id
            or primary.use_operation_id != self.use_operation_id
            or primary.outcome_operation_id != self.outcome_operation_id
        ):
            raise ValueError("live feedback operation join differs from dataset")
        identity_digest = hashlib.sha256(
            _canonical(self.identity_payload()).encode("utf-8")
        ).hexdigest()
        expected = f"live-extraction-feedback.{identity_digest[:40]}"
        if self.record_id != expected:
            raise ValueError("live extraction feedback record ID mismatch")

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "family_id": self.family_id,
            "stage": self.stage,
            "run_id": self.run_id,
            "trace_id": self.trace_id,
            "episode_id": self.episode_id,
            "session_id": self.session_id,
            "task_id": self.task_id,
            "deployment_observation_id": self.deployment_observation_id,
            "source_record_id": self.source_record_id,
            "opportunity_operation_id": self.opportunity_operation_id,
            "use_operation_id": self.use_operation_id,
            "outcome_operation_id": self.outcome_operation_id,
            "dataset_id": self.dataset.dataset_id,
            "source_projection_digest": self.dataset.source_projection_digest,
            "contract_digest": self.dataset.contract_digest,
            "evidence_plane": self.evidence_plane.value,
            "evidence_source": self.evidence_source.value,
        }

    def payload(self) -> dict[str, object]:
        return {
            **self.identity_payload(),
            "record_id": self.record_id,
            "dataset": self.dataset.payload(),
        }

    @classmethod
    def create(
        cls,
        *,
        family_id: str,
        stage: str,
        run_id: str,
        trace_id: str,
        episode_id: str,
        session_id: str,
        task_id: str,
        deployment_observation_id: str,
        source_record_id: str,
        opportunity_operation_id: str,
        use_operation_id: str,
        outcome_operation_id: str,
        dataset: ExtractionFeedbackDataset,
    ) -> "LiveExtractionFeedbackRecord":
        identity = {
            "schema_version": LIVE_EXTRACTION_FEEDBACK_SCHEMA_VERSION,
            "family_id": family_id,
            "stage": stage,
            "run_id": run_id,
            "trace_id": trace_id,
            "episode_id": episode_id,
            "session_id": session_id,
            "task_id": task_id,
            "deployment_observation_id": deployment_observation_id,
            "source_record_id": source_record_id,
            "opportunity_operation_id": opportunity_operation_id,
            "use_operation_id": use_operation_id,
            "outcome_operation_id": outcome_operation_id,
            "dataset_id": dataset.dataset_id,
            "source_projection_digest": dataset.source_projection_digest,
            "contract_digest": dataset.contract_digest,
            "evidence_plane": EvidencePlane.BENCHMARK_AUDIT.value,
            "evidence_source": EvidenceSourceKind.BENCHMARK_CONTRACT.value,
        }
        record_id = (
            "live-extraction-feedback."
            + hashlib.sha256(_canonical(identity).encode("utf-8")).hexdigest()[:40]
        )
        return cls(
            record_id,
            family_id,
            stage,
            run_id,
            trace_id,
            episode_id,
            session_id,
            task_id,
            deployment_observation_id,
            source_record_id,
            opportunity_operation_id,
            use_operation_id,
            outcome_operation_id,
            dataset,
            evidence_plane=EvidencePlane.BENCHMARK_AUDIT,
            evidence_source=EvidenceSourceKind.BENCHMARK_CONTRACT,
        )

    @classmethod
    def from_payload(cls, value: object) -> "LiveExtractionFeedbackRecord":
        identity_fields = {
            "schema_version",
            "family_id",
            "stage",
            "run_id",
            "trace_id",
            "episode_id",
            "session_id",
            "task_id",
            "deployment_observation_id",
            "source_record_id",
            "opportunity_operation_id",
            "use_operation_id",
            "outcome_operation_id",
            "dataset_id",
            "source_projection_digest",
            "contract_digest",
            "evidence_plane",
            "evidence_source",
        }
        if not isinstance(value, dict) or set(value) != identity_fields | {
            "record_id",
            "dataset",
        }:
            raise ValueError("malformed live extraction feedback record")
        try:
            dataset = ExtractionFeedbackDataset.from_payload(value["dataset"])
            if (
                value["dataset_id"] != dataset.dataset_id
                or value["source_projection_digest"]
                != dataset.source_projection_digest
                or value["contract_digest"] != dataset.contract_digest
            ):
                raise ValueError("live feedback dataset identity mismatch")
            return cls(
                value["record_id"],
                value["family_id"],
                value["stage"],
                value["run_id"],
                value["trace_id"],
                value["episode_id"],
                value["session_id"],
                value["task_id"],
                value["deployment_observation_id"],
                value["source_record_id"],
                value["opportunity_operation_id"],
                value["use_operation_id"],
                value["outcome_operation_id"],
                dataset,
                evidence_plane=value["evidence_plane"],
                evidence_source=value["evidence_source"],
                schema_version=value["schema_version"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("malformed live extraction feedback record") from exc


class JsonLiveExtractionFeedbackRecordLog:
    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().resolve()
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")

    def _read_unlocked(self) -> tuple[LiveExtractionFeedbackRecord, ...]:
        if not self.path.exists():
            return ()
        records = []
        identities: dict[str, str] = {}
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = LiveExtractionFeedbackRecord.from_payload(json.loads(line))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError("malformed live extraction feedback log") from exc
            canonical = _canonical(record.payload())
            prior = identities.get(record.record_id)
            if prior is not None and prior != canonical:
                raise ValueError("conflicting live extraction feedback record")
            if prior is None:
                records.append(record)
            identities[record.record_id] = canonical
        return tuple(records)

    def records(self) -> tuple[LiveExtractionFeedbackRecord, ...]:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_SH)
            try:
                return self._read_unlocked()
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def append(self, record: LiveExtractionFeedbackRecord) -> bool:
        serialized = _canonical(record.payload())
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                existing = self._read_unlocked()
                current = next(
                    (value for value in existing if value.record_id == record.record_id),
                    None,
                )
                if current is not None:
                    if _canonical(current.payload()) != serialized:
                        raise ValueError("conflicting live extraction feedback record")
                    return False
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(serialized + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                return True
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


class Mem0FlatExtractionSourceProjector:
    """Build content-free source evidence from the actual policy and executor trace."""

    def __init__(self, registry: FeedbackContractRegistry | None = None) -> None:
        self.registry = registry or default_feedback_contract_registry()

    def project(
        self,
        boundary: StaticSemanticBoundaryResult,
        policy: Mem0FlatSemanticPolicy,
        *,
        family_id: str,
        available_semantic_keys: tuple[str, ...],
    ) -> ExtractionSourceEvidence:
        if boundary.duplicate or boundary.writeback is None:
            raise ValueError("extraction projection requires an original writeback result")
        ingestion = boundary.writeback.ingestion
        if ingestion is None:
            raise ValueError("extraction projection requires an ingestion result")
        trace = policy.operation_trace(ingestion.idempotency_key)
        if trace is None:
            raise ValueError("extraction projection requires a Mem0-flat operation trace")
        contract = self.registry.resolver(family_id).contract
        allowed_keys = set(contract.opportunity.memory_scope_keys)
        available_keys = tuple(available_semantic_keys)
        if len(available_keys) != len(set(available_keys)):
            raise ValueError("available extraction semantic keys must be unique")
        if set(available_keys) - allowed_keys:
            raise ValueError("available extraction semantic keys escape family contract")
        if (
            boundary.receipt is not None
            and boundary.receipt.source_projection_digest != ingestion.source_digest
        ):
            raise ValueError("compilation receipt and ingestion source digest disagree")

        operations = ingestion.operations
        executions = boundary.writeback.executions
        accepted_index = 0
        facts = []
        for extraction in trace.fact_extractions:
            fact = policy.fact_for_digest(extraction.content_digest)
            if fact is None or fact.fact_id != extraction.fact_id:
                raise ValueError("policy fact owner disagrees with extraction trace")
            semantic_keys = detect_extracted_fact_semantic_keys(
                family_id,
                fact.content,
            )
            if set(semantic_keys) - allowed_keys:
                raise ValueError("extracted fact semantic keys escape family contract")
            quality_issue = (
                ExtractionQualityIssue.UNSUPPORTED
                if semantic_keys and not set(semantic_keys).issubset(available_keys)
                else None
            )
            artifact_id = None
            if not extraction.accepted:
                disposition = FactDisposition.FILTERED
            else:
                operation = (
                    operations[accepted_index]
                    if accepted_index < len(operations)
                    else None
                )
                execution = (
                    executions[accepted_index]
                    if accepted_index < len(executions)
                    else None
                )
                accepted_index += 1
                if operation is None or ingestion.status != MemoryIngestStatus.SUCCESS:
                    disposition = FactDisposition.MUTATION_FAILED
                elif operation.action in {
                    InternalMemoryAction.NONE,
                    InternalMemoryAction.DELETE,
                }:
                    disposition = FactDisposition.NONE
                elif (
                    execution is not None
                    and execution.status in {
                        MutationExecutionStatus.COMMITTED,
                        MutationExecutionStatus.DUPLICATE,
                    }
                    and execution.artifact_id is not None
                ):
                    disposition = FactDisposition.PERSISTED
                    artifact_id = execution.artifact_id
                else:
                    disposition = FactDisposition.MUTATION_FAILED
            facts.append(ExtractedFactEvidence(
                extraction.fact_id,
                semantic_keys,
                disposition,
                artifact_id=artifact_id,
                quality_issue=quality_issue,
            ))

        dispositions = {fact.disposition for fact in facts}
        if not facts:
            status = (
                ExtractionSetStatus.EMPTY
                if ingestion.status == MemoryIngestStatus.SUCCESS
                and any(
                    operation.action == InternalMemoryAction.NONE
                    for operation in operations
                )
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
        return ExtractionSourceEvidence(
            trace.source_artifact_id,
            ingestion.source_digest,
            trace.extraction_operation_id,
            status,
            available_keys,
            tuple(facts),
        )

    def project_record(
        self,
        boundary: StaticSemanticBoundaryResult,
        policy: Mem0FlatSemanticPolicy,
        runtime_binding: ExtractionRuntimeBinding,
        *,
        family_id: str,
        stage: str,
        available_semantic_keys: tuple[str, ...],
    ) -> ExtractionSourceRecord:
        source = self.project(
            boundary,
            policy,
            family_id=family_id,
            available_semantic_keys=available_semantic_keys,
        )
        assert boundary.writeback is not None
        ingestion = boundary.writeback.ingestion
        assert ingestion is not None
        trace = policy.operation_trace(ingestion.idempotency_key)
        assert trace is not None
        context = trace.context
        extraction_output_digest = hashlib.sha256(_canonical([
                {
                    "fact_id": fact.fact_id,
                    "content_digest": fact.content_digest,
                    "accepted": fact.accepted,
                    "reason_code": fact.reason_code,
                }
                for fact in trace.fact_extractions
            ]).encode("utf-8")).hexdigest()
        invocation = policy.extraction_invocation(ingestion.idempotency_key)
        if invocation is None:
            raise ValueError("extraction projection requires an invocation fingerprint")
        executions = boundary.writeback.executions
        activation = ExtractionActivationFingerprint.create(
            compilation_id=boundary.compilation_id,
            extraction_operation_id=trace.extraction_operation_id,
            runtime_binding=runtime_binding,
            semantic_policy=policy.semantic_manifest,
            invocation=invocation,
            parsed_output_digest=extraction_output_digest,
            mutation_ids=tuple(dict.fromkeys(
                execution.mutation_id for execution in executions
            )),
            persisted_artifact_ids=tuple(dict.fromkeys(
                execution.artifact_id
                for execution in executions
                if execution.artifact_id is not None
                and execution.status in {
                    MutationExecutionStatus.COMMITTED,
                    MutationExecutionStatus.DUPLICATE,
                }
            )),
        )
        return ExtractionSourceRecord.create(
            family_id=family_id,
            stage=stage,
            run_id=context.run_id,
            episode_id=context.episode_id,
            session_id=context.session_id,
            task_id=context.task_id,
            compilation_id=boundary.compilation_id,
            extraction_artifact_id=(
                policy.semantic_manifest.extraction_component_id
            ),
            extraction_artifact_digest=(
                policy.semantic_manifest.extraction_component_digest
            ),
            extraction_output_digest=extraction_output_digest,
            source=source,
            activation=activation,
        )
