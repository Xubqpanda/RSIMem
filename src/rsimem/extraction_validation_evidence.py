"""Assemble extraction matched-trial evidence from formal validation runs."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .extraction_experiment_manifest import (
    EXTRACTION_METHOD_VARIANTS,
    extraction_acceptance_criteria_from_manifest,
    load_extraction_manifest_for_phase,
)
from .memory.extraction_matched_activation import (
    ExtractionMatchedTrialDecision,
    ExtractionMatchedTrialEvaluator,
)
from .memory.extraction_offline_validation import (
    ExtractionOfflineValidationDecision,
)
from .memory.extraction_policy_artifact import ExtractionPromptPolicyArtifact
from .memory.extraction_projection import (
    ExtractionSourceRecord,
    JsonExtractionSourceRecordStore,
    JsonLiveExtractionFeedbackRecordLog,
    LiveExtractionFeedbackRecord,
)
from .memory.extraction_prompt_validation import (
    JsonExtractionValidationObservationStore,
    ExtractionPromptValidationSplit,
    ExtractionValidationObservation,
    ExtractionValidationSafetyEvidence,
    ExtractionValidationSplitRole,
    ExtractionValidationVariant,
)
from .memory.extraction_validation_adapter import (
    ExtractionValidationObservationAssembler,
)
from .memory.live_writeback import ExtractionPromptRuntimeScope
from .memory.process_corpus import (
    JsonProcessCorpusStore,
    ensure_process_corpus_has_no_evaluation_fields,
)
from .memory.process_feedback import audit_process_events
from .memory.prompt_components import (
    SemanticPolicyManifest,
    canonical_json,
    content_digest,
)
from .memory_systems.mem0_flat import MEM0_FLAT_EXTRACTION_SLOT


EXTRACTION_MATCHED_EVIDENCE_SCHEMA_VERSION = 1
EXTRACTION_MATCHED_EVIDENCE_SCHEMA = "extraction-matched-evidence-batch-v1"
_IGNORED_ACCOUNTING_ISSUES = {"incomplete_model_usage"}
_SCHEMA_ISSUES = {
    "missing_trace",
    "missing_trace_end",
    "trace_total_mismatch",
    "static_utility_ingestion_join_mismatch",
    "billing_call_count_mismatch",
}
_PROMPT_LEAKAGE_ISSUES = {
    "memory_text_leak",
    "credential_pattern",
    "absolute_source_path",
    "prompt_leakage",
}
_NATIVE_WRITER_ISSUES = {
    "native_writer_contamination",
    "native_memory_writer_bypass",
}


def _file_digest(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ValueError(f"validation evidence file cannot be read: {path.name}") from exc


def _read_json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} cannot be read") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _deduplicate(
    values: Iterable[Any],
    *,
    identity: str,
) -> tuple[Any, ...]:
    result = []
    canonical_by_id: dict[str, str] = {}
    for value in values:
        key = getattr(value, identity)
        canonical = canonical_json(value.payload())
        previous = canonical_by_id.get(key)
        if previous is not None and previous != canonical:
            raise ValueError(f"conflicting validation evidence identity: {key}")
        if previous is None:
            result.append(value)
        canonical_by_id[key] = canonical
    return tuple(result)


def _source_records(run_dir: Path) -> tuple[ExtractionSourceRecord, ...]:
    return _deduplicate(
        (
            record
            for path in sorted(run_dir.rglob("extraction_sources.jsonl"))
            for record in JsonExtractionSourceRecordStore(path).records()
        ),
        identity="record_id",
    )


def _feedback_records(run_dir: Path) -> tuple[LiveExtractionFeedbackRecord, ...]:
    return _deduplicate(
        (
            record
            for path in sorted(run_dir.rglob("rsimem_extraction_feedback.jsonl"))
            for record in JsonLiveExtractionFeedbackRecordLog(path).records()
        ),
        identity="record_id",
    )


def _completed_attempts(manifest: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    latest_by_run: dict[str, dict[str, Any]] = {}
    for event in manifest["attemptHistory"]:
        latest_by_run[event["runName"]] = event
    by_slot: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for event in latest_by_run.values():
        if event["status"] == "completed":
            by_slot.setdefault((event["replicate"], event["method"]), []).append(
                event
            )
    expected = {
        (replicate, method)
        for replicate in range(1, manifest["replicates"] + 1)
        for method in manifest["methods"]
    }
    if set(by_slot) != expected or any(len(values) != 1 for values in by_slot.values()):
        raise ValueError("validation batch does not have one completed run per slot")
    return tuple(
        by_slot[key][0]
        for key in sorted(by_slot, key=lambda value: (value[0], value[1]))
    )


def _issue_weight(issue: Mapping[str, object]) -> int:
    count = issue.get("count", 1)
    if type(count) is not int or count < 1:
        raise ValueError("validation audit issue count is invalid")
    return count


def _audit_safety(
    audit_path: Path,
) -> tuple[str, str, tuple[int, int, int, int]]:
    audit = _read_json(audit_path, "validation audit")
    issues = audit.get("issues")
    if not isinstance(issues, list) or type(audit.get("ok")) is not bool:
        raise ValueError("validation audit status is incomplete")
    if audit["ok"] != (not issues):
        raise ValueError("validation audit status and issues disagree")
    counts = [0, 0, 0, 0]
    for issue in issues:
        if not isinstance(issue, Mapping) or not isinstance(issue.get("kind"), str):
            raise ValueError("validation audit issue is malformed")
        kind = issue["kind"]
        if kind in _IGNORED_ACCOUNTING_ISSUES:
            continue
        weight = _issue_weight(issue)
        if kind in _SCHEMA_ISSUES:
            counts[0] += weight
        elif kind in _PROMPT_LEAKAGE_ISSUES:
            counts[2] += weight
        elif kind in _NATIVE_WRITER_ISSUES:
            counts[3] += weight
        else:
            counts[1] += weight
    digest = _file_digest(audit_path)
    return f"audit.{digest[:40]}", digest, tuple(counts)


@dataclass(frozen=True, slots=True)
class ExtractionObservationEvidenceJoin:
    observation_id: str
    replicate: int
    method: str
    run_name: str
    source_record_id: str
    live_feedback_record_id: str
    safety_evidence_id: str
    audit_digest: str

    def __post_init__(self) -> None:
        for value in (
            self.observation_id,
            self.run_name,
            self.source_record_id,
            self.live_feedback_record_id,
            self.safety_evidence_id,
        ):
            if not isinstance(value, str) or not value:
                raise ValueError("extraction observation evidence join is incomplete")
        if self.method not in EXTRACTION_METHOD_VARIANTS:
            raise ValueError("extraction observation evidence method is invalid")
        if type(self.replicate) is not int or self.replicate < 1:
            raise ValueError("extraction observation evidence replicate is invalid")
        if (
            not isinstance(self.audit_digest, str)
            or len(self.audit_digest) != 64
            or any(value not in "0123456789abcdef" for value in self.audit_digest)
        ):
            raise ValueError("extraction observation audit digest is invalid")

    def payload(self) -> dict[str, object]:
        return {
            "observation_id": self.observation_id,
            "replicate": self.replicate,
            "method": self.method,
            "run_name": self.run_name,
            "source_record_id": self.source_record_id,
            "live_feedback_record_id": self.live_feedback_record_id,
            "safety_evidence_id": self.safety_evidence_id,
            "audit_digest": self.audit_digest,
        }

    @classmethod
    def from_payload(cls, value: object) -> "ExtractionObservationEvidenceJoin":
        fields = {
            "observation_id",
            "replicate",
            "method",
            "run_name",
            "source_record_id",
            "live_feedback_record_id",
            "safety_evidence_id",
            "audit_digest",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError("malformed extraction observation evidence join")
        try:
            return cls(**value)
        except (TypeError, ValueError) as exc:
            raise ValueError("malformed extraction observation evidence join") from exc


@dataclass(frozen=True, slots=True)
class ExtractionMatchedEvidenceBatch:
    evidence_batch_id: str
    batch_id: str
    experiment_id: str
    manifest_digest: str
    offline_decision_id: str
    split_id: str
    criteria_digest: str
    parent_artifact_id: str
    parent_artifact_digest: str
    candidate_artifact_id: str
    candidate_artifact_digest: str
    parent_runtime_artifact_id: str
    candidate_runtime_artifact_id: str
    observations: tuple[ExtractionValidationObservation, ...]
    safety_evidence: tuple[ExtractionValidationSafetyEvidence, ...]
    evidence_joins: tuple[ExtractionObservationEvidenceJoin, ...]
    decision: ExtractionMatchedTrialDecision
    batch_schema: str = EXTRACTION_MATCHED_EVIDENCE_SCHEMA
    schema_version: int = EXTRACTION_MATCHED_EVIDENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            self.schema_version != EXTRACTION_MATCHED_EVIDENCE_SCHEMA_VERSION
            or self.batch_schema != EXTRACTION_MATCHED_EVIDENCE_SCHEMA
        ):
            raise ValueError("unsupported extraction matched evidence batch")
        for value in (
            self.evidence_batch_id,
            self.batch_id,
            self.experiment_id,
            self.offline_decision_id,
            self.split_id,
            self.parent_artifact_id,
            self.candidate_artifact_id,
            self.parent_runtime_artifact_id,
            self.candidate_runtime_artifact_id,
        ):
            if not isinstance(value, str) or not value:
                raise ValueError("matched evidence batch identity is incomplete")
        for value in (
            self.manifest_digest,
            self.criteria_digest,
            self.parent_artifact_digest,
            self.candidate_artifact_digest,
        ):
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError("matched evidence batch digest is invalid")
        observation_ids = tuple(value.observation_id for value in self.observations)
        if not observation_ids or len(observation_ids) != len(set(observation_ids)):
            raise ValueError("matched evidence observations must be unique")
        safety_by_id = {value.evidence_id: value for value in self.safety_evidence}
        joins_by_id = {value.observation_id: value for value in self.evidence_joins}
        if (
            len(safety_by_id) != len(self.safety_evidence)
            or len(joins_by_id) != len(self.evidence_joins)
            or set(joins_by_id) != set(observation_ids)
            or set(safety_by_id)
            != {value.safety_evidence_id for value in self.evidence_joins}
        ):
            raise ValueError("matched evidence joins are incomplete")
        observations = {value.observation_id: value for value in self.observations}
        for observation_id, join in joins_by_id.items():
            safety = safety_by_id.get(join.safety_evidence_id)
            if (
                safety is None
                or safety.source_record_id != join.source_record_id
                or safety.live_feedback_record_id != join.live_feedback_record_id
                or safety.audit_digest != join.audit_digest
                or observations[observation_id].failure_counts != safety.failure_counts
                or not safety.complete
                or observations[observation_id].replicate != join.replicate
                or observations[observation_id].run_id != join.run_name
                or (
                    observations[observation_id].variant
                    == ExtractionValidationVariant.PARENT
                )
                != (join.method == EXTRACTION_METHOD_VARIANTS[0])
            ):
                raise ValueError("matched evidence safety join differs")
        decision = self.decision
        if (
            decision.offline_decision_id != self.offline_decision_id
            or decision.split_id != self.split_id
            or decision.criteria_digest != self.criteria_digest
            or decision.parent_artifact_id != self.parent_artifact_id
            or decision.parent_artifact_digest != self.parent_artifact_digest
            or decision.candidate_artifact_id != self.candidate_artifact_id
            or decision.candidate_artifact_digest != self.candidate_artifact_digest
            or decision.parent_runtime_artifact_id
            != self.parent_runtime_artifact_id
            or decision.candidate_runtime_artifact_id
            != self.candidate_runtime_artifact_id
            or set(decision.quality_decision.observation_ids) != set(observation_ids)
        ):
            raise ValueError("matched evidence decision join differs")
        expected = f"extraction-evidence.{content_digest(self.identity_payload())[:40]}"
        if self.evidence_batch_id != expected:
            raise ValueError("extraction matched evidence batch ID mismatch")

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "batch_schema": self.batch_schema,
            "batch_id": self.batch_id,
            "experiment_id": self.experiment_id,
            "manifest_digest": self.manifest_digest,
            "offline_decision_id": self.offline_decision_id,
            "split_id": self.split_id,
            "criteria_digest": self.criteria_digest,
            "parent_artifact_id": self.parent_artifact_id,
            "parent_artifact_digest": self.parent_artifact_digest,
            "candidate_artifact_id": self.candidate_artifact_id,
            "candidate_artifact_digest": self.candidate_artifact_digest,
            "parent_runtime_artifact_id": self.parent_runtime_artifact_id,
            "candidate_runtime_artifact_id": self.candidate_runtime_artifact_id,
            "observations": [value.payload() for value in self.observations],
            "safety_evidence": [value.payload() for value in self.safety_evidence],
            "evidence_joins": [value.payload() for value in self.evidence_joins],
            "decision": self.decision.payload(),
        }

    def payload(self) -> dict[str, object]:
        return {"evidence_batch_id": self.evidence_batch_id, **self.identity_payload()}

    @classmethod
    def from_payload(cls, value: object) -> "ExtractionMatchedEvidenceBatch":
        fields = {
            "evidence_batch_id",
            "schema_version",
            "batch_schema",
            "batch_id",
            "experiment_id",
            "manifest_digest",
            "offline_decision_id",
            "split_id",
            "criteria_digest",
            "parent_artifact_id",
            "parent_artifact_digest",
            "candidate_artifact_id",
            "candidate_artifact_digest",
            "parent_runtime_artifact_id",
            "candidate_runtime_artifact_id",
            "observations",
            "safety_evidence",
            "evidence_joins",
            "decision",
        }
        if not isinstance(value, Mapping) or set(value) != fields or not all(
            isinstance(value[field], list)
            for field in ("observations", "safety_evidence", "evidence_joins")
        ):
            raise ValueError("malformed extraction matched evidence batch")
        try:
            return cls(
                evidence_batch_id=value["evidence_batch_id"],
                batch_id=value["batch_id"],
                experiment_id=value["experiment_id"],
                manifest_digest=value["manifest_digest"],
                offline_decision_id=value["offline_decision_id"],
                split_id=value["split_id"],
                criteria_digest=value["criteria_digest"],
                parent_artifact_id=value["parent_artifact_id"],
                parent_artifact_digest=value["parent_artifact_digest"],
                candidate_artifact_id=value["candidate_artifact_id"],
                candidate_artifact_digest=value["candidate_artifact_digest"],
                parent_runtime_artifact_id=value["parent_runtime_artifact_id"],
                candidate_runtime_artifact_id=value["candidate_runtime_artifact_id"],
                observations=tuple(
                    ExtractionValidationObservation.from_payload(item)
                    for item in value["observations"]
                ),
                safety_evidence=tuple(
                    ExtractionValidationSafetyEvidence.from_payload(item)
                    for item in value["safety_evidence"]
                ),
                evidence_joins=tuple(
                    ExtractionObservationEvidenceJoin.from_payload(item)
                    for item in value["evidence_joins"]
                ),
                decision=ExtractionMatchedTrialDecision.from_payload(
                    value["decision"]
                ),
                batch_schema=value["batch_schema"],
                schema_version=value["schema_version"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("malformed extraction matched evidence batch") from exc


def _write_immutable(path: Path, batch: ExtractionMatchedEvidenceBatch) -> None:
    serialized = canonical_json(batch.payload()) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + ".lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            if path.exists():
                if path.read_text(encoding="utf-8") != serialized:
                    raise ValueError(
                        "extraction matched evidence file conflicts with its ID"
                    )
                return
            descriptor, temporary = tempfile.mkstemp(
                prefix=f".{path.name}.",
                dir=path.parent,
            )
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    handle.write(serialized)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, path)
                directory = os.open(path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            except BaseException:
                try:
                    os.unlink(temporary)
                except OSError:
                    pass
                raise
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def assemble_extraction_matched_evidence_batch(
    batch_root: Path,
    *,
    parent: ExtractionPromptPolicyArtifact,
    candidate: ExtractionPromptPolicyArtifact,
    offline_decision: ExtractionOfflineValidationDecision,
    split: ExtractionPromptValidationSplit,
    output_path: Path | None = None,
    observation_store_path: Path | None = None,
) -> ExtractionMatchedEvidenceBatch:
    root = batch_root.expanduser().resolve()
    manifest_path = root / "batch_manifest.json"
    manifest = load_extraction_manifest_for_phase(
        manifest_path,
        required_phase="validation",
    )
    criteria = extraction_acceptance_criteria_from_manifest(manifest)
    validation_assignment = next(
        (
            value
            for value in split.assignments
            if value.role == ExtractionValidationSplitRole.VALIDATION
            and value.family_id == manifest["split"]["familyId"]
            and value.task_template_group_id
            == manifest["split"]["taskTemplateGroupId"]
            and value.task_manifest_digest
            == manifest["split"]["taskManifestDigest"]
        ),
        None,
    )
    if validation_assignment is None:
        raise ValueError("validation batch is outside the declared split")
    parent_component = parent.to_prompt_component(MEM0_FLAT_EXTRACTION_SLOT)
    candidate_component = candidate.to_prompt_component(MEM0_FLAT_EXTRACTION_SLOT)
    expected = manifest["semanticPolicy"]["activeArtifactByMethod"]
    expected_semantic = {
        EXTRACTION_METHOD_VARIANTS[0]: SemanticPolicyManifest.from_payload(
            manifest["semanticPolicy"]["parent"]
        ),
        EXTRACTION_METHOD_VARIANTS[1]: SemanticPolicyManifest.from_payload(
            manifest["semanticPolicy"]["active"]
        ),
    }
    expected_policy_artifacts = {
        EXTRACTION_METHOD_VARIANTS[0]: parent,
        EXTRACTION_METHOD_VARIANTS[1]: candidate,
    }
    expected_components = {
        EXTRACTION_METHOD_VARIANTS[0]: parent_component,
        EXTRACTION_METHOD_VARIANTS[1]: candidate_component,
    }
    if expected != {
        EXTRACTION_METHOD_VARIANTS[0]: {
            "artifactId": parent_component.artifact_id,
            "artifactDigest": parent.body_digest,
        },
        EXTRACTION_METHOD_VARIANTS[1]: {
            "artifactId": candidate_component.artifact_id,
            "artifactDigest": candidate.body_digest,
        },
    }:
        raise ValueError("validation batch runtime artifact identity differs")

    observations = []
    safety_values = []
    joins = []
    for attempt in _completed_attempts(manifest):
        run_dir = (root / attempt["outputDirectory"]).resolve()
        if not run_dir.is_relative_to(root):
            raise ValueError("validation run directory escapes its batch")
        sources = _source_records(run_dir)
        feedback = _feedback_records(run_dir)
        if not sources or not feedback:
            raise ValueError("validation run has incomplete extraction evidence")
        process_corpus = JsonProcessCorpusStore(
            run_dir / "process_corpus.json"
        ).get()
        if process_corpus is None:
            raise ValueError("validation run has incomplete process evidence")
        if (
            process_corpus.split_role != "validation"
            or process_corpus.family_id != manifest["split"]["familyId"]
            or process_corpus.task_template_group_id
            != manifest["split"]["taskTemplateGroupId"]
            or process_corpus.task_manifest_digest
            != manifest["split"]["taskManifestDigest"]
        ):
            raise ValueError("validation process corpus identity differs")
        process_errors = audit_process_events(process_corpus.events)
        if process_errors:
            raise ValueError(
                "validation process evidence audit failed: "
                + "; ".join(process_errors)
            )
        ensure_process_corpus_has_no_evaluation_fields(process_corpus.payload())
        audit_id, audit_digest, failure_counts = _audit_safety(
            run_dir / "audit.json"
        )
        expected_artifact = expected[attempt["method"]]
        expected_policy = expected_semantic[attempt["method"]]
        expected_policy_artifact = expected_policy_artifacts[attempt["method"]]
        expected_component = expected_components[attempt["method"]]
        expected_scope = (
            ExtractionPromptRuntimeScope.ROOT_STATIC
            if attempt["method"] == EXTRACTION_METHOD_VARIANTS[0]
            else ExtractionPromptRuntimeScope.MATCHED_VALIDATION
        )
        sources_by_id = {value.record_id: value for value in sources}
        if any(
            source.run_id != attempt["runName"]
            or source.family_id != manifest["split"]["familyId"]
            or source.extraction_artifact_id != expected_artifact["artifactId"]
            or source.extraction_artifact_digest
            != expected_artifact["artifactDigest"]
            or source.activation.runtime_binding.policy_artifact_id
            != expected_policy_artifact.artifact_id
            or source.activation.runtime_binding.policy_artifact_digest
            != expected_policy_artifact.artifact_digest
            or source.activation.runtime_binding.deployment_scope != expected_scope
            or source.activation.runtime_binding.adapter_id
            != expected_component.owner_adapter_id
            or source.activation.runtime_binding.slot_id
            != expected_policy_artifact.slot_id
            or source.activation.runtime_binding.slot_contract_digest
            != expected_policy_artifact.slot_contract_digest
            or source.activation.runtime_binding.frozen_wrapper_digest
            != expected_policy_artifact.frozen_wrapper_digest
            or source.activation.runtime_binding.input_schema_digest
            != expected_policy_artifact.input_schema_digest
            or source.activation.runtime_binding.output_schema_digest
            != expected_policy_artifact.output_schema_digest
            or source.activation.runtime_binding.model_profile
            != expected_policy_artifact.model_profile
            or source.activation.semantic_policy != expected_policy
            or source.activation.invocation.binding_id
            != source.activation.runtime_binding.binding_id
            or source.activation.invocation.rendered_template_digest
            != source.activation.runtime_binding.rendered_template_digest
            or source.activation.persisted_artifact_ids != source.artifact_ids
            for source in sources
        ):
            raise ValueError(
                "validation source identity differs from its run activation"
            )
        if {value.source_record_id for value in feedback} != set(sources_by_id):
            raise ValueError("validation sources lack complete feedback closure")
        variant = (
            ExtractionValidationVariant.PARENT
            if attempt["method"] == EXTRACTION_METHOD_VARIANTS[0]
            else ExtractionValidationVariant.PROPOSAL
        )
        for live in feedback:
            source = sources_by_id.get(live.source_record_id)
            if (
                source is None
                or live.run_id != attempt["runName"]
                or live.family_id != manifest["split"]["familyId"]
                or live.dataset.contract_digest
                != manifest["feedbackContract"]["contractDigest"]
            ):
                raise ValueError("validation live feedback identity differs")
            safety = ExtractionValidationSafetyEvidence.create(
                live_feedback_record_id=live.record_id,
                source_record_id=source.record_id,
                audit_id=audit_id,
                audit_digest=audit_digest,
                evidence_cutoff_operation_id=live.outcome_operation_id,
                complete=True,
                schema_failure_count=failure_counts[0],
                safety_failure_count=failure_counts[1],
                prompt_leakage_failure_count=failure_counts[2],
                native_writer_failure_count=failure_counts[3],
            )
            observation = ExtractionValidationObservationAssembler().assemble(
                live_feedback=live,
                source=source,
                safety=safety,
                variant=variant,
                replicate=attempt["replicate"],
                task_template_group_id=manifest["split"]["taskTemplateGroupId"],
                task_manifest_digest=manifest["split"]["taskManifestDigest"],
                model_profile_digest=manifest["modelProfile"]["profileDigest"],
                budget_id=manifest["requestBudget"]["budgetId"],
                persistence_state_digest=manifest["persistenceIsolation"][
                    "profileDigest"
                ],
            )
            observations.append(observation)
            safety_values.append(safety)
            joins.append(ExtractionObservationEvidenceJoin(
                observation.observation_id,
                attempt["replicate"],
                attempt["method"],
                attempt["runName"],
                source.record_id,
                live.record_id,
                safety.evidence_id,
                audit_digest,
            ))
    ordered = tuple(sorted(observations, key=lambda value: value.observation_id))
    # Persist raw observations before deriving the matched decision.  The
    # decision evaluator below intentionally consumes the reloaded records so
    # a restart/replay cannot silently substitute an in-memory observation.
    observation_store = JsonExtractionValidationObservationStore(
        observation_store_path or (root / "validation_observations"),
        split=split,
    )
    for observation in ordered:
        observation_store.put(observation)
    persisted = observation_store.records()
    if persisted != ordered:
        raise ValueError("persisted validation observations differ from assembly")
    ordered = persisted
    safety_ordered = tuple(sorted(safety_values, key=lambda value: value.evidence_id))
    joins_ordered = tuple(sorted(joins, key=lambda value: value.observation_id))
    decision = ExtractionMatchedTrialEvaluator().evaluate(
        parent=parent,
        candidate=candidate,
        offline_decision=offline_decision,
        split=split,
        observations=ordered,
        criteria=criteria,
        parent_runtime_artifact_id=parent_component.artifact_id,
        candidate_runtime_artifact_id=candidate_component.artifact_id,
    )
    values = {
        "batch_id": manifest["batchId"],
        "experiment_id": manifest["experimentId"],
        "manifest_digest": _file_digest(manifest_path),
        "offline_decision_id": offline_decision.decision_id,
        "split_id": split.split_id,
        "criteria_digest": criteria.digest,
        "parent_artifact_id": parent.artifact_id,
        "parent_artifact_digest": parent.artifact_digest,
        "candidate_artifact_id": candidate.artifact_id,
        "candidate_artifact_digest": candidate.artifact_digest,
        "parent_runtime_artifact_id": parent_component.artifact_id,
        "candidate_runtime_artifact_id": candidate_component.artifact_id,
        "observations": ordered,
        "safety_evidence": safety_ordered,
        "evidence_joins": joins_ordered,
        "decision": decision,
        "batch_schema": EXTRACTION_MATCHED_EVIDENCE_SCHEMA,
        "schema_version": EXTRACTION_MATCHED_EVIDENCE_SCHEMA_VERSION,
    }
    identity = {
        "schema_version": values["schema_version"],
        "batch_schema": values["batch_schema"],
        "batch_id": values["batch_id"],
        "experiment_id": values["experiment_id"],
        "manifest_digest": values["manifest_digest"],
        "offline_decision_id": values["offline_decision_id"],
        "split_id": values["split_id"],
        "criteria_digest": values["criteria_digest"],
        "parent_artifact_id": values["parent_artifact_id"],
        "parent_artifact_digest": values["parent_artifact_digest"],
        "candidate_artifact_id": values["candidate_artifact_id"],
        "candidate_artifact_digest": values["candidate_artifact_digest"],
        "parent_runtime_artifact_id": values["parent_runtime_artifact_id"],
        "candidate_runtime_artifact_id": values["candidate_runtime_artifact_id"],
        "observations": [value.payload() for value in ordered],
        "safety_evidence": [value.payload() for value in safety_ordered],
        "evidence_joins": [value.payload() for value in joins_ordered],
        "decision": decision.payload(),
    }
    batch = ExtractionMatchedEvidenceBatch(
        evidence_batch_id=(
            f"extraction-evidence.{content_digest(identity)[:40]}"
        ),
        **values,
    )
    if output_path is not None:
        _write_immutable(output_path.expanduser().resolve(), batch)
    return batch


def load_extraction_matched_evidence_batch(
    path: Path,
) -> ExtractionMatchedEvidenceBatch:
    return ExtractionMatchedEvidenceBatch.from_payload(
        _read_json(path.expanduser().resolve(), "extraction matched evidence batch")
    )
