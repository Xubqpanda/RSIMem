"""Audit live extraction feedback and build an optimizer-only corpus."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, TypeVar

from .memory.extraction_feedback import (
    AttributionConfidence,
    ExtractionFeedbackLabel,
)
from .memory.extraction_optimizer_audit import audit_optimizer_corpus_isolation
from .memory.extraction_optimizer_builder import (
    DelayedEvidenceContent,
    ExtractionOptimizerCorpusBuilder,
)
from .memory.extraction_optimizer_capture import (
    ExtractionOptimizerFeedbackCapture,
    ExtractionOptimizerSourceCapture,
    JsonExtractionOptimizerCaptureLog,
)
from .memory.extraction_optimizer_contracts import (
    FROZEN_EXTRACTION_OPTIMIZER_CONFIG,
)
from .memory.extraction_optimizer_corpus import (
    ExtractionOptimizerCorpus,
    OptimizerComponentOwnership,
    OptimizerCorpusRetention,
    OptimizerCorpusSplit,
    PROCESS_SIGNAL_GATE_NO_SIGNAL,
    PROCESS_SIGNAL_GATE_NOT_BOUND,
    PROCESS_SIGNAL_GATE_READY,
)
from .memory.process_signal import (
    JsonProcessSignalCaseStore,
    ProcessSignalCaseStatus,
    census_process_signal_cases,
)
from .memory.extraction_optimizer_store import JsonExtractionOptimizerCorpusStore
from .memory.extraction_projection import (
    EXTRACTION_SOURCE_RECORD_SCHEMA_VERSION,
    LIVE_EXTRACTION_FEEDBACK_SCHEMA_VERSION,
    ExtractionSourceRecord,
    JsonExtractionSourceRecordStore,
    JsonLiveExtractionFeedbackRecordLog,
    LiveExtractionFeedbackRecord,
)
from .memory.operation_graph import (
    AppendOnlyOperationEvidenceLog,
    OperationGraph,
    materialize_operation_graph,
)
from .memory.prompt_components import canonical_json, content_digest


EXTRACTION_PREPARATION_SCHEMA_VERSION = 1
EXTRACTION_PREPARATION_SCHEMA = "extraction-optimizer-preparation-v1"
_ACTIONABLE_LABELS = {
    ExtractionFeedbackLabel.USEFUL.value,
    ExtractionFeedbackLabel.HARMFUL.value,
    ExtractionFeedbackLabel.MISSED.value,
}
_ACTIONABLE_CONFIDENCE = {
    AttributionConfidence.HIGH.value,
    AttributionConfidence.MEDIUM.value,
}
T = TypeVar("T")


def _jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    values = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            raise ValueError(f"empty JSONL record in {path.name}:{line_number}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"malformed JSONL record in {path.name}:{line_number}"
            ) from exc
        if not isinstance(value, dict):
            raise ValueError(f"JSONL record must be an object in {path.name}")
        values.append(value)
    return tuple(values)


def _deduplicate(
    values: Iterable[T],
    *,
    identity: str,
    payload,
) -> tuple[T, ...]:
    result = []
    canonical_by_id: dict[str, str] = {}
    for value in values:
        key = getattr(value, identity)
        canonical = canonical_json(payload(value))
        previous = canonical_by_id.get(key)
        if previous is not None and previous != canonical:
            raise ValueError(f"conflicting extraction preparation identity: {key}")
        if previous is None:
            result.append(value)
        canonical_by_id[key] = canonical
    return tuple(result)


def _source_paths(batch_dir: Path) -> tuple[Path, ...]:
    return tuple(sorted(batch_dir.rglob("extraction_sources.jsonl")))


def _feedback_paths(batch_dir: Path) -> tuple[Path, ...]:
    return tuple(sorted(batch_dir.rglob("rsimem_extraction_feedback.jsonl")))


def _capture_paths(batch_dir: Path) -> tuple[Path, ...]:
    return tuple(sorted(batch_dir.rglob("extraction_optimizer_capture.jsonl")))


def _operation_paths(batch_dir: Path) -> tuple[Path, ...]:
    return tuple(sorted(batch_dir.rglob("rsimem_semantic_operations.jsonl")))


def _process_signal_paths(batch_dir: Path) -> tuple[Path, ...]:
    return tuple(sorted(batch_dir.rglob("process_signal_cases.jsonl")))


def _process_signal_observation_window(batch_dir: Path) -> str:
    """Return the frozen window identity for optimizer join provenance."""

    windows = {
        case.observation_window
        for path in _process_signal_paths(batch_dir)
        for case in JsonProcessSignalCaseStore(path).records()
        if case.observation_window is not None
    }
    if len(windows) > 1:
        raise ValueError("process-signal cases mix observation windows")
    return next(iter(windows), "window.unbound")


def _process_signal_gate(
    batch_dir: Path,
) -> tuple[str, str | None, str | None, int, int, str | None]:
    """Return the bound process-signal gate and its case counts.

    A batch without process-signal case files is a legacy/unit fixture and is
    intentionally marked ``not_bound``. Once a case store is present, the
    optimizer must observe at least one explicit optimization signal before a
    provider request can be made.
    """

    paths = _process_signal_paths(batch_dir)
    if not paths:
        return PROCESS_SIGNAL_GATE_NOT_BOUND, None, None, 0, 0, None
    cases = []
    for path in paths:
        cases.extend(JsonProcessSignalCaseStore(path).records())
    if not cases:
        return PROCESS_SIGNAL_GATE_NO_SIGNAL, None, None, 0, 0, None
    # A logical case may be observed several times (for example across
    # provider replicates or retrieval boundaries).  Never treat conflicting
    # statuses as independent optimization evidence.  The census also checks
    # that a physical observation identity is not reused across cases; that
    # is a malformed store and must fail closed rather than being silently
    # deduplicated here.
    census = census_process_signal_cases(cases)
    has_conflicting_logical_case = census.conflict_case_count > 0
    metadata = {
        (
            case.analysis_protocol_id,
            case.replicate_id,
            case.observation_window,
        )
        for case in cases
    }
    if any(any(value is None for value in item) for item in metadata):
        raise ValueError("process-signal cases are not fully protocol bound")
    protocols = {item[0] for item in metadata}
    windows = {item[2] for item in metadata}
    if len(protocols) != 1 or len(windows) != 1:
        raise ValueError("process-signal cases mix frozen protocols")
    protocol_id = next(iter(protocols))
    case_digest = content_digest(
        [case.payload() for case in sorted(cases, key=lambda item: item.case_id)]
    )
    if has_conflicting_logical_case:
        return (
            PROCESS_SIGNAL_GATE_NO_SIGNAL,
            protocol_id,
            case_digest,
            len(cases),
            0,
            None,
        )
    optimization_cases = [
        case
        for case in cases
        if case.status is ProcessSignalCaseStatus.OPTIMIZATION_SIGNAL
    ]
    optimization = len(optimization_cases)
    by_hypothesis: dict[str, set[str]] = {}
    for case in optimization_cases:
        if case.abstract_hypothesis_digest is not None:
            by_hypothesis.setdefault(case.abstract_hypothesis_digest, set()).add(
                case.logical_case_id
            )
    supporting_hypotheses = {
        hypothesis
        for hypothesis, logical_case_ids in by_hypothesis.items()
        if len(logical_case_ids) >= 2
    }
    supports_general_edit = len(supporting_hypotheses) == 1
    hypothesis_digest = next(iter(supporting_hypotheses), None)
    gate = (
        PROCESS_SIGNAL_GATE_READY
        if supports_general_edit
        else PROCESS_SIGNAL_GATE_NO_SIGNAL
    )
    return (
        gate,
        protocol_id,
        case_digest,
        len(cases),
        optimization,
        hypothesis_digest,
    )


def _raw_source_versions(batch_dir: Path) -> tuple[int, ...]:
    versions = []
    for path in _source_paths(batch_dir):
        versions.extend(value.get("schema_version") for value in _jsonl(path))
    return tuple(sorted({value for value in versions if type(value) is int}))


def _raw_feedback(batch_dir: Path) -> tuple[dict[str, Any], ...]:
    return tuple(
        value
        for path in _feedback_paths(batch_dir)
        for value in _jsonl(path)
    )


def _capture_records(batch_dir: Path):
    values = (
        value
        for path in _capture_paths(batch_dir)
        for value in JsonExtractionOptimizerCaptureLog(path).records()
    )
    return _deduplicate(
        values,
        identity="capture_id",
        payload=lambda value: value.payload(),
    )


def _source_records(batch_dir: Path) -> tuple[ExtractionSourceRecord, ...]:
    return _deduplicate(
        (
            value
            for path in _source_paths(batch_dir)
            for value in JsonExtractionSourceRecordStore(path).records()
        ),
        identity="record_id",
        payload=lambda value: value.payload(),
    )


def _feedback_records(
    batch_dir: Path,
) -> tuple[LiveExtractionFeedbackRecord, ...]:
    return _deduplicate(
        (
            value
            for path in _feedback_paths(batch_dir)
            for value in JsonLiveExtractionFeedbackRecordLog(path).records()
        ),
        identity="record_id",
        payload=lambda value: value.payload(),
    )


def _operation_graph(batch_dir: Path) -> OperationGraph:
    merged = AppendOnlyOperationEvidenceLog()
    for path in _operation_paths(batch_dir):
        for event in AppendOnlyOperationEvidenceLog(path).events:
            merged.append(event)
    return materialize_operation_graph(merged.events)


def _primary_feedback_counts(
    feedback_payloads: tuple[dict[str, Any], ...],
) -> tuple[dict[str, int], int]:
    counts = {label.value: 0 for label in ExtractionFeedbackLabel}
    actionable = 0
    for record in feedback_payloads:
        dataset = record.get("dataset")
        examples = dataset.get("examples") if isinstance(dataset, Mapping) else None
        if not isinstance(examples, list):
            continue
        for example in examples:
            if not isinstance(example, Mapping) or example.get("primary") is not True:
                continue
            label = example.get("label")
            if label in counts:
                counts[label] += 1
            if (
                label in _ACTIONABLE_LABELS
                and example.get("attribution_confidence")
                in _ACTIONABLE_CONFIDENCE
            ):
                actionable += 1
    return counts, actionable


def _optimizer_signal_is_ready(
    *,
    corpus_ready: bool,
    actionable_primary_count: int,
    process_signal_gate: str,
) -> bool:
    """Require a bound pure-process signal before optimizer readiness."""

    return (
        corpus_ready
        and actionable_primary_count
        >= FROZEN_EXTRACTION_OPTIMIZER_CONFIG.minimum_actionable_primary_examples
        and process_signal_gate == PROCESS_SIGNAL_GATE_READY
    )


def _build_optimizer_examples(
    root: Path,
    *,
    sources: tuple[ExtractionSourceRecord, ...],
    feedback: tuple[LiveExtractionFeedbackRecord, ...],
    captures: tuple[object, ...],
) -> tuple[object, ...]:
    source_by_id = {value.record_id: value for value in sources}
    source_capture_by_id = {
        value.source_record_id: value
        for value in captures
        if isinstance(value, ExtractionOptimizerSourceCapture)
    }
    feedback_capture_by_id = {
        value.feedback_record_id: value
        for value in captures
        if isinstance(value, ExtractionOptimizerFeedbackCapture)
    }
    graph = _operation_graph(root)
    builder = ExtractionOptimizerCorpusBuilder()
    observation_window = _process_signal_observation_window(root)
    examples = []
    for feedback_record in feedback:
        source = source_by_id.get(feedback_record.source_record_id)
        source_capture = source_capture_by_id.get(feedback_record.source_record_id)
        feedback_capture = feedback_capture_by_id.get(feedback_record.record_id)
        if source is None or source_capture is None or feedback_capture is None:
            raise ValueError("optimizer corpus capture join is incomplete")
        if source_capture.source_record_digest != source.content_digest:
            raise ValueError("optimizer source capture record digest mismatch")
        if feedback_capture.source_record_id != source.record_id:
            raise ValueError("optimizer feedback capture source mismatch")
        examples.extend(builder.build_examples(
            projection=source_capture.projection,
            source_record=source,
            feedback_record=feedback_record,
            observation=feedback_capture.observation,
            operation_graph=graph,
            fact_contents=source_capture.fact_contents,
            delayed_content=DelayedEvidenceContent(
                source_capture.captured_at,
                feedback_capture.captured_at,
                feedback_capture.current_input,
                observation_window,
            ),
        ))
    return tuple(examples)


@dataclass(frozen=True, slots=True)
class ExtractionFeedbackBatchAudit:
    audit_id: str
    batch_id: str
    source_count: int
    feedback_count: int
    source_schema_versions: tuple[int, ...]
    feedback_schema_versions: tuple[int, ...]
    source_capture_count: int
    feedback_capture_count: int
    primary_label_counts: Mapping[str, int]
    actionable_primary_count: int
    corpus_ready: bool
    optimizer_signal_ready: bool
    reason_codes: tuple[str, ...]
    process_signal_gate: str = PROCESS_SIGNAL_GATE_NOT_BOUND
    process_signal_protocol_id: str | None = None
    process_signal_case_digest: str | None = None
    process_signal_case_count: int = 0
    process_signal_optimization_count: int = 0
    process_signal_hypothesis_digest: str | None = None
    schema_version: int = EXTRACTION_PREPARATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != EXTRACTION_PREPARATION_SCHEMA_VERSION:
            raise ValueError("unsupported extraction preparation audit schema")
        if not self.reason_codes:
            raise ValueError("extraction preparation audit requires reasons")
        if self.process_signal_gate not in {
            PROCESS_SIGNAL_GATE_NOT_BOUND,
            PROCESS_SIGNAL_GATE_NO_SIGNAL,
            PROCESS_SIGNAL_GATE_READY,
        }:
            raise ValueError("extraction preparation process-signal gate is invalid")
        if self.process_signal_protocol_id is not None:
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:+-]{0,255}", self.process_signal_protocol_id):
                raise ValueError("process signal protocol identity is invalid")
        for value, name in (
            (self.process_signal_case_digest, "process signal case digest"),
            (self.process_signal_hypothesis_digest, "process signal hypothesis digest"),
        ):
            if value is not None and not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError(f"{name} must be sha256")
        for value, name in (
            (self.process_signal_case_count, "process signal case count"),
            (self.process_signal_optimization_count, "process signal optimization count"),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.process_signal_optimization_count > self.process_signal_case_count:
            raise ValueError("process signal optimization count exceeds case count")
        if self.process_signal_case_count == 0 and self.process_signal_case_digest is not None:
            raise ValueError("empty process signal gate cannot carry case digest")
        if self.process_signal_case_count and (
            self.process_signal_protocol_id is None
            or self.process_signal_case_digest is None
        ):
            raise ValueError("bound process signal gate requires provenance")
        if self.process_signal_gate == PROCESS_SIGNAL_GATE_NOT_BOUND and any((
            self.process_signal_protocol_id is not None,
            self.process_signal_case_digest is not None,
            self.process_signal_case_count,
            self.process_signal_optimization_count,
            self.process_signal_hypothesis_digest is not None,
        )):
            raise ValueError("unbound process signal gate cannot carry evidence")
        if self.process_signal_gate == PROCESS_SIGNAL_GATE_READY:
            if self.process_signal_optimization_count < 2 or self.process_signal_hypothesis_digest is None:
                raise ValueError("ready process signal gate requires replicated hypothesis")
        if self.optimizer_signal_ready and self.process_signal_gate != PROCESS_SIGNAL_GATE_READY:
            raise ValueError(
                "optimizer signal readiness requires a ready process-signal gate"
            )
        expected = f"extraction-preparation-audit.{content_digest(self.identity_payload())[:40]}"
        if self.audit_id != expected:
            raise ValueError("extraction preparation audit ID mismatch")

    def identity_payload(self) -> dict[str, object]:
        return {
            "schemaVersion": self.schema_version,
            "auditSchema": EXTRACTION_PREPARATION_SCHEMA,
            "batchId": self.batch_id,
            "sourceCount": self.source_count,
            "feedbackCount": self.feedback_count,
            "sourceSchemaVersions": list(self.source_schema_versions),
            "feedbackSchemaVersions": list(self.feedback_schema_versions),
            "sourceCaptureCount": self.source_capture_count,
            "feedbackCaptureCount": self.feedback_capture_count,
            "primaryLabelCounts": dict(sorted(self.primary_label_counts.items())),
            "actionablePrimaryCount": self.actionable_primary_count,
            "minimumActionablePrimaryExamples": (
                FROZEN_EXTRACTION_OPTIMIZER_CONFIG
                .minimum_actionable_primary_examples
            ),
            "corpusReady": self.corpus_ready,
            "optimizerSignalReady": self.optimizer_signal_ready,
            "processSignalGate": self.process_signal_gate,
            "processSignalProtocolId": self.process_signal_protocol_id,
            "processSignalCaseDigest": self.process_signal_case_digest,
            "processSignalCaseCount": self.process_signal_case_count,
            "processSignalOptimizationCount": self.process_signal_optimization_count,
            "processSignalHypothesisDigest": self.process_signal_hypothesis_digest,
            "reasonCodes": list(self.reason_codes),
        }

    def payload(self) -> dict[str, object]:
        return {"auditId": self.audit_id, **self.identity_payload()}


def audit_extraction_feedback_batch(
    batch_dir: Path,
    *,
    batch_id: str,
) -> ExtractionFeedbackBatchAudit:
    root = batch_dir.expanduser().resolve()
    if not root.is_dir():
        raise ValueError("extraction feedback batch directory is missing")
    reasons = []
    source_paths = _source_paths(root)
    feedback_payloads = _raw_feedback(root)
    source_versions = _raw_source_versions(root)
    feedback_versions = tuple(sorted({
        value["schema_version"]
        for value in feedback_payloads
        if type(value.get("schema_version")) is int
    }))
    raw_sources = tuple(
        value for path in source_paths for value in _jsonl(path)
    )
    captures = _capture_records(root) if _capture_paths(root) else ()
    source_captures = tuple(
        value for value in captures
        if isinstance(value, ExtractionOptimizerSourceCapture)
    )
    feedback_captures = tuple(
        value for value in captures
        if isinstance(value, ExtractionOptimizerFeedbackCapture)
    )
    process_signal_gate = PROCESS_SIGNAL_GATE_NOT_BOUND
    process_signal_protocol_id = None
    process_signal_case_digest = None
    process_signal_case_count = 0
    process_signal_optimization_count = 0
    process_signal_hypothesis_digest = None
    process_signal_paths = _process_signal_paths(root)
    if process_signal_paths:
        try:
            (
                process_signal_gate,
                process_signal_protocol_id,
                process_signal_case_digest,
                process_signal_case_count,
                process_signal_optimization_count,
                process_signal_hypothesis_digest,
            ) = _process_signal_gate(root)
        except (OSError, ValueError):
            reasons.append("process_signal_evidence_invalid")
            process_signal_gate = PROCESS_SIGNAL_GATE_NO_SIGNAL
            process_signal_protocol_id = None
            process_signal_case_digest = None
            process_signal_case_count = 0
            process_signal_optimization_count = 0
            process_signal_hypothesis_digest = None
        if (
            process_signal_gate == PROCESS_SIGNAL_GATE_NO_SIGNAL
            and "process_signal_evidence_invalid" not in reasons
        ):
            reasons.append("no_optimization_process_signal")
    primary_counts, actionable = _primary_feedback_counts(feedback_payloads)
    if not raw_sources:
        reasons.append("source_evidence_missing")
    if not feedback_payloads:
        reasons.append("feedback_evidence_missing")
    if source_versions != (EXTRACTION_SOURCE_RECORD_SCHEMA_VERSION,):
        reasons.append("source_schema_not_current")
    if feedback_versions != (LIVE_EXTRACTION_FEEDBACK_SCHEMA_VERSION,):
        reasons.append("feedback_schema_not_current")
    relevant_source_ids = {
        value.get("source_record_id") for value in feedback_payloads
        if isinstance(value.get("source_record_id"), str)
    }
    captured_source_ids = {value.source_record_id for value in source_captures}
    captured_feedback_ids = {
        value.feedback_record_id for value in feedback_captures
    }
    feedback_ids = {
        value.get("record_id") for value in feedback_payloads
        if isinstance(value.get("record_id"), str)
    }
    if not relevant_source_ids.issubset(captured_source_ids):
        reasons.append("source_optimizer_capture_missing")
    if not feedback_ids.issubset(captured_feedback_ids):
        reasons.append("feedback_optimizer_capture_missing")
    if not reasons:
        try:
            _build_optimizer_examples(
                root,
                sources=_source_records(root),
                feedback=_feedback_records(root),
                captures=captures,
            )
        except (TypeError, ValueError):
            reasons.append("optimizer_corpus_join_invalid")
    corpus_ready = not reasons
    optimizer_signal_ready = _optimizer_signal_is_ready(
        corpus_ready=corpus_ready,
        actionable_primary_count=actionable,
        process_signal_gate=process_signal_gate,
    )
    if corpus_ready and not optimizer_signal_ready:
        if actionable < FROZEN_EXTRACTION_OPTIMIZER_CONFIG.minimum_actionable_primary_examples:
            reasons.append("insufficient_actionable_extraction_signal")
        elif process_signal_gate != PROCESS_SIGNAL_GATE_READY:
            reasons.append("process_signal_gate_not_ready")
    if not reasons:
        reasons.append("optimizer_signal_ready")
    values = {
        "batch_id": batch_id,
        "source_count": len({
            value.get("record_id") for value in raw_sources
            if isinstance(value.get("record_id"), str)
        }),
        "feedback_count": len(feedback_ids),
        "source_schema_versions": source_versions,
        "feedback_schema_versions": feedback_versions,
        "source_capture_count": len(source_captures),
        "feedback_capture_count": len(feedback_captures),
        "primary_label_counts": primary_counts,
        "actionable_primary_count": actionable,
        "corpus_ready": corpus_ready,
        "optimizer_signal_ready": optimizer_signal_ready,
        "reason_codes": tuple(reasons),
        "process_signal_gate": process_signal_gate,
        "process_signal_protocol_id": process_signal_protocol_id,
        "process_signal_case_digest": process_signal_case_digest,
        "process_signal_case_count": process_signal_case_count,
        "process_signal_optimization_count": process_signal_optimization_count,
        "process_signal_hypothesis_digest": process_signal_hypothesis_digest,
    }
    identity = {
        "schemaVersion": EXTRACTION_PREPARATION_SCHEMA_VERSION,
        "auditSchema": EXTRACTION_PREPARATION_SCHEMA,
        "batchId": values["batch_id"],
        "sourceCount": values["source_count"],
        "feedbackCount": values["feedback_count"],
        "sourceSchemaVersions": list(values["source_schema_versions"]),
        "feedbackSchemaVersions": list(values["feedback_schema_versions"]),
        "sourceCaptureCount": values["source_capture_count"],
        "feedbackCaptureCount": values["feedback_capture_count"],
        "primaryLabelCounts": dict(sorted(primary_counts.items())),
        "actionablePrimaryCount": actionable,
        "minimumActionablePrimaryExamples": (
            FROZEN_EXTRACTION_OPTIMIZER_CONFIG
            .minimum_actionable_primary_examples
        ),
        "corpusReady": corpus_ready,
        "optimizerSignalReady": optimizer_signal_ready,
        "processSignalGate": process_signal_gate,
        "processSignalProtocolId": process_signal_protocol_id,
        "processSignalCaseDigest": process_signal_case_digest,
        "processSignalCaseCount": process_signal_case_count,
        "processSignalOptimizationCount": process_signal_optimization_count,
        "processSignalHypothesisDigest": process_signal_hypothesis_digest,
        "reasonCodes": list(values["reason_codes"]),
    }
    return ExtractionFeedbackBatchAudit(
        audit_id=(
            f"extraction-preparation-audit.{content_digest(identity)[:40]}"
        ),
        **values,
    )


def build_extraction_optimizer_corpus(
    batch_dir: Path,
    *,
    batch_id: str,
    attempt_id: str,
    observation_cutoff: str,
    attempt_root: Path,
    owner_controlled_root: Path,
    retention: OptimizerCorpusRetention = (
        OptimizerCorpusRetention.DELETE_AFTER_POLICY_DECISION
    ),
) -> tuple[
    ExtractionFeedbackBatchAudit,
    ExtractionOptimizerCorpus,
    JsonExtractionOptimizerCorpusStore,
]:
    root = batch_dir.expanduser().resolve()
    audit = audit_extraction_feedback_batch(root, batch_id=batch_id)
    if not audit.corpus_ready:
        raise ValueError(
            "extraction optimizer corpus is not reconstructable: "
            + ",".join(audit.reason_codes)
        )
    sources = _source_records(root)
    feedback = _feedback_records(root)
    captures = _capture_records(root)
    graph = _operation_graph(root)
    examples = _build_optimizer_examples(
        root,
        sources=sources,
        feedback=feedback,
        captures=captures,
    )
    corpus = ExtractionOptimizerCorpus.create(
        batch_id=batch_id,
        attempt_id=attempt_id,
        split=OptimizerCorpusSplit.TRAIN,
        observation_cutoff=observation_cutoff,
        retention=retention,
        examples=tuple(examples),
        process_signal_gate=audit.process_signal_gate,
        process_signal_protocol_id=audit.process_signal_protocol_id,
        process_signal_case_digest=audit.process_signal_case_digest,
        process_signal_case_count=audit.process_signal_case_count,
        process_signal_optimization_count=audit.process_signal_optimization_count,
        process_signal_hypothesis_digest=audit.process_signal_hypothesis_digest,
    )
    public_payloads = {
        "source_records": [value.payload() for value in sources],
        "feedback_records": [value.payload() for value in feedback],
        "operation_graph": {
            "artifacts": [value.to_payload() for value in graph.artifacts],
            "operations": [value.to_payload() for value in graph.operations],
            "mutations": [value.to_payload() for value in graph.mutations],
        },
    }
    isolation_issues = audit_optimizer_corpus_isolation(corpus, public_payloads)
    if isolation_issues:
        raise ValueError(
            "optimizer corpus content leaked into public evidence: "
            + ",".join(isolation_issues)
        )
    store = JsonExtractionOptimizerCorpusStore(
        attempt_root,
        owner_controlled_root=owner_controlled_root,
        attempt_id=attempt_id,
        split=OptimizerCorpusSplit.TRAIN,
    )
    store.write(corpus)
    return audit, corpus, store


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = canonical_json(value) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != serialized:
        raise ValueError("extraction preparation report conflicts with existing output")
    path.write_text(serialized, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit-batch")
    audit.add_argument("batch_dir", type=Path)
    audit.add_argument("--batch-id", required=True)
    audit.add_argument("--output", type=Path)
    build = subparsers.add_parser("build-corpus")
    build.add_argument("batch_dir", type=Path)
    build.add_argument("--batch-id", required=True)
    build.add_argument("--attempt-id", required=True)
    build.add_argument("--observation-cutoff", required=True)
    build.add_argument("--owner-controlled-root", type=Path, required=True)
    build.add_argument("--attempt-root", type=Path, required=True)
    build.add_argument("--audit-output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "audit-batch":
        audit = audit_extraction_feedback_batch(
            args.batch_dir,
            batch_id=args.batch_id,
        )
        if args.output is not None:
            _write_json(args.output, audit.payload())
        print(canonical_json(audit.payload()))
        return 0
    audit, corpus, store = build_extraction_optimizer_corpus(
        args.batch_dir,
        batch_id=args.batch_id,
        attempt_id=args.attempt_id,
        observation_cutoff=args.observation_cutoff,
        owner_controlled_root=args.owner_controlled_root,
        attempt_root=args.attempt_root,
    )
    if args.audit_output is not None:
        _write_json(args.audit_output, audit.payload())
    print(canonical_json({
        "auditId": audit.audit_id,
        "corpusId": corpus.corpus_id,
        "corpusDigest": corpus.corpus_digest,
        "corpusPath": str(store.path.relative_to(store.owner_controlled_root)),
        "optimizerSignalReady": audit.optimizer_signal_ready,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
