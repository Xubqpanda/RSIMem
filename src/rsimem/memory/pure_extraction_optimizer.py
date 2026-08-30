"""Pure-process content boundary for the extraction optimizer.

The regular optimizer corpus predates the pure-process evidence plane and
contains benchmark audit joins.  This module is the deployment-only request
boundary: stable identities come from :mod:`pure_extraction`, while bounded
text is supplied explicitly by an owner-controlled capture.  No family, stage,
grader, or answer metadata is accepted here.
"""

from __future__ import annotations

import fcntl
import json
import os
import stat
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

from .evidence_planes import EvidencePlane, require_optimizer_plane, validate_pure_process_payload
from .extraction_optimizer_contracts import (
    EXTRACTION_OPTIMIZER_SCHEMA_VERSION,
    EXTRACTION_OPTIMIZER_REQUEST_SCHEMA,
    EXTRACTION_OPTIMIZER_SYSTEM_INSTRUCTION,
    FROZEN_EXTRACTION_OPTIMIZER_CONFIG,
    ExtractionOptimizerConfig,
    ExtractionOptimizerRequest,
)
from .extraction_optimizer_corpus import (
    OptimizerDelayedEvidence,
    OptimizerExtractedFact,
    OptimizerSourceMessage,
)
from .extraction_source import ExtractionSourceProjection
from .extraction_policy_artifact import ExtractionPromptPolicyArtifact
from .prompt_components import canonical_json, content_digest, text_digest
from .pure_extraction import (
    PureExtractionAttribution,
    PureExtractionOptimizerCorpus,
    PureExtractionOptimizerExample,
)


_LABELS = {
    PureExtractionAttribution.ATTRIBUTABLE_SUCCESS: "useful",
    PureExtractionAttribution.ATTRIBUTABLE_FAILURE: "harmful",
    PureExtractionAttribution.UNRESOLVED: "unresolved",
    PureExtractionAttribution.CENSORED: "censored",
}
_ACTIONABLE = {
    PureExtractionAttribution.ATTRIBUTABLE_SUCCESS,
    PureExtractionAttribution.ATTRIBUTABLE_FAILURE,
}
PURE_EXTRACTION_OPTIMIZER_CAPTURE_SCHEMA_VERSION = 1
PURE_EXTRACTION_OPTIMIZER_CAPTURE_SCHEMA = (
    "rsimem-pure-extraction-optimizer-capture-v1"
)


def _id(value: object, name: str) -> None:
    import re

    if not isinstance(value, str) or re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.:+-]{0,255}", value
    ) is None:
        raise ValueError(f"{name} must be a stable identifier")


def _utc(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{name} must be an ISO UTC timestamp")
    try:
        datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO UTC timestamp") from exc


@dataclass(frozen=True, slots=True)
class PureExtractionOptimizerContentCapture:
    """Owner-controlled text for one pure optimizer example.

    The capture deliberately carries no benchmark identity.  The optimizer
    request only accepts it after all stable IDs and digests have been checked
    against the corresponding pure-process example.
    """

    example_id: str
    logical_case_id: str
    physical_observation_ids: tuple[str, ...]
    source_projection: ExtractionSourceProjection
    source_messages: tuple[OptimizerSourceMessage, ...]
    extracted_facts: tuple[OptimizerExtractedFact, ...]
    delayed_evidence: OptimizerDelayedEvidence

    def __post_init__(self) -> None:
        _id(self.example_id, "pure optimizer capture example ID")
        _id(self.logical_case_id, "pure optimizer capture logical case ID")
        if not self.physical_observation_ids:
            raise ValueError("pure optimizer capture requires physical observations")
        if len(self.physical_observation_ids) != len(set(self.physical_observation_ids)):
            raise ValueError("pure optimizer capture observations must be unique")
        for value in self.physical_observation_ids:
            _id(value, "pure optimizer capture physical observation ID")
        if not isinstance(self.source_projection, ExtractionSourceProjection):
            raise TypeError("pure optimizer capture source projection has the wrong type")
        if not self.source_messages:
            raise ValueError("pure optimizer capture requires source messages")
        if any(not isinstance(value, OptimizerSourceMessage) for value in self.source_messages):
            raise TypeError("pure optimizer capture source message has the wrong type")
        if any(not isinstance(value, OptimizerExtractedFact) for value in self.extracted_facts):
            raise TypeError("pure optimizer capture fact has the wrong type")
        if not isinstance(self.delayed_evidence, OptimizerDelayedEvidence):
            raise TypeError("pure optimizer capture delayed evidence has the wrong type")

    @property
    def capture_id(self) -> str:
        return "pure-optimizer-capture." + content_digest(self.identity_payload())[:40]

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": PURE_EXTRACTION_OPTIMIZER_CAPTURE_SCHEMA_VERSION,
            "example_id": self.example_id,
            "logical_case_id": self.logical_case_id,
            "physical_observation_ids": list(self.physical_observation_ids),
            "source_projection": self.source_projection.payload(),
            "source_messages": [value.payload() for value in self.source_messages],
            "extracted_facts": [value.payload() for value in self.extracted_facts],
            "delayed_evidence": self.delayed_evidence.payload(),
        }

    def payload(self) -> dict[str, object]:
        return {
            "schema": PURE_EXTRACTION_OPTIMIZER_CAPTURE_SCHEMA,
            "capture_id": self.capture_id,
            **self.identity_payload(),
        }

    @classmethod
    def from_payload(cls, value: object) -> "PureExtractionOptimizerContentCapture":
        fields = {
            "schema", "capture_id", "schema_version", "example_id",
            "logical_case_id", "physical_observation_ids", "source_projection",
            "source_messages", "extracted_facts", "delayed_evidence",
        }
        if (
            not isinstance(value, Mapping)
            or set(value) != fields
            or value.get("schema") != PURE_EXTRACTION_OPTIMIZER_CAPTURE_SCHEMA
        ):
            raise ValueError("malformed pure optimizer content capture")
        if not isinstance(value["physical_observation_ids"], list):
            raise ValueError("malformed pure optimizer capture observations")
        if not isinstance(value["source_messages"], list) or not isinstance(
            value["extracted_facts"], list
        ):
            raise ValueError("malformed pure optimizer capture collections")
        try:
            result = cls(
                example_id=value["example_id"],
                logical_case_id=value["logical_case_id"],
                physical_observation_ids=tuple(value["physical_observation_ids"]),
                source_projection=ExtractionSourceProjection.from_payload(
                    value["source_projection"]
                ),
                source_messages=tuple(
                    OptimizerSourceMessage.from_payload(item)
                    for item in value["source_messages"]
                ),
                extracted_facts=tuple(
                    OptimizerExtractedFact.from_payload(item)
                    for item in value["extracted_facts"]
                ),
                delayed_evidence=OptimizerDelayedEvidence.from_payload(
                    value["delayed_evidence"]
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("malformed pure optimizer content capture") from exc
        if value["schema_version"] != PURE_EXTRACTION_OPTIMIZER_CAPTURE_SCHEMA_VERSION:
            raise ValueError("unsupported pure optimizer content capture schema")
        if value["capture_id"] != result.capture_id:
            raise ValueError("pure optimizer content capture ID mismatch")
        if result.payload() != dict(value):
            raise ValueError("non-canonical pure optimizer content capture")
        return result


class JsonPureExtractionOptimizerContentCaptureStore:
    """Crash-safe owner-controlled persistence for pure optimizer captures."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.lock_path = self.path.with_name(self.path.name + ".lock")

    @contextmanager
    def _locked(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        os.chmod(self.lock_path, 0o600)
        with os.fdopen(descriptor, "r+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _read_unlocked(self) -> dict[str, PureExtractionOptimizerContentCapture]:
        if self.path.is_symlink():
            raise ValueError("pure optimizer capture file cannot be a symlink")
        if not self.path.exists():
            return {}
        if stat.S_IMODE(self.path.stat().st_mode) & 0o077:
            raise PermissionError("pure optimizer capture file permissions are too broad")
        records: dict[str, PureExtractionOptimizerContentCapture] = {}
        for line_number, line in enumerate(
            self.path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                raise ValueError(
                    f"empty pure optimizer capture record at line {line_number}"
                )
            try:
                capture = PureExtractionOptimizerContentCapture.from_payload(
                    json.loads(line)
                )
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"malformed pure optimizer capture at line {line_number}"
                ) from exc
            previous = records.get(capture.example_id)
            if previous is not None and previous.payload() != capture.payload():
                raise ValueError("conflicting pure optimizer content capture")
            records[capture.example_id] = capture
        return records

    def records(self) -> tuple[PureExtractionOptimizerContentCapture, ...]:
        with self._locked():
            records = self._read_unlocked()
        return tuple(records[key] for key in sorted(records))

    def append(self, capture: PureExtractionOptimizerContentCapture) -> bool:
        if not isinstance(capture, PureExtractionOptimizerContentCapture):
            raise TypeError("pure optimizer capture store received the wrong type")
        serialized = json.dumps(
            capture.payload(), ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ) + "\n"
        with self._locked():
            records = self._read_unlocked()
            previous = records.get(capture.example_id)
            if previous is not None:
                if previous.payload() != capture.payload():
                    raise ValueError("conflicting pure optimizer content capture")
                return False
            descriptor = os.open(
                self.path,
                os.O_CREAT | os.O_APPEND | os.O_WRONLY,
                0o600,
            )
            try:
                os.chmod(self.path, 0o600)
                encoded = serialized.encode("utf-8")
                offset = 0
                while offset < len(encoded):
                    offset += os.write(descriptor, encoded[offset:])
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            return True


def _capture_for(
    captures: Sequence[PureExtractionOptimizerContentCapture],
) -> dict[str, PureExtractionOptimizerContentCapture]:
    result: dict[str, PureExtractionOptimizerContentCapture] = {}
    for capture in captures:
        if not isinstance(capture, PureExtractionOptimizerContentCapture):
            raise TypeError("pure optimizer content captures have the wrong type")
        previous = result.get(capture.example_id)
        if previous is not None and previous != capture:
            raise ValueError("conflicting pure optimizer content capture")
        result[capture.example_id] = capture
    return result


def _validate_capture(
    example: PureExtractionOptimizerExample,
    capture: PureExtractionOptimizerContentCapture,
    *,
    parent: ExtractionPromptPolicyArtifact,
) -> None:
    if capture.example_id != example.example_id:
        raise ValueError("pure optimizer capture/example identity mismatch")
    if capture.source_projection.projection_id != example.source_projection_id:
        raise ValueError("pure optimizer capture source projection identity mismatch")
    if capture.source_projection.projection_digest != example.source_projection_digest:
        raise ValueError("pure optimizer capture source projection digest mismatch")
    if example.extraction_artifact_digest != parent.body_digest:
        raise ValueError("pure optimizer example was not produced by the parent policy")
    fact_ids = tuple(value.fact_id for value in capture.extracted_facts)
    if fact_ids != example.fact_ids:
        raise ValueError("pure optimizer capture fact identity mismatch")
    if content_digest([
        value.trace_payload() for value in capture.extracted_facts
    ]) != example.extraction_output_digest:
        raise ValueError("pure optimizer capture extraction output digest mismatch")
    if len(fact_ids) != len(set(fact_ids)):
        raise ValueError("pure optimizer capture fact IDs must be unique")
    source_projection = {
        value.segment_id for value in capture.source_messages
    }
    if len(source_projection) != len(capture.source_messages):
        raise ValueError("pure optimizer capture source segments must be unique")
    projected_ids = tuple(value.source_message_id for value in capture.source_messages)
    if projected_ids != capture.source_projection.source_message_ids:
        raise ValueError("pure optimizer capture source message identity mismatch")
    for projected, original in zip(
        capture.source_messages,
        capture.source_projection.messages,
    ):
        if projected.content.source_digest != text_digest(original.content):
            raise ValueError("pure optimizer capture source content digest mismatch")
    delayed = capture.delayed_evidence
    if example.opportunity_evidence_id is not None and (
        delayed.future_opportunity_id != example.opportunity_evidence_id
    ):
        raise ValueError("pure optimizer capture opportunity identity mismatch")
    for expected, actual, name in (
        (example.opportunity_operation_id, delayed.opportunity_operation_id, "opportunity"),
        (example.memory_use_operation_id, delayed.use_operation_id, "use"),
        (example.outcome_operation_id, delayed.outcome_operation_id, "outcome"),
    ):
        if expected is not None and actual != expected:
            raise ValueError(f"pure optimizer capture {name} operation identity mismatch")
    # Delayed evidence is intentionally content-bearing but must remain
    # attributable to a closed observation window.
    _utc(capture.delayed_evidence.source_completed_at, "pure optimizer source completion time")
    _utc(capture.delayed_evidence.observed_at, "pure optimizer observation time")
    validate_pure_process_payload({
        "source_messages": [value.payload() for value in capture.source_messages],
        "extracted_facts": [value.payload() for value in capture.extracted_facts],
        "delayed_evidence": capture.delayed_evidence.payload(),
    })


def _logical_groups(
    corpus: PureExtractionOptimizerCorpus,
    captures: Mapping[str, PureExtractionOptimizerContentCapture],
) -> dict[str, list[PureExtractionOptimizerExample]]:
    groups: dict[str, list[PureExtractionOptimizerExample]] = {}
    for example in corpus.examples:
        capture = captures.get(example.example_id)
        # Diagnostic unresolved/censored observations may intentionally have
        # no owner-controlled text.  Keep them as one physical unit; only
        # actionable examples need a content capture and replicate identity.
        logical_id = capture.logical_case_id if capture is not None else (
            "logical-case." + content_digest({"example_id": example.example_id})[:40]
        )
        groups.setdefault(logical_id, []).append(example)
    return groups


def _unit_payload(
    logical_case_id: str,
    values: list[PureExtractionOptimizerExample],
    captures: Mapping[str, PureExtractionOptimizerContentCapture],
    *,
    parent: ExtractionPromptPolicyArtifact,
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    primaries = list(values)
    if not primaries:
        raise ValueError("pure optimizer logical case is empty")
    labels = {value.attribution for value in primaries}
    if len(labels) != 1:
        raise ValueError("pure optimizer logical case has conflicting attribution")
    primary = min(primaries, key=lambda value: value.example_id)
    capture = captures.get(primary.example_id)
    if primary.attribution in _ACTIONABLE:
        if capture is None:
            raise ValueError("actionable pure optimizer example lacks content capture")
        _validate_capture(primary, capture, parent=parent)
    for value in primaries:
        other = captures.get(value.example_id)
        if other is not None:
            _validate_capture(value, other, parent=parent)
            if other.logical_case_id != logical_case_id:
                raise ValueError("pure optimizer capture logical case mismatch")
    label = _LABELS[primary.attribution]
    source_ref = fact_ref = delayed_ref = None
    source_catalog: dict[str, object] = {}
    fact_catalog: dict[str, object] = {}
    delayed_catalog: dict[str, object] = {}
    if capture is not None and primary.attribution in _ACTIONABLE:
        source_payload = [value.payload() for value in capture.source_messages]
        fact_payload = [value.payload() for value in capture.extracted_facts]
        delayed_payload = capture.delayed_evidence.payload()
        source_ref = "optimizer-source." + content_digest(source_payload)[:40]
        fact_ref = "optimizer-facts." + content_digest(fact_payload)[:40]
        delayed_ref = "optimizer-evidence." + content_digest(delayed_payload)[:40]
        source_catalog[source_ref] = source_payload
        fact_catalog[fact_ref] = fact_payload
        delayed_catalog[delayed_ref] = delayed_payload
    unit = {
        "logical_case_id": logical_case_id,
        "primary_example_id": primary.example_id,
        "label": label,
        "reason_codes": list(primary.reason_codes),
        "source_projection_ref": source_ref,
        "extracted_fact_set_ref": fact_ref,
        "delayed_evidence_ref": delayed_ref,
        "replica_count": len(primaries),
        "replica_primary_example_ids": [
            value.example_id for value in sorted(primaries, key=lambda item: item.example_id)
        ],
        "replica_physical_observation_ids": [
            observation_id
            for value in sorted(primaries, key=lambda item: item.example_id)
            for observation_id in (
                captures[value.example_id].physical_observation_ids
                if value.example_id in captures else ()
            )
        ],
    }
    return unit, source_catalog, fact_catalog, delayed_catalog


def build_pure_extraction_optimizer_request(
    parent: ExtractionPromptPolicyArtifact,
    corpus: PureExtractionOptimizerCorpus,
    *,
    captures: Sequence[PureExtractionOptimizerContentCapture] = (),
    config: ExtractionOptimizerConfig = FROZEN_EXTRACTION_OPTIMIZER_CONFIG,
) -> ExtractionOptimizerRequest:
    """Build an optimizer request from pure-process evidence only.

    ``captures`` is owner-controlled and never persisted in the public
    process corpus.  Every actionable example must have a capture; unresolved
    and censored examples remain content-free diagnostics.  Replicates are
    grouped by the capture's logical case identity and are represented as one
    optimizer unit with explicit physical observation IDs.
    """

    if not isinstance(parent, ExtractionPromptPolicyArtifact):
        raise TypeError("pure optimizer request requires a policy artifact")
    if not isinstance(corpus, PureExtractionOptimizerCorpus):
        raise TypeError("pure optimizer request requires a pure extraction corpus")
    if corpus.split != "train":
        raise ValueError("pure optimizer request requires the training corpus")
    if corpus.process_signal_gate != "ready":
        raise ValueError("pure optimizer request requires a ready process-signal gate")
    require_optimizer_plane(EvidencePlane.PURE_PROCESS)
    capture_by_id = _capture_for(captures)
    groups = _logical_groups(corpus, capture_by_id)
    units: list[dict[str, object]] = []
    source_catalog: dict[str, object] = {}
    fact_catalog: dict[str, object] = {}
    delayed_catalog: dict[str, object] = {}
    primary_ids: list[str] = []
    for logical_case_id in sorted(groups):
        unit, source, facts, delayed = _unit_payload(
            logical_case_id,
            groups[logical_case_id],
            capture_by_id,
            parent=parent,
        )
        units.append(unit)
        source_catalog.update(source)
        fact_catalog.update(facts)
        delayed_catalog.update(delayed)
        primary_ids.append(str(unit["primary_example_id"]))
    if len(units) > config.maximum_primary_examples:
        raise ValueError("pure optimizer corpus exceeds the primary sample budget")
    input_payload = {
        "schema_version": EXTRACTION_OPTIMIZER_SCHEMA_VERSION,
        "parent_policy": {
            "artifact_id": parent.artifact_id,
            "artifact_digest": parent.artifact_digest,
            "policy_version": parent.policy_version,
            "spec": parent.spec.payload(),
            "protected_rule_ids": [
                rule.rule_id for rule in parent.spec.rules if rule.protected
            ],
        },
        "objective": {
            "primary_unit": "extraction_set_future_opportunity",
            "resolved_target": "increase_future_resolved_useful_proportion",
            "constraints": [
                "do_not_increase_harmful_rate",
                "preserve_nonempty_coverage",
                "do_not_increase_empty_rate",
                "unresolved_and_censored_are_not_negative",
                "cost_is_not_an_optimization_signal",
            ],
            "maximum_candidates": 1,
            "maximum_rule_edits": config.maximum_rule_edits,
        },
        "process_signal": {
            "gate": corpus.process_signal_gate,
            "protocol_id": corpus.process_signal_protocol_id,
            "case_digest": corpus.process_signal_case_digest,
            "case_count": corpus.process_signal_case_count,
            "optimization_count": corpus.process_signal_optimization_count,
        },
        "evidence_groups": {
            label: [unit for unit in units if unit["label"] == label]
            for label in ("useful", "harmful", "missed", "unresolved", "censored")
        },
        "content_catalog": {
            "source_projections": dict(sorted(source_catalog.items())),
            "extracted_fact_sets": dict(sorted(fact_catalog.items())),
            "delayed_evidence": dict(sorted(delayed_catalog.items())),
        },
    }
    validate_pure_process_payload(input_payload)
    input_json = canonical_json(input_payload)
    if len(input_json) > config.maximum_input_chars:
        raise ValueError("pure optimizer request exceeds the input character budget")
    values = {
        "parent_artifact_id": parent.artifact_id,
        "parent_artifact_digest": parent.artifact_digest,
        "corpus_id": corpus.corpus_id,
        "corpus_digest": corpus.corpus_digest,
        "optimizer_config_digest": config.config_digest,
        "primary_example_ids": tuple(sorted(primary_ids)),
        "system_instruction": EXTRACTION_OPTIMIZER_SYSTEM_INSTRUCTION,
        "input_json": input_json,
        "provider_eligible": True,
        "request_schema": EXTRACTION_OPTIMIZER_REQUEST_SCHEMA,
        "schema_version": EXTRACTION_OPTIMIZER_SCHEMA_VERSION,
    }
    identity = {
        "schema_version": values["schema_version"],
        "request_schema": values["request_schema"],
        "parent_artifact_id": values["parent_artifact_id"],
        "parent_artifact_digest": values["parent_artifact_digest"],
        "corpus_id": values["corpus_id"],
        "corpus_digest": values["corpus_digest"],
        "optimizer_config_digest": values["optimizer_config_digest"],
        "primary_example_ids": list(values["primary_example_ids"]),
        "system_instruction_digest": text_digest(values["system_instruction"]),
        "input_json_digest": text_digest(values["input_json"]),
        "provider_eligible": values["provider_eligible"],
    }
    digest = content_digest(identity)
    return ExtractionOptimizerRequest(
        request_id=f"optimizer-request.{digest[:40]}",
        request_digest=digest,
        **values,
    )


def build_pure_extraction_optimizer_gate_request(
    parent: ExtractionPromptPolicyArtifact,
    corpus: PureExtractionOptimizerCorpus,
    *,
    reason_codes: Sequence[str],
    config: ExtractionOptimizerConfig = FROZEN_EXTRACTION_OPTIMIZER_CONFIG,
) -> ExtractionOptimizerRequest:
    """Build a content-free NO_PROPOSAL request for a pure corpus.

    The gate request is useful when a caller wants one replay-stable audit
    record for an unresolved, censored, stale, or insufficient corpus.  It
    never carries owner-controlled text and is never eligible for a provider
    call.
    """

    if not isinstance(parent, ExtractionPromptPolicyArtifact):
        raise TypeError("pure optimizer gate requires a policy artifact")
    if not isinstance(corpus, PureExtractionOptimizerCorpus):
        raise TypeError("pure optimizer gate requires a pure extraction corpus")
    if corpus.split != "train":
        raise ValueError("pure optimizer gate requires the training corpus")
    reasons = tuple(reason_codes)
    if not reasons or len(reasons) != len(set(reasons)):
        raise ValueError("pure optimizer gate requires unique reason codes")
    for reason in reasons:
        _id(reason, "pure optimizer gate reason code")
    if any(
        example.extraction_artifact_digest != parent.body_digest
        for example in corpus.examples
    ):
        raise ValueError("pure optimizer corpus was not produced by the parent policy")
    primary_ids = tuple(sorted(value.example_id for value in corpus.examples))
    input_payload = {
        "schema_version": EXTRACTION_OPTIMIZER_SCHEMA_VERSION,
        "request_mode": "deterministic_signal_gate",
        "decision": "NO_PROPOSAL",
        "reason_codes": list(reasons),
        "parent_artifact_id": parent.artifact_id,
        "parent_artifact_digest": parent.artifact_digest,
        "corpus_id": corpus.corpus_id,
        "corpus_digest": corpus.corpus_digest,
        "process_signal_gate": corpus.process_signal_gate,
        "primary_example_ids": list(primary_ids),
    }
    input_json = canonical_json(input_payload)
    if len(input_json) > config.maximum_input_chars:
        raise ValueError("pure optimizer gate request exceeds the input character budget")
    values = {
        "parent_artifact_id": parent.artifact_id,
        "parent_artifact_digest": parent.artifact_digest,
        "corpus_id": corpus.corpus_id,
        "corpus_digest": corpus.corpus_digest,
        "optimizer_config_digest": config.config_digest,
        "primary_example_ids": primary_ids,
        "system_instruction": EXTRACTION_OPTIMIZER_SYSTEM_INSTRUCTION,
        "input_json": input_json,
        "provider_eligible": False,
        "request_schema": EXTRACTION_OPTIMIZER_REQUEST_SCHEMA,
        "schema_version": EXTRACTION_OPTIMIZER_SCHEMA_VERSION,
    }
    identity = {
        "schema_version": values["schema_version"],
        "request_schema": values["request_schema"],
        "parent_artifact_id": values["parent_artifact_id"],
        "parent_artifact_digest": values["parent_artifact_digest"],
        "corpus_id": values["corpus_id"],
        "corpus_digest": values["corpus_digest"],
        "optimizer_config_digest": values["optimizer_config_digest"],
        "primary_example_ids": list(primary_ids),
        "system_instruction_digest": text_digest(values["system_instruction"]),
        "input_json_digest": text_digest(input_json),
        "provider_eligible": False,
    }
    digest = content_digest(identity)
    return ExtractionOptimizerRequest(
        request_id=f"optimizer-request.{digest[:40]}",
        request_digest=digest,
        **values,
    )


__all__ = [
    "PURE_EXTRACTION_OPTIMIZER_CAPTURE_SCHEMA",
    "PURE_EXTRACTION_OPTIMIZER_CAPTURE_SCHEMA_VERSION",
    "JsonPureExtractionOptimizerContentCaptureStore",
    "PureExtractionOptimizerContentCapture",
    "build_pure_extraction_optimizer_gate_request",
    "build_pure_extraction_optimizer_request",
]
