"""Build a privacy-preserving lifecycle ledger from PAST-Bench evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import threading
from typing import Any, Iterable, Mapping

from .lifecycle.snapshot import ContextSnapshot
from .lifecycle.writeback import WritebackEvent
from .memory.contracts import MemoryEvent

SCHEMA_VERSION = 1

_MEMORY_RUNTIME_EVENT_FIELDS = {
    "schemaVersion",
    "eventId",
    "runId",
    "variant",
    "traceId",
    "episodeId",
    "sessionId",
    "taskId",
    "familyId",
    "stage",
    "snapshotId",
    "kind",
    "source",
    "data",
}
_MEMORY_RUNTIME_DATA_FIELDS = {
    "executionMode",
    "memoryKind",
    "backend",
    "artifactIds",
    "queryChars",
    "contentChars",
    "reasonCode",
    "attributes",
}
_MEMORY_RUNTIME_ATTRIBUTE_FIELDS = {
    "action",
    "count",
    "equivalent",
    "failure_type",
    "limit",
    "namespace",
    "surface",
    "execution_id",
    "operation_id",
    "snapshot_id",
    "mutation_id",
    "receipt_id",
    "writer_identity",
}
_MEMORY_RUNTIME_ID_ATTRIBUTES = {
    "execution_id",
    "operation_id",
    "snapshot_id",
    "mutation_id",
    "receipt_id",
    "writer_identity",
}
_RSIMEM_EXECUTION_MODES = {"native+ledger", "native+adapter+ledger"}
_LIFECYCLE_EVENT_KINDS = {
    "context_snapshot",
    "evaluation_accepted",
    "evaluation_rejected",
    "boundary_rejected",
    "plan_created",
    "plan_rejected",
    "plan_validated",
    "dry_run_mutation",
    "dry_run_duplicate",
    "memory_ingestion",
    "static_utility_decisions",
}
_LIFECYCLE_SNAPSHOT_DATA_FIELDS = {
    "segmentCount",
    "activeSegmentCount",
    "protectedSegmentCount",
    "toolClosureCount",
    "openToolClosureCount",
    "totalTokens",
    "taskState",
    "lifecycleState",
}
_LIFECYCLE_EVALUATION_DATA_FIELDS = {
    "evaluationId",
    "trigger",
    "evaluator",
    "policyVersion",
    "status",
    "reasonCodes",
}
_LIFECYCLE_BOUNDARY_DATA_FIELDS = {
    "evaluationId",
    "boundaryId",
    "trigger",
    "status",
    "reasonCodes",
}
_LIFECYCLE_PLAN_DATA_FIELDS = {
    "evaluationId",
    "planId",
    "mutationId",
    "contextAction",
    "memoryAction",
    "memoryKind",
    "targetBackend",
    "targetArtifactId",
    "compilerVersion",
    "sourceSegmentCount",
    "status",
    "reasonCodes",
    "resources",
}
_LIFECYCLE_RESOURCE_FIELDS = {
    "schemaVersion",
    "inputTokens",
    "outputTokens",
    "cacheReadTokens",
    "cacheWriteTokens",
    "reasoningTokens",
    "modelRequests",
    "retryCount",
    "durationMs",
    "storageBytes",
}
_LIFECYCLE_INGESTION_DATA_FIELDS = {
    "executionId",
    "status",
    "outcome",
    "routeBackend",
    "memoryKind",
    "policyProvider",
    "policyVersion",
    "frameworkVersion",
    "promptVersion",
    "featureSchemaVersion",
    "operationIds",
    "operationActions",
    "sourceDigest",
    "contentDigests",
    "reasonCodes",
    "resources",
}
_LIFECYCLE_UTILITY_DATA_FIELDS = {
    "executionId",
    "operationIds",
    "requestId",
    "gateVersion",
    "gateDigest",
    "featureSchemaVersion",
    "decisionCount",
    "decisions",
}
_LIFECYCLE_UTILITY_DECISION_FIELDS = {
    "schema_version",
    "target",
    "disposition",
    "score",
    "predicted_benefit",
    "lifecycle_cost",
    "risk",
    "contributions",
    "reason_codes",
    "feature_digest",
    "cost_digest",
    "feature_schema",
    "cost_schema",
    "policy_version",
    "cutoff",
}
_MACHINE_REASON_CODE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$")


def _json_hash(value: Any, *, length: int = 24) -> str:
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:length]


def resolve_comparison_evidence_path(raw_path: Any, root: Path) -> Path:
    """Resolve absolute or run-anchored PAST evidence paths without cwd state."""

    root = root.resolve()
    path = Path(str(raw_path)).expanduser()
    if path.is_absolute():
        return path.resolve()
    anchor_indexes = [index for index, part in enumerate(path.parts) if part == root.name]
    if anchor_indexes:
        suffix = path.parts[anchor_indexes[-1] + 1 :]
        if suffix:
            return root.joinpath(*suffix).resolve()
    candidate = (root / path).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError("relative comparison evidence path escapes the run directory")
    return candidate


def _validate_memory_runtime_event(value: dict[str, Any], source_path: Path) -> None:
    if set(value) != _MEMORY_RUNTIME_EVENT_FIELDS:
        raise ValueError(f"invalid RSIMem runtime event fields in {source_path}")
    if value.get("source") != {"type": "rsimem_memory_runtime"}:
        raise ValueError(f"invalid RSIMem runtime event source in {source_path}")
    data = value.get("data")
    if not isinstance(data, dict) or set(data) != _MEMORY_RUNTIME_DATA_FIELDS:
        raise ValueError(f"invalid RSIMem runtime event data fields in {source_path}")
    attributes = data.get("attributes")
    if not isinstance(attributes, dict) or not set(attributes).issubset(
        _MEMORY_RUNTIME_ATTRIBUTE_FIELDS
    ):
        raise ValueError(f"invalid RSIMem runtime event attributes in {source_path}")
    if "equivalent" in attributes and not isinstance(attributes["equivalent"], bool):
        raise ValueError(f"invalid RSIMem projection result in {source_path}")
    if any(
        key in attributes
        and attributes[key] is not None
        and (
            not isinstance(attributes[key], str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}", attributes[key])
        )
        for key in _MEMORY_RUNTIME_ID_ATTRIBUTES
    ):
        raise ValueError(f"invalid RSIMem runtime operation identity in {source_path}")
    if value.get("kind") == "mutation_committed":
        action = attributes.get("action")
        if not isinstance(attributes.get("mutation_id"), str) or not isinstance(
            attributes.get("receipt_id"), str
        ):
            raise ValueError(f"incomplete committed mutation identity in {source_path}")
        writer = attributes.get("writer_identity")
        if (action == "none") != (writer is None):
            raise ValueError(f"invalid committed mutation writer in {source_path}")
    if value.get("kind") == "projection_check" and (
        not isinstance(attributes.get("surface"), str)
        or not attributes["surface"]
        or not isinstance(attributes.get("equivalent"), bool)
    ):
        raise ValueError(f"incomplete RSIMem projection check in {source_path}")
    if data.get("executionMode") not in _RSIMEM_EXECUTION_MODES:
        raise ValueError(f"invalid RSIMem execution mode in {source_path}")


def _validate_lifecycle_contract_event(value: dict[str, Any], source_path: Path) -> None:
    if set(value) != _MEMORY_RUNTIME_EVENT_FIELDS:
        raise ValueError(f"invalid RSIMem lifecycle event fields in {source_path}")
    if value.get("source") != {"type": "rsimem_lifecycle_contract"}:
        raise ValueError(f"invalid RSIMem lifecycle event source in {source_path}")
    kind = value.get("kind")
    if kind not in _LIFECYCLE_EVENT_KINDS:
        raise ValueError(f"invalid RSIMem lifecycle event kind in {source_path}")
    snapshot_id = value.get("snapshotId")
    if kind == "boundary_rejected":
        if snapshot_id is not None:
            raise ValueError(f"pre-snapshot lifecycle rejection has snapshotId in {source_path}")
    elif not isinstance(snapshot_id, str) or not snapshot_id:
        raise ValueError(f"RSIMem lifecycle event requires snapshotId in {source_path}")

    data = value.get("data")
    expected_fields = (
        _LIFECYCLE_SNAPSHOT_DATA_FIELDS
        if kind == "context_snapshot"
        else _LIFECYCLE_EVALUATION_DATA_FIELDS
        if kind in {"evaluation_accepted", "evaluation_rejected"}
        else _LIFECYCLE_BOUNDARY_DATA_FIELDS
        if kind == "boundary_rejected"
        else _LIFECYCLE_INGESTION_DATA_FIELDS
        if kind == "memory_ingestion"
        else _LIFECYCLE_UTILITY_DATA_FIELDS
        if kind == "static_utility_decisions"
        else _LIFECYCLE_PLAN_DATA_FIELDS
    )
    if not isinstance(data, dict) or set(data) != expected_fields:
        raise ValueError(f"invalid RSIMem lifecycle event data fields in {source_path}")
    reason_codes = data.get("reasonCodes", [])
    if not isinstance(reason_codes, list) or any(
        not isinstance(code, str) or not _MACHINE_REASON_CODE.fullmatch(code)
        for code in reason_codes
    ):
        raise ValueError(f"invalid RSIMem lifecycle reason codes in {source_path}")
    if kind == "static_utility_decisions":
        _validate_static_utility_data(data, source_path)
    elif kind not in {
        "context_snapshot",
        "evaluation_accepted",
        "evaluation_rejected",
        "boundary_rejected",
    }:
        resources = data.get("resources")
        if not isinstance(resources, dict) or set(resources) != _LIFECYCLE_RESOURCE_FIELDS:
            raise ValueError(f"invalid RSIMem lifecycle resources in {source_path}")


def _validate_static_utility_data(data: dict[str, Any], source_path: Path) -> None:
    string_fields = (
        "executionId",
        "requestId",
        "gateVersion",
        "featureSchemaVersion",
    )
    if any(
        not isinstance(data.get(field), str)
        or not _IDENTIFIER.fullmatch(data[field])
        for field in string_fields
    ) or not isinstance(data.get("gateDigest"), str) or not _SHA256.fullmatch(
        data["gateDigest"]
    ):
        raise ValueError(f"invalid static utility identity in {source_path}")
    operation_ids = data.get("operationIds")
    decisions = data.get("decisions")
    if (
        not isinstance(operation_ids, list)
        or len(operation_ids) != len(set(operation_ids))
        or any(
            not isinstance(item, str) or not _IDENTIFIER.fullmatch(item)
            for item in operation_ids
        )
        or not isinstance(decisions, list)
        or type(data.get("decisionCount")) is not int
        or data["decisionCount"] != len(decisions)
    ):
        raise ValueError(f"invalid static utility decision collection in {source_path}")
    for decision in decisions:
        if not isinstance(decision, dict) or set(decision) != (
            _LIFECYCLE_UTILITY_DECISION_FIELDS
        ):
            raise ValueError(f"invalid static utility decision fields in {source_path}")
        if (
            decision.get("schema_version") != 1
            or decision.get("target")
            not in {"generation", "internal_operation", "retrieval"}
            or decision.get("disposition") not in {"accept", "defer", "reject"}
            or type(decision.get("cutoff")) is not int
            or decision["cutoff"] < 0
        ):
            raise ValueError(f"invalid static utility decision contract in {source_path}")
        for field, lower, upper in (
            ("score", -1.0, 1.0),
            ("predicted_benefit", 0.0, 1.0),
            ("lifecycle_cost", 0.0, 1.0),
            ("risk", 0.0, 1.0),
        ):
            number = decision.get(field)
            if (
                not isinstance(number, (int, float))
                or isinstance(number, bool)
                or not math.isfinite(float(number))
                or not lower <= float(number) <= upper
            ):
                raise ValueError(f"invalid static utility numeric evidence in {source_path}")
        if any(
            not isinstance(decision.get(field), str)
            or not _SHA256.fullmatch(decision[field])
            for field in ("feature_digest", "cost_digest")
        ) or any(
            not isinstance(decision.get(field), str)
            or not _IDENTIFIER.fullmatch(decision[field])
            for field in (
                "feature_schema",
                "cost_schema",
                "policy_version",
            )
        ):
            raise ValueError(f"invalid static utility schema evidence in {source_path}")
        contributions = decision.get("contributions")
        if not isinstance(contributions, dict) or any(
            not isinstance(name, str)
            or not _IDENTIFIER.fullmatch(name)
            or not isinstance(amount, (int, float))
            or isinstance(amount, bool)
            or not math.isfinite(float(amount))
            for name, amount in contributions.items()
        ):
            raise ValueError(f"invalid static utility contributions in {source_path}")
        reasons = decision.get("reason_codes")
        if not isinstance(reasons, list) or not reasons or any(
            not isinstance(reason, str) or not _MACHINE_REASON_CODE.fullmatch(reason)
            for reason in reasons
        ):
            raise ValueError(f"invalid static utility reason codes in {source_path}")


def load_episode_lifecycle_events(comparison_path: Path) -> tuple[dict[str, Any], ...]:
    """Load content-free RSIMem evidence adjacent to comparison-owned traces."""

    comparison_path = comparison_path.resolve()
    root = comparison_path.parent
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    evidence_identities: dict[Path, tuple[str, set[tuple[Any, ...]]]] = {}
    for variant in ("with_persistence", "without_persistence"):
        payload = comparison.get(variant, {})
        episodes = payload.get("episodes", []) if isinstance(payload, dict) else []
        for episode in episodes:
            if not isinstance(episode, dict):
                continue
            trace_value = str(episode.get("trace") or "").strip()
            if not trace_value:
                continue
            trace_path = resolve_comparison_evidence_path(trace_value, root)
            identity = (
                root.name,
                variant,
                str(episode.get("trace_id") or ""),
                episode.get("task_id"),
                episode.get("family_id"),
                episode.get("stage"),
            )
            artifacts = trace_path.resolve().parent / "artifacts"
            for evidence_name, evidence_type in (
                ("rsimem_memory_events.jsonl", "memory"),
                ("rsimem_lifecycle_events.jsonl", "lifecycle"),
            ):
                evidence_path = artifacts / evidence_name
                registered_type, identities = evidence_identities.setdefault(
                    evidence_path,
                    (evidence_type, set()),
                )
                if registered_type != evidence_type:
                    raise ValueError("conflicting RSIMem evidence path type")
                identities.add(identity)

    events: list[dict[str, Any]] = []
    events_by_id: dict[str, str] = {}
    for evidence_path in sorted(evidence_identities, key=str):
        if not evidence_path.exists():
            continue
        evidence_type, allowed_identities = evidence_identities[evidence_path]
        for line_number, line in enumerate(
            evidence_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"malformed RSIMem evidence at {evidence_path}:{line_number}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(
                    f"RSIMem evidence must be an object at {evidence_path}:{line_number}"
                )
            if evidence_type == "memory":
                _validate_memory_runtime_event(value, evidence_path)
            else:
                _validate_lifecycle_contract_event(value, evidence_path)
            identity = (
                value.get("runId"),
                value.get("variant"),
                value.get("traceId"),
                value.get("taskId"),
                value.get("familyId"),
                value.get("stage"),
            )
            if identity not in allowed_identities:
                raise ValueError(
                    f"RSIMem evidence identity does not match {evidence_path}"
                )
            event_id = value.get("eventId")
            if not isinstance(event_id, str) or not event_id:
                raise ValueError(f"RSIMem evidence requires eventId in {evidence_path}")
            canonical = json.dumps(
                value,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            existing = events_by_id.get(event_id)
            if existing is not None:
                if existing != canonical:
                    raise ValueError(f"conflicting ledger eventId: {event_id}")
                continue
            events_by_id[event_id] = canonical
            events.append(value)
    return tuple(events)


class _RecordIdRegistry:
    """Link memory evidence without exposing a content-derived fingerprint."""

    def __init__(self, run_id: str) -> None:
        self._run_id = run_id
        self._ids: dict[str, str] = {}

    def resolve(self, content: str) -> str:
        normalized = " ".join(content.split())
        record_id = self._ids.get(normalized)
        if record_id is None:
            record_id = f"mem_{_json_hash({'runId': self._run_id, 'ordinal': len(self._ids)})}"
            self._ids[normalized] = record_id
        return record_id


class LifecycleLedgerObserver:
    """Join content-free snapshot and writeback evidence to the ledger schema."""

    def __init__(
        self,
        *,
        variant: str,
        trace_id: str,
        family_id: str | None = None,
        stage: str | None = None,
        output_path: Path | None = None,
    ) -> None:
        if not variant.strip() or not trace_id.strip():
            raise ValueError("lifecycle ledger variant and trace_id must not be empty")
        self.variant = variant
        self.trace_id = trace_id
        self.family_id = family_id
        self.stage = stage
        # Preserve the final path component so a runtime evidence path cannot
        # redirect writes through a symlink.  The observer intentionally starts
        # a fresh per-attempt file, but a symlink must still fail closed before
        # any truncation occurs.
        self.output_path = (
            Path(os.path.abspath(os.path.expanduser(os.fspath(output_path))))
            if output_path
            else None
        )
        self._events: list[dict[str, Any]] = []
        self._events_by_id: dict[str, str] = {}
        self._lock = threading.RLock()
        if self.output_path is not None:
            if self.output_path.is_symlink():
                raise ValueError("lifecycle ledger path cannot be a symlink")
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self._load_existing()

    @property
    def events(self) -> tuple[dict[str, Any], ...]:
        with self._lock:
            return tuple(self._events)

    def _load_existing(self) -> None:
        assert self.output_path is not None
        if not self.output_path.exists():
            return
        for line_number, line in enumerate(
            self.output_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"malformed lifecycle ledger event at line {line_number}"
                ) from exc
            if not isinstance(event, dict):
                raise ValueError(
                    f"malformed lifecycle ledger event at line {line_number}"
                )
            event_id = event.get("eventId")
            if not isinstance(event_id, str) or not event_id.strip():
                raise ValueError(
                    f"lifecycle ledger event has no eventId at line {line_number}"
                )
            canonical = json.dumps(
                event,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            existing = self._events_by_id.get(event_id)
            if existing is not None:
                if existing != canonical:
                    raise ValueError(f"conflicting lifecycle ledger event: {event_id}")
                continue
            self._events_by_id[event_id] = canonical
            self._events.append(event)

    def _append(
        self,
        *,
        kind: str,
        run_id: str,
        episode_id: str,
        session_id: str,
        task_id: str,
        snapshot_id: str | None,
        data: dict[str, Any],
    ) -> None:
        with self._lock:
            identity = {
                "runId": run_id,
                "variant": self.variant,
                "traceId": self.trace_id,
                "snapshotId": snapshot_id,
                "kind": kind,
                "evaluationId": data.get("evaluationId"),
                "planId": data.get("planId"),
                "mutationId": data.get("mutationId"),
            }
            if data.get("executionId") is not None:
                identity["executionId"] = data["executionId"]
            event_id = f"evt_{_json_hash(identity)}"
            event = {
                "schemaVersion": SCHEMA_VERSION,
                "eventId": event_id,
                "runId": run_id,
                "variant": self.variant,
                "traceId": self.trace_id,
                "episodeId": episode_id,
                "sessionId": session_id,
                "taskId": task_id,
                "familyId": self.family_id,
                "stage": self.stage,
                "snapshotId": snapshot_id,
                "kind": kind,
                "source": {"type": "rsimem_lifecycle_contract"},
                "data": data,
            }
            canonical = json.dumps(
                event,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            existing = self._events_by_id.get(event_id)
            if existing is not None:
                if existing != canonical:
                    raise ValueError(f"conflicting lifecycle ledger event: {event_id}")
                return
            if self.output_path is not None:
                with self.output_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(event, ensure_ascii=True, sort_keys=True) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
            self._events_by_id[event_id] = canonical
            self._events.append(event)

    def record_snapshot(self, snapshot: ContextSnapshot) -> None:
        self._append(
            kind="context_snapshot",
            run_id=snapshot.run_id,
            episode_id=snapshot.episode_id,
            session_id=snapshot.session_id,
            task_id=snapshot.task_id,
            snapshot_id=snapshot.snapshot_id,
            data={
                "segmentCount": len(snapshot.segments),
                "activeSegmentCount": len(snapshot.active_segment_ids),
                "protectedSegmentCount": len(snapshot.protected_segment_ids),
                "toolClosureCount": len(snapshot.tool_closures),
                "openToolClosureCount": sum(
                    1 for closure in snapshot.tool_closures if not closure.closed
                ),
                "totalTokens": snapshot.total_token_count,
                "taskState": snapshot.task_state.value,
                "lifecycleState": snapshot.lifecycle_state,
            },
        )

    def record_evaluation(
        self,
        snapshot: ContextSnapshot,
        *,
        evaluation_id: str,
        trigger: str,
        evaluator: str,
        policy_version: str,
        status: str,
        reason_codes: tuple[str, ...] = (),
    ) -> None:
        """Record evaluator disposition without prompt or response content."""

        if status not in {"accepted", "rejected"}:
            raise ValueError("lifecycle evaluation status must be accepted or rejected")
        self._append(
            kind=f"evaluation_{status}",
            run_id=snapshot.run_id,
            episode_id=snapshot.episode_id,
            session_id=snapshot.session_id,
            task_id=snapshot.task_id,
            snapshot_id=snapshot.snapshot_id,
            data={
                "evaluationId": evaluation_id,
                "trigger": trigger,
                "evaluator": evaluator,
                "policyVersion": policy_version,
                "status": status,
                "reasonCodes": list(reason_codes),
            },
        )

    def record_boundary_rejection(
        self,
        *,
        run_id: str,
        episode_id: str,
        session_id: str,
        task_id: str,
        boundary_id: str,
        trigger: str,
        reason_code: str,
    ) -> None:
        """Record a host failure that occurs before a snapshot can exist."""

        self._append(
            kind="boundary_rejected",
            run_id=run_id,
            episode_id=episode_id,
            session_id=session_id,
            task_id=task_id,
            snapshot_id=None,
            data={
                "evaluationId": boundary_id,
                "boundaryId": boundary_id,
                "trigger": trigger,
                "status": "rejected",
                "reasonCodes": [reason_code],
            },
        )

    def record(self, event: WritebackEvent) -> None:
        """Implement WritebackObserver without retaining plan or source content."""

        resources = event.resources
        self._append(
            kind=event.kind.value,
            run_id=event.run_id,
            episode_id=event.episode_id,
            session_id=event.session_id,
            task_id=event.task_id,
            snapshot_id=event.snapshot_id,
            data={
                "evaluationId": event.evaluation_id,
                "planId": event.plan_id,
                "mutationId": event.mutation_id,
                "contextAction": (
                    event.context_action.value if event.context_action is not None else None
                ),
                "memoryAction": (
                    event.memory_action.value if event.memory_action is not None else None
                ),
                "memoryKind": event.memory_kind.value if event.memory_kind is not None else None,
                "targetBackend": event.target_backend,
                "targetArtifactId": event.target_artifact_id,
                "compilerVersion": event.compiler_version,
                "sourceSegmentCount": event.source_segment_count,
                "status": event.status,
                "reasonCodes": list(event.reason_codes),
                "resources": {
                    "schemaVersion": resources.schema_version,
                    "inputTokens": resources.input_tokens,
                    "outputTokens": resources.output_tokens,
                    "cacheReadTokens": resources.cache_read_tokens,
                    "cacheWriteTokens": resources.cache_write_tokens,
                    "reasoningTokens": resources.reasoning_tokens,
                    "modelRequests": resources.model_requests,
                    "retryCount": resources.retry_count,
                    "durationMs": resources.duration_ms,
                    "storageBytes": resources.storage_bytes,
                },
            },
        )

    def record_ingestion(self, request: Any, result: Any) -> None:
        """Record content-free ingestion result and complete raw usage buckets."""

        if result.idempotency_key != request.idempotency_key:
            raise ValueError("ingestion result does not match request identity")
        provenance = request.provenance.source
        resources = result.usage
        self._append(
            kind="memory_ingestion",
            run_id=provenance.run_id,
            episode_id=provenance.episode_id,
            session_id=provenance.session_id,
            task_id=provenance.task_id,
            snapshot_id=provenance.snapshot_id,
            data={
                "executionId": result.execution_id,
                "status": result.status.value,
                "outcome": result.outcome.value,
                "routeBackend": result.fixed_route.backend,
                "memoryKind": result.fixed_route.kind.value,
                "policyProvider": result.policy_provider,
                "policyVersion": result.policy_version,
                "frameworkVersion": result.framework_version,
                "promptVersion": result.prompt_version,
                "featureSchemaVersion": result.feature_schema_version,
                "operationIds": [item.operation_id for item in result.operations],
                "operationActions": [item.action.value for item in result.operations],
                "sourceDigest": result.source_digest,
                "contentDigests": list(result.content_digests),
                "reasonCodes": list(result.reason_codes),
                "resources": {
                    "schemaVersion": resources.schema_version,
                    "inputTokens": resources.input_tokens,
                    "outputTokens": resources.output_tokens,
                    "cacheReadTokens": resources.cache_read_tokens,
                    "cacheWriteTokens": resources.cache_write_tokens,
                    "reasoningTokens": resources.reasoning_tokens,
                    "modelRequests": resources.model_requests,
                    "retryCount": resources.retry_count,
                    "durationMs": resources.duration_ms,
                    "storageBytes": resources.storage_bytes,
                },
            },
        )

    def record_utility_decisions(
        self,
        request: Any,
        result: Any,
        evidence: Mapping[str, Any],
    ) -> None:
        """Persist frozen utility decisions without policy input content."""

        if result.idempotency_key != request.idempotency_key:
            raise ValueError("utility evidence does not match ingestion request")
        if evidence.get("request_id") != request.idempotency_key:
            raise ValueError("utility evidence request identity is invalid")
        decisions = evidence.get("decisions")
        if not isinstance(decisions, list) or not all(
            isinstance(item, dict) for item in decisions
        ):
            raise ValueError("utility evidence decisions must be objects")
        provenance = request.provenance.source
        self._append(
            kind="static_utility_decisions",
            run_id=provenance.run_id,
            episode_id=provenance.episode_id,
            session_id=provenance.session_id,
            task_id=provenance.task_id,
            snapshot_id=provenance.snapshot_id,
            data={
                "executionId": result.execution_id,
                "operationIds": [item.operation_id for item in result.operations],
                "requestId": request.idempotency_key,
                "gateVersion": evidence.get("gate_version"),
                "gateDigest": evidence.get("gate_digest"),
                "featureSchemaVersion": evidence.get("feature_schema"),
                "decisionCount": len(decisions),
                "decisions": decisions,
            },
        )

    def write(self, output_path: Path) -> None:
        with self._lock:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                "".join(
                    json.dumps(event, ensure_ascii=True, sort_keys=True) + "\n"
                    for event in self._events
                ),
                encoding="utf-8",
            )


class MemoryLedgerObserver:
    """Convert content-free memory runtime events to the experiment ledger schema."""

    def __init__(
        self,
        *,
        run_id: str,
        variant: str,
        trace_id: str,
        episode_id: str,
        session_id: str,
        task_id: str,
        family_id: str | None = None,
        stage: str | None = None,
        snapshot_id: str | None = None,
        execution_mode: str | None = None,
        output_path: Path | None = None,
    ) -> None:
        required = (run_id, variant, trace_id, episode_id, session_id, task_id)
        if any(not value.strip() for value in required):
            raise ValueError("memory ledger identity fields must not be empty")
        self.run_id = run_id
        self.variant = variant
        self.trace_id = trace_id
        self.episode_id = episode_id
        self.session_id = session_id
        self.task_id = task_id
        self.family_id = family_id
        self.stage = stage
        self.snapshot_id = snapshot_id
        self.execution_mode = execution_mode
        # Preserve the final path component so a runtime evidence path cannot
        # redirect writes through a symlink.  The observer intentionally starts
        # a fresh per-attempt file, but a symlink must still fail closed before
        # any truncation occurs.
        self.output_path = (
            Path(os.path.abspath(os.path.expanduser(os.fspath(output_path))))
            if output_path
            else None
        )
        self._events: list[dict[str, Any]] = []
        self._occurrences: dict[str, int] = {}
        self._lock = threading.Lock()
        if self.output_path is not None:
            if self.output_path.is_symlink():
                raise ValueError("memory runtime evidence path cannot be a symlink")
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self.output_path.write_text("", encoding="utf-8")

    @property
    def events(self) -> tuple[dict[str, Any], ...]:
        with self._lock:
            return tuple(self._events)

    def record(self, event: MemoryEvent) -> None:
        with self._lock:
            attributes = dict(event.attributes)
            logical_identity = {
                "runId": self.run_id,
                "variant": self.variant,
                "traceId": self.trace_id,
                "snapshotId": self.snapshot_id,
                "executionMode": self.execution_mode,
                "kind": event.kind.value,
                "memoryKind": event.memory_kind.value,
                "backend": event.backend,
                "artifactIds": event.artifact_ids,
                "queryChars": event.query_chars,
                "contentChars": event.content_chars,
                "reasonCode": event.reason_code,
                "attributes": attributes,
            }
            logical_key = _json_hash(logical_identity)
            occurrence = self._occurrences.get(logical_key, 0)
            event_id = f"evt_{_json_hash({**logical_identity, 'occurrence': occurrence})}"
            value = {
                "schemaVersion": SCHEMA_VERSION,
                "eventId": event_id,
                "runId": self.run_id,
                "variant": self.variant,
                "traceId": self.trace_id,
                "episodeId": self.episode_id,
                "sessionId": self.session_id,
                "taskId": self.task_id,
                "familyId": self.family_id,
                "stage": self.stage,
                "snapshotId": self.snapshot_id,
                "kind": event.kind.value,
                "source": {"type": "rsimem_memory_runtime"},
                "data": {
                    "executionMode": self.execution_mode,
                    "memoryKind": event.memory_kind.value,
                    "backend": event.backend,
                    "artifactIds": list(event.artifact_ids),
                    "queryChars": event.query_chars,
                    "contentChars": event.content_chars,
                    "reasonCode": event.reason_code,
                    "attributes": attributes,
                },
            }
            if self.output_path is not None:
                with self.output_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(value, ensure_ascii=True, sort_keys=True) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
            self._occurrences[logical_key] = occurrence + 1
            self._events.append(value)


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return path.name


def _directory_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else None


def _session_evidence(
    episode: dict[str, Any],
    root: Path,
) -> tuple[dict[str, Any] | None, Path | None]:
    raw_path = episode.get("internal_tools", {}).get("session_file")
    if not raw_path:
        return None, None
    path = resolve_comparison_evidence_path(raw_path, root)
    return _read_json(path), path


def _model_call_events(
    *,
    run_id: str,
    variant: str,
    episode: dict[str, Any],
    root: Path,
) -> Iterable[dict[str, Any]]:
    trace_path = resolve_comparison_evidence_path(episode.get("trace", ""), root)
    if not trace_path.is_file():
        return
    ordinal = 0
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(raw, dict) or raw.get("type") != "model_call_usage":
            continue
        usage = raw.get("usage") if isinstance(raw.get("usage"), dict) else {}
        usage_available = raw.get("usage_available", False) is True
        yield _event(
            run_id=run_id,
            variant=variant,
            episode=episode,
            kind="model_call_usage",
            ordinal=ordinal,
            source={"type": "past_bench_model_call_usage", "path": _relative(trace_path, root)},
            data={
                "callId": raw.get("call_id"),
                "sequence": raw.get("sequence"),
                "component": raw.get("component"),
                "purpose": raw.get("purpose"),
                "provider": raw.get("provider"),
                "model": raw.get("model"),
                "apiMode": raw.get("api_mode"),
                "attempt": raw.get("attempt"),
                "status": raw.get("status"),
                "inputTokens": usage.get("input_tokens") if usage_available else None,
                "outputTokens": usage.get("output_tokens") if usage_available else None,
                "cacheReadTokens": usage.get("cache_read_tokens"),
                "cacheWriteTokens": usage.get("cache_write_tokens"),
                "reasoningTokens": usage.get("reasoning_tokens"),
                "usageAvailable": usage_available,
                "durationMs": raw.get("duration_ms"),
                "httpStatus": raw.get("http_status"),
                "errorCategory": raw.get("error_category"),
                "billingExecutionId": f"{episode.get('trace_id')}:{raw.get('call_id')}",
            },
        )
        ordinal += 1


def _event(
    *,
    run_id: str,
    variant: str,
    episode: dict[str, Any],
    kind: str,
    ordinal: int,
    source: dict[str, Any],
    data: dict[str, Any],
) -> dict[str, Any]:
    """Create a content-free ledger event with a stable logical identity.

    ``ordinal`` is only a last-resort disambiguator for legacy evidence that
    carries no event-level identity.  Events which expose a durable identity
    (model call, tool message, memory record, snapshot/plan/mutation, etc.) use
    that identity instead, so replaying or reordering evidence does not change
    their event IDs.
    """

    identity = {
        "runId": run_id,
        "variant": variant,
        "traceId": episode.get("trace_id"),
        "kind": kind,
    }
    # Keep the owning episode identity in every generated event.  Trace IDs
    # are expected to be unique, but these fields make the contract robust to
    # fixtures which replay a trace under a different lifecycle identity.
    for field in ("episode_id", "task_id", "family_id", "stage"):
        value = episode.get(field)
        if value is not None:
            identity[field] = value

    # Prefer stable logical IDs over position in an evidence file.  Payload
    # fields which are mutable accounting evidence are intentionally excluded:
    # if they change for the same logical event, the existing event-id conflict
    # checks can detect the conflicting canonical payload.
    logical_fields: tuple[str, ...]
    if kind == "model_call_usage":
        logical_fields = ("callId", "sequence", "attempt")
    elif kind in {"tool_call", "memory_operation"}:
        logical_fields = ("callId", "operationId")
    elif kind == "memory_injection":
        logical_fields = ("recordId",)
    else:
        logical_fields = (
            "snapshotId",
            "evaluationId",
            "planId",
            "mutationId",
            "executionId",
            "operationId",
        )

    found = False
    for field in logical_fields:
        value = data.get(field)
        if value is not None and value != "":
            identity[field] = value
            found = True

    # Tool evidence historically stores its durable message identity in the
    # source envelope.  Keep it separate from data so the public event payload
    # remains unchanged.
    if kind in {"tool_call", "memory_operation"}:
        message_index = source.get("messageIndex")
        if message_index is not None:
            identity["messageIndex"] = message_index
            found = True

    if not found and kind == "model_call_usage":
        # A malformed/legacy usage record can lack call_id and sequence.  The
        # ordinal fallback is deterministic but deliberately not preferred.
        identity["ordinal"] = ordinal
    elif not found and kind in {"tool_call", "memory_operation"}:
        identity["ordinal"] = ordinal

    return {
        "schemaVersion": SCHEMA_VERSION,
        "eventId": f"evt_{_json_hash(identity)}",
        "runId": run_id,
        "variant": variant,
        "traceId": episode.get("trace_id"),
        "taskId": episode.get("task_id"),
        "familyId": episode.get("family_id"),
        "stage": episode.get("stage"),
        "kind": kind,
        "source": source,
        "data": data,
    }


def _tool_events(
    *,
    run_id: str,
    variant: str,
    episode: dict[str, Any],
    root: Path,
    record_ids: _RecordIdRegistry,
) -> Iterable[dict[str, Any]]:
    calls = episode.get("internal_tools", {}).get("calls", [])
    if not isinstance(calls, list):
        return
    trace_path = resolve_comparison_evidence_path(episode.get("trace", ""), root)
    for index, call in enumerate(calls):
        if not isinstance(call, dict):
            continue
        name = str(call.get("name") or "unknown")
        args = call.get("args") if isinstance(call.get("args"), dict) else {}
        source = {
            "type": "hermes_session_tool_call",
            "path": _relative(trace_path.parent / "artifacts" / "session_current.json", root),
            "messageIndex": call.get("message_index"),
        }
        tool_data: dict[str, Any] = {
            "name": name,
            "argumentKeys": sorted(args),
            "durationAvailable": False,
        }
        if name == "memory":
            tool_data["action"] = args.get("action")
            tool_data["target"] = args.get("target")
        yield _event(
            run_id=run_id,
            variant=variant,
            episode=episode,
            kind="tool_call",
            ordinal=index,
            source=source,
            data=tool_data,
        )

        if name != "memory":
            continue
        content = str(args.get("content") or args.get("new_content") or "")
        memory_data: dict[str, Any] = {
            "action": args.get("action"),
            "target": args.get("target"),
            "contentChars": len(content),
        }
        if content:
            memory_data["recordId"] = record_ids.resolve(content)
        yield _event(
            run_id=run_id,
            variant=variant,
            episode=episode,
            kind="memory_operation",
            ordinal=index,
            source=source,
            data=memory_data,
        )


def _injection_events(
    *,
    run_id: str,
    variant: str,
    episode: dict[str, Any],
    root: Path,
    session: dict[str, Any] | None,
    session_path: Path | None,
    record_ids: _RecordIdRegistry,
) -> Iterable[dict[str, Any]]:
    reported = int(episode.get("retrieval_signals", {}).get("memory_injection_count") or 0)
    if reported == 0:
        return
    visible = ""
    if session is not None:
        visible = str(session.get("system_prompt") or "")
        messages = session.get("messages")
        if isinstance(messages, list) and messages and isinstance(messages[0], dict):
            visible += "\n" + str(messages[0].get("content") or "")
    artifacts = episode.get("artifacts", {})
    raw_entries = [
        *artifacts.get("memory_entries", []),
        *artifacts.get("user_entries", []),
    ]
    entries = list(dict.fromkeys(
        entry for entry in raw_entries if isinstance(entry, str) and entry
    ))
    matched = [entry for entry in entries if isinstance(entry, str) and entry and entry in visible]
    source = {
        "type": "hermes_model_visible_context",
        "path": _relative(session_path, root) if session_path is not None else None,
    }
    for index, entry in enumerate(matched):
        yield _event(
            run_id=run_id,
            variant=variant,
            episode=episode,
            kind="memory_injection",
            ordinal=index,
            source=source,
            data={
                "recordId": record_ids.resolve(entry),
                "contentChars": len(entry),
                "matchEvidence": "exact_model_visible_text",
            },
        )
    if reported > len(matched):
        yield _event(
            run_id=run_id,
            variant=variant,
            episode=episode,
            kind="memory_injection_unresolved",
            ordinal=len(matched),
            source=source,
            data={"reportedCount": reported, "matchedCount": len(matched)},
        )


def build_events(
    comparison_path: Path,
    *,
    judge_enabled: bool | None = None,
    lifecycle_events: Iterable[dict[str, Any]] = (),
) -> list[dict[str, Any]]:
    """Build deterministic ledger events from one sequence comparison."""
    comparison_path = comparison_path.resolve()
    root = comparison_path.parent
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    run_id = root.name
    record_ids = _RecordIdRegistry(run_id)
    trace_occurrences: dict[str, int] = {}
    episode_identities: dict[str, dict[str, dict[str, Any]]] = {}
    for variant, payload in comparison.items():
        if not isinstance(payload, dict) or not isinstance(payload.get("episodes"), list):
            continue
        by_trace: dict[str, dict[str, Any]] = {}
        for episode in payload["episodes"]:
            if not isinstance(episode, dict):
                continue
            trace_id = str(episode.get("trace_id") or "")
            if trace_id:
                by_trace[trace_id] = episode
        episode_identities[variant] = by_trace
    for variant in ("with_persistence", "without_persistence"):
        for episode in comparison.get(variant, {}).get("episodes", []):
            trace_id = str(episode.get("trace_id") or "")
            trace_occurrences[trace_id] = trace_occurrences.get(trace_id, 0) + 1

    events: list[dict[str, Any]] = []
    for variant in ("with_persistence", "without_persistence"):
        episodes = comparison.get(variant, {}).get("episodes", [])
        for episode_index, episode in enumerate(episodes):
            if not isinstance(episode, dict):
                continue
            trace_path = resolve_comparison_evidence_path(episode.get("trace", ""), root)
            trace_source = {"type": "past_bench_sequence_comparison", "path": _relative(comparison_path, root)}
            events.append(_event(
                run_id=run_id,
                variant=variant,
                episode=episode,
                kind="episode_outcome",
                ordinal=episode_index,
                source=trace_source,
                data={
                    "bucket": episode.get("bucket"),
                    "taskScore": episode.get("task_score"),
                    "passed": episode.get("passed"),
                    "judgeScore": episode.get("judge_score"),
                    "judgeEnabled": judge_enabled,
                    "judgeConfigurationEvidence": (
                        "launcher_explicit" if judge_enabled is not None else "unavailable"
                    ),
                    "infraBlocked": episode.get("infra_blocked", False),
                },
            ))

            session, session_path = _session_evidence(episode, root)
            messages = session.get("messages", []) if session else []
            request_count = sum(
                1 for message in messages
                if isinstance(message, dict) and message.get("role") == "assistant"
            )
            usage = episode.get("token_usage", {})
            timing = episode.get("timing", {})
            trace_id = str(episode.get("trace_id") or "")
            events.append(_event(
                run_id=run_id,
                variant=variant,
                episode=episode,
                kind="model_usage",
                ordinal=0,
                source={
                    "type": "past_bench_trace_and_hermes_session",
                    "tracePath": _relative(trace_path, root),
                    "sessionPath": _relative(session_path, root) if session_path else None,
                },
                data={
                    "model": session.get("model") if session else None,
                    "inputTokens": int(usage.get("input_tokens") or 0),
                    "outputTokens": int(usage.get("output_tokens") or 0),
                    "totalTokens": int(usage.get("total_tokens") or 0),
                    "requestCount": (
                        usage.get("model_request_count")
                        if usage.get("model_request_count") is not None
                        else request_count
                    ),
                    "requestCountEvidence": (
                        "model_call_usage_events"
                        if usage.get("model_request_count") is not None
                        else "assistant_message_count"
                    ),
                    "modelTimeSeconds": timing.get("model_time_s"),
                    "wallTimeSeconds": timing.get("wall_time_s"),
                    "cacheReadTokens": usage.get("cache_read_tokens"),
                    "cacheWriteTokens": usage.get("cache_write_tokens"),
                    "reasoningTokens": usage.get("reasoning_tokens"),
                    "retryCount": usage.get("model_retry_count"),
                    "detailedUsageAvailable": usage.get("model_usage_complete", False),
                    "retryEvidenceAvailable": usage.get("model_retry_count") is not None,
                    "billingExecutionId": trace_id,
                    "sharedExecution": trace_occurrences.get(trace_id, 0) > 1,
                },
            ))

            events.extend(_model_call_events(
                run_id=run_id,
                variant=variant,
                episode=episode,
                root=root,
            ))

            events.extend(_tool_events(
                run_id=run_id,
                variant=variant,
                episode=episode,
                root=root,
                record_ids=record_ids,
            ))
            events.extend(_injection_events(
                run_id=run_id,
                variant=variant,
                episode=episode,
                root=root,
                session=session,
                session_path=session_path,
                record_ids=record_ids,
            ))

            artifact_dir = trace_path.parent / "artifacts"
            events.append(_event(
                run_id=run_id,
                variant=variant,
                episode=episode,
                kind="storage_snapshot",
                ordinal=0,
                source={"type": "hermes_artifacts", "path": _relative(artifact_dir, root)},
                data={
                    "memoryChars": int(episode.get("artifacts", {}).get("memory_chars") or 0),
                    "userProfileChars": int(episode.get("artifacts", {}).get("user_chars") or 0),
                    "memoryEntryCount": len(episode.get("artifacts", {}).get("memory_entries", [])),
                    "skillCount": int(episode.get("artifacts", {}).get("skill_count") or 0),
                    "memoryFilesBytes": _directory_bytes(artifact_dir / "memories"),
                    "skillFilesBytes": _directory_bytes(artifact_dir / "skills"),
                    "stateDbBytes": _directory_bytes(artifact_dir / "state.db"),
                },
            ))
    # Validate the generated portion with the same idempotency/conflict
    # semantics used for persisted evidence.  A plain dict comprehension here
    # would silently overwrite a conflicting payload when two physical rows
    # derive the same logical event identity.
    events_by_id: dict[str, str] = {}
    unique_events: list[dict[str, Any]] = []
    for event in events:
        event_id = event.get("eventId")
        canonical = json.dumps(
            event,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        existing = events_by_id.get(event_id)
        if existing is not None:
            if existing != canonical:
                raise ValueError(f"conflicting ledger eventId: {event_id}")
            continue
        events_by_id[event_id] = canonical
        unique_events.append(event)
    events = unique_events
    joined_events = (*load_episode_lifecycle_events(comparison_path), *lifecycle_events)
    for event in joined_events:
        value = dict(event)
        if value.get("schemaVersion") != SCHEMA_VERSION:
            raise ValueError("lifecycle event schema version does not match the ledger")
        if value.get("runId") != run_id:
            raise ValueError("lifecycle event runId does not match the comparison run")
        variant = value.get("variant")
        if not isinstance(variant, str) or variant not in episode_identities:
            raise ValueError("lifecycle event variant does not match the comparison")
        trace_id = value.get("traceId")
        episode = episode_identities[variant].get(str(trace_id))
        if episode is None:
            raise ValueError("lifecycle event traceId does not match the comparison variant")
        expected_identity = {
            "taskId": episode.get("task_id"),
            "familyId": episode.get("family_id"),
            "stage": episode.get("stage"),
        }
        for field, expected in expected_identity.items():
            if value.get(field) != expected:
                raise ValueError(f"lifecycle event {field} does not match its episode")
        event_id = value.get("eventId")
        if not isinstance(event_id, str) or not event_id:
            raise ValueError("lifecycle event requires eventId")
        canonical = json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        existing = events_by_id.get(event_id)
        if existing is not None:
            if existing != canonical:
                raise ValueError(f"conflicting ledger eventId: {event_id}")
            continue
        events_by_id[event_id] = canonical
        events.append(value)
    return events


def write_ledger(
    comparison_path: Path,
    output_path: Path,
    *,
    judge_enabled: bool | None = None,
    lifecycle_events: Iterable[dict[str, Any]] = (),
) -> list[dict[str, Any]]:
    events = build_events(
        comparison_path,
        judge_enabled=judge_enabled,
        lifecycle_events=lifecycle_events,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(json.dumps(event, ensure_ascii=True, sort_keys=True) + "\n" for event in events)
    output_path.write_text(content, encoding="utf-8")
    return events


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparison", type=Path, help="Path to sequence_comparison.json")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL ledger")
    judge_group = parser.add_mutually_exclusive_group()
    judge_group.add_argument(
        "--judge-enabled",
        dest="judge_enabled",
        action="store_true",
        default=None,
        help="Record explicit launcher evidence that the LLM judge was enabled",
    )
    judge_group.add_argument(
        "--judge-disabled",
        dest="judge_enabled",
        action="store_false",
        help="Record explicit launcher evidence that the LLM judge was disabled",
    )
    args = parser.parse_args(argv)
    events = write_ledger(
        args.comparison,
        args.output,
        judge_enabled=args.judge_enabled,
    )
    counts: dict[str, int] = {}
    for event in events:
        kind = str(event["kind"])
        counts[kind] = counts.get(kind, 0) + 1
    print(f"Ledger: {args.output} ({len(events)} events)")
    print("Kinds: " + ", ".join(f"{key}={counts[key]}" for key in sorted(counts)))


if __name__ == "__main__":
    main()
