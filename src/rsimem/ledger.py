"""Build a privacy-preserving lifecycle ledger from PAST-Bench evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import threading
from typing import Any, Iterable

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
    "failure_type",
    "limit",
    "namespace",
    "surface",
}
_RSIMEM_EXECUTION_MODES = {"native+ledger", "native+adapter+ledger"}


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
    if data.get("executionMode") not in _RSIMEM_EXECUTION_MODES:
        raise ValueError(f"invalid RSIMem execution mode in {source_path}")


def load_episode_lifecycle_events(comparison_path: Path) -> tuple[dict[str, Any], ...]:
    """Load content-free RSIMem evidence adjacent to comparison-owned traces."""

    comparison_path = comparison_path.resolve()
    root = comparison_path.parent
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    evidence_identities: dict[Path, set[tuple[Any, ...]]] = {}
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
            evidence_path = (
                trace_path.resolve().parent
                / "artifacts"
                / "rsimem_memory_events.jsonl"
            )
            evidence_identities.setdefault(evidence_path, set()).add((
                root.name,
                variant,
                str(episode.get("trace_id") or ""),
                episode.get("task_id"),
                episode.get("family_id"),
                episode.get("stage"),
            ))

    events: list[dict[str, Any]] = []
    events_by_id: dict[str, str] = {}
    for evidence_path in sorted(evidence_identities, key=str):
        if not evidence_path.exists():
            continue
        allowed_identities = evidence_identities[evidence_path]
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
            _validate_memory_runtime_event(value, evidence_path)
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
    ) -> None:
        if not variant.strip() or not trace_id.strip():
            raise ValueError("lifecycle ledger variant and trace_id must not be empty")
        self.variant = variant
        self.trace_id = trace_id
        self.family_id = family_id
        self.stage = stage
        self._events: list[dict[str, Any]] = []
        self._events_by_id: dict[str, str] = {}

    @property
    def events(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._events)

    def _append(
        self,
        *,
        kind: str,
        run_id: str,
        episode_id: str,
        session_id: str,
        task_id: str,
        snapshot_id: str,
        data: dict[str, Any],
    ) -> None:
        identity = {
            "runId": run_id,
            "variant": self.variant,
            "traceId": self.trace_id,
            "snapshotId": snapshot_id,
            "kind": kind,
            "evaluationId": data.get("evaluationId"),
            "planId": data.get("planId"),
            "mutationId": data.get("mutationId"),
            "status": data.get("status"),
            "reasonCodes": data.get("reasonCodes"),
        }
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
        canonical = json.dumps(event, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        existing = self._events_by_id.get(event_id)
        if existing is not None:
            if existing != canonical:
                raise ValueError(f"conflicting lifecycle ledger event: {event_id}")
            return
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

    def write(self, output_path: Path) -> None:
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
        self.output_path = output_path.expanduser().resolve() if output_path else None
        self._events: list[dict[str, Any]] = []
        self._occurrences: dict[str, int] = {}
        self._lock = threading.Lock()
        if self.output_path is not None:
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
    if not trace_path.exists():
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
    identity = {
        "runId": run_id,
        "variant": variant,
        "traceId": episode.get("trace_id"),
        "kind": kind,
        "ordinal": ordinal,
    }
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
    entries = episode.get("artifacts", {}).get("memory_entries", [])
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
    events_by_id = {
        event["eventId"]: json.dumps(
            event,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        for event in events
    }
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
