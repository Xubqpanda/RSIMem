"""Build a privacy-preserving lifecycle ledger from PAST-Bench evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = 1


def _json_hash(value: Any, *, length: int = 24) -> str:
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:length]


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


def _session_evidence(episode: dict[str, Any]) -> tuple[dict[str, Any] | None, Path | None]:
    raw_path = episode.get("internal_tools", {}).get("session_file")
    if not raw_path:
        return None, None
    path = Path(str(raw_path))
    return _read_json(path), path


def _model_call_events(
    *,
    run_id: str,
    variant: str,
    episode: dict[str, Any],
    root: Path,
) -> Iterable[dict[str, Any]]:
    trace_path = Path(str(episode.get("trace", "")))
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
                "inputTokens": usage.get("input_tokens"),
                "outputTokens": usage.get("output_tokens"),
                "cacheReadTokens": usage.get("cache_read_tokens"),
                "cacheWriteTokens": usage.get("cache_write_tokens"),
                "reasoningTokens": usage.get("reasoning_tokens"),
                "usageAvailable": raw.get("usage_available", False),
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
    trace_path = Path(str(episode.get("trace", "")))
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
) -> list[dict[str, Any]]:
    """Build deterministic ledger events from one sequence comparison."""
    comparison_path = comparison_path.resolve()
    root = comparison_path.parent
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    run_id = root.name
    record_ids = _RecordIdRegistry(run_id)
    trace_occurrences: dict[str, int] = {}
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
            trace_path = Path(str(episode.get("trace", "")))
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

            session, session_path = _session_evidence(episode)
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
    return events


def write_ledger(
    comparison_path: Path,
    output_path: Path,
    *,
    judge_enabled: bool | None = None,
) -> list[dict[str, Any]]:
    events = build_events(comparison_path, judge_enabled=judge_enabled)
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
