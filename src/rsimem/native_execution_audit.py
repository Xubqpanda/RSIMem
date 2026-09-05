"""Fail-closed audit of manifest-bound native execution artifacts."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence
from urllib.parse import urlsplit

from .native_attribution_run import (
    NativeAttributionRunManifestStore,
    NativeAttributionRunSpec,
)


AUDIT_SCHEMA = "rsimem-native-execution-audit-v1"


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _tree_digest(root: Path) -> str:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("audited directory is missing or symlinked")
    entries: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("audited directory cannot contain symlinks")
        if path.is_file():
            entries.append({
                "path": path.relative_to(root).as_posix(),
                "digest": hashlib.sha256(path.read_bytes()).hexdigest(),
                "size": path.stat().st_size,
            })
    return _digest(entries)


def _read_json(path: Path) -> object:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"audit artifact is missing or symlinked: {path.name}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"audit artifact is unreadable: {path.name}") from exc


@dataclass(frozen=True, slots=True)
class NativeExecutionAudit:
    run_id: str
    trace_digest: str
    state_digest: str
    hermes_home_digest: str
    artifact_digest: str
    service_identity_digest: str
    episode_identity_digest: str
    provider_id: str
    model_id: str
    usage_complete: bool
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    reasoning_tokens: int
    request_count: int
    retry_count: int
    usage_digest: str
    trace_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id:
            raise ValueError("audit run ID is required")
        for value, name in (
            (self.trace_digest, "trace digest"),
            (self.state_digest, "state digest"),
            (self.hermes_home_digest, "Hermes home digest"),
            (self.artifact_digest, "artifact digest"),
            (self.service_identity_digest, "service identity digest"),
            (self.episode_identity_digest, "episode identity digest"),
            (self.usage_digest, "usage digest"),
        ):
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"invalid {name}")
        if not isinstance(self.provider_id, str) or not self.provider_id:
            raise ValueError("provider ID is required")
        if not isinstance(self.model_id, str) or not self.model_id:
            raise ValueError("model ID is required")
        if type(self.usage_complete) is not bool or not self.usage_complete:
            raise ValueError("native execution usage must be complete")
        for value in (
            self.input_tokens, self.output_tokens, self.cache_read_tokens,
            self.cache_write_tokens, self.reasoning_tokens, self.request_count,
            self.retry_count,
        ):
            if type(value) is not int or value < 0:
                raise ValueError("native execution usage totals must be nonnegative integers")
        if self.request_count < 1:
            raise ValueError("native execution requires model request evidence")
        if not self.trace_ids or any(not isinstance(value, str) or not value for value in self.trace_ids):
            raise ValueError("native execution requires trace IDs")

    def payload(self) -> dict[str, object]:
        return {
            "schema": AUDIT_SCHEMA,
            "run_id": self.run_id,
            "trace_digest": self.trace_digest,
            "state_digest": self.state_digest,
            "hermes_home_digest": self.hermes_home_digest,
            "artifact_digest": self.artifact_digest,
            "service_identity_digest": self.service_identity_digest,
            "episode_identity_digest": self.episode_identity_digest,
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "usage_complete": self.usage_complete,
            "usage": {
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "cache_read_tokens": self.cache_read_tokens,
                "cache_write_tokens": self.cache_write_tokens,
                "reasoning_tokens": self.reasoning_tokens,
                "request_count": self.request_count,
                "retry_count": self.retry_count,
            },
            "usage_digest": self.usage_digest,
            "trace_ids": list(self.trace_ids),
        }


class NativeExecutionAuditStore:
    """Append-once content-free audit receipts keyed by logical run ID."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).expanduser().resolve()

    def put(self, audit: NativeExecutionAudit) -> bool:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{audit.run_id}.json"
        lock_path = self.root / f"{audit.run_id}.lock"
        serialized = _canonical(audit.payload()) + "\n"
        with lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if path.is_symlink() or lock_path.is_symlink():
                raise ValueError("native execution audit store cannot be symlinked")
            if path.exists():
                if path.read_text(encoding="utf-8") != serialized:
                    raise ValueError("native run already has a different execution audit")
                return False
            temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
            temporary.write_text(serialized, encoding="utf-8")
            temporary.replace(path)
            return True


def audit_native_execution(
    *, run: NativeAttributionRunSpec, output_root: Path
) -> NativeExecutionAudit:
    """Audit one run directory against its frozen manifest identity."""

    root = Path(output_root).expanduser().resolve()
    state = root / run.state_directory
    home = root / run.hermes_home_directory
    sessions = root / run.session_directory
    artifacts = root / run.artifact_directory
    trace = root / run.trace_directory
    if not trace.is_dir():
        raise ValueError("native trace directory is missing")
    runtime_identity = _read_json(trace / "native_runtime_identity.json")
    if not isinstance(runtime_identity, Mapping) or runtime_identity.get("schema") != "past-bench-native-runtime-identity-v1":
        raise ValueError("native runtime identity is malformed")
    expected_runtime = {
        "sequence": f"native-attribution.{run.run_id}",
        "method_task_id": run.method_case_id,
        "model_id": run.model_id,
        "port_offset": run.port_offset,
        "state_directory": str(state),
        "hermes_home_directory": str(home),
        "session_directory": str(sessions),
        "artifact_directory": str(artifacts),
        "trace_directory": str(trace),
        "initial_home_digest": run.initial_home_digest,
    }
    for key, expected in expected_runtime.items():
        if runtime_identity.get(key) != expected:
            raise ValueError(f"native runtime {key} does not match manifest")
    base_url = runtime_identity.get("base_url")
    if not isinstance(base_url, str):
        raise ValueError("native runtime provider URL is missing")
    parsed = urlsplit(base_url)
    observed_provider = (parsed.netloc + parsed.path).rstrip("/")
    if parsed.scheme != "https" or observed_provider != run.provider_id:
        raise ValueError("native runtime provider does not match manifest")
    result_paths = sorted(trace.glob("**/sequence_results.json"))
    if len(result_paths) != 1:
        raise ValueError("native sequence results are missing or ambiguous")

    result = _read_json(result_paths[0])
    if not isinstance(result, Mapping):
        raise ValueError("native sequence result is malformed")
    if result.get("sequence") != f"native-attribution.{run.run_id}":
        raise ValueError("native sequence identity does not match manifest")
    if result.get("variant") != "with_persistence":
        raise ValueError("native execution variant does not match manifest")
    episodes = result.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("native sequence result has no episodes")

    trace_ids: list[str] = []
    usage_totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "reasoning_tokens": 0,
        "request_count": 0,
        "retry_count": 0,
    }
    service_evidence: list[object] = []
    episode_identities: list[object] = []
    episode_dirs: set[Path] = set()
    observed_task_ids: list[str] = []
    for episode in episodes:
        if not isinstance(episode, Mapping):
            raise ValueError("native episode result is malformed")
        trace_id = episode.get("trace_id")
        usage = episode.get("token_usage")
        trace_path = Path(str(episode.get("trace", ""))).expanduser().resolve()
        if not isinstance(trace_id, str) or not trace_id:
            raise ValueError("native episode trace identity is missing")
        if not isinstance(usage, Mapping) or usage.get("model_usage_complete") is not True:
            raise ValueError("native execution usage is incomplete")
        if trace not in trace_path.parents or not trace_path.is_file() or trace_path.is_symlink():
            raise ValueError("native episode trace escaped the manifest directory")
        events = []
        for line in trace_path.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError("native trace event is malformed") from exc
            if not isinstance(event, Mapping):
                raise ValueError("native trace event is malformed")
            events.append(event)
        starts = [event for event in events if event.get("type") == "trace_start"]
        ends = [event for event in events if event.get("type") == "trace_end"]
        if len(starts) != 1 or len(ends) != 1:
            raise ValueError("native trace terminal identity is incomplete")
        if starts[0].get("trace_id") != trace_id or ends[0].get("trace_id") != trace_id:
            raise ValueError("native trace ID does not match sequence result")
        if starts[0].get("model") != run.model_id:
            raise ValueError("native trace model does not match manifest")
        if ends[0].get("model_usage_complete") is not True:
            raise ValueError("native trace usage is incomplete")
        episode_identity = _read_json(trace_path.parent / "native_episode_identity.json")
        if (
            not isinstance(episode_identity, Mapping)
            or episode_identity.get("schema") != "past-bench-native-episode-identity-v1"
            or episode_identity.get("trace_id") != trace_id
            or episode_identity.get("task_id") != starts[0].get("task_id")
            or episode_identity.get("family_id") != run.family_id
        ):
            raise ValueError("native episode state identity is malformed")
        for field in ("state_before_digest", "state_after_digest"):
            value = episode_identity.get(field)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError("native episode state digest is malformed")
        for field in ("artifact_before", "artifact_after"):
            value = episode_identity.get(field)
            if not isinstance(value, Mapping) or set(value) != {
                "artifact_ids", "memory_entry_count", "user_entry_count",
                "skill_count", "digest",
            }:
                raise ValueError("native episode artifact identity is malformed")
            if not isinstance(value.get("artifact_ids"), list):
                raise ValueError("native episode artifact IDs are malformed")
            digest = value.get("digest")
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError("native episode artifact digest is malformed")
        episode_identities.append(episode_identity)
        calls = [event for event in events if event.get("type") == "model_call_usage"]
        if not calls:
            raise ValueError("native trace has no model request usage")
        trace_usage: dict[str, int] = {}
        for field, end_field in (
            ("input_tokens", "model_input_tokens"),
            ("output_tokens", "model_output_tokens"),
            ("cache_read_tokens", "cache_read_tokens"),
            ("cache_write_tokens", "cache_write_tokens"),
            ("reasoning_tokens", "reasoning_tokens"),
        ):
            values = [
                event.get("usage", {}).get(field)
                if isinstance(event.get("usage"), Mapping)
                else None
                for event in calls
            ]
            if not all(type(value) is int and value >= 0 for value in values):
                raise ValueError("native model request usage is incomplete")
            trace_usage[field] = sum(values)
            if ends[0].get(end_field) != trace_usage[field] or usage.get(field) != trace_usage[field]:
                raise ValueError(f"native usage total mismatch: {field}")
        trace_usage["request_count"] = len(calls)
        trace_usage["retry_count"] = sum(int(event.get("attempt") or 1) - 1 for event in calls)
        if (
            ends[0].get("model_request_count") != trace_usage["request_count"]
            or usage.get("model_request_count") != trace_usage["request_count"]
        ):
            raise ValueError("native usage total mismatch: request_count")
        if (
            ends[0].get("model_retry_count") != trace_usage["retry_count"]
            or usage.get("model_retry_count") != trace_usage["retry_count"]
        ):
            raise ValueError("native usage total mismatch: retry_count")
        if any(
            event.get("usage_available") is not True
            or not isinstance(event.get("usage"), Mapping)
            or event["usage"].get("usage_complete") is not True
            for event in calls
        ):
            raise ValueError("native model request usage is incomplete")
        for field, value in trace_usage.items():
            usage_totals[field] += value
        if episode.get("episode_kind") != "reflection":
            observed_task_ids.append(str(starts[0].get("task_id") or ""))
            service_path = trace_path.parent / "service_identity.json"
            service = _read_json(service_path)
            if not isinstance(service, Mapping) or service.get("schema") != "past-bench-episode-service-identity-v1":
                raise ValueError("native service identity evidence is malformed")
            if service.get("task_id") != starts[0].get("task_id"):
                raise ValueError("native service identity task does not match trace")
            values = service.get("services")
            if not isinstance(values, list):
                raise ValueError("native service identity evidence is malformed")
            for value in values:
                if not isinstance(value, Mapping) or value.get("schema") != "past-bench-verified-service-identity-v1":
                    raise ValueError("native verified service identity is malformed")
            service_evidence.append(service)
        trace_ids.append(trace_id)
        episode_dirs.add(trace_path.parent)

    if len(episode_dirs) != len(episodes):
        raise ValueError("native episodes do not have distinct trace directories")
    if tuple(observed_task_ids) != run.native_task_ids:
        raise ValueError("native task execution order does not match manifest")
    observed_services = sorted(
        (item["task_id"], value["service"], value["port"], value["fixture_digest"])
        for item in service_evidence for value in item["services"]
    )
    expected_services = sorted(
        (value.task_id, value.service, value.port, value.fixture_digest)
        for value in run.service_identities
    )
    if observed_services != expected_services:
        raise ValueError("native service fixture identity does not match manifest")

    return NativeExecutionAudit(
        run_id=run.run_id,
        trace_digest=_tree_digest(trace),
        state_digest=_tree_digest(state),
        hermes_home_digest=_tree_digest(home),
        artifact_digest=_tree_digest(artifacts),
        service_identity_digest=_digest(service_evidence),
        episode_identity_digest=_digest(episode_identities),
        provider_id=run.provider_id,
        model_id=run.model_id,
        usage_complete=True,
        input_tokens=usage_totals["input_tokens"],
        output_tokens=usage_totals["output_tokens"],
        cache_read_tokens=usage_totals["cache_read_tokens"],
        cache_write_tokens=usage_totals["cache_write_tokens"],
        reasoning_tokens=usage_totals["reasoning_tokens"],
        request_count=usage_totals["request_count"],
        retry_count=usage_totals["retry_count"],
        usage_digest=_digest(usage_totals),
        trace_ids=tuple(trace_ids),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--receipt-root", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = NativeAttributionRunManifestStore(args.manifest).get()
    matches = [run for run in manifest.runs if run.run_id == args.run_id]
    if len(matches) != 1:
        raise ValueError("native run ID is not uniquely registered in manifest")
    audit = audit_native_execution(run=matches[0], output_root=args.output_root)
    NativeExecutionAuditStore(args.receipt_root).put(audit)
    print(_canonical(audit.payload()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUDIT_SCHEMA", "NativeExecutionAudit", "NativeExecutionAuditStore",
    "audit_native_execution",
]
