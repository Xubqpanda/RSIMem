"""Deterministic scheduling and provenance for matched RSIMem runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


EXECUTION_MODES = (
    "native",
    "native+ledger",
    "native+adapter+ledger",
)
_ATTEMPT_STATUSES = {"running", "completed", "failed"}


def execution_order(replicate: int) -> tuple[str, ...]:
    if replicate < 1:
        raise ValueError("replicate must be positive")
    offset = (replicate - 1) % len(EXECUTION_MODES)
    return EXECUTION_MODES[offset:] + EXECUTION_MODES[:offset]


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def initialize_batch_manifest(
    path: Path,
    *,
    replicates: int,
    rsimem_commit: str,
    past_bench_commit: str,
    past_bench_tree: str,
    working_tree_dirty: bool,
) -> None:
    if replicates < 1:
        raise ValueError("replicates must be positive")
    required_revisions = (rsimem_commit, past_bench_commit, past_bench_tree)
    if any(not value.strip() for value in required_revisions):
        raise ValueError("experiment revisions must not be empty")
    _write_json(path, {
        "schemaVersion": 1,
        "family": "memory_ability/SM01_preference_adoption",
        "agent": "hermes-luna",
        "model": "gpt-5.6-luna",
        "runtime": "local",
        "temperature": 0.0,
        "judgeEnabled": False,
        "compareNoPersistence": True,
        "adapterFailurePolicy": "fail_closed",
        "executionModes": list(EXECUTION_MODES),
        "executionOrderByReplicate": {
            str(replicate): list(execution_order(replicate))
            for replicate in range(1, replicates + 1)
        },
        "replicates": replicates,
        "seedControl": "not_exposed_by_current_runtime",
        "rsimemCommit": rsimem_commit,
        "pastBenchCommit": past_bench_commit,
        "pastBenchTree": past_bench_tree,
        "workingTreeDirty": working_tree_dirty,
        "resourceEvidence": "Each run writes raw ledger.jsonl and audit.json evidence.",
        "attempts": [],
    })


def record_attempt(
    path: Path,
    *,
    replicate: int,
    ordinal: int,
    mode: str,
    run_name: str,
    status: str,
    failure_stage: str | None = None,
) -> None:
    if status not in _ATTEMPT_STATUSES:
        raise ValueError(f"invalid attempt status: {status}")
    path = path.expanduser().resolve()
    value = json.loads(path.read_text(encoding="utf-8"))
    if mode not in EXECUTION_MODES or not run_name.strip():
        raise ValueError("attempt mode and run_name must be valid")
    expected_order = value["executionOrderByReplicate"].get(str(replicate))
    if expected_order is None or ordinal < 1 or ordinal > len(expected_order):
        raise ValueError("attempt does not belong to the scheduled replicate")
    if expected_order[ordinal - 1] != mode:
        raise ValueError("attempt mode does not match the scheduled order")

    attempts = value.setdefault("attempts", [])
    matches = [
        attempt
        for attempt in attempts
        if attempt.get("replicate") == replicate and attempt.get("ordinal") == ordinal
    ]
    if len(matches) > 1:
        raise ValueError("manifest has duplicate attempt identity")
    if matches:
        attempt = matches[0]
        if attempt.get("mode") != mode or attempt.get("runName") != run_name:
            raise ValueError("attempt identity conflicts with the manifest")
        if attempt.get("status") != "running" or status == "running":
            raise ValueError("invalid attempt status transition")
    else:
        if status != "running":
            raise ValueError("new attempt must start in running state")
        attempt = {
            "replicate": replicate,
            "ordinal": ordinal,
            "mode": mode,
            "runName": run_name,
        }
        attempts.append(attempt)
    attempt["status"] = status
    attempt["failureStage"] = failure_stage if status == "failed" else None
    _write_json(path, value)
