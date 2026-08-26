"""Validated experiment identity, scheduling, and attempt provenance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


EXECUTION_MODES = (
    "native",
    "native+ledger",
    "native+adapter+ledger",
)
_ATTEMPT_STATUSES = {"running", "completed", "failed"}
_REQUIRED_CONFIGURATION = {
    "taskFamily",
    "agent",
    "runtime",
    "model",
    "judge",
    "budget",
    "executionModes",
    "persistenceIsolation",
    "adapterFailurePolicy",
    "seedControl",
}
_REQUIRED_REVISIONS = {
    "rsimemCommit",
    "rsimemWorkingTreeDirty",
    "pastBenchCommit",
    "pastBenchTree",
    "pastBenchDirty",
}


def execution_order(replicate: int) -> tuple[str, ...]:
    if replicate < 1:
        raise ValueError("replicate must be positive")
    offset = (replicate - 1) % len(EXECUTION_MODES)
    return EXECUTION_MODES[offset:] + EXECUTION_MODES[:offset]


def _canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("configuration YAML must contain an object")
    return value


def resolved_model_profile(registry_path: Path, agent: str) -> dict[str, Any]:
    """Read the effective non-secret model settings from the agent registry."""
    registry = _load_yaml(registry_path)
    agents = registry.get("agents")
    profile = agents.get(agent) if isinstance(agents, dict) else None
    model = profile.get("default_model") if isinstance(profile, dict) else None
    if not isinstance(model, dict):
        raise ValueError("agent model profile is missing")
    model_id = model.get("model_id")
    base_url = model.get("base_url")
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("effective model ID is missing")
    if not isinstance(base_url, str) or not base_url:
        raise ValueError("effective provider base URL is missing")
    return {
        "profile": f"{agent}/default_model",
        "modelId": model_id,
        "providerBaseUrl": base_url,
    }


def resolved_family_budget(family_root: Path) -> dict[str, Any]:
    """Read actual task limits in declared episode order."""
    family = _load_yaml(family_root / "family.yaml")
    episode_order = family.get("episode_order")
    if not isinstance(episode_order, list) or not episode_order:
        raise ValueError("family episode order is missing")
    tasks: list[dict[str, Any]] = []
    for episode in episode_order:
        if not isinstance(episode, str) or not episode:
            raise ValueError("family episode identity is invalid")
        task = _load_yaml(family_root / episode / "task.yaml")
        environment = task.get("environment")
        task_id = task.get("task_id")
        max_turns = environment.get("max_turns") if isinstance(environment, dict) else None
        timeout = environment.get("timeout_seconds") if isinstance(environment, dict) else None
        if (
            not isinstance(task_id, str)
            or not task_id
            or not isinstance(max_turns, int)
            or max_turns < 1
            or not isinstance(timeout, int)
            or timeout < 1
        ):
            raise ValueError("task budget is incomplete")
        tasks.append({
            "episode": episode,
            "taskId": task_id,
            "maxTurns": max_turns,
            "timeoutSeconds": timeout,
        })
    return {
        "source": "task_manifest",
        "taskTimeoutOverrideSeconds": None,
        "tasks": tasks,
        "taskManifestDigest": _canonical_digest(tasks),
    }


def _validate_non_empty_object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"manifest {name} must be a non-empty object")
    return value


def _require_exact_fields(value: dict[str, Any], fields: set[str], name: str) -> None:
    if set(value) != fields:
        raise ValueError(f"manifest {name} fields are incomplete or unknown")


def _require_non_empty_strings(value: dict[str, Any], fields: tuple[str, ...], name: str) -> None:
    if any(not isinstance(value.get(field), str) or not value[field] for field in fields):
        raise ValueError(f"manifest {name} is invalid")


def _validate_attempts(value: dict[str, Any]) -> None:
    attempts = value.get("attempts")
    if not isinstance(attempts, list):
        raise ValueError("manifest attempts must be a list")
    run_names: set[str] = set()
    actual_ordinals: set[int] = set()
    slot_counts: dict[tuple[int, int], int] = {}
    for attempt in attempts:
        if not isinstance(attempt, dict):
            raise ValueError("manifest attempt must be an object")
        _require_exact_fields(attempt, {
            "attemptNumber",
            "actualOrdinal",
            "replicate",
            "ordinal",
            "mode",
            "runName",
            "outputDirectory",
            "status",
            "failureStage",
        }, "attempt")
        replicate = attempt.get("replicate")
        ordinal = attempt.get("ordinal")
        mode = attempt.get("mode")
        if not isinstance(replicate, int) or not isinstance(ordinal, int) or not isinstance(mode, str):
            raise ValueError("manifest attempt slot is invalid")
        _validate_slot(value, replicate, ordinal, mode)
        run_name = attempt.get("runName")
        if (
            not isinstance(run_name, str)
            or not run_name
            or run_name in run_names
            or attempt.get("outputDirectory") != run_name
        ):
            raise ValueError("manifest attempt output identity is invalid")
        run_names.add(run_name)
        actual_ordinal = attempt.get("actualOrdinal")
        if not isinstance(actual_ordinal, int) or actual_ordinal < 1 or actual_ordinal in actual_ordinals:
            raise ValueError("manifest actual attempt order is invalid")
        actual_ordinals.add(actual_ordinal)
        slot = (replicate, ordinal)
        slot_counts[slot] = slot_counts.get(slot, 0) + 1
        if attempt.get("attemptNumber") != slot_counts[slot]:
            raise ValueError("manifest slot attempt order is invalid")
        status = attempt.get("status")
        failure_stage = attempt.get("failureStage")
        if status not in _ATTEMPT_STATUSES:
            raise ValueError("manifest attempt status is invalid")
        if status == "failed":
            if not isinstance(failure_stage, str) or not failure_stage:
                raise ValueError("failed manifest attempt is missing its failure stage")
        elif failure_stage is not None:
            raise ValueError("non-failed manifest attempt has a failure stage")
    if actual_ordinals != set(range(1, len(attempts) + 1)):
        raise ValueError("manifest actual attempt order is incomplete")


def validate_manifest(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schemaVersion") != 2:
        raise ValueError("unsupported experiment manifest schema")
    if not isinstance(value.get("experimentId"), str) or not value["experimentId"]:
        raise ValueError("manifest experiment identity is missing")
    if not isinstance(value.get("replicates"), int) or value["replicates"] < 1:
        raise ValueError("manifest replicate count is invalid")

    configuration = _validate_non_empty_object(value.get("configuration"), "configuration")
    if set(configuration) != _REQUIRED_CONFIGURATION:
        raise ValueError("manifest configuration fields are incomplete or unknown")
    for field in ("taskFamily", "agent", "runtime", "adapterFailurePolicy", "seedControl"):
        if not isinstance(configuration.get(field), str) or not configuration[field]:
            raise ValueError(f"manifest {field} is invalid")
    model = _validate_non_empty_object(configuration.get("model"), "model")
    _require_exact_fields(model, {"profile", "modelId", "providerBaseUrl"}, "model")
    _require_non_empty_strings(model, ("profile", "modelId", "providerBaseUrl"), "model")
    judge = _validate_non_empty_object(configuration.get("judge"), "judge")
    _require_exact_fields(judge, {"enabled", "profile", "modelId"}, "judge")
    if not isinstance(judge.get("enabled"), bool) or not isinstance(judge.get("profile"), str):
        raise ValueError("manifest judge is invalid")
    if judge["enabled"] and (not isinstance(judge.get("modelId"), str) or not judge["modelId"]):
        raise ValueError("enabled manifest judge requires a model ID")
    if not judge["enabled"] and judge.get("modelId") is not None:
        raise ValueError("disabled manifest judge cannot have a model ID")
    budget = _validate_non_empty_object(configuration.get("budget"), "budget")
    _require_exact_fields(budget, {
        "source",
        "taskTimeoutOverrideSeconds",
        "tasks",
        "taskManifestDigest",
    }, "budget")
    if budget.get("source") != "task_manifest" or budget.get("taskTimeoutOverrideSeconds") is not None:
        raise ValueError("manifest budget source or override is invalid")
    if not isinstance(budget.get("taskManifestDigest"), str) or not budget["taskManifestDigest"]:
        raise ValueError("manifest task digest is invalid")
    tasks = budget.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("manifest task budgets are missing")
    task_ids: set[str] = set()
    for task in tasks:
        if not isinstance(task, dict):
            raise ValueError("manifest task budget must be an object")
        _require_exact_fields(task, {"episode", "taskId", "maxTurns", "timeoutSeconds"}, "task budget")
        _require_non_empty_strings(task, ("episode", "taskId"), "task budget")
        if task["taskId"] in task_ids:
            raise ValueError("manifest contains duplicate task budgets")
        task_ids.add(task["taskId"])
        if (
            not isinstance(task.get("maxTurns"), int)
            or task["maxTurns"] < 1
            or not isinstance(task.get("timeoutSeconds"), int)
            or task["timeoutSeconds"] < 1
        ):
            raise ValueError("manifest task limits are invalid")
    isolation = _validate_non_empty_object(
        configuration.get("persistenceIsolation"),
        "persistence isolation",
    )
    _require_exact_fields(isolation, {"strategy", "compareNoPersistence"}, "persistence isolation")
    if (
        isolation.get("strategy") != "per_attempt_trace_directory"
        or not isinstance(isolation.get("compareNoPersistence"), bool)
    ):
        raise ValueError("manifest persistence isolation is invalid")
    modes = configuration.get("executionModes")
    if not isinstance(modes, list) or not modes or any(mode not in EXECUTION_MODES for mode in modes):
        raise ValueError("manifest contains an unknown execution mode")
    if len(set(modes)) != len(modes):
        raise ValueError("manifest contains duplicate execution modes")

    revisions = _validate_non_empty_object(value.get("revisions"), "revisions")
    if set(revisions) != _REQUIRED_REVISIONS:
        raise ValueError("manifest revision fields are incomplete or unknown")
    for field in ("rsimemCommit", "pastBenchCommit", "pastBenchTree"):
        if not isinstance(revisions.get(field), str) or not revisions[field]:
            raise ValueError(f"manifest {field} is invalid")
    if not isinstance(revisions.get("rsimemWorkingTreeDirty"), bool):
        raise ValueError("manifest RSIMem dirty state is invalid")
    if revisions.get("pastBenchDirty") is not False:
        raise ValueError("dirty PAST-Bench checkout is not allowed")

    schedule = value.get("executionOrderByReplicate")
    if not isinstance(schedule, dict) or len(schedule) != value["replicates"]:
        raise ValueError("manifest execution schedule is incomplete")
    expected_modes = tuple(modes)
    for replicate in range(1, value["replicates"] + 1):
        order = schedule.get(str(replicate))
        rotated = execution_order(replicate)
        expected = [mode for mode in rotated if mode in expected_modes]
        if order != expected:
            raise ValueError("manifest execution schedule is invalid")
    _validate_attempts(value)

    identity_payload = {
        "configuration": configuration,
        "replicates": value["replicates"],
        "revisions": revisions,
        "executionOrderByReplicate": schedule,
    }
    if value["experimentId"] != _canonical_digest(identity_payload):
        raise ValueError("manifest experiment identity does not match its configuration")
    return value


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("experiment manifest cannot be read") from exc
    return validate_manifest(value)


def initialize_batch_manifest(
    path: Path,
    *,
    replicates: int,
    task_family: str,
    agent: str,
    runtime: str,
    model: dict[str, Any],
    judge: dict[str, Any],
    budget: dict[str, Any],
    persistence_isolation: dict[str, Any],
    rsimem_commit: str,
    rsimem_working_tree_dirty: bool,
    past_bench_commit: str,
    past_bench_tree: str,
    past_bench_dirty: bool,
) -> str:
    if replicates < 1:
        raise ValueError("replicates must be positive")
    if past_bench_dirty:
        raise ValueError("dirty PAST-Bench checkout is not allowed")
    configuration = {
        "taskFamily": task_family,
        "agent": agent,
        "runtime": runtime,
        "model": model,
        "judge": judge,
        "budget": budget,
        "executionModes": list(EXECUTION_MODES),
        "persistenceIsolation": persistence_isolation,
        "adapterFailurePolicy": "fail_closed",
        "seedControl": "independent_unseeded_replicates",
    }
    revisions = {
        "rsimemCommit": rsimem_commit,
        "rsimemWorkingTreeDirty": rsimem_working_tree_dirty,
        "pastBenchCommit": past_bench_commit,
        "pastBenchTree": past_bench_tree,
        "pastBenchDirty": past_bench_dirty,
    }
    schedule = {
        str(replicate): list(execution_order(replicate))
        for replicate in range(1, replicates + 1)
    }
    identity_payload = {
        "configuration": configuration,
        "replicates": replicates,
        "revisions": revisions,
        "executionOrderByReplicate": schedule,
    }
    value = {
        "schemaVersion": 2,
        "experimentId": _canonical_digest(identity_payload),
        **identity_payload,
        "attempts": [],
    }
    validate_manifest(value)
    resolved_path = path.expanduser().resolve()
    if resolved_path.exists():
        existing = load_manifest(resolved_path)
        if existing["experimentId"] != value["experimentId"]:
            raise ValueError("existing manifest belongs to a different experiment")
        return existing["experimentId"]
    _write_json(resolved_path, value)
    return value["experimentId"]


def next_attempt_name(
    path: Path,
    *,
    replicate: int,
    ordinal: int,
    mode: str,
    base_run_name: str,
) -> str | None:
    value = load_manifest(path)
    _validate_slot(value, replicate, ordinal, mode)
    attempts = _slot_attempts(value, replicate, ordinal)
    if any(attempt.get("status") == "completed" for attempt in attempts):
        return None
    if any(attempt.get("status") == "running" for attempt in attempts):
        raise ValueError("scheduled slot already has a running attempt")
    attempt_number = len(attempts) + 1
    return base_run_name if attempt_number == 1 else f"{base_run_name}_attempt{attempt_number:02d}"


def _validate_slot(value: dict[str, Any], replicate: int, ordinal: int, mode: str) -> None:
    if mode not in EXECUTION_MODES:
        raise ValueError("attempt mode is invalid")
    expected_order = value["executionOrderByReplicate"].get(str(replicate))
    if expected_order is None or ordinal < 1 or ordinal > len(expected_order):
        raise ValueError("attempt does not belong to the scheduled replicate")
    if expected_order[ordinal - 1] != mode:
        raise ValueError("attempt mode does not match the scheduled order")


def _slot_attempts(value: dict[str, Any], replicate: int, ordinal: int) -> list[dict[str, Any]]:
    return [
        attempt
        for attempt in value["attempts"]
        if attempt.get("replicate") == replicate and attempt.get("ordinal") == ordinal
    ]


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
    if (
        not isinstance(run_name, str)
        or not run_name.strip()
        or "/" in run_name
        or run_name in {".", ".."}
    ):
        raise ValueError("attempt run name is invalid")
    value = load_manifest(path)
    _validate_slot(value, replicate, ordinal, mode)
    matches = [attempt for attempt in value["attempts"] if attempt.get("runName") == run_name]
    if len(matches) > 1:
        raise ValueError("manifest has duplicate attempt identity")
    if matches:
        attempt = matches[0]
        if attempt.get("replicate") != replicate or attempt.get("ordinal") != ordinal or attempt.get("mode") != mode:
            raise ValueError("attempt identity conflicts with the manifest")
        if attempt.get("status") != "running" or status == "running":
            raise ValueError("invalid attempt status transition")
    else:
        if status != "running":
            raise ValueError("new attempt must start in running state")
        slot_attempts = _slot_attempts(value, replicate, ordinal)
        if any(attempt.get("status") in {"running", "completed"} for attempt in slot_attempts):
            raise ValueError("scheduled slot does not accept another attempt")
        attempt = {
            "attemptNumber": len(slot_attempts) + 1,
            "actualOrdinal": len(value["attempts"]) + 1,
            "replicate": replicate,
            "ordinal": ordinal,
            "mode": mode,
            "runName": run_name,
            "outputDirectory": run_name,
        }
        value["attempts"].append(attempt)
    attempt["status"] = status
    attempt["failureStage"] = failure_stage if status == "failed" else None
    _write_json(path, value)
