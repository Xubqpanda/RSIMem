"""Validated experiment identity, scheduling, and attempt provenance."""

from __future__ import annotations

import hashlib
import json
import sys
from importlib.metadata import distributions
from pathlib import Path
from typing import Any


EXECUTION_MODES = (
    "native",
    "native+ledger",
    "native+adapter+ledger",
)
STATIC_METHOD_VARIANTS = (
    "no-persistence",
    "native-hermes",
    "static-rsimem",
)
STATIC_UTILITY_METHOD_VARIANTS = (
    "static-rsimem",
    "static-utility-rsimem",
)
ADAPTIVE_METHOD_VARIANTS = (
    "no-persistence",
    "native-hermes",
    "native-ledger",
    "static-rsimem",
    "adaptive-rsimem",
)
_ADAPTIVE_METHOD_EXECUTION = {
    "no-persistence": {
        "persistenceVariant": "without_persistence",
        "rsimemMode": "native",
        "lifecycleEvaluatorMode": "disabled",
        "semanticWritebackMode": "disabled",
        "adaptiveConfigRequired": False,
    },
    "native-hermes": {
        "persistenceVariant": "with_persistence",
        "rsimemMode": "native",
        "lifecycleEvaluatorMode": "disabled",
        "semanticWritebackMode": "disabled",
        "adaptiveConfigRequired": False,
    },
    "native-ledger": {
        "persistenceVariant": "with_persistence",
        "rsimemMode": "native+ledger",
        "lifecycleEvaluatorMode": "deterministic",
        "semanticWritebackMode": "disabled",
        "adaptiveConfigRequired": False,
    },
    "static-rsimem": {
        "persistenceVariant": "with_persistence",
        "rsimemMode": "native+ledger",
        "lifecycleEvaluatorMode": "deterministic",
        "semanticWritebackMode": "static_utility",
        "adaptiveConfigRequired": False,
    },
    "adaptive-rsimem": {
        "persistenceVariant": "with_persistence",
        "rsimemMode": "native+ledger",
        "lifecycleEvaluatorMode": "deterministic",
        "semanticWritebackMode": "adaptive_utility",
        "adaptiveConfigRequired": True,
    },
}
_KNOWN_MODES = frozenset((
    *EXECUTION_MODES,
    *STATIC_METHOD_VARIANTS,
    *STATIC_UTILITY_METHOD_VARIANTS,
    *ADAPTIVE_METHOD_VARIANTS,
))
_ATTEMPT_STATUSES = {"running", "completed", "failed"}
_REQUIRED_CONFIGURATION = {
    "taskFamily",
    "agent",
    "runtime",
    "model",
    "judge",
    "budget",
    "environment",
    "executionModes",
    "persistenceIsolation",
    "adapterFailurePolicy",
    "adapterProjectionVerification",
    "seedControl",
    "adaptivePolicy",
}
_REQUIRED_REVISIONS = {
    "rsimemCommit",
    "rsimemWorkingTreeDirty",
    "pastBenchCommit",
    "pastBenchTree",
    "pastBenchDirty",
}


def execution_order(
    replicate: int,
    modes: tuple[str, ...] = EXECUTION_MODES,
) -> tuple[str, ...]:
    if replicate < 1:
        raise ValueError("replicate must be positive")
    if (
        not modes
        or len(modes) != len(set(modes))
        or any(mode not in _KNOWN_MODES for mode in modes)
    ):
        raise ValueError("execution modes must be unique known modes")
    offset = (replicate - 1) % len(modes)
    return modes[offset:] + modes[:offset]


def adaptive_method_execution_profile(method: str) -> dict[str, object]:
    """Return the frozen PAST execution contract for one Phase 2K method."""

    if method not in ADAPTIVE_METHOD_VARIANTS:
        raise ValueError("unknown adaptive experiment method")
    return dict(_ADAPTIVE_METHOD_EXECUTION[method])


def _canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def resolved_adaptive_policy_profile(config_path: Path) -> dict[str, Any]:
    """Resolve and verify one prepared ACTIVE policy without exposing content."""

    resolved_config = config_path.expanduser().resolve()
    try:
        config = json.loads(resolved_config.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("adaptive policy config cannot be read") from exc
    expected = {
        "schema_version",
        "prepared_policy_store_file",
        "adaptive_policy_store_path",
        "adaptive_trusted_roots",
        "adaptive_parameters",
    }
    if not isinstance(config, dict) or set(config) != expected:
        raise ValueError("adaptive policy config fields are incomplete or unknown")
    if config.get("schema_version") != 1:
        raise ValueError("unsupported adaptive policy config schema")
    prepared_file = config.get("prepared_policy_store_file")
    if not isinstance(prepared_file, str) or not prepared_file.strip():
        raise ValueError("prepared adaptive policy store file is invalid")
    prepared_path = Path(prepared_file)
    if (
        prepared_path.is_absolute()
        or ".." in prepared_path.parts
        or prepared_path.name != prepared_path.as_posix()
    ):
        raise ValueError("prepared adaptive policy store must be a sibling file")
    source_store = (resolved_config.parent / prepared_path).resolve()
    if not source_store.is_relative_to(resolved_config.parent):
        raise ValueError("prepared adaptive policy store escapes config directory")
    try:
        store_payload = json.loads(source_store.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("prepared adaptive policy store cannot be read") from exc

    from .memory.adaptive_mem0_binding import ActiveAdaptiveMem0Binder
    from .memory.adaptive_policy_store import JsonAdaptivePolicyStore
    from .memory.live_writeback import StaticSemanticWritebackConfig
    from .memory_systems.mem0_flat import FrozenMem0UtilityGate
    from .memory_systems.mem0_flat.policy import Mem0FlatSemanticPolicy

    runtime = StaticSemanticWritebackConfig.from_mapping({
        "mode": "adaptive_utility",
        "adaptive_policy_store_path": config["adaptive_policy_store_path"],
        "adaptive_trusted_roots": config["adaptive_trusted_roots"],
        "adaptive_parameters": config["adaptive_parameters"],
    })
    destination = Path(runtime.adaptive_policy_store_path or "")
    if destination.is_absolute() or ".." in destination.parts:
        raise ValueError("adaptive policy store must be relative to Hermes home")
    store = JsonAdaptivePolicyStore(
        source_store,
        trusted_root_policy_versions=runtime.adaptive_trusted_roots,
    )
    snapshot = store.snapshot()
    if snapshot.active is None:
        raise ValueError("prepared adaptive policy store has no ACTIVE policy")
    base_gate = FrozenMem0UtilityGate()
    base_policy = Mem0FlatSemanticPolicy(object(), utility_gate=base_gate)
    binding = ActiveAdaptiveMem0Binder(runtime.adaptive_parameters).bind(
        store,
        base_gate,
        expected_parent_policy_version=base_policy.descriptor.policy_version,
    )
    if not binding.adaptive or binding.artifact_id != snapshot.active.artifact_id:
        raise ValueError("prepared adaptive policy does not bind to Mem0 runtime")
    runtime_identity = {
        key: value
        for key, value in config.items()
        if key != "prepared_policy_store_file"
    }
    return {
        "configSchemaVersion": config["schema_version"],
        "configDigest": _canonical_digest(runtime_identity),
        "storeSchemaVersion": store_payload.get("schema_version"),
        "storeDigest": _canonical_digest(store_payload),
        "activePolicyVersion": snapshot.active.policy_version,
        "activeArtifactId": snapshot.active.artifact_id,
        "activeArtifactDigest": snapshot.active.content_digest,
        "preparation": "external_audited_active_store",
    }


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


def resolved_model_profile(
    registry_path: Path,
    agent: str,
    *,
    temperature: float,
) -> dict[str, Any]:
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
        "temperature": temperature,
    }


def resolved_run_profile(config_path: Path) -> dict[str, Any]:
    """Read effective runtime and judge settings from the PAST run config."""
    config = _load_yaml(config_path)
    runtime_config = config.get("runtime")
    judge_config = config.get("judge")
    runtime = runtime_config.get("mode") if isinstance(runtime_config, dict) else None
    temperature = runtime_config.get("temperature") if isinstance(runtime_config, dict) else None
    judge_enabled = judge_config.get("enabled") if isinstance(judge_config, dict) else None
    if not isinstance(runtime, str) or not runtime:
        raise ValueError("effective runtime mode is missing")
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
        raise ValueError("effective model temperature is missing")
    if not isinstance(judge_enabled, bool):
        raise ValueError("effective judge state is missing")
    judge_model = judge_config.get("model_id") if judge_enabled else None
    if judge_enabled and (not isinstance(judge_model, str) or not judge_model):
        raise ValueError("enabled judge model is missing")
    return {
        "runtime": runtime,
        "temperature": float(temperature),
        "judge": {
            "enabled": judge_enabled,
            "profile": "config/judge" if judge_enabled else "disabled",
            "modelId": judge_model,
        },
    }


def resolved_environment_profile() -> dict[str, Any]:
    """Capture exact package versions without editable paths or machine details."""
    installed: dict[str, str] = {}
    for distribution in distributions():
        name = distribution.metadata.get("Name")
        if not isinstance(name, str) or not name:
            continue
        normalized = name.lower().replace("_", "-")
        package_version = distribution.version
        previous = installed.get(normalized)
        if previous is not None and previous != package_version:
            raise ValueError("environment contains conflicting distribution versions")
        installed[normalized] = package_version
    for required in ("rsimem", "past-bench", "hermes-agent"):
        if required not in installed:
            raise ValueError("experiment environment is missing a required distribution")
    return {
        "pythonVersion": ".".join(str(value) for value in sys.version_info[:3]),
        "distributions": dict(sorted(installed.items())),
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
    if not isinstance(value, dict) or value.get("schemaVersion") != 3:
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
    if configuration.get("adapterProjectionVerification") is not True:
        raise ValueError("matched manifest requires native projection verification")
    model = _validate_non_empty_object(configuration.get("model"), "model")
    _require_exact_fields(
        model,
        {"profile", "modelId", "providerBaseUrl", "temperature"},
        "model",
    )
    _require_non_empty_strings(model, ("profile", "modelId", "providerBaseUrl"), "model")
    if not isinstance(model.get("temperature"), (int, float)) or isinstance(model["temperature"], bool):
        raise ValueError("manifest model temperature is invalid")
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
    environment = _validate_non_empty_object(configuration.get("environment"), "environment")
    _require_exact_fields(environment, {"pythonVersion", "distributions"}, "environment")
    _require_non_empty_strings(environment, ("pythonVersion",), "environment")
    installed = environment.get("distributions")
    if not isinstance(installed, dict) or not installed:
        raise ValueError("manifest environment distributions are missing")
    if any(
        not isinstance(name, str)
        or not name
        or not isinstance(package_version, str)
        or not package_version
        for name, package_version in installed.items()
    ):
        raise ValueError("manifest environment distribution is invalid")
    if any(required not in installed for required in ("rsimem", "past-bench", "hermes-agent")):
        raise ValueError("manifest environment is missing a required distribution")
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
    if not isinstance(modes, list) or not modes or any(mode not in _KNOWN_MODES for mode in modes):
        raise ValueError("manifest contains an unknown execution mode")
    if len(set(modes)) != len(modes):
        raise ValueError("manifest contains duplicate execution modes")
    adaptive_policy = configuration.get("adaptivePolicy")
    if "adaptive-rsimem" in modes:
        adaptive_policy = _validate_non_empty_object(
            adaptive_policy,
            "adaptive policy profile",
        )
        _require_exact_fields(adaptive_policy, {
            "configSchemaVersion",
            "configDigest",
            "storeSchemaVersion",
            "storeDigest",
            "activePolicyVersion",
            "activeArtifactId",
            "activeArtifactDigest",
            "preparation",
        }, "adaptive policy profile")
        if (
            adaptive_policy.get("configSchemaVersion") != 1
            or adaptive_policy.get("storeSchemaVersion") != 1
            or adaptive_policy.get("preparation")
            != "external_audited_active_store"
        ):
            raise ValueError("manifest adaptive policy schema is invalid")
        _require_non_empty_strings(adaptive_policy, (
            "configDigest",
            "storeDigest",
            "activePolicyVersion",
            "activeArtifactId",
            "activeArtifactDigest",
        ), "adaptive policy profile")
    elif adaptive_policy is not None:
        raise ValueError("non-adaptive manifest cannot bind an adaptive policy")

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
        expected = list(execution_order(replicate, expected_modes))
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
    environment: dict[str, Any],
    persistence_isolation: dict[str, Any],
    adapter_projection_verification: bool,
    rsimem_commit: str,
    rsimem_working_tree_dirty: bool,
    past_bench_commit: str,
    past_bench_tree: str,
    past_bench_dirty: bool,
    execution_modes: tuple[str, ...] = EXECUTION_MODES,
    adaptive_policy: dict[str, Any] | None = None,
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
        "environment": environment,
        "executionModes": list(execution_modes),
        "persistenceIsolation": persistence_isolation,
        "adapterFailurePolicy": "fail_closed",
        "adapterProjectionVerification": adapter_projection_verification,
        "seedControl": "independent_unseeded_replicates",
        "adaptivePolicy": adaptive_policy,
    }
    revisions = {
        "rsimemCommit": rsimem_commit,
        "rsimemWorkingTreeDirty": rsimem_working_tree_dirty,
        "pastBenchCommit": past_bench_commit,
        "pastBenchTree": past_bench_tree,
        "pastBenchDirty": past_bench_dirty,
    }
    schedule = {
        str(replicate): list(execution_order(replicate, execution_modes))
        for replicate in range(1, replicates + 1)
    }
    identity_payload = {
        "configuration": configuration,
        "replicates": replicates,
        "revisions": revisions,
        "executionOrderByReplicate": schedule,
    }
    value = {
        "schemaVersion": 3,
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
    if mode not in value["configuration"]["executionModes"]:
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
