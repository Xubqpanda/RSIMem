"""Agent registry loader and model-resolution helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from .protocol import RuntimeModelConfig

_ENV_REPLACEMENT_PREFIX = "${"


def _expand_env(value: Any) -> Any:
    if isinstance(value, str) and value.startswith(_ENV_REPLACEMENT_PREFIX) and value.endswith("}"):
        return os.environ.get(value[2:-1])
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def env_name_candidates(name: str | None) -> list[str]:
    """Return lookup candidates for one env var name.

    We prefer the exact spelling first, but also support shell-safe aliases so a
    config entry like ``kimi-code-api-key`` can be satisfied by
    ``KIMI_CODE_API_KEY``.
    """

    if not name:
        return []

    variants = [
        name,
        name.replace("-", "_"),
        name.upper(),
        name.upper().replace("-", "_"),
    ]
    seen: set[str] = set()
    ordered: list[str] = []
    for variant in variants:
        if variant and variant not in seen:
            ordered.append(variant)
            seen.add(variant)
    return ordered


def resolve_env_value(name: str | None) -> str | None:
    """Resolve one env var using exact name first, then shell-safe aliases."""

    for candidate in env_name_candidates(name):
        value = os.environ.get(candidate)
        if value:
            return value
    return None


class AgentModelDefaults(BaseModel):
    """Default model wiring for one agent entry."""

    model_id: str = ""
    api_key: str | None = None
    api_key_env: str | None = None
    base_url: str | None = None
    extra_body: dict[str, Any] | None = None


class AgentSpec(BaseModel):
    """A named agent entry exposed by the bench."""

    name: str
    description: str = ""
    adapter: str = "openai_compat_chat"
    required_env: list[str] = Field(default_factory=list)
    default_profile: str | None = None
    default_model: AgentModelDefaults = Field(default_factory=AgentModelDefaults)
    model_profiles: dict[str, AgentModelDefaults] = Field(default_factory=dict)
    runtime_image: str | None = None
    install_policy: str = "prebaked"
    bootstrap_commands: list[str] = Field(default_factory=list)
    capabilities: list[str] = Field(default_factory=list)


def _candidate_registry_paths(path: str | Path | None = None) -> list[Path]:
    if path is not None:
        return [Path(path)]
    return [
        Path.cwd() / "configs" / "agents.yaml",
        Path(__file__).resolve().parents[3] / "configs" / "agents.yaml",
    ]


def load_agent_registry(path: str | Path | None = None) -> dict[str, AgentSpec]:
    """Load the agent registry YAML."""

    for candidate in _candidate_registry_paths(path):
        if candidate.exists():
            with open(candidate) as fh:
                raw = yaml.safe_load(fh) or {}
            expanded = _expand_env(raw)
            agent_map = expanded.get("agents", expanded)
            specs = {}
            for name, data in agent_map.items():
                payload = {"name": name, **(data or {})}
                specs[name] = AgentSpec.model_validate(payload)
            return specs
    raise FileNotFoundError("No agent registry found. Expected configs/agents.yaml")


def get_agent_spec(agent_name: str, registry: dict[str, AgentSpec]) -> AgentSpec:
    """Return one agent entry or raise a clear error."""

    try:
        return registry[agent_name]
    except KeyError as exc:
        available = ", ".join(sorted(registry))
        raise KeyError(f"Unknown agent {agent_name!r}. Available agents: {available}") from exc


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _merge_dicts(current, value)
        else:
            merged[key] = value
    return merged


def resolve_model_defaults(spec: AgentSpec, *, profile: str | None = None) -> tuple[str | None, AgentModelDefaults]:
    """Resolve one agent profile into an effective default-model config."""

    selected_profile = profile or spec.default_profile
    resolved = spec.default_model.model_copy(deep=True)
    if not selected_profile:
        return None, resolved

    try:
        profile_defaults = spec.model_profiles[selected_profile]
    except KeyError as exc:
        available = ", ".join(sorted(spec.model_profiles))
        raise KeyError(
            f"Unknown profile {selected_profile!r} for agent {spec.name!r}. "
            f"Available profiles: {available or '-'}"
        ) from exc

    updates = profile_defaults.model_dump(exclude_none=True, exclude_defaults=True)
    if "extra_body" in updates and isinstance(resolved.extra_body, dict) and isinstance(updates["extra_body"], dict):
        updates["extra_body"] = _merge_dicts(resolved.extra_body, updates["extra_body"])
    resolved = resolved.model_copy(update=updates)
    return selected_profile, resolved


def required_env_names(spec: AgentSpec, *, profile: str | None = None) -> list[str]:
    """Return all env vars required for one agent/profile combination."""

    _, defaults = resolve_model_defaults(spec, profile=profile)
    ordered: list[str] = []
    seen: set[str] = set()
    for name in [*spec.required_env, defaults.api_key_env]:
        if not name or name in seen:
            continue
        ordered.append(name)
        seen.add(name)
    return ordered


def resolve_model_config(
    spec: AgentSpec,
    *,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    profile: str | None = None,
) -> RuntimeModelConfig:
    """Resolve final model settings from registry defaults and CLI overrides."""

    _, default = resolve_model_defaults(spec, profile=profile)
    resolved_api_key = api_key or default.api_key
    if not resolved_api_key and default.api_key_env:
        resolved_api_key = resolve_env_value(default.api_key_env)
    return RuntimeModelConfig(
        model_id=model or default.model_id,
        api_key=resolved_api_key,
        base_url=base_url or default.base_url,
        extra_body=default.extra_body,
    )


def missing_required_env(spec: AgentSpec, *, profile: str | None = None) -> list[str]:
    """Return any required env vars that are currently unset."""

    return [name for name in required_env_names(spec, profile=profile) if not resolve_env_value(name)]
