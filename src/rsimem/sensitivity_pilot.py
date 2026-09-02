"""Run one registered, bounded Stage 3 sensitivity pilot.

The runner is intentionally limited to one declared family and replicate. It
persists an audit-plane plan before any provider request, probes the configured
provider once, then executes isolated PAST commands in a rotated condition
order. It does not inspect task scores, grader data, or oracle contents.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from .provider_probe import ProviderProbeResult, probe_provider
from .research_protocol import SensitivityCondition
from .sensitivity import SensitivityPanel
from .sensitivity_prepare import PreparedSensitivityBatch, _matrix, prepare_registered_sensitivity_batch


SENSITIVITY_PILOT_SCHEMA = "rsimem-sensitivity-pilot-v1"
SENSITIVITY_PILOT_SCHEMA_VERSION = 1

Runner = Callable[[tuple[str, ...], Path], int]
Probe = Callable[[str, str, str], ProviderProbeResult]


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _rotated_conditions(replicate: int) -> tuple[SensitivityCondition, ...]:
    if type(replicate) is not int or replicate < 1:
        raise ValueError("sensitivity pilot replicate must be positive")
    conditions = tuple(SensitivityCondition)
    offset = (replicate - 1) % len(conditions)
    return conditions[offset:] + conditions[:offset]


def _run_command(command: tuple[str, ...], cwd: Path) -> int:
    """Delegate output ownership to PAST traces while retaining only status."""

    completed = subprocess.run(command, cwd=cwd, check=False)
    return completed.returncode


def _probe(base_url: str, api_key: str, model: str) -> ProviderProbeResult:
    return probe_provider(base_url, api_key, model)


@dataclass(frozen=True, slots=True)
class SensitivityPilotPlan:
    pilot_id: str
    family_id: str
    panel: SensitivityPanel
    replicate: int
    condition_order: tuple[SensitivityCondition, ...]
    run_ids: tuple[str, ...]
    command_digests: tuple[str, ...]
    schema: str = SENSITIVITY_PILOT_SCHEMA
    schema_version: int = SENSITIVITY_PILOT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema != SENSITIVITY_PILOT_SCHEMA or self.schema_version != SENSITIVITY_PILOT_SCHEMA_VERSION:
            raise ValueError("unsupported sensitivity pilot schema")
        object.__setattr__(self, "panel", SensitivityPanel(self.panel))
        order = tuple(SensitivityCondition(item) for item in self.condition_order)
        if order != _rotated_conditions(self.replicate):
            raise ValueError("sensitivity pilot condition order is invalid")
        if len(self.run_ids) != len(order) or len(self.command_digests) != len(order):
            raise ValueError("sensitivity pilot runs are incomplete")
        values = self.identity_payload()
        expected = "sensitivity-pilot." + _digest(values)[:40]
        if self.pilot_id != expected:
            raise ValueError("sensitivity pilot ID mismatch")

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "family_id": self.family_id,
            "panel": self.panel.value,
            "replicate": self.replicate,
            "condition_order": [item.value for item in self.condition_order],
            "run_ids": list(self.run_ids),
            "command_digests": list(self.command_digests),
        }

    def payload(self) -> dict[str, object]:
        return {"pilot_id": self.pilot_id, **self.identity_payload()}


def _write_once(path: Path, value: dict[str, object]) -> None:
    rendered = json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        existing = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        path.write_text(rendered, encoding="utf-8")
        return
    if existing != rendered:
        raise ValueError("sensitivity pilot file conflicts with prior run")


def _event_path(output_root: Path) -> Path:
    return output_root / "sensitivity_pilot_events.jsonl"


def _completed_run_ids(path: Path) -> set[str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return set()
    completed: set[str] = set()
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError("sensitivity pilot events are unreadable") from exc
        if not isinstance(value, dict) or not isinstance(value.get("run_id"), str):
            raise ValueError("sensitivity pilot event is malformed")
        if value.get("status") == "completed":
            completed.add(value["run_id"])
    return completed


def _append_event(path: Path, value: dict[str, object]) -> None:
    required = {"pilot_id", "run_id", "condition", "command_digest", "status"}
    if set(value) - {"pilot_id", "run_id", "condition", "command_digest", "status", "return_code"} or not required <= set(value):
        raise ValueError("sensitivity pilot event fields are invalid")
    if value["status"] not in {"started", "completed", "failed", "planned"}:
        raise ValueError("sensitivity pilot event status is invalid")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(_canonical(value) + "\n")


def _command_with_runtime_options(
    prepared: PreparedSensitivityBatch,
    *,
    config_path: Path,
    registry_path: Path,
) -> tuple[str, ...]:
    return prepared.launch.command + (
        "--runtime", "local",
        "--sandbox",
        "--sandbox-tools",
        "--no-judge",
        "--config", str(config_path),
        "--registry", str(registry_path),
        "--rsimem-mode", "native+ledger",
        "--rsimem-adapter-failure-policy", "fail_closed",
    )


def run_sensitivity_pilot(
    *,
    panel: SensitivityPanel,
    family_id: str,
    replicate: int,
    batch_id: str,
    rsimem_root: Path,
    past_bench_root: Path,
    registry_path: Path,
    trusted_seed_root: Path,
    output_root: Path,
    config_path: Path,
    agent_registry_path: Path,
    api_key: str,
    execute: bool,
    runner: Runner = _run_command,
    provider_probe: Probe = _probe,
) -> SensitivityPilotPlan:
    """Prepare and optionally execute one selected family/replicate pilot."""

    panel = SensitivityPanel(panel)
    _, matrix = _matrix(panel)
    cases = {
        case.condition: case
        for case in matrix.cases
        if case.family_id == family_id
    }
    if set(cases) != set(SensitivityCondition):
        raise ValueError("selected sensitivity family is not in the requested panel")
    output_root = Path(output_root).expanduser().resolve()
    prepared: list[tuple[SensitivityCondition, PreparedSensitivityBatch, tuple[str, ...]]] = []
    for condition in _rotated_conditions(replicate):
        batch = prepare_registered_sensitivity_batch(
            panel=panel,
            case_id=cases[condition].case_id,
            replicate=replicate,
            batch_id=batch_id,
            rsimem_root=rsimem_root,
            past_bench_root=past_bench_root,
            registry_path=registry_path,
            trusted_seed_root=trusted_seed_root,
            output_root=output_root,
            past_bench_binary=str(Path(rsimem_root) / ".venv" / "bin" / "past-bench"),
        )
        prepared.append((condition, batch, _command_with_runtime_options(
            batch, config_path=config_path, registry_path=agent_registry_path
        )))
    values = {
        "schema": SENSITIVITY_PILOT_SCHEMA,
        "schema_version": SENSITIVITY_PILOT_SCHEMA_VERSION,
        "family_id": family_id,
        "panel": panel.value,
        "replicate": replicate,
        "condition_order": [condition.value for condition, _, _ in prepared],
        "run_ids": [batch.launch.run_id for _, batch, _ in prepared],
        "command_digests": [_digest(command) for _, _, command in prepared],
    }
    plan = SensitivityPilotPlan(
        pilot_id="sensitivity-pilot." + _digest(values)[:40],
        family_id=family_id,
        panel=panel,
        replicate=replicate,
        condition_order=tuple(condition for condition, _, _ in prepared),
        run_ids=tuple(batch.launch.run_id for _, batch, _ in prepared),
        command_digests=tuple(_digest(command) for _, _, command in prepared),
    )
    _write_once(output_root / "sensitivity_pilot_plan.json", plan.payload())
    events = _event_path(output_root)
    for condition, batch, command in prepared:
        _append_event(events, {
            "pilot_id": plan.pilot_id,
            "run_id": batch.launch.run_id,
            "condition": condition.value,
            "command_digest": _digest(command),
            "status": "planned",
        })
    if not execute:
        return plan
    protocol, _ = _matrix(panel)
    probe = provider_probe("https://" + protocol.provider_id, api_key, protocol.model_id)
    _write_once(output_root / "provider_probe.json", probe.payload())
    if not probe.ok:
        raise ValueError("provider probe failed; no sensitivity command was executed")
    completed = _completed_run_ids(events)
    for condition, batch, command in prepared:
        if batch.launch.run_id in completed:
            continue
        event = {
            "pilot_id": plan.pilot_id,
            "run_id": batch.launch.run_id,
            "condition": condition.value,
            "command_digest": _digest(command),
        }
        _append_event(events, {**event, "status": "started"})
        return_code = runner(command, Path(past_bench_root))
        _append_event(events, {
            **event,
            "status": "completed" if return_code == 0 else "failed",
            "return_code": return_code,
        })
        if return_code != 0:
            raise RuntimeError("PAST sensitivity command failed")
    return plan


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", choices=[item.value for item in SensitivityPanel], required=True)
    parser.add_argument("--family-id", required=True)
    parser.add_argument("--replicate", type=int, default=1)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--rsimem-root", type=Path, required=True)
    parser.add_argument("--past-bench-root", type=Path, required=True)
    parser.add_argument("--oracle-registry", type=Path, required=True)
    parser.add_argument("--trusted-seed-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--agent-registry", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    plan = run_sensitivity_pilot(
        panel=SensitivityPanel(args.panel), family_id=args.family_id, replicate=args.replicate,
        batch_id=args.batch_id, rsimem_root=args.rsimem_root, past_bench_root=args.past_bench_root,
        registry_path=args.oracle_registry, trusted_seed_root=args.trusted_seed_root,
        output_root=args.output_root, config_path=args.config, agent_registry_path=args.agent_registry,
        api_key=os.environ.get("GPT_LUNA_API_KEY", ""), execute=args.execute,
    )
    print(plan.pilot_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["SensitivityPilotPlan", "run_sensitivity_pilot"]
