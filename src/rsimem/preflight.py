"""Secret-free environment checks for reproducible RSIMem experiments."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Callable, Mapping, Sequence


REQUIRED_PYTHON = (3, 11)
REQUIRED_DISTRIBUTIONS = ("rsimem", "past-bench", "hermes-agent")


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str
    code: str


@dataclass(frozen=True)
class PreflightReport:
    schema_version: int
    ok: bool
    checks: tuple[CheckResult, ...]

    def to_json(self) -> str:
        return json.dumps(
            {
                "schemaVersion": self.schema_version,
                "ok": self.ok,
                "checks": [asdict(check) for check in self.checks],
            },
            ensure_ascii=True,
            sort_keys=True,
        )


def _probe_state_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path, prefix=".preflight-", delete=True):
        pass


def _provider_configuration(
    registry_path: Path,
    agent: str,
    environ: Mapping[str, str],
) -> str:
    try:
        import yaml
    except ImportError:
        return "provider_dependency_missing"
    try:
        registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, yaml.YAMLError):
        return "provider_registry_invalid"
    if not isinstance(registry, dict):
        return "provider_registry_invalid"
    agents = registry.get("agents")
    profile = agents.get(agent) if isinstance(agents, dict) else None
    model = profile.get("default_model") if isinstance(profile, dict) else None
    if not isinstance(model, dict):
        return "provider_profile_missing"
    key_env = model.get("api_key_env")
    base_url = model.get("base_url")
    if not isinstance(key_env, str) or not key_env or not isinstance(base_url, str) or not base_url:
        return "provider_profile_incomplete"
    if not environ.get(key_env):
        return "provider_credential_missing"
    return "provider_configured"


def run_preflight(
    *,
    state_directory: Path,
    past_bench_root: Path,
    registry_path: Path,
    agent: str = "hermes-luna",
    require_provider: bool = False,
    python_version: tuple[int, int] | None = None,
    distribution_version: Callable[[str], str] = version,
    state_probe: Callable[[Path], None] = _probe_state_directory,
    environ: Mapping[str, str] | None = None,
) -> PreflightReport:
    checks: list[CheckResult] = []
    actual_python = python_version or (sys.version_info.major, sys.version_info.minor)
    python_ok = actual_python == REQUIRED_PYTHON
    checks.append(CheckResult(
        name="python",
        status="passed" if python_ok else "failed",
        code="python_supported" if python_ok else "python_version_mismatch",
    ))

    missing: list[str] = []
    for distribution in REQUIRED_DISTRIBUTIONS:
        try:
            distribution_version(distribution)
        except PackageNotFoundError:
            missing.append(distribution)
    checks.append(CheckResult(
        name="dependencies",
        status="passed" if not missing else "failed",
        code="dependencies_installed" if not missing else "dependency_missing",
    ))

    checkout_ok = (past_bench_root / "pyproject.toml").is_file()
    checks.append(CheckResult(
        name="past_bench_checkout",
        status="passed" if checkout_ok else "failed",
        code="past_bench_checkout_found" if checkout_ok else "past_bench_checkout_missing",
    ))

    try:
        state_probe(state_directory)
    except OSError:
        state_ok = False
    else:
        state_ok = True
    checks.append(CheckResult(
        name="state_directory",
        status="passed" if state_ok else "failed",
        code="state_directory_writable" if state_ok else "state_directory_unwritable",
    ))

    provider_code = _provider_configuration(
        registry_path,
        agent,
        environ if environ is not None else os.environ,
    )
    provider_ok = provider_code == "provider_configured"
    checks.append(CheckResult(
        name="provider",
        status="passed" if provider_ok else ("failed" if require_provider else "optional"),
        code=provider_code,
    ))

    return PreflightReport(
        schema_version=1,
        ok=all(check.status != "failed" for check in checks),
        checks=tuple(checks),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate the RSIMem experiment environment.")
    parser.add_argument("--state-dir", type=Path, default=Path.home() / ".local/state/rsimem")
    parser.add_argument("--past-bench-root", type=Path, default=Path("benchmarks/past-bench"))
    parser.add_argument("--registry", type=Path, default=Path("configs/agents.yaml"))
    parser.add_argument("--agent", default="hermes-luna")
    parser.add_argument("--require-provider", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run_preflight(
        state_directory=args.state_dir,
        past_bench_root=args.past_bench_root,
        registry_path=args.registry,
        agent=args.agent,
        require_provider=args.require_provider,
    )
    print(report.to_json())
    return 0 if report.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
