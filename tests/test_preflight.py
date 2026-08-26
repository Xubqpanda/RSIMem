from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError
from pathlib import Path

from rsimem.preflight import main, run_preflight


def _fixture_paths(tmp_path: Path) -> tuple[Path, Path]:
    past_root = tmp_path / "past-bench"
    past_root.mkdir()
    (past_root / "pyproject.toml").write_text("[project]\nname='past-bench'\n", encoding="utf-8")
    registry = tmp_path / "agents.yaml"
    registry.write_text(
        "agents:\n"
        "  hermes-luna:\n"
        "    default_model:\n"
        "      api_key_env: TEST_PROVIDER_KEY\n"
        "      base_url: https://provider.invalid/v1\n",
        encoding="utf-8",
    )
    return past_root, registry


def _codes(report: object) -> dict[str, str]:
    return {check.name: check.code for check in report.checks}  # type: ignore[attr-defined]


def test_preflight_passes_with_clean_home_and_optional_provider(tmp_path: Path) -> None:
    past_root, registry = _fixture_paths(tmp_path)
    report = run_preflight(
        state_directory=tmp_path / "home/.local/state/rsimem",
        past_bench_root=past_root,
        registry_path=registry,
        python_version=(3, 11),
        distribution_version=lambda _: "1.0",
        environ={},
    )

    assert report.ok
    assert (tmp_path / "home/.local/state/rsimem").is_dir()
    assert _codes(report)["provider"] == "provider_credential_missing"


def test_preflight_reports_wrong_python_missing_dependency_and_unwritable_state(
    tmp_path: Path,
) -> None:
    past_root, registry = _fixture_paths(tmp_path)

    def distribution_version(name: str) -> str:
        if name == "hermes-agent":
            raise PackageNotFoundError(name)
        return "1.0"

    def reject_state(_: Path) -> None:
        raise PermissionError("private machine path must not escape")

    report = run_preflight(
        state_directory=tmp_path / "private-state",
        past_bench_root=past_root,
        registry_path=registry,
        require_provider=True,
        python_version=(3, 12),
        distribution_version=distribution_version,
        state_probe=reject_state,
        environ={},
    )

    assert not report.ok
    assert _codes(report) == {
        "python": "python_version_mismatch",
        "dependencies": "dependency_missing",
        "past_bench_checkout": "past_bench_checkout_found",
        "state_directory": "state_directory_unwritable",
        "provider": "provider_credential_missing",
    }
    serialized = report.to_json()
    assert "private-state" not in serialized
    assert "PermissionError" not in serialized


def test_cli_never_prints_provider_secret_or_machine_paths(
    tmp_path: Path,
    monkeypatch: object,
    capsys: object,
) -> None:
    past_root, registry = _fixture_paths(tmp_path)
    secret = "sentinel-provider-secret"
    monkeypatch.setenv("TEST_PROVIDER_KEY", secret)  # type: ignore[attr-defined]

    exit_code = main([
        "--state-dir", str(tmp_path / "state"),
        "--past-bench-root", str(past_root),
        "--registry", str(registry),
        "--require-provider",
    ])

    captured = capsys.readouterr()  # type: ignore[attr-defined]
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["ok"] is True
    assert secret not in captured.out
    assert str(tmp_path) not in captured.out
    assert "Authorization" not in captured.out
