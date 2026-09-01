from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path

import rsimem.baseline as baseline


ROOT = Path(__file__).resolve().parents[1]


def _manifest_copy(tmp_path: Path) -> Path:
    source = json.loads(
        (ROOT / "docs/baseline_manifest_20260901.json").read_text(encoding="utf-8")
    )
    source["repository"]["commit"] = "test-baseline-commit"
    source["repository"]["allowedPostBaselinePaths"] = []
    source["runtime"]["python"] = sys.version.split()[0]
    source["runtime"]["dependencyLock"]["pipFreezeDigest"] = baseline._pip_freeze_digest()
    for relative in source["runtime"]["dependencyLock"]["requirementsDigests"]:
        source["runtime"]["dependencyLock"]["requirementsDigests"][relative] = hashlib.sha256(
            (ROOT / relative).read_bytes()
        ).hexdigest()
    for section, relative in (
        ("rsimemSource", "src/rsimem"),
        ("hermesAgent", "benchmarks/past-bench/agents/hermes-agent"),
        ("pastBench", "benchmarks/past-bench/src/past_bench"),
    ):
        source["vendoredIdentity"][section]["gitTreeDigest"] = baseline._git_tree_digest(
            ROOT, relative
        )
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(source), encoding="utf-8")
    return path


def test_missing_baseline_manifest_fails_closed(tmp_path: Path) -> None:
    report = baseline.verify_baseline_manifest(
        tmp_path / "missing.json",
        ROOT,
    )
    assert not report.ok
    assert report.checks[0].name == "manifest"
    assert report.checks[0].code == "baseline manifest is missing or not a regular file"


def test_pip_freeze_normalizes_editable_vcs_revision() -> None:
    first = [
        "-e git+https://example.invalid/repo.git@0123456789abcdef0123456789abcdef01234567#egg=demo",
        "PyYAML==6.0.3",
    ]
    second = [
        "PyYAML==6.0.3",
        "-e git+https://example.invalid/repo.git@fedcba9876543210fedcba9876543210fedcba98#egg=demo",
    ]
    assert baseline._normalize_pip_freeze(first) == baseline._normalize_pip_freeze(second)


def test_malformed_baseline_shape_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps({"schemaVersion": 1}), encoding="utf-8")
    report = baseline.verify_baseline_manifest(path, ROOT)
    assert not report.ok
    assert report.checks[0].code == "manifest_shape_invalid"


def test_matching_baseline_identity_passes(monkeypatch, tmp_path: Path) -> None:
    path = _manifest_copy(tmp_path)
    original_git = baseline._git

    def fake_git(repo_root: Path, *args: str) -> str:
        if args == ("status", "--porcelain=v1"):
            return ""
        if args == ("rev-parse", "HEAD"):
            return "test-baseline-commit\n"
        if args and args[0] == "diff":
            return ""
        return original_git(repo_root, *args)

    monkeypatch.setattr(baseline, "_git", fake_git)
    report = baseline.verify_baseline_manifest(path, ROOT, git=fake_git)
    assert report.ok, report.to_json()
    assert all(check.status == "passed" for check in report.checks)


def test_source_drift_and_dirty_tree_fail_closed(monkeypatch, tmp_path: Path) -> None:
    path = _manifest_copy(tmp_path)
    original_git = baseline._git

    def fake_git(repo_root: Path, *args: str) -> str:
        if args == ("status", "--porcelain=v1"):
            return " M src/rsimem/example.py\n"
        if args == ("rev-parse", "HEAD"):
            return "different-commit\n"
        if args and args[0] == "diff":
            return "src/rsimem/example.py\n"
        return original_git(repo_root, *args)

    monkeypatch.setattr(baseline, "_git", fake_git)
    report = baseline.verify_baseline_manifest(path, ROOT, git=fake_git)
    codes = {check.name: check.code for check in report.checks}
    assert not report.ok
    assert codes["working_tree"] == "working_tree_dirty_or_git_unavailable"
    assert codes["source_commit"] == "source_commit_drift"
