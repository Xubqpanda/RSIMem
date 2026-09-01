"""Fail-closed verification for a recorded pre-cleanup baseline.

The verifier is intentionally independent of the experiment runners.  A
baseline describes the exact source/runtime identity at a review point; after
that point only explicitly listed metadata files may change.  Any source,
dependency, vendored-tree, import-origin, or working-tree drift is reported as
a failure before cleanup is allowed to proceed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from dataclasses import dataclass, asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Callable, Mapping, Sequence


BASELINE_SCHEMA_VERSION = 1
DEFAULT_MANIFEST = Path("docs/baseline_manifest_20260901.json")


@dataclass(frozen=True, slots=True)
class BaselineCheck:
    name: str
    status: str
    code: str


@dataclass(frozen=True, slots=True)
class BaselineReport:
    schema_version: int
    ok: bool
    checks: tuple[BaselineCheck, ...]

    def payload(self) -> dict[str, object]:
        return {
            "schemaVersion": self.schema_version,
            "ok": self.ok,
            "checks": [asdict(check) for check in self.checks],
        }

    def to_json(self) -> str:
        return json.dumps(self.payload(), ensure_ascii=True, sort_keys=True)


def _git(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_tree_digest(repo_root: Path, path: str) -> str:
    listing = _git(repo_root, "ls-tree", "-r", "HEAD", path)
    return hashlib.sha256(listing.encode("utf-8")).hexdigest()


def _pip_freeze_digest() -> str:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        check=True,
        capture_output=True,
        text=True,
    )
    normalized = "\n".join(sorted(result.stdout.splitlines()))
    if normalized:
        normalized += "\n"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _load_manifest(path: Path) -> Mapping[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("baseline manifest is missing or not a regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("baseline manifest is not valid JSON") from exc
    if not isinstance(value, dict) or value.get("schemaVersion") != BASELINE_SCHEMA_VERSION:
        raise ValueError("unsupported baseline manifest schema")
    return value


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"baseline manifest {name} must be an object")
    return value


def verify_baseline_manifest(
    manifest_path: Path,
    repo_root: Path,
    *,
    package_version: Callable[[str], str] = version,
    pip_freeze_digest: Callable[[], str] = _pip_freeze_digest,
    git: Callable[..., str] = _git,
) -> BaselineReport:
    """Verify a baseline without exposing machine paths or credentials."""

    checks: list[BaselineCheck] = []
    try:
        manifest = _load_manifest(manifest_path)
    except ValueError as exc:
        return BaselineReport(
            BASELINE_SCHEMA_VERSION,
            False,
            (BaselineCheck("manifest", "failed", str(exc)),),
        )

    try:
        repository = _mapping(manifest.get("repository"), "repository")
        runtime = _mapping(manifest.get("runtime"), "runtime")
        dependencies = _mapping(runtime.get("dependencyLock"), "dependencyLock")
        vendored = _mapping(manifest.get("vendoredIdentity"), "vendoredIdentity")
    except ValueError:
        return BaselineReport(
            BASELINE_SCHEMA_VERSION,
            False,
            (BaselineCheck("manifest", "failed", "manifest_shape_invalid"),),
        )

    def add(name: str, ok: bool, passed: str, failed: str) -> None:
        checks.append(BaselineCheck(name, "passed" if ok else "failed", passed if ok else failed))

    try:
        status = git(repo_root, "status", "--porcelain=v1")
    except (OSError, subprocess.CalledProcessError):
        status = "__git_unavailable__"
    add("working_tree", status == "", "clean", "working_tree_dirty_or_git_unavailable")

    try:
        current_commit = git(repo_root, "rev-parse", "HEAD").strip()
    except (OSError, subprocess.CalledProcessError):
        current_commit = ""
    expected_commit = repository.get("commit")
    allowed = repository.get("allowedPostBaselinePaths", ())
    allowed_paths = set(allowed) if isinstance(allowed, list) and all(isinstance(v, str) for v in allowed) else set()
    commit_ok = current_commit == expected_commit
    if not commit_ok and isinstance(expected_commit, str) and expected_commit:
        try:
            changed = git(repo_root, "diff", "--name-only", f"{expected_commit}..HEAD").splitlines()
            commit_ok = all(path in allowed_paths for path in changed)
        except (OSError, subprocess.CalledProcessError):
            changed = ["__git_unavailable__"]
    add("source_commit", commit_ok, "baseline_commit_or_allowed_metadata", "source_commit_drift")

    expected_python = str(runtime.get("python") or "")
    add("python", sys.version.split()[0] == expected_python, "python_matches", "python_version_mismatch")

    expected_packages = runtime.get("packageVersions")
    package_ok = True
    if isinstance(expected_packages, Mapping):
        for name, expected in expected_packages.items():
            if not isinstance(name, str) or not isinstance(expected, str):
                package_ok = False
                break
            try:
                if package_version(name) != expected:
                    package_ok = False
                    break
            except PackageNotFoundError:
                package_ok = False
                break
    else:
        package_ok = False
    add("packages", package_ok, "package_versions_match", "package_version_mismatch")

    expected_freeze = dependencies.get("pipFreezeDigest")
    try:
        actual_freeze = pip_freeze_digest()
    except (OSError, subprocess.CalledProcessError):
        actual_freeze = ""
    add("pip_freeze", actual_freeze == expected_freeze, "pip_freeze_matches", "pip_freeze_drift")

    requirement_ok = True
    expected_requirements = dependencies.get("requirementsDigests")
    if not isinstance(expected_requirements, Mapping):
        requirement_ok = False
    else:
        for relative, expected in expected_requirements.items():
            path = repo_root / str(relative)
            if not isinstance(relative, str) or not isinstance(expected, str) or not path.is_file():
                requirement_ok = False
                break
            try:
                actual = _sha256_file(path)
            except OSError:
                requirement_ok = False
                break
            if actual != expected:
                requirement_ok = False
                break
    add("requirements", requirement_ok, "requirements_match", "requirements_drift")

    tree_specs = (
        ("rsimem_source", "rsimemSource", "gitTreeDigest", "src/rsimem"),
        ("hermes_tree", "hermesAgent", "gitTreeDigest", "benchmarks/past-bench/agents/hermes-agent"),
        ("past_bench_tree", "pastBench", "gitTreeDigest", "benchmarks/past-bench/src/past_bench"),
    )
    for name, section, key, relative in tree_specs:
        section_value = _mapping(vendored.get(section), section)
        expected = section_value.get(key)
        try:
            actual = _git_tree_digest(repo_root, relative)
        except (OSError, subprocess.CalledProcessError):
            actual = ""
        current_ok = actual == expected
        add(name, current_ok, "tree_digest_matches", "tree_digest_drift")

    editable_path = dependencies.get("editableHermesPath")
    import_path_ok = editable_path == "benchmarks/past-bench/agents/hermes-agent"
    try:
        spec = importlib.util.find_spec("run_agent")
        origin = Path(spec.origin).resolve() if spec is not None and spec.origin else None
        expected_origin = (repo_root / str(editable_path)).resolve() / "run_agent.py"
        import_path_ok = import_path_ok and origin == expected_origin
    except (ImportError, OSError, ValueError):
        import_path_ok = False
    add("hermes_import_origin", import_path_ok, "editable_origin_matches", "hermes_import_origin_mismatch")

    baseline_clean = repository.get("workingTreeClean") is True and dependencies.get("editableHermesPathVerified") is True
    add("manifest_claims", baseline_clean, "baseline_claims_valid", "baseline_claims_invalid")
    add(
        "deletion_guard",
        manifest.get("deletionAuthorized") is False,
        "deletion_locked",
        "deletion_not_locked",
    )

    return BaselineReport(
        BASELINE_SCHEMA_VERSION,
        all(check.status != "failed" for check in checks),
        tuple(checks),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify a frozen RSIMem baseline manifest.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = verify_baseline_manifest(
        args.manifest.expanduser().resolve(),
        args.repo_root.expanduser().resolve(),
    )
    print(report.to_json())
    return 0 if report.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
