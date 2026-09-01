"""Fail-closed credential scan for Git-tracked repository files.

The scanner intentionally operates on ``git ls-files`` instead of walking the
working tree.  Local operator credentials, ignored experiment outputs, and
untracked drafts are outside the repository acceptance boundary.
"""

from __future__ import annotations

import argparse
import re
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


_PATTERNS: tuple[tuple[str, re.Pattern[bytes]], ...] = (
    ("openai_style", re.compile(rb"(?<![A-Za-z0-9_-])sk-[A-Za-z0-9_-]{20,}\b")),
    ("github_pat", re.compile(rb"(?<![A-Za-z0-9_-])gh[pousr]_[A-Za-z0-9]{20,}\b")),
    ("aws_access_key", re.compile(rb"(?<![A-Za-z0-9])AKIA[0-9A-Z]{16}\b")),
    # Require a following body line.  This avoids flagging a redaction rule
    # that contains only the literal PEM header it is designed to detect.
    (
        "private_key",
        re.compile(
            rb"-----BEGIN(?: [A-Z0-9]+)? PRIVATE KEY-----\r?\n[A-Za-z0-9+/=]{20,}"
        ),
    ),
)


@dataclass(frozen=True, slots=True)
class SecretScanFinding:
    """A content-free tracked-file scan finding."""

    path: str
    pattern: str

    def __post_init__(self) -> None:
        if not self.path or Path(self.path).is_absolute() or ".." in Path(self.path).parts:
            raise ValueError("secret scan finding path must be repository-relative")
        if not self.pattern:
            raise ValueError("secret scan finding pattern must not be empty")


def tracked_files(root: Path) -> tuple[Path, ...]:
    """Return canonical Git-tracked paths without scanning local-only files."""

    repository = Path(root).expanduser().resolve()
    try:
        output = subprocess.run(
            ("git", "-C", str(repository), "ls-files", "-z"),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("tracked secret scan requires a Git worktree") from exc
    names = tuple(item for item in output.decode("utf-8", "strict").split("\0") if item)
    paths = tuple(Path(name) for name in names)
    if any(path.is_absolute() or ".." in path.parts for path in paths):
        raise ValueError("Git returned an unsafe tracked path")
    return paths


def scan_paths(root: Path, paths: Iterable[Path]) -> tuple[SecretScanFinding, ...]:
    """Scan explicit repository-relative regular files without exposing text."""

    repository = Path(root).expanduser().resolve()
    findings: list[SecretScanFinding] = []
    for relative in sorted((Path(path) for path in paths), key=lambda path: path.as_posix()):
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("secret scan path must be repository-relative")
        candidate = repository / relative
        try:
            mode = candidate.lstat().st_mode
        except FileNotFoundError as exc:
            raise ValueError("tracked secret scan file is missing") from exc
        if not stat.S_ISREG(mode):
            raise ValueError("tracked secret scan requires regular files")
        payload = candidate.read_bytes()
        for pattern_name, pattern in _PATTERNS:
            if pattern.search(payload) is not None:
                findings.append(SecretScanFinding(relative.as_posix(), pattern_name))
    return tuple(findings)


def scan_tracked_files(root: Path) -> tuple[SecretScanFinding, ...]:
    return scan_paths(root, tracked_files(root))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="scan Git-tracked files for credentials")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    try:
        findings = scan_tracked_files(args.root)
    except ValueError as exc:
        print(f"tracked-secret-scan: failed: {exc}")
        return 2
    if findings:
        for finding in findings:
            print(f"tracked-secret-scan: {finding.pattern}: {finding.path}")
        return 1
    print("tracked-secret-scan: clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
