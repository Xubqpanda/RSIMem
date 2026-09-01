from __future__ import annotations

from pathlib import Path

import pytest

from rsimem.secret_scan import scan_paths


def test_scans_only_repository_relative_regular_files(tmp_path: Path) -> None:
    (tmp_path / "tracked.txt").write_text("safe content", encoding="utf-8")
    assert scan_paths(tmp_path, (Path("tracked.txt"),)) == ()

    with pytest.raises(ValueError, match="repository-relative"):
        scan_paths(tmp_path, (Path("../outside.txt"),))

    (tmp_path / "directory").mkdir()
    with pytest.raises(ValueError, match="regular files"):
        scan_paths(tmp_path, (Path("directory"),))


@pytest.mark.parametrize(
    ("name", "payload", "pattern"),
    (
        (
            "openai.txt",
            b"token = " + b"sk-" + b"abcdefghijklmnopqrstuvwxyz123456",
            "openai_style",
        ),
        (
            "github.txt",
            b"token = " + b"ghp_" + b"abcdefghijklmnopqrstuvwx",
            "github_pat",
        ),
        ("aws.txt", b"AK" + b"IA" + b"ABCDEFGHIJKLMNOP", "aws_access_key"),
        (
            "key.pem",
            b"-----BEGIN " + b"PRIVATE KEY-----\n" + b"ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+/=",
            "private_key",
        ),
    ),
)
def test_reports_pattern_and_path_without_returning_secret(
    tmp_path: Path,
    name: str,
    payload: bytes,
    pattern: str,
) -> None:
    (tmp_path / name).write_bytes(payload)
    findings = scan_paths(tmp_path, (Path(name),))
    assert [(item.path, item.pattern) for item in findings] == [(name, pattern)]
    assert payload.decode("utf-8") not in repr(findings)


def test_does_not_confuse_noncredential_substrings(tmp_path: Path) -> None:
    (tmp_path / "source.py").write_text(
        'boundary = "task-completion-or-session-end-v1"\n',
        encoding="utf-8",
    )
    assert scan_paths(tmp_path, (Path("source.py"),)) == ()
