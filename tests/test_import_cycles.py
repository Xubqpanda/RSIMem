from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_lifecycle_import_is_safe_in_a_fresh_interpreter() -> None:
    """Memory contracts must not make the public lifecycle import circular."""

    source_root = Path(__file__).resolve().parents[1] / "src"
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(source_root) if not existing else f"{source_root}{os.pathsep}{existing}"
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import rsimem.lifecycle; from rsimem.lifecycle import RawResourceUsage",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
