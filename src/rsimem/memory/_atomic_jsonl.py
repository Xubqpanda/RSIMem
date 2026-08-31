"""Small owner-controlled atomic JSONL writer used by runtime evidence logs."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Iterable


def replace_jsonl(path: Path, lines: Iterable[str], *, error_name: str) -> None:
    """Replace ``path`` with complete newline-terminated ``lines`` atomically.

    Callers must hold their store lock and perform any schema/idempotency
    checks before invoking this helper.  The destination is never opened for
    append, so a crash during serialization cannot expose a partial final
    record to a later reader.
    """

    path = Path(path)
    if path.is_symlink():
        raise ValueError(f"{error_name} cannot be a symlink")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for line in lines:
                if not isinstance(line, str):
                    raise TypeError("atomic JSONL lines must be strings")
                handle.write(line.rstrip("\n") + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if path.is_symlink():
            raise ValueError(f"{error_name} cannot be a symlink")
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


__all__ = ["replace_jsonl"]
