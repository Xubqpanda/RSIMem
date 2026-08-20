"""Append-only JSONL trace writer.

Modified by RSIMem to accept request-level model usage events.
"""

from __future__ import annotations

from pathlib import Path
from typing import IO

from ..models.trace import (
    AuditSnapshot,
    MediaLoad,
    ModelCallUsage,
    RuntimeRequest,
    RuntimeResponse,
    TraceEnd,
    TraceEvent,
    TraceMessage,
    TraceStart,
    ToolDispatch,
)


class TraceWriter:
    """Writes trace events as JSONL (one JSON object per line)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh: IO[str] | None = None

    def _ensure_open(self) -> IO[str]:
        if self._fh is None or self._fh.closed:
            self._fh = open(self.path, "a")
        return self._fh

    def write_event(
        self,
        event: TraceStart | TraceMessage | ModelCallUsage | ToolDispatch | RuntimeRequest | RuntimeResponse | AuditSnapshot | MediaLoad | TraceEnd,
    ) -> None:
        fh = self._ensure_open()
        fh.write(event.model_dump_json() + "\n")
        fh.flush()

    def close(self) -> None:
        if self._fh and not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> TraceWriter:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
