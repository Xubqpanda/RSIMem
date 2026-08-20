"""Trace I/O utilities with lazy exports."""

from __future__ import annotations

from typing import Any

__all__ = ["TraceWriter", "load_trace", "read_events"]


def __getattr__(name: str) -> Any:
    if name == "TraceWriter":
        from .writer import TraceWriter

        return TraceWriter
    if name in {"load_trace", "read_events"}:
        from .reader import load_trace, read_events

        return {"load_trace": load_trace, "read_events": read_events}[name]
    raise AttributeError(name)
