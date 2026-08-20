"""JSONL trace reader with type-based partitioning.

Modified by RSIMem to parse request-level model usage evidence.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from ..models.trace import (
    AuditSnapshot,
    GradingResult,
    MediaLoad,
    ModelCallUsage,
    RuntimeRequest,
    RuntimeResponse,
    ToolDispatch,
    TraceEnd,
    TraceMessage,
    TraceStart,
)

_EVENT_MAP = {
    "trace_start": TraceStart,
    "message": TraceMessage,
    "tool_dispatch": ToolDispatch,
    "runtime_request": RuntimeRequest,
    "runtime_response": RuntimeResponse,
    "audit_snapshot": AuditSnapshot,
    "media_load": MediaLoad,
    "model_call_usage": ModelCallUsage,
    "trace_end": TraceEnd,
    "grading_result": GradingResult,
}


def read_events(path: str | Path) -> Iterator[TraceStart | TraceMessage | ModelCallUsage | ToolDispatch | RuntimeRequest | RuntimeResponse | AuditSnapshot | MediaLoad | TraceEnd]:
    """Parse each JSONL line by its ``type`` discriminator field."""
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            event_type = raw.get("type")
            cls = _EVENT_MAP.get(event_type)
            if cls is None:
                raise ValueError(f"Unknown trace event type: {event_type!r}")
            yield cls.model_validate(raw)


def load_trace(
    path: str | Path,
) -> tuple[TraceStart, list[TraceMessage], list[ToolDispatch], list[MediaLoad], TraceEnd | None, dict[str, dict]]:
    """Load a full trace file and partition by event type.

    Returns (start, messages, dispatches, media_events, end, audit_data).
    audit_data is keyed by service_name from AuditSnapshot events.
    GradingResult events are silently skipped (they are post-hoc additions).
    """
    start: TraceStart | None = None
    messages: list[TraceMessage] = []
    dispatches: list[ToolDispatch] = []
    media_events: list[MediaLoad] = []
    end: TraceEnd | None = None
    audit_data: dict[str, dict] = {}

    for event in read_events(path):
        if isinstance(event, TraceStart):
            start = event
        elif isinstance(event, TraceMessage):
            messages.append(event)
        elif isinstance(event, ToolDispatch):
            dispatches.append(event)
        elif isinstance(event, ModelCallUsage):
            pass
        elif isinstance(event, RuntimeRequest):
            pass
        elif isinstance(event, RuntimeResponse):
            pass
        elif isinstance(event, AuditSnapshot):
            audit_data[event.service_name] = event.audit_data
        elif isinstance(event, MediaLoad):
            media_events.append(event)
        elif isinstance(event, TraceEnd):
            end = event
        elif isinstance(event, GradingResult):
            pass  # post-hoc grading data; not needed for re-grading

    if start is None:
        raise ValueError(f"No TraceStart event found in {path}")
    return start, messages, dispatches, media_events, end, audit_data
