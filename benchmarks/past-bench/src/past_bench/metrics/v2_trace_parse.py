"""Self-Evolve-Tasks-V2 trace-level metric extraction.

Parses the JSONL traces the runner already writes to harvest signals that
Layers 2/3 currently tag `needs_runner_collection`:

- `first_use_turn`        — turn index at which a given rule/phrase first
                             appears in the agent's tool input or final response
- `repeat_violation_count` — number of times the agent re-uses a phrase after
                             it was explicitly corrected in the same trace
- `stale_leak_events`     — count of episodes whose final response still
                             contains a "stale" phrase that a conflict_update
                             trigger was supposed to overwrite

All three are pure trace parsers with no runner changes required. They read
whatever already exists under `traces/<run>/<episode>/*.jsonl`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def _iter_events(trace_path: Path) -> Iterable[dict[str, Any]]:
    for line in trace_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def _event_text(event: dict[str, Any]) -> str:
    """Flatten text-bearing fields of one trace event into lowercase for grep."""
    parts: list[str] = []
    for key in ("content", "final_response", "response", "text", "prompt"):
        v = event.get(key)
        if isinstance(v, str):
            parts.append(v)
    tool_input = event.get("tool_input") or event.get("input")
    if isinstance(tool_input, dict):
        parts.append(json.dumps(tool_input))
    elif isinstance(tool_input, str):
        parts.append(tool_input)
    tool_output = event.get("tool_output") or event.get("output")
    if isinstance(tool_output, str):
        parts.append(tool_output)
    elif isinstance(tool_output, (dict, list)):
        parts.append(json.dumps(tool_output))
    return "\n".join(parts).lower()


def first_use_turn(trace_path: str | Path, phrases: list[str]) -> int | None:
    """Return the 1-indexed turn on which any of `phrases` first appears.

    `None` if no phrase appears in the whole trace.
    """
    phrases_lc = [p.lower() for p in phrases if p]
    if not phrases_lc:
        return None
    turn = 0
    for event in _iter_events(Path(trace_path)):
        etype = (event.get("type") or event.get("event") or "").lower()
        if etype in {"assistant_message", "llm_response", "turn_end"}:
            turn += 1
        text = _event_text(event)
        if any(p in text for p in phrases_lc):
            return max(turn, 1)
    return None


def repeat_violation_count(
    trace_path: str | Path,
    banned_phrases: list[str],
) -> int:
    """Count occurrences of banned phrases *after* the first correction event.

    A correction event is any event whose text contains 'correction',
    'do not', 'stop using', or 'must not'. Anything before the first such
    event is the agent's default behavior; anything after is a repeat violation.
    """
    banned_lc = [p.lower() for p in banned_phrases if p]
    if not banned_lc:
        return 0
    corrected = False
    correction_markers = ("correction", "do not", "stop using", "must not")
    count = 0
    for event in _iter_events(Path(trace_path)):
        text = _event_text(event)
        if not corrected and any(m in text for m in correction_markers):
            corrected = True
            continue
        if corrected and any(p in text for p in banned_lc):
            count += 1
    return count


def stale_leak_events(
    trace_path: str | Path,
    stale_phrases: list[str],
) -> int:
    """Count events in which a 'stale' phrase appears.

    For `conflict_update` families: after the overwrite notice, the agent
    should never re-emit the old value. Each post-overwrite event containing
    a stale phrase counts as a leak.
    """
    stale_lc = [p.lower() for p in stale_phrases if p]
    if not stale_lc:
        return 0
    overwrite_seen = False
    markers = ("overwrite", "no longer valid", "supersed", "replaces", "new rule")
    count = 0
    for event in _iter_events(Path(trace_path)):
        text = _event_text(event)
        if not overwrite_seen and any(m in text for m in markers):
            overwrite_seen = True
            continue
        if overwrite_seen and any(p in text for p in stale_lc):
            count += 1
    return count


def scan_trace_dir(
    trace_dir: str | Path,
    *,
    trigger_phrases: list[str] | None = None,
    banned_phrases: list[str] | None = None,
    stale_phrases: list[str] | None = None,
) -> dict[str, dict[str, int | None]]:
    """Run all three extractors across every .jsonl file in `trace_dir`."""
    root = Path(trace_dir)
    out: dict[str, dict[str, int | None]] = {}
    for jsonl in sorted(root.rglob("*.jsonl")):
        rel = str(jsonl.relative_to(root))
        out[rel] = {
            "first_use_turn": first_use_turn(jsonl, trigger_phrases or []),
            "repeat_violation_count": repeat_violation_count(jsonl, banned_phrases or []),
            "stale_leak_events": stale_leak_events(jsonl, stale_phrases or []),
        }
    return out
