"""Content-free acceptance checks for completed isolated execution phases."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping


def require_complete_phase(root: Path, *, phase_name: str = "execution") -> None:
    """Fail closed unless every phase episode reports complete model usage."""

    results_path = root / "sequence_results.json"
    try:
        payload = json.loads(results_path.read_text(encoding="utf-8"))
        episodes = payload["episodes"]
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"{phase_name} phase has no readable sequence results") from exc
    if not isinstance(episodes, list) or not episodes:
        raise RuntimeError(f"{phase_name} phase has no episodes")
    rejected = []
    for episode in episodes:
        usage = episode.get("token_usage") if isinstance(episode, Mapping) else None
        if not isinstance(usage, Mapping) or usage.get("model_usage_complete") is not True:
            rejected.append(
                str(episode.get("task_id", "unknown"))
                if isinstance(episode, Mapping)
                else "unknown"
            )
    if rejected:
        raise RuntimeError(
            f"{phase_name} infrastructure failure: incomplete model usage for "
            + ", ".join(rejected)
        )


__all__ = ["require_complete_phase"]
