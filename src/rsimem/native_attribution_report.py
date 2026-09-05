"""Content-free metrics for a frozen native attribution corpus."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any

from .native_attribution_corpus import NativeAttributionCorpus


REPORT_SCHEMA = "rsimem-native-attribution-report-v1"


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _counts(values: list[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def build_attribution_report(corpus: NativeAttributionCorpus) -> dict[str, Any]:
    """Summarize attribution coverage without inspecting evaluation content."""

    observations = tuple(corpus.observations)
    candidates = tuple(corpus.candidates)
    observed_events = sum(
        event.status.value == "observed"
        for observation in observations
        for event in observation.events
    )
    total_events = sum(len(observation.events) for observation in observations)
    unresolved = corpus.unresolved_count
    actionable = corpus.actionable_count
    candidate_count = len(candidates)
    values: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "corpus_id": corpus.corpus_id,
        "protocol_id": corpus.protocol_id,
        "accepted_run_count": len(corpus.accepted_run_ids),
        "excluded_run_count": len(corpus.excluded_runs),
        "observation_count": len(observations),
        "candidate_count": candidate_count,
        "unresolved_count": unresolved,
        "actionable_count": actionable,
        "unresolved_rate": unresolved / candidate_count if candidate_count else 0.0,
        "actionability_rate": actionable / candidate_count if candidate_count else 0.0,
        "evidence_completeness": observed_events / total_events if total_events else 0.0,
        "surface_counts": _counts([item.primary_failure_surface.value for item in candidates]),
        "memory_kind_counts": _counts([
            item.memory_kind or "none" for item in candidates
        ]),
        "family_counts": _counts([item.family_id for item in candidates]),
        "excluded_reasons": _counts([
            str(item.get("reason"))
            for item in corpus.excluded_runs
            if item.get("reason") is not None
        ]),
    }
    values["report_id"] = "native-attribution-report." + _digest(values)[:40]
    return values


__all__ = ["REPORT_SCHEMA", "build_attribution_report"]
