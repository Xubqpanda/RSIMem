"""Content-free metrics for a frozen native attribution corpus."""

from __future__ import annotations

import hashlib
import json
import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from .native_attribution_corpus import NativeAttributionCorpus, NativeAttributionCorpusStore


REPORT_SCHEMA = "rsimem-native-attribution-report-v1"


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _counts(values: list[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _panel(family_id: str) -> str:
    if family_id.startswith("SM"):
        return "semantic"
    if family_id.startswith("EP"):
        return "episodic"
    if family_id.startswith("PC"):
        return "procedural"
    if family_id.startswith("PG"):
        return "auxiliary"
    return "unknown"


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
        "attribution_coverage": candidate_count / len(observations) if observations else 0.0,
        "unresolved_count": unresolved,
        "actionable_count": actionable,
        "unresolved_rate": unresolved / candidate_count if candidate_count else 0.0,
        "actionability_rate": actionable / candidate_count if candidate_count else 0.0,
        "non_memory_exclusion_rate": (
            sum(item.primary_failure_surface.value == "non_memory_failure" for item in candidates)
            / candidate_count
            if candidate_count else 0.0
        ),
        "evidence_completeness": observed_events / total_events if total_events else 0.0,
        "surface_counts": _counts([item.primary_failure_surface.value for item in candidates]),
        "memory_kind_counts": _counts([
            item.memory_kind or "none" for item in candidates
        ]),
        "family_counts": _counts([item.family_id for item in candidates]),
        "replicate_counts": _counts([
            str(item.replicate_id) if item.replicate_id is not None else "unknown"
            for item in candidates
        ]),
        "panel_counts": _counts([_panel(item.family_id) for item in candidates]),
        "family_surface_counts": {
            family: _counts([
                item.primary_failure_surface.value
                for item in candidates
                if item.family_id == family
            ])
            for family in sorted({item.family_id for item in candidates})
        },
        "case_ids_by_surface": {
            surface: sorted(
                item.case_id
                for item in candidates
                if item.primary_failure_surface.value == surface
            )
            for surface in sorted({item.primary_failure_surface.value for item in candidates})
        },
        "evidence_refs_by_case": {
            item.case_id: list(item.evidence_refs)
            for item in sorted(candidates, key=lambda value: value.case_id)
        },
        "excluded_reasons": _counts([
            str(item.get("reason"))
            for item in corpus.excluded_runs
            if item.get("reason") is not None
        ]),
    }
    grouped_surfaces = {
        tuple(sorted(report.values()))
        for report in values["family_surface_counts"].values()
    }
    values["cross_family_consistency"] = len(grouped_surfaces) <= 1
    values["report_id"] = "native-attribution-report." + _digest(values)[:40]
    return values


def assess_stage2_gate(
    corpus: NativeAttributionCorpus,
    *,
    reviewer_two_reviewer_count: int = 0,
) -> dict[str, object]:
    """Return a conservative, deterministic decision for opening repairs."""

    reasons: list[str] = []
    total_events = sum(len(observation.events) for observation in corpus.observations)
    observed_events = sum(
        event.status.value == "observed"
        for observation in corpus.observations
        for event in observation.events
    )
    evidence_completeness = observed_events / total_events if total_events else 0.0
    if not corpus.observations:
        reasons.append("no_observations")
    if corpus.unresolved_count == len(corpus.candidates):
        reasons.append("unresolved_only")
    if corpus.actionable_count == 0:
        reasons.append("no_actionable_candidate")
    if evidence_completeness < 1.0:
        reasons.append("incomplete_evidence")
    if reviewer_two_reviewer_count <= 0:
        reasons.append("no_two_reviewer_case")
    decision = "OPEN_STAGE2" if not reasons else "STOP_NO_ACTIONABLE_SIGNAL"
    return {
        "schema": "rsimem-stage2-gate-v1",
        "corpus_id": corpus.corpus_id,
        "decision": decision,
        "reasons": reasons,
        "actionable_count": corpus.actionable_count,
        "two_reviewer_candidate_count": reviewer_two_reviewer_count,
        "evidence_completeness": evidence_completeness,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", help="path to a frozen native attribution corpus")
    parser.add_argument("--review-store", help="optional canonical reviewer JSONL store")
    args = parser.parse_args(argv)
    corpus = NativeAttributionCorpusStore(Path(args.corpus)).load()
    report = build_attribution_report(corpus)
    reviewer_two_reviewer_count = 0
    if args.review_store:
        from .native_attribution_review import (
            NativeAttributionReviewStore,
            build_review_summary,
        )
        store = NativeAttributionReviewStore(Path(args.review_store))
        summary = build_review_summary(corpus, store.load_all(corpus=corpus))
        report["review_summary"] = summary
        reviewer_two_reviewer_count = int(summary["two_reviewer_candidate_count"])
    report["stage2_gate"] = assess_stage2_gate(
        corpus, reviewer_two_reviewer_count=reviewer_two_reviewer_count
    )
    print(json.dumps(report, ensure_ascii=True, sort_keys=True))
    return 0


__all__ = ["REPORT_SCHEMA", "build_attribution_report", "assess_stage2_gate", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
