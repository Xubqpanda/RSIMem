"""Explicit boundaries between runtime process, benchmark audit, and scoring evidence."""

from __future__ import annotations

from enum import StrEnum
import re
from typing import Mapping


class EvidencePlane(StrEnum):
    PURE_PROCESS = "pure_process"
    BENCHMARK_AUDIT = "benchmark_audit"
    FINAL_EVALUATION = "final_evaluation"


class EvidenceSourceKind(StrEnum):
    RUNTIME_OBSERVATION = "runtime_observation"
    APPLICATION_CONTRACT = "application_contract"
    BENCHMARK_CONTRACT = "benchmark_contract"
    FINAL_REPORTER = "final_reporter"


def validate_plane_source(
    plane: EvidencePlane | str,
    source: EvidenceSourceKind | str,
) -> tuple[EvidencePlane, EvidenceSourceKind]:
    resolved_plane = EvidencePlane(plane)
    resolved_source = EvidenceSourceKind(source)
    allowed = {
        EvidencePlane.PURE_PROCESS: {
            EvidenceSourceKind.RUNTIME_OBSERVATION,
            EvidenceSourceKind.APPLICATION_CONTRACT,
        },
        EvidencePlane.BENCHMARK_AUDIT: {EvidenceSourceKind.BENCHMARK_CONTRACT},
        EvidencePlane.FINAL_EVALUATION: {EvidenceSourceKind.FINAL_REPORTER},
    }
    if resolved_source not in allowed[resolved_plane]:
        raise ValueError("evidence plane and source identity are inconsistent")
    return resolved_plane, resolved_source


_FORBIDDEN_PROCESS_KEYS = frozenset({
    "family_id", "familyId", "stage", "grader", "answer_key", "answerKey",
    "hidden_expectation", "hiddenExpectation", "official_score", "officialScore",
    "official_evaluation", "officialEvaluation", "task_score", "taskScore",
    "score", "answer", "judge", "judge_feedback", "expectation",
})
_FORBIDDEN_PROCESS_KEYS_NORMALIZED = frozenset(
    re.sub(r"[^a-z0-9]", "", key.lower())
    for key in _FORBIDDEN_PROCESS_KEYS
)


def validate_pure_process_payload(value: object) -> None:
    """Reject benchmark/scoring metadata in a pure-process learner payload."""

    def walk(item: object) -> None:
        if isinstance(item, Mapping):
            # Treat casing and separators as presentation details.  This
            # closes aliases such as ``Task-Score``/``official.Score`` while
            # still allowing non-evaluation fields such as ``score_digest``.
            overlap = _FORBIDDEN_PROCESS_KEYS_NORMALIZED.intersection(
                re.sub(r"[^a-z0-9]", "", str(key).lower())
                for key in item
            )
            if overlap:
                raise ValueError(
                    "pure-process evidence contains forbidden evaluation fields: "
                    + ", ".join(sorted(overlap))
                )
            for child in item.values():
                walk(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                walk(child)

    walk(value)


def require_optimizer_plane(plane: EvidencePlane | str) -> EvidencePlane:
    resolved = EvidencePlane(plane)
    if resolved != EvidencePlane.PURE_PROCESS:
        raise ValueError(
            "optimizer requires pure_process evidence; benchmark/final evidence "
            "is diagnostic-only"
        )
    return resolved


__all__ = [
    "EvidencePlane",
    "EvidenceSourceKind",
    "require_optimizer_plane",
    "validate_plane_source",
    "validate_pure_process_payload",
]
