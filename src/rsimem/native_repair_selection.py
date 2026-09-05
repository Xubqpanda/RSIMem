"""Select and freeze evidence-backed native repair cases."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Mapping, Sequence

from .native_attribution_corpus import NativeAttributionCorpus


SCHEMA = "rsimem-native-repair-case-list-v1"
_AXES = {"formation", "persistence", "maintenance", "retrieval", "application"}


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class NativeRepairCase:
    case_id: str
    attribution_id: str
    corpus_id: str
    family_id: str
    memory_kind: str | None
    repair_axis: str
    evidence_refs: tuple[str, ...]

    def identity_payload(self) -> dict[str, object]:
        return {
            "case_id": self.case_id, "attribution_id": self.attribution_id,
            "corpus_id": self.corpus_id, "family_id": self.family_id,
            "memory_kind": self.memory_kind, "repair_axis": self.repair_axis,
            "evidence_refs": list(self.evidence_refs),
        }

    def payload(self) -> dict[str, object]:
        return self.identity_payload()


def select_native_repair_cases(
    corpus: NativeAttributionCorpus,
    *,
    two_reviewer_candidate_ids: Sequence[str],
) -> tuple[NativeRepairCase, ...]:
    """Return only fully reviewed, high-confidence, evidence-backed cases."""

    reviewed = set(two_reviewer_candidate_ids)
    candidates = []
    for candidate in corpus.candidates:
        if not (candidate.is_actionable and candidate.confidence == "high"):
            continue
        if candidate.candidate_repair_axis not in _AXES:
            raise ValueError("actionable attribution has invalid repair axis")
        if not candidate.evidence_refs:
            raise ValueError("actionable attribution has incomplete evidence")
        if candidate.attribution_id not in reviewed:
            raise ValueError("actionable attribution lacks two-reviewer coverage")
        candidates.append(NativeRepairCase(
            case_id=candidate.case_id,
            attribution_id=candidate.attribution_id,
            corpus_id=corpus.corpus_id,
            family_id=candidate.family_id,
            memory_kind=candidate.memory_kind,
            repair_axis=candidate.candidate_repair_axis,
            evidence_refs=candidate.evidence_refs,
        ))
    if not candidates:
        raise ValueError("no eligible native repair cases")
    return tuple(sorted(candidates, key=lambda item: item.attribution_id))


def build_case_list_payload(
    corpus: NativeAttributionCorpus,
    cases: Sequence[NativeRepairCase],
) -> dict[str, object]:
    if any(case.corpus_id != corpus.corpus_id for case in cases):
        raise ValueError("repair case list corpus mismatch")
    identity = {
        "schema": SCHEMA, "corpus_id": corpus.corpus_id,
        "cases": [case.payload() for case in cases],
    }
    return {"case_list_id": "native-repair-cases." + _digest(identity)[:40], **identity}


__all__ = ["NativeRepairCase", "SCHEMA", "build_case_list_payload", "select_native_repair_cases"]
