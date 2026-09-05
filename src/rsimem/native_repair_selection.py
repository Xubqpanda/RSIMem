"""Select and freeze evidence-backed native repair cases."""

from __future__ import annotations

import hashlib
import fcntl
import json
import os
from dataclasses import dataclass
from pathlib import Path
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

    @classmethod
    def from_payload(cls, value: object) -> "NativeRepairCase":
        if not isinstance(value, Mapping) or set(value) != {
            "case_id", "attribution_id", "corpus_id", "family_id", "memory_kind",
            "repair_axis", "evidence_refs",
        }:
            raise ValueError("native repair case fields are invalid")
        if not all(isinstance(value[field], str) and value[field] for field in
                   ("case_id", "attribution_id", "corpus_id", "family_id", "repair_axis")):
            raise ValueError("native repair case identity is invalid")
        if value["memory_kind"] is not None and not isinstance(value["memory_kind"], str):
            raise ValueError("native repair case memory kind is invalid")
        refs = value["evidence_refs"]
        if not isinstance(refs, list) or not refs or any(not isinstance(item, str) or not item for item in refs):
            raise ValueError("native repair case evidence is invalid")
        return cls(
            case_id=value["case_id"], attribution_id=value["attribution_id"],
            corpus_id=value["corpus_id"], family_id=value["family_id"],
            memory_kind=value["memory_kind"], repair_axis=value["repair_axis"],
            evidence_refs=tuple(refs),
        )


@dataclass(frozen=True, slots=True)
class NativeRepairCaseList:
    case_list_id: str
    corpus_id: str
    cases: tuple[NativeRepairCase, ...]
    schema: str = SCHEMA

    def payload(self) -> dict[str, object]:
        identity = {"schema": self.schema, "corpus_id": self.corpus_id,
                    "cases": [case.payload() for case in self.cases]}
        return {"case_list_id": self.case_list_id, **identity}

    @classmethod
    def from_payload(cls, value: object) -> "NativeRepairCaseList":
        if not isinstance(value, Mapping) or set(value) != {"case_list_id", "schema", "corpus_id", "cases"}:
            raise ValueError("native repair case list fields are invalid")
        raw = value["cases"]
        if not isinstance(raw, list) or not raw:
            raise ValueError("native repair case list is empty or malformed")
        cases = tuple(NativeRepairCase.from_payload(item) for item in raw)
        result = cls(value["case_list_id"], value["corpus_id"], cases, value["schema"])
        if result.payload() != dict(value):
            raise ValueError("native repair case list is not canonical")
        return result


class NativeRepairCaseListStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path).expanduser().resolve()

    def put(self, case_list: NativeRepairCaseList) -> bool:
        serialized = _canonical(case_list.payload()) + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_name(self.path.name + ".lock")
        with lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if self.path.is_symlink() or lock_path.is_symlink():
                raise ValueError("native repair case list store cannot be symlinked")
            if self.path.exists():
                if self.path.read_text(encoding="utf-8") != serialized:
                    raise ValueError("native repair case list conflicts with existing list")
                return False
            temporary = self.path.with_name(f".{self.path.name}.tmp-{os.getpid()}")
            temporary.write_text(serialized, encoding="utf-8")
            os.replace(temporary, self.path)
            return True

    def load(self) -> NativeRepairCaseList:
        if self.path.is_symlink() or not self.path.is_file():
            raise ValueError("native repair case list is missing or symlinked")
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
            result = NativeRepairCaseList.from_payload(value)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ValueError("malformed native repair case list") from exc
        if self.path.read_text(encoding="utf-8") != _canonical(result.payload()) + "\n":
            raise ValueError("native repair case list is not canonical")
        return result


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


__all__ = ["NativeRepairCase", "NativeRepairCaseList", "NativeRepairCaseListStore", "SCHEMA", "build_case_list_payload", "select_native_repair_cases"]
