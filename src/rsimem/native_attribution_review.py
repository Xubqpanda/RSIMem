"""Independent, content-free reviewer records for native attribution candidates."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Mapping, Sequence

from .native_attribution_corpus import NativeAttributionCorpus


REVIEW_SCHEMA = "rsimem-native-attribution-review-v1"


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


class ReviewDecision(StrEnum):
    CONFIRM = "confirm"
    REJECT = "reject"
    ESCALATE = "escalate"


@dataclass(frozen=True, slots=True)
class NativeAttributionReviewRecord:
    record_id: str
    corpus_id: str
    candidate_id: str
    reviewer_id: str
    decision: ReviewDecision
    reviewed_evidence_refs: tuple[str, ...]
    rationale_codes: tuple[str, ...]
    adjudication_id: str | None = None
    schema: str = REVIEW_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != REVIEW_SCHEMA:
            raise ValueError("unsupported native attribution review schema")
        if any(
            not isinstance(value, str) or not value.strip()
            for value in (self.corpus_id, self.candidate_id, self.reviewer_id)
        ):
            raise ValueError("native attribution review identity is incomplete")
        object.__setattr__(self, "decision", ReviewDecision(self.decision))
        for field in ("reviewed_evidence_refs", "rationale_codes"):
            values = getattr(self, field)
            if not isinstance(values, tuple) or not values or any(
                not isinstance(value, str) or not value.strip() for value in values
            ) or len(values) != len(set(values)):
                raise ValueError("native attribution review evidence is invalid")
        if self.adjudication_id is not None and (
            not isinstance(self.adjudication_id, str) or not self.adjudication_id.strip()
        ):
            raise ValueError("native attribution adjudication identity is invalid")
        if self.record_id != "native-attribution-review." + _digest(self.identity_payload())[:40]:
            raise ValueError("native attribution review ID mismatch")

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "corpus_id": self.corpus_id,
            "candidate_id": self.candidate_id,
            "reviewer_id": self.reviewer_id,
            "decision": self.decision.value,
            "reviewed_evidence_refs": list(self.reviewed_evidence_refs),
            "rationale_codes": list(self.rationale_codes),
            "adjudication_id": self.adjudication_id,
        }

    def payload(self) -> dict[str, object]:
        return {"record_id": self.record_id, **self.identity_payload()}

    @classmethod
    def create(
        cls,
        *,
        corpus_id: str,
        candidate_id: str,
        reviewer_id: str,
        decision: ReviewDecision,
        reviewed_evidence_refs: Sequence[str],
        rationale_codes: Sequence[str],
        adjudication_id: str | None = None,
    ) -> "NativeAttributionReviewRecord":
        values = {
            "schema": REVIEW_SCHEMA,
            "corpus_id": corpus_id,
            "candidate_id": candidate_id,
            "reviewer_id": reviewer_id,
            "decision": ReviewDecision(decision).value,
            "reviewed_evidence_refs": sorted(set(reviewed_evidence_refs)),
            "rationale_codes": sorted(set(rationale_codes)),
            "adjudication_id": adjudication_id,
        }
        return cls(
            record_id="native-attribution-review." + _digest(values)[:40],
            corpus_id=corpus_id,
            candidate_id=candidate_id,
            reviewer_id=reviewer_id,
            decision=ReviewDecision(decision),
            reviewed_evidence_refs=tuple(values["reviewed_evidence_refs"]),
            rationale_codes=tuple(values["rationale_codes"]),
            adjudication_id=adjudication_id,
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "NativeAttributionReviewRecord":
        expected = {
            "record_id", "schema", "corpus_id", "candidate_id", "reviewer_id", "decision",
            "reviewed_evidence_refs", "rationale_codes", "adjudication_id",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("native attribution review payload fields are invalid")
        if any(not isinstance(payload[field], str) for field in (
            "record_id", "schema", "corpus_id", "candidate_id", "reviewer_id", "decision"
        )):
            raise ValueError("native attribution review payload scalar types are invalid")
        if any(
            not isinstance(payload[field], list)
            or any(not isinstance(value, str) for value in payload[field])
            for field in ("reviewed_evidence_refs", "rationale_codes")
        ):
            raise ValueError("native attribution review payload collections are invalid")
        if payload["adjudication_id"] is not None and not isinstance(payload["adjudication_id"], str):
            raise ValueError("native attribution review adjudication type is invalid")
        return cls(
            record_id=payload["record_id"], schema=payload["schema"],
            corpus_id=payload["corpus_id"], candidate_id=payload["candidate_id"],
            reviewer_id=payload["reviewer_id"], decision=ReviewDecision(payload["decision"]),
            reviewed_evidence_refs=tuple(payload["reviewed_evidence_refs"]),
            rationale_codes=tuple(payload["rationale_codes"]),
            adjudication_id=payload["adjudication_id"],
        )


def build_review_packet(corpus: NativeAttributionCorpus) -> tuple[dict[str, object], ...]:
    """Create reviewer inputs containing IDs and evidence references only."""

    return tuple({
        "corpus_id": corpus.corpus_id,
        "candidate_id": candidate.attribution_id,
        "case_id": candidate.case_id,
        "family_id": candidate.family_id,
        "memory_kind": candidate.memory_kind,
        "primary_failure_surface": candidate.primary_failure_surface.value,
        "candidate_repair_axis": candidate.candidate_repair_axis,
        "evidence_refs": list(candidate.evidence_refs),
        "review_status": candidate.review_status,
    } for candidate in corpus.candidates)


def validate_review_record(
    record: NativeAttributionReviewRecord,
    corpus: NativeAttributionCorpus,
) -> None:
    """Bind a reviewer record to an existing frozen candidate."""

    if record.corpus_id != corpus.corpus_id:
        raise ValueError("native attribution review corpus mismatch")
    candidate = next(
        (item for item in corpus.candidates if item.attribution_id == record.candidate_id),
        None,
    )
    if candidate is None:
        raise ValueError("native attribution review candidate is not in corpus")
    allowed = set(candidate.evidence_refs)
    if not allowed:
        allowed.add("no_evidence")
    if not set(record.reviewed_evidence_refs).issubset(allowed):
        raise ValueError("native attribution review cites unknown evidence")


class NativeAttributionReviewStore:
    """Append-once JSONL store for independent reviewer records."""

    def __init__(self, path: Path, *, corpus: NativeAttributionCorpus | None = None) -> None:
        self.path = Path(path).expanduser().resolve()
        self.corpus = corpus

    def append(self, record: NativeAttributionReviewRecord) -> bool:
        if self.corpus is not None:
            validate_review_record(record, self.corpus)
        line = _canonical(record.payload()) + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_name(self.path.name + ".lock")
        with lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if self.path.is_symlink() or lock_path.is_symlink():
                raise ValueError("native attribution review store cannot be symlinked")
            existing = self.path.read_text(encoding="utf-8") if self.path.exists() else ""
            if any(json.loads(item).get("record_id") == record.record_id for item in existing.splitlines()):
                if line.rstrip("\n") in existing.splitlines():
                    return False
                raise ValueError("native attribution review ID conflicts with existing record")
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
            return True


__all__ = [
    "REVIEW_SCHEMA", "ReviewDecision", "NativeAttributionReviewRecord",
    "NativeAttributionReviewStore", "build_review_packet", "validate_review_record",
]
