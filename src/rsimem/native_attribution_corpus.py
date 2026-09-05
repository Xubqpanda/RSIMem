"""Restart-safe, content-free corpus for native failure-attribution review."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .native_attribution import NativeAttributionCandidate
from .native_observation import NativeEpisodeObservation


CORPUS_SCHEMA = "rsimem-native-attribution-corpus-v1"

_FORBIDDEN_KEYS = {
    "answer", "answer_key", "hidden_answer", "grader", "judge", "score",
    "task_score", "official_score", "final_response_text", "prompt",
    "completion", "reference_answer", "future_evaluation",
}


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _assert_content_free(value: object) -> None:
    def visit(item: object) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                normalized = str(key).strip().lower().replace("-", "_")
                if normalized in _FORBIDDEN_KEYS:
                    raise ValueError(f"attribution corpus contains forbidden field: {normalized}")
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
    visit(value)


def _validate_excluded_runs(values: Sequence[Mapping[str, object]]) -> tuple[dict[str, str], ...]:
    validated: list[dict[str, str]] = []
    seen: set[str] = set()
    for value in values:
        if set(value) != {"run_id", "reason"}:
            raise ValueError("native attribution exclusion fields are invalid")
        run_id, reason = value.get("run_id"), value.get("reason")
        if (
            not isinstance(run_id, str) or not run_id
            or not isinstance(reason, str) or not reason
            or run_id in seen
        ):
            raise ValueError("native attribution exclusion identity is invalid")
        seen.add(run_id)
        validated.append({"run_id": run_id, "reason": reason})
    return tuple(validated)


@dataclass(frozen=True, slots=True)
class NativeAttributionCorpus:
    corpus_id: str
    protocol_id: str
    accepted_run_ids: tuple[str, ...]
    observations: tuple[NativeEpisodeObservation, ...]
    candidates: tuple[NativeAttributionCandidate, ...]
    excluded_runs: tuple[dict[str, object], ...] = ()
    schema: str = CORPUS_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != CORPUS_SCHEMA:
            raise ValueError("unsupported native attribution corpus schema")
        if not self.protocol_id or not self.accepted_run_ids:
            raise ValueError("native attribution corpus identity is incomplete")
        if len(self.accepted_run_ids) != len(set(self.accepted_run_ids)):
            raise ValueError("native attribution corpus runs must be unique")
        observations = tuple(self.observations)
        candidates = tuple(self.candidates)
        observation_ids = {value.observation_id for value in observations}
        candidate_ids = {value.observation_id for value in candidates}
        if not observation_ids or candidate_ids != observation_ids:
            raise ValueError("native attribution corpus observation/candidate coverage mismatch")
        if any(value.run_id not in self.accepted_run_ids for value in observations):
            raise ValueError("native attribution observation references an unaccepted run")
        _assert_content_free({
            "protocol_id": self.protocol_id,
            "accepted_run_ids": list(self.accepted_run_ids),
            "observations": [value.payload() for value in observations],
            "candidates": [value.payload() for value in candidates],
            "excluded_runs": [dict(value) for value in self.excluded_runs],
        })
        excluded = _validate_excluded_runs(self.excluded_runs)
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "excluded_runs", excluded)
        if self.corpus_id != "native-corpus." + _digest(self.identity_payload())[:40]:
            raise ValueError("native attribution corpus ID mismatch")

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "protocol_id": self.protocol_id,
            "accepted_run_ids": list(self.accepted_run_ids),
            "observations": [value.payload() for value in self.observations],
            "candidates": [value.payload() for value in self.candidates],
            "excluded_runs": list(self.excluded_runs),
        }

    def payload(self) -> dict[str, object]:
        return {"corpus_id": self.corpus_id, **self.identity_payload()}

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "NativeAttributionCorpus":
        if not isinstance(payload, Mapping):
            raise ValueError("native attribution corpus payload is malformed")
        expected = {
            "corpus_id", "schema", "protocol_id", "accepted_run_ids", "observations",
            "candidates", "excluded_runs",
        }
        if set(payload) != expected or payload.get("schema") != CORPUS_SCHEMA:
            raise ValueError("native attribution corpus payload fields are invalid")
        if any(not isinstance(payload[field], str) for field in ("corpus_id", "protocol_id", "schema")):
            raise ValueError("native attribution corpus scalar types are invalid")
        accepted = payload.get("accepted_run_ids")
        observations = payload.get("observations")
        candidates = payload.get("candidates")
        excluded = payload.get("excluded_runs")
        if (
            not isinstance(accepted, list)
            or any(not isinstance(value, str) for value in accepted)
            or not isinstance(observations, list)
            or not isinstance(candidates, list)
            or not isinstance(excluded, list)
            or any(not isinstance(value, Mapping) for value in excluded)
        ):
            raise ValueError("native attribution corpus payload collections are invalid")
        return cls(
            corpus_id=payload["corpus_id"],
            protocol_id=payload["protocol_id"],
            accepted_run_ids=tuple(accepted),
            observations=tuple(
                NativeEpisodeObservation.from_payload(value) for value in observations
            ),
            candidates=tuple(
                NativeAttributionCandidate.from_payload(value) for value in candidates
            ),
            excluded_runs=tuple(dict(value) for value in excluded),
            schema=payload["schema"],
        )

    @property
    def actionable_count(self) -> int:
        return sum(value.is_actionable for value in self.candidates)

    @property
    def unresolved_count(self) -> int:
        return sum(value.primary_failure_surface.value == "unresolved" for value in self.candidates)

    @classmethod
    def create(
        cls,
        *,
        protocol_id: str,
        accepted_run_ids: Sequence[str],
        observations: Sequence[NativeEpisodeObservation],
        candidates: Sequence[NativeAttributionCandidate],
        excluded_runs: Sequence[Mapping[str, object]] = (),
    ) -> "NativeAttributionCorpus":
        values = {
            "schema": CORPUS_SCHEMA,
            "protocol_id": protocol_id,
            "accepted_run_ids": sorted(set(accepted_run_ids)),
            "observations": [value.payload() for value in sorted(observations, key=lambda item: item.observation_id)],
            "candidates": [value.payload() for value in sorted(candidates, key=lambda item: item.attribution_id)],
            "excluded_runs": [dict(value) for value in excluded_runs],
        }
        return cls(
            corpus_id="native-corpus." + _digest(values)[:40],
            protocol_id=protocol_id,
            accepted_run_ids=tuple(values["accepted_run_ids"]),
            observations=tuple(sorted(observations, key=lambda item: item.observation_id)),
            candidates=tuple(sorted(candidates, key=lambda item: item.attribution_id)),
            excluded_runs=tuple(values["excluded_runs"]),
        )


class NativeAttributionCorpusStore:
    """Append-once JSON store for a frozen attribution corpus."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path).expanduser().resolve()

    def put(self, corpus: NativeAttributionCorpus) -> bool:
        serialized = _canonical(corpus.payload()) + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_name(self.path.name + ".lock")
        with lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if self.path.is_symlink() or lock_path.is_symlink():
                raise ValueError("native attribution corpus store cannot be symlinked")
            if self.path.exists():
                if self.path.read_text(encoding="utf-8") != serialized:
                    raise ValueError("native attribution corpus conflicts with existing corpus")
                return False
            temporary = self.path.with_name(f".{self.path.name}.tmp-{os.getpid()}")
            temporary.write_text(serialized, encoding="utf-8")
            temporary.replace(self.path)
            return True

    def load(self) -> NativeAttributionCorpus:
        if not self.path.is_file() or self.path.is_symlink():
            raise ValueError("native attribution corpus store is missing or symlinked")
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("native attribution corpus store is unreadable") from exc
        corpus = NativeAttributionCorpus.from_payload(payload)
        if self.path.read_text(encoding="utf-8") != _canonical(corpus.payload()) + "\n":
            raise ValueError("native attribution corpus store is not canonical")
        return corpus


__all__ = [
    "CORPUS_SCHEMA", "NativeAttributionCorpus", "NativeAttributionCorpusStore",
]
