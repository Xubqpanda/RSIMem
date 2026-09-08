"""Frozen B0/B1/B2 comparison contract for AdaMem Semantic RSI."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping

from .adamem_adapter import ADAMEM_UPSTREAM_COMMIT, AdaMemFeedbackView


SCHEMA = "rsimem-adamem-comparison-manifest-v1"
PROTOCOL_ID = "adamem-trajectory-baseline-v1"


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 256:
        raise ValueError(f"{name} must be a nonempty stable identifier")
    return value


class AdaMemCondition(StrEnum):
    MEM0_STATIC = "B0_mem0_static"
    ADAMEM_TERMINAL = "B1_mem0_adamem_terminal"
    ADAMEM_FULL_TRAJECTORY = "B2_mem0_adamem_full_trajectory"


@dataclass(frozen=True, slots=True)
class AdaMemRunSpec:
    run_id: str
    condition: AdaMemCondition
    replicate: int
    state_directory: str
    trace_directory: str
    artifact_directory: str
    mem0_collection: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "condition", AdaMemCondition(self.condition))
        _identifier(self.run_id, "AdaMem run ID")
        if type(self.replicate) is not int or self.replicate < 1:
            raise ValueError("AdaMem replicate must be positive")
        for value in (
            self.state_directory, self.trace_directory, self.artifact_directory,
            self.mem0_collection,
        ):
            _identifier(value, "AdaMem isolated run identity")

    def payload(self) -> dict[str, object]:
        return {
            "run_id": self.run_id, "condition": self.condition.value,
            "replicate": self.replicate, "state_directory": self.state_directory,
            "trace_directory": self.trace_directory, "artifact_directory": self.artifact_directory,
            "mem0_collection": self.mem0_collection,
        }


@dataclass(frozen=True, slots=True)
class AdaMemComparisonManifest:
    manifest_id: str
    family_id: str
    train_sequence_id: str
    n_plus_one_sequence_id: str
    fixture_digest: str
    base_model: str
    meta_agent_model: str
    temperature: float
    token_budget: int
    update_budget: int
    mem0_backend: str
    policy_update_space: str
    upstream_commit: str
    runs: tuple[AdaMemRunSpec, ...]
    schema: str = SCHEMA
    protocol_id: str = PROTOCOL_ID

    def __post_init__(self) -> None:
        if self.schema != SCHEMA or self.protocol_id != PROTOCOL_ID:
            raise ValueError("unsupported AdaMem comparison protocol")
        for value, name in (
            (self.family_id, "family ID"), (self.train_sequence_id, "train sequence"),
            (self.n_plus_one_sequence_id, "N+1 sequence"), (self.fixture_digest, "fixture digest"),
            (self.base_model, "base model"), (self.meta_agent_model, "meta-agent model"),
            (self.mem0_backend, "Mem0 backend"), (self.policy_update_space, "policy update space"),
            (self.upstream_commit, "AdaMem upstream commit"),
        ):
            _identifier(value, name)
        if type(self.temperature) is not float or self.temperature < 0:
            raise ValueError("AdaMem temperature is invalid")
        if type(self.token_budget) is not int or self.token_budget < 1:
            raise ValueError("AdaMem token budget is invalid")
        if type(self.update_budget) is not int or self.update_budget < 1:
            raise ValueError("AdaMem update budget is invalid")
        if self.policy_update_space != "versioned_semantic_extraction_policy_only":
            raise ValueError("AdaMem update space must be extraction-policy only")
        if self.upstream_commit != ADAMEM_UPSTREAM_COMMIT:
            raise ValueError("AdaMem manifest must bind the audited upstream commit")
        runs = tuple(self.runs)
        if not runs or len({run.run_id for run in runs}) != len(runs):
            raise ValueError("AdaMem manifest runs must be unique")
        expected = {(condition, replicate) for condition in AdaMemCondition for replicate in range(1, max(run.replicate for run in runs) + 1)}
        observed = {(run.condition, run.replicate) for run in runs}
        if observed != expected:
            raise ValueError("AdaMem manifest must contain every condition for every replicate")
        identities = [item for run in runs for item in (
            run.state_directory, run.trace_directory, run.artifact_directory, run.mem0_collection,
        )]
        if len(identities) != len(set(identities)):
            raise ValueError("AdaMem manifest isolation identities collide")
        if self.manifest_id != "adamem-manifest." + _digest(self.identity_payload())[:40]:
            raise ValueError("AdaMem manifest ID mismatch")
        object.__setattr__(self, "runs", runs)

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema, "protocol_id": self.protocol_id,
            "family_id": self.family_id, "train_sequence_id": self.train_sequence_id,
            "n_plus_one_sequence_id": self.n_plus_one_sequence_id,
            "fixture_digest": self.fixture_digest, "base_model": self.base_model,
            "meta_agent_model": self.meta_agent_model, "temperature": self.temperature,
            "token_budget": self.token_budget, "update_budget": self.update_budget,
            "mem0_backend": self.mem0_backend, "policy_update_space": self.policy_update_space,
            "upstream_commit": self.upstream_commit,
            "runs": [run.payload() for run in self.runs],
        }

    def payload(self) -> dict[str, object]:
        return {"manifest_id": self.manifest_id, **self.identity_payload()}

    @classmethod
    def create(
        cls,
        *, family_id: str, train_sequence_id: str, n_plus_one_sequence_id: str,
        fixture_digest: str, base_model: str, meta_agent_model: str,
        token_budget: int, update_budget: int, replicate_count: int = 1,
        temperature: float = 0.0, mem0_backend: str = "mem0-flat-hermes-v1",
    ) -> "AdaMemComparisonManifest":
        if type(replicate_count) is not int or replicate_count < 1:
            raise ValueError("AdaMem replicate count must be positive")
        provisional: list[AdaMemRunSpec] = []
        for condition in AdaMemCondition:
            for replicate in range(1, replicate_count + 1):
                root = f"runs/{condition.value}/replicate-{replicate:02d}"
                core = {"condition": condition.value, "replicate": replicate, "root": root}
                provisional.append(AdaMemRunSpec(
                    run_id="adamem-run." + _digest(core)[:40], condition=condition,
                    replicate=replicate, state_directory=root + "/state",
                    trace_directory=root + "/trace", artifact_directory=root + "/artifacts",
                    mem0_collection="adamem-mem0-" + _digest(core)[:24],
                ))
        values = {
            "schema": SCHEMA, "protocol_id": PROTOCOL_ID, "family_id": family_id,
            "train_sequence_id": train_sequence_id, "n_plus_one_sequence_id": n_plus_one_sequence_id,
            "fixture_digest": fixture_digest, "base_model": base_model,
            "meta_agent_model": meta_agent_model, "temperature": temperature,
            "token_budget": token_budget, "update_budget": update_budget,
            "mem0_backend": mem0_backend,
            "policy_update_space": "versioned_semantic_extraction_policy_only",
            "upstream_commit": ADAMEM_UPSTREAM_COMMIT,
            "runs": [run.payload() for run in provisional],
        }
        return cls(manifest_id="adamem-manifest." + _digest(values)[:40], runs=tuple(provisional), **{
            key: values[key] for key in values if key not in {"schema", "protocol_id", "runs"}
        })


def feedback_view_for_condition(condition: AdaMemCondition) -> AdaMemFeedbackView | None:
    condition = AdaMemCondition(condition)
    return {
        AdaMemCondition.MEM0_STATIC: None,
        AdaMemCondition.ADAMEM_TERMINAL: AdaMemFeedbackView.TERMINAL,
        AdaMemCondition.ADAMEM_FULL_TRAJECTORY: AdaMemFeedbackView.FULL_TRAJECTORY,
    }[condition]


__all__ = [
    "AdaMemComparisonManifest", "AdaMemCondition", "AdaMemRunSpec", "PROTOCOL_ID",
    "SCHEMA", "feedback_view_for_condition",
]
