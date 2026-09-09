"""Deterministic preparation contracts for the AdaMem PAST trajectory baseline.

This module does not execute PAST or grade a task.  It turns one frozen family
manifest into the only admissible causal ordering for an extraction-policy
update: prefix writes under P_n, then update/abstain, then later writes and
N+1 evaluation under P_n+1.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .adamem_adapter import AdaMemFeedbackView, AdaMemPolicy, AdaMemUpdateResult


SCHEMA = "rsimem-adamem-runtime-v1"
_FORBIDDEN_KEYS = {
    "answer", "gold_answer", "golden_feedback", "grader", "judge",
    "official_score", "score", "hidden_answer", "reference_answer",
    "future_evaluation", "task_score",
}


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _require_identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 256:
        raise ValueError(f"{name} must be non-empty text")
    return value


def _assert_content_safe(value: object) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            if normalized in _FORBIDDEN_KEYS:
                raise ValueError(f"pure-process feedback contains forbidden field: {normalized}")
            _assert_content_safe(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _assert_content_safe(child)


@dataclass(frozen=True, slots=True)
class AdaMemTrajectorySplit:
    """A frozen PAST family split with a post-update extraction opportunity."""

    family_id: str
    source_digest: str
    cutover_label: str
    prefix_episodes: tuple[dict[str, object], ...]
    suffix_episodes: tuple[dict[str, object], ...]
    require_post_update_learning: bool = True

    def __post_init__(self) -> None:
        _require_identifier(self.family_id, "AdaMem family ID")
        _require_identifier(self.cutover_label, "AdaMem cutover label")
        if len(self.source_digest) != 64:
            raise ValueError("AdaMem source digest is invalid")
        if not self.prefix_episodes or not self.suffix_episodes:
            raise ValueError("AdaMem trajectory split requires prefix and suffix episodes")
        if self.require_post_update_learning and not any(item.get("bucket") == "learn" for item in self.suffix_episodes):
            raise ValueError("AdaMem suffix needs a post-update learning episode")
        if not any(item.get("bucket") == "evaluation" for item in self.suffix_episodes):
            raise ValueError("AdaMem suffix needs an N+1 evaluation episode")

    @property
    def split_id(self) -> str:
        return "adamem-split." + _digest(self.payload())[:40]

    def payload(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "family_id": self.family_id,
            "source_digest": self.source_digest,
            "cutover_label": self.cutover_label,
            "require_post_update_learning": self.require_post_update_learning,
            "prefix_episodes": list(self.prefix_episodes),
            "suffix_episodes": list(self.suffix_episodes),
        }


def split_family_manifest(
    source: Mapping[str, object], *, cutover_label: str,
    require_post_update_learning: bool = True,
) -> AdaMemTrajectorySplit:
    """Split after an explicit prefix episode; never infer a causal cutover."""

    episodes = source.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("PAST sequence requires episodes")
    copied = tuple(copy.deepcopy(item) for item in episodes)
    if any(not isinstance(item, dict) for item in copied):
        raise ValueError("PAST sequence episode is invalid")
    matches = [index for index, item in enumerate(copied) if item.get("label") == cutover_label]
    if len(matches) != 1:
        raise ValueError("AdaMem cutover label must identify one episode")
    index = matches[0]
    if copied[index].get("bucket") != "learn":
        raise ValueError("AdaMem cutover must end on a learning episode")
    family_ids = {item.get("family_id") for item in copied}
    if len(family_ids) != 1 or not isinstance(next(iter(family_ids)), str):
        raise ValueError("AdaMem sequence must contain one family")
    # Controls neither create the train feedback nor define matched N+1.
    suffix = tuple(item for item in copied[index + 1:] if item.get("bucket") in {"learn", "evaluation"})
    return AdaMemTrajectorySplit(
        family_id=next(iter(family_ids)), source_digest=_digest(source),
        cutover_label=cutover_label, prefix_episodes=copied[:index + 1], suffix_episodes=suffix,
        require_post_update_learning=require_post_update_learning,
    )


def materialize_phase_manifest(
    source: Mapping[str, object], *, split: AdaMemTrajectorySplit, phase: str,
    initial_home_fixture_dir: str = "",
) -> dict[str, object]:
    """Create a PAST manifest with exactly the frozen phase episodes."""

    if _digest(source) != split.source_digest:
        raise ValueError("AdaMem source manifest differs from split")
    if phase not in {"prefix", "suffix"}:
        raise ValueError("AdaMem phase is invalid")
    document = copy.deepcopy(dict(source))
    hermes = document.get("hermes")
    if isinstance(hermes, dict):
        # Keep Luna tool requests compatible with the OpenAI-compatible chat
        # endpoint across all AdaMem phases and feedback conditions.
        hermes["reasoning_effort"] = "none"
    episodes = split.prefix_episodes if phase == "prefix" else split.suffix_episodes
    document["episodes"] = [copy.deepcopy(item) for item in episodes]
    for episode in document["episodes"]:
        if isinstance(episode, dict):
            # Formal trajectories own their full state; never import a shared
            # cold result from a different replicate/condition.
            episode["shared_cold_run"] = False
    document["name"] = f"{source.get('name', split.family_id)}_adamem_{phase}"
    if phase == "suffix":
        if not initial_home_fixture_dir:
            raise ValueError("AdaMem suffix requires isolated prefix state")
        first = document["episodes"][0]
        if not isinstance(first, dict):
            raise ValueError("AdaMem suffix episode is invalid")
        first["initial_home_fixture_dir"] = initial_home_fixture_dir
    return document


def build_pure_process_feedback(
    *, feedback_view: AdaMemFeedbackView, trace_paths: Sequence[Path],
    operation_paths: Sequence[Path] = (),
) -> dict[str, object]:
    """Collect only model-visible trace and memory-operation evidence.

    PAST's grader appends a ``grading_result`` event to the same trace.  It is
    intentionally skipped before any feedback object is constructed.
    """

    view = AdaMemFeedbackView(feedback_view)
    messages: list[dict[str, object]] = []
    tool_calls: list[dict[str, object]] = []
    visible_output: list[str] = []
    for path in trace_paths:
        for raw in path.read_text(encoding="utf-8").splitlines():
            event = json.loads(raw)
            if not isinstance(event, Mapping):
                raise ValueError("trace event must be an object")
            event_type = event.get("type")
            if event_type == "grading_result":
                continue
            if event_type == "message":
                message = event.get("message")
                if not isinstance(message, Mapping):
                    raise ValueError("trace message is invalid")
                role = message.get("role")
                content = message.get("content")
                if role not in {"user", "assistant", "tool"} or not isinstance(content, list):
                    raise ValueError("trace message is invalid")
                record = {"role": role, "content": copy.deepcopy(content)}
                messages.append(record)
                if role == "assistant":
                    visible_output.extend(
                        str(part["text"]) for part in content
                        if isinstance(part, Mapping) and part.get("type") == "text"
                        and isinstance(part.get("text"), str)
                    )
            elif event_type in {"runtime_request", "runtime_response"}:
                payload = event.get("payload")
                if not isinstance(payload, Mapping):
                    raise ValueError("runtime tool event is invalid")
                tool_calls.append({"type": event_type, "payload": copy.deepcopy(payload)})
    operations: list[dict[str, object]] = []
    for path in operation_paths:
        for raw in path.read_text(encoding="utf-8").splitlines():
            event = json.loads(raw)
            if not isinstance(event, Mapping) or event.get("evidenceKind") != "operation":
                continue
            payload = event.get("payload")
            if not isinstance(payload, Mapping):
                raise ValueError("memory operation event is invalid")
            operations.append(copy.deepcopy(dict(payload)))
    result: dict[str, object] = {"outcome": "train_prefix_completed"}
    if view == AdaMemFeedbackView.FULL_TRAJECTORY:
        result.update({
            "messages": messages,
            "tool_calls": tool_calls,
            "memory_operations": operations,
            "memory_state": {"operation_count": len(operations)},
            "visible_output": visible_output,
        })
    elif view == AdaMemFeedbackView.NATIVE:
        result = {"messages": messages}
    _assert_content_safe(result)
    return result


@dataclass(frozen=True, slots=True)
class AdaMemPolicyReceipt:
    split_id: str
    feedback_view: AdaMemFeedbackView | None
    parent_policy_version: str
    candidate_policy_version: str
    request_digest: str | None
    patch_digest: str | None
    outcome: str
    reason_code: str
    activation: str

    def __post_init__(self) -> None:
        _require_identifier(self.split_id, "AdaMem split ID")
        _require_identifier(self.parent_policy_version, "AdaMem parent policy version")
        _require_identifier(self.candidate_policy_version, "AdaMem candidate policy version")
        if self.feedback_view is not None:
            object.__setattr__(self, "feedback_view", AdaMemFeedbackView(self.feedback_view))
        if self.outcome not in {"updated", "no_update", "static"}:
            raise ValueError("AdaMem policy receipt outcome is invalid")
        if self.activation not in {"activated", "retained"}:
            raise ValueError("AdaMem policy receipt activation is invalid")

    @classmethod
    def from_update(cls, *, split: AdaMemTrajectorySplit, result: AdaMemUpdateResult) -> "AdaMemPolicyReceipt":
        return cls(
            split_id=split.split_id, feedback_view=result.feedback_view,
            parent_policy_version=result.parent_policy.version,
            candidate_policy_version=result.candidate_policy.version,
            request_digest=result.request_digest, patch_digest=result.patch_digest,
            outcome=result.outcome, reason_code=result.reason_code,
            activation="activated" if result.outcome == "updated" else "retained",
        )

    @classmethod
    def static(cls, *, split: AdaMemTrajectorySplit, policy: AdaMemPolicy) -> "AdaMemPolicyReceipt":
        return cls(split.split_id, None, policy.version, policy.version, None, None, "static", "updater_disabled", "retained")

    def payload(self) -> dict[str, object]:
        return {
            "schema": SCHEMA, "split_id": self.split_id,
            "feedback_view": self.feedback_view.value if self.feedback_view else None,
            "parent_policy_version": self.parent_policy_version,
            "candidate_policy_version": self.candidate_policy_version,
            "request_digest": self.request_digest, "patch_digest": self.patch_digest,
            "outcome": self.outcome, "reason_code": self.reason_code,
            "activation": self.activation,
        }


__all__ = [
    "AdaMemPolicyReceipt", "AdaMemTrajectorySplit", "SCHEMA", "build_pure_process_feedback",
    "materialize_phase_manifest", "split_family_manifest",
]
