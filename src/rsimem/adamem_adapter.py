"""Host-neutral AdaMem policy contract for the Semantic RSI baseline.

This module intentionally ports only AdaMem's policy-state mechanism. Hermes
and PAST-Bench remain responsible for execution, storage, and evaluation.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Callable, Mapping, Sequence

from .memory.prompt_components import PromptBindingFingerprint, PromptComponentArtifact
from .memory_systems.mem0_flat.prompt_adapter import (
    MEM0_FLAT_EXTRACTION_SLOT,
    Mem0FlatPromptAdapter,
)
from .memory_systems.mem0_flat.prompts import POLICY_FACT_EXTRACTION_ROOT_BODY


ADAMEM_SCHEMA = "rsimem-adamem-policy-v1"
ADAMEM_UPSTREAM_COMMIT = "9dba25d"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$")
_FORBIDDEN_KEYS = {
    "answer", "gold_answer", "golden_feedback", "grader", "judge",
    "official_score", "score", "hidden_answer", "reference_answer",
    "future_evaluation", "task_score",
}


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{name} must be a stable identifier")
    return value


def _content_free(value: object) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            if normalized in _FORBIDDEN_KEYS:
                raise ValueError(f"AdaMem feedback contains forbidden field: {normalized}")
            _content_free(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _content_free(child)


class AdaMemFeedbackView(StrEnum):
    TERMINAL = "terminal"
    FULL_TRAJECTORY = "full_trajectory"
    NATIVE = "native"


@dataclass(frozen=True, slots=True)
class AdaMemPolicy:
    general_policy: str
    by_character: tuple[tuple[str, str], ...]
    version: str
    parent_version: str | None = None
    schema: str = ADAMEM_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != ADAMEM_SCHEMA:
            raise ValueError("unsupported AdaMem policy schema")
        _identifier(self.version, "AdaMem policy version")
        if self.parent_version is not None:
            _identifier(self.parent_version, "AdaMem parent policy version")
        if not isinstance(self.general_policy, str):
            raise ValueError("AdaMem general policy must be text")
        pairs = tuple(self.by_character)
        if any(
            not isinstance(name, str) or not name.strip()
            or not isinstance(rule, str)
            for name, rule in pairs
        ):
            raise ValueError("AdaMem character rules are invalid")
        if tuple(sorted(name for name, _ in pairs)) != tuple(name for name, _ in pairs):
            raise ValueError("AdaMem character rules must be sorted")
        if len({name for name, _ in pairs}) != len(pairs):
            raise ValueError("AdaMem character rules must be unique")
        object.__setattr__(self, "by_character", pairs)

    @classmethod
    def root(cls, *, version: str = "adamem-root-v1") -> "AdaMemPolicy":
        return cls(
            general_policy=(
                "Track relationships, stated preferences, commitments, recurring "
                "topics, and emotional dynamics; attribute facts to the person involved."
            ),
            by_character=(),
            version=version,
        )

    def policy_object(self) -> dict[str, object]:
        return {"general_policy": self.general_policy, "by_character": dict(self.by_character)}

    @property
    def digest(self) -> str:
        return _digest(self.payload())

    def payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "version": self.version,
            "parent_version": self.parent_version,
            "general_policy": self.general_policy,
            "by_character": {name: rule for name, rule in self.by_character},
        }


@dataclass(frozen=True, slots=True)
class AdaMemUpdateResult:
    parent_policy: AdaMemPolicy
    candidate_policy: AdaMemPolicy
    feedback_view: AdaMemFeedbackView
    feedback_digest: str
    request_digest: str
    patch_digest: str | None
    outcome: str
    reason_code: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "feedback_view", AdaMemFeedbackView(self.feedback_view))
        if self.outcome not in {"updated", "no_update"}:
            raise ValueError("AdaMem update outcome is invalid")
        _identifier(self.reason_code, "AdaMem update reason")
        for value in (self.feedback_digest, self.request_digest):
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError("AdaMem update digest is invalid")
        if self.patch_digest is not None and len(self.patch_digest) != 64:
            raise ValueError("AdaMem patch digest is invalid")
        if self.outcome == "updated" and self.candidate_policy.parent_version != self.parent_policy.version:
            raise ValueError("AdaMem update parent binding is invalid")
        if self.outcome == "no_update" and self.candidate_policy != self.parent_policy:
            raise ValueError("AdaMem no-update must retain parent policy")


@dataclass(frozen=True, slots=True)
class AdaMemMem0Binding:
    """The only permitted AdaMem effect on Mem0Static extraction."""

    policy_version: str
    policy_digest: str
    extraction_component: PromptComponentArtifact
    binding: PromptBindingFingerprint

    def __post_init__(self) -> None:
        _identifier(self.policy_version, "AdaMem binding policy version")
        if self.policy_digest != self.extraction_component.source_provenance:
            raise ValueError("AdaMem binding provenance differs from policy digest")
        if self.extraction_component.slot_id != MEM0_FLAT_EXTRACTION_SLOT.slot_id:
            raise ValueError("AdaMem binding uses an unexpected extraction slot")
        if self.binding.artifact_id != self.extraction_component.artifact_id:
            raise ValueError("AdaMem binding component identity differs")


def render_extraction_instructions(policy: AdaMemPolicy) -> str:
    """Render only the policy-controlled extraction preference boundary."""

    if policy.by_character:
        preferences = "\n".join(f"- {name}: {rule}" for name, rule in policy.by_character)
    else:
        preferences = "(none; use the default extraction behavior for each person.)"
    return (
        "User-specific memory extraction preferences:\n"
        f"{preferences}\n\n"
        "Treat every rule as an additional positive priority. Preserve relevant "
        "facts outside a listed preference when they concern the user's plans, "
        "commitments, schedule, decisions, or preferences."
    )


def bind_to_mem0_flat(policy: AdaMemPolicy) -> AdaMemMem0Binding:
    """Bind policy rendering to Mem0 extraction without changing retrieval."""

    adapter = Mem0FlatPromptAdapter()
    policy_digest = policy.digest
    component = PromptComponentArtifact.create(
        slot=MEM0_FLAT_EXTRACTION_SLOT,
        version=policy.version,
        policy_body=(
            POLICY_FACT_EXTRACTION_ROOT_BODY
            + "\n\nAdaMem extraction preferences:\n"
            + render_extraction_instructions(policy)
        ),
        source_provenance=policy_digest,
    )
    binding = adapter.bind(MEM0_FLAT_EXTRACTION_SLOT.slot_id, component)
    return AdaMemMem0Binding(policy.version, policy_digest, component, binding)


def build_feedback_request(
    *, policy: AdaMemPolicy, feedback_view: AdaMemFeedbackView, feedback: Mapping[str, object]
) -> dict[str, object]:
    _content_free(feedback)
    view = AdaMemFeedbackView(feedback_view)
    allowed = {
        AdaMemFeedbackView.TERMINAL: {"outcome", "user_feedback", "environment_feedback"},
        AdaMemFeedbackView.FULL_TRAJECTORY: {
            "outcome", "user_feedback", "environment_feedback", "messages",
            "tool_calls", "memory_operations", "memory_state", "visible_output",
        },
        AdaMemFeedbackView.NATIVE: {"messages", "user_feedback", "environment_feedback"},
    }[view]
    if set(feedback) - allowed:
        raise ValueError("AdaMem feedback view contains disallowed fields")
    return {
        "schema": "rsimem-adamem-feedback-request-v1",
        "upstream_commit": ADAMEM_UPSTREAM_COMMIT,
        "feedback_view": view.value,
        "policy": policy.policy_object(),
        "feedback": dict(feedback),
    }


def apply_policy_patch(parent: AdaMemPolicy, patch: object) -> AdaMemPolicy | None:
    """Apply AdaMem's incremental patch or reject malformed shape."""

    if not isinstance(patch, Mapping) or set(patch) - {"general_policy", "set", "remove"}:
        return None
    general = parent.general_policy
    if "general_policy" in patch:
        if not isinstance(patch["general_policy"], str):
            return None
        general = patch["general_policy"].strip()
    values = dict(parent.by_character)
    if "set" in patch:
        if not isinstance(patch["set"], Mapping):
            return None
        for name, rule in patch["set"].items():
            if not isinstance(name, str) or not name.strip() or not isinstance(rule, str):
                return None
            if rule.strip():
                values[name] = rule.strip()
            else:
                values.pop(name, None)
    if "remove" in patch:
        if not isinstance(patch["remove"], list) or any(not isinstance(name, str) for name in patch["remove"]):
            return None
        for name in patch["remove"]:
            values.pop(name, None)
    if not patch:
        return parent
    version = "adamem-policy." + _digest({"parent": parent.version, "patch": patch})[:24]
    return AdaMemPolicy(
        general_policy=general,
        by_character=tuple(sorted(values.items())),
        version=version,
        parent_version=parent.version,
    )


def update_policy(
    *,
    parent: AdaMemPolicy,
    feedback_view: AdaMemFeedbackView,
    feedback: Mapping[str, object],
    reflect: Callable[[Mapping[str, object]], str],
) -> AdaMemUpdateResult:
    request = build_feedback_request(policy=parent, feedback_view=feedback_view, feedback=feedback)
    feedback_digest = _digest(request["feedback"])
    request_digest = _digest(request)
    try:
        raw = reflect(request)
        patch = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return AdaMemUpdateResult(parent, parent, feedback_view, feedback_digest, request_digest, None, "no_update", "reflect_parse_failed")
    candidate = apply_policy_patch(parent, patch)
    if candidate is None:
        return AdaMemUpdateResult(parent, parent, feedback_view, feedback_digest, request_digest, _digest(patch), "no_update", "patch_shape_invalid")
    if candidate == parent:
        return AdaMemUpdateResult(parent, parent, feedback_view, feedback_digest, request_digest, _digest(patch), "no_update", "empty_patch")
    return AdaMemUpdateResult(parent, candidate, feedback_view, feedback_digest, request_digest, _digest(patch), "updated", "patch_applied")


__all__ = [
    "ADAMEM_SCHEMA", "ADAMEM_UPSTREAM_COMMIT", "AdaMemFeedbackView", "AdaMemPolicy",
    "AdaMemMem0Binding", "AdaMemUpdateResult", "apply_policy_patch", "bind_to_mem0_flat", "build_feedback_request",
    "render_extraction_instructions", "update_policy",
]
