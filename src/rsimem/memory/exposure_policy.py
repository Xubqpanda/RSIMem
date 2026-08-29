"""Deterministic memory exposure policy and injection receipt contract (2D.3)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Sequence

from .policy_contracts import (
    DecisionAction,
    ExecutionStatus,
    ExposureDecision,
    ExposureMode,
    PolicyArtifactIdentity,
    PolicyArtifactKind,
    PolicyLayer,
    SafetyBoundary,
    TriggerEvent,
    content_digest,
)


class InjectionReceiptStatus(StrEnum):
    COMMITTED = "committed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class InjectionReceipt:
    receipt_id: str
    decision_id: str
    context_revision: str
    artifact_ids: tuple[str, ...]
    render_fingerprint: str
    status: InjectionReceiptStatus = InjectionReceiptStatus.COMMITTED

    def __post_init__(self) -> None:
        for value, name in ((self.receipt_id, "injection receipt ID"), (self.decision_id, "exposure decision ID"), (self.context_revision, "context revision"), (self.render_fingerprint, "render fingerprint")):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must not be empty")
        values = tuple(self.artifact_ids)
        if len(values) != len(set(values)) or any(not item.strip() for item in values):
            raise ValueError("injection artifact IDs must be unique non-empty strings")
        object.__setattr__(self, "artifact_ids", values)
        object.__setattr__(self, "status", InjectionReceiptStatus(self.status))

    @classmethod
    def create(cls, decision: ExposureDecision, *, context_revision: str, render_fingerprint: str, status: InjectionReceiptStatus = InjectionReceiptStatus.COMMITTED) -> "InjectionReceipt":
        if decision.action != DecisionAction.RUN:
            raise ValueError("only run exposure decisions may produce injection receipts")
        if not context_revision.strip() or context_revision != decision.source_revision:
            raise ValueError("injection context revision does not match exposure decision")
        identity = {
            "decision_id": decision.decision_id,
            "context_revision": context_revision,
            "artifact_ids": list(decision.selected_artifact_ids),
            "render_fingerprint": render_fingerprint,
            "status": InjectionReceiptStatus(status).value,
        }
        return cls(
            receipt_id=f"injection-receipt.{content_digest(identity)[:40]}",
            decision_id=decision.decision_id,
            context_revision=context_revision,
            artifact_ids=decision.selected_artifact_ids,
            render_fingerprint=render_fingerprint,
            status=status,
        )


@dataclass(frozen=True, slots=True)
class ExposurePolicyConfig:
    policy_version: str = "fixed.exposure.parent.v1"
    mode: ExposureMode = ExposureMode.EAGER_SYSTEM_PROMPT
    max_artifacts: int | None = None
    max_tokens: int | None = None
    injection_position: str = "system.memory"

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", ExposureMode(self.mode))
        if not self.policy_version.strip() or not self.injection_position.strip():
            raise ValueError("exposure policy identity and position must not be empty")
        for value, name in ((self.max_artifacts, "max artifacts"), (self.max_tokens, "max tokens")):
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{name} must be positive")


class DeterministicExposurePolicy:
    layer = PolicyLayer.EXPOSURE

    def __init__(self, config: ExposurePolicyConfig | None = None) -> None:
        self.config = config or ExposurePolicyConfig()

    @property
    def artifact_identity(self) -> PolicyArtifactIdentity:
        return PolicyArtifactIdentity.create(
            policy_version=self.config.policy_version,
            kind=PolicyArtifactKind.FIXED,
            layers=(PolicyLayer.EXPOSURE,),
        )

    def decide(
        self,
        event: TriggerEvent,
        artifact_ids: Sequence[str],
        *,
        artifact_token_counts: Sequence[int] | None = None,
        budget_tokens: int | None = None,
        safety: SafetyBoundary | None = None,
    ) -> ExposureDecision:
        if budget_tokens is not None and (type(budget_tokens) is not int or budget_tokens < 0):
            raise ValueError("exposure budget must be non-negative")
        ids = tuple(artifact_ids)
        if len(ids) != len(set(ids)) or any(not item.strip() for item in ids):
            raise ValueError("artifact IDs must be unique non-empty strings")
        if artifact_token_counts is not None:
            token_counts = tuple(artifact_token_counts)
            if len(token_counts) != len(ids) or any(type(item) is not int or item < 0 for item in token_counts):
                raise ValueError("artifact token counts must match artifact IDs")
        else:
            token_counts = (0,) * len(ids)
        limit = budget_tokens if budget_tokens is not None else self.config.max_tokens
        if safety is not None and not safety.safe:
            return ExposureDecision.create(
                policy_version=self.config.policy_version,
                source_revision=event.source_revision,
                input_payload={
                    "event_id": event.event_id,
                    "artifact_ids": list(ids),
                    "safety_digest": safety.digest,
                },
                output_payload={
                    "selected": [],
                    "skipped": list(ids),
                    "reason": "safety_boundary_invalid",
                },
                action=DecisionAction.SKIP,
                execution_status=ExecutionStatus.SKIPPED,
                reason_codes=("safety_boundary_invalid",),
                lineage_id=f"lineage.{event.event_id}",
                trigger_event_id=event.event_id,
                exposure_mode=ExposureMode.NOT_EXPOSED,
                selected_artifact_ids=(),
                ordering=(),
                budget_tokens=limit,
            )
        selected: list[str] = []
        used = 0
        skipped: list[str] = []
        for index, artifact_id in enumerate(ids):
            if self.config.max_artifacts is not None and len(selected) >= self.config.max_artifacts:
                skipped.append(artifact_id)
                continue
            if limit is not None and used + token_counts[index] > limit:
                skipped.append(artifact_id)
                continue
            selected.append(artifact_id)
            used += token_counts[index]
        if not selected:
            return ExposureDecision.create(
                policy_version=self.config.policy_version,
                source_revision=event.source_revision,
                input_payload={"event_id": event.event_id, "artifact_ids": list(ids)},
                output_payload={"selected": [], "skipped": list(skipped or ids), "reason": "empty_memory" if not ids else "budget_exhausted"},
                action=DecisionAction.SKIP,
                execution_status=ExecutionStatus.SKIPPED,
                reason_codes=("empty_memory" if not ids else "budget_exhausted",),
                lineage_id=f"lineage.{event.event_id}",
                trigger_event_id=event.event_id,
                exposure_mode=ExposureMode.NOT_EXPOSED,
                selected_artifact_ids=(),
                ordering=(),
                budget_tokens=limit,
            )
        return ExposureDecision.create(
            policy_version=self.config.policy_version,
            source_revision=event.source_revision,
            input_payload={"event_id": event.event_id, "artifact_ids": list(ids), "token_counts": list(token_counts)},
            output_payload={"selected": selected, "skipped": skipped, "reason": "expose_eager"},
            action=DecisionAction.RUN,
            execution_status=ExecutionStatus.PENDING,
            reason_codes=("expose_eager",),
            lineage_id=f"lineage.{event.event_id}",
            trigger_event_id=event.event_id,
            exposure_mode=self.config.mode,
            selected_artifact_ids=tuple(selected),
            ordering=tuple(selected),
            injection_position=self.config.injection_position,
            budget_tokens=limit,
        )

    def decide_exposure(self, event: TriggerEvent, artifact_ids: Sequence[str], **kwargs: object) -> ExposureDecision:
        return self.decide(event, artifact_ids, **kwargs)  # type: ignore[arg-type]

    @staticmethod
    def bind_injection(decision: ExposureDecision, *, context_revision: str, render_fingerprint: str) -> InjectionReceipt:
        receipt = InjectionReceipt.create(decision, context_revision=context_revision, render_fingerprint=render_fingerprint)
        if tuple(receipt.artifact_ids) != tuple(decision.selected_artifact_ids):
            raise ValueError("injection receipt artifact join mismatch")
        return receipt


__all__ = [
    "InjectionReceiptStatus",
    "InjectionReceipt",
    "ExposurePolicyConfig",
    "DeterministicExposurePolicy",
]
