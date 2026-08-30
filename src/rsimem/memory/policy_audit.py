"""Cross-ledger audit for host lifecycle and six-layer policy evidence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .policy_evidence import JsonPolicyDecisionLedger, PolicyDecisionEvidence
from .policy_contracts import FORMATION_LAYERS


@dataclass(frozen=True, slots=True)
class PolicyAuditReport:
    ok: bool
    event_count: int
    lineage_count: int
    layers: tuple[str, ...]
    errors: tuple[str, ...] = ()

    def payload(self) -> dict[str, object]:
        return {
            "schemaVersion": 1,
            "ok": self.ok,
            "eventCount": self.event_count,
            "lineageCount": self.lineage_count,
            "layers": list(self.layers),
            "errors": list(self.errors),
        }


def _read_lifecycle_events(value: Path | Iterable[dict[str, Any]] | None) -> tuple[dict[str, Any], ...]:
    if value is None:
        return ()
    if isinstance(value, Path):
        if not value.exists():
            raise ValueError("lifecycle evidence path does not exist")
        events: list[dict[str, Any]] = []
        for line_number, line in enumerate(value.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"malformed lifecycle evidence at line {line_number}") from exc
            if not isinstance(event, dict):
                raise ValueError("lifecycle evidence event must be an object")
            events.append(event)
        return tuple(events)
    return tuple(value)


def audit_policy_evidence(
    path: Path,
    *,
    run_id: str | None = None,
    variant: str | None = None,
    trace_id: str | None = None,
    episode_id: str | None = None,
    session_id: str | None = None,
    task_id: str | None = None,
    family_id: str | None = None,
    stage: str | None = None,
    lifecycle_events: Path | Iterable[dict[str, Any]] | None = None,
    required_layers: Iterable[str] = (),
) -> PolicyAuditReport:
    """Validate policy evidence identity and its lifecycle snapshot join.

    The function is deliberately content-free.  It checks only stable IDs,
    digests, policy decisions, and receipt references; source or memory text is
    never loaded into the report.
    """

    ledger = JsonPolicyDecisionLedger(
        path,
        variant=variant or "unscoped",
        trace_id=trace_id or "unscoped",
        family_id=family_id,
        stage=stage,
    )
    errors: list[str] = []
    evidence: list[PolicyDecisionEvidence] = []
    expected = {
        "run_id": run_id,
        "variant": variant,
        "trace_id": trace_id,
        "episode_id": episode_id,
        "session_id": session_id,
        "task_id": task_id,
        "family_id": family_id,
        "stage": stage,
    }
    try:
        for event in ledger.events:
            item = PolicyDecisionEvidence.from_payload(event)
            evidence.append(item)
            for field, expected_value in expected.items():
                if expected_value is not None and getattr(item, field) != expected_value:
                    errors.append(f"{item.event_id}: {field} does not match expected identity")
    except ValueError as exc:
        return PolicyAuditReport(False, 0, 0, (), (str(exc),))
    if not evidence:
        return PolicyAuditReport(False, 0, 0, (), ("policy evidence is empty",))

    lineage_groups: dict[str, list[PolicyDecisionEvidence]] = {}
    for item in evidence:
        lineage_groups.setdefault(item.lineage_id, []).append(item)
    required = set(required_layers)
    if required:
        observed = {item.layer for item in evidence}
        missing = sorted(required.difference(observed))
        if missing:
            errors.append("missing policy layers: " + ",".join(missing))

    lifecycle = _read_lifecycle_events(lifecycle_events)
    # Lifecycle evidence is joined independently of the ledger builder too.
    # Validate the complete host identity here so callers cannot bypass the
    # run/variant/trace/task/family/stage boundary by invoking this audit
    # helper directly.
    if lifecycle:
        owner = evidence[0]
        lifecycle_identity = {
            "runId": run_id if run_id is not None else owner.run_id,
            "variant": variant if variant is not None else owner.variant,
            "traceId": trace_id if trace_id is not None else owner.trace_id,
            "taskId": task_id if task_id is not None else owner.task_id,
            "familyId": family_id if family_id is not None else owner.family_id,
            "stage": stage if stage is not None else owner.stage,
        }
        for index, event in enumerate(lifecycle):
            for field, expected_value in lifecycle_identity.items():
                if field not in event or event.get(field) != expected_value:
                    errors.append(
                        f"lifecycle event {field} does not match policy evidence "
                        f"at index {index}"
                    )
    snapshot_ids = {
        str(event.get("snapshotId"))
        for event in lifecycle
        if event.get("kind") == "context_snapshot" and event.get("snapshotId")
    }
    if snapshot_ids:
        for item in evidence:
            if item.snapshot_id not in snapshot_ids and not item.snapshot_id.startswith("exposure.snapshot."):
                errors.append(f"{item.event_id}: snapshot is absent from lifecycle evidence")

    for lineage_id, items in lineage_groups.items():
        trigger_ids = {item.trigger_event_id for item in items if item.trigger_event_id}
        if len(trigger_ids) > 1:
            errors.append(f"{lineage_id}: decisions reference multiple trigger events")
        revisions = {item.source_revision for item in items}
        if len(revisions) > 1:
            errors.append(f"{lineage_id}: decisions reference multiple source revisions")

    layers = tuple(sorted({item.layer for item in evidence}))
    return PolicyAuditReport(not errors, len(evidence), len(lineage_groups), layers, tuple(errors))


def audit_policy_evidence_or_raise(*args: object, **kwargs: object) -> PolicyAuditReport:
    report = audit_policy_evidence(*args, **kwargs)  # type: ignore[arg-type]
    if not report.ok:
        raise ValueError("; ".join(report.errors))
    return report


__all__ = ["PolicyAuditReport", "audit_policy_evidence", "audit_policy_evidence_or_raise"]
