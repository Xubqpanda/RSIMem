"""Content-free audit of mutation receipts against current backend state."""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass

from .contracts import MemoryQuery
from .ingestion import InternalMemoryAction
from .receipts import (
    JsonMutationReceiptStore,
    MutationReceipt,
    MutationReceiptStatus,
)
from .runtime import MemoryBackendRegistry


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class MutationReceiptAuditIssue:
    kind: str
    receipt_id: str | None = None
    mutation_id: str | None = None


@dataclass(frozen=True, slots=True)
class MutationReceiptAuditReport:
    ok: bool
    receipt_count: int
    status_counts: tuple[tuple[str, int], ...]
    issues: tuple[MutationReceiptAuditIssue, ...]

    def observer_evidence(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "receipt_count": self.receipt_count,
            "status_counts": dict(self.status_counts),
            "issues": [
                {
                    "kind": issue.kind,
                    "receipt_id": issue.receipt_id,
                    "mutation_id": issue.mutation_id,
                }
                for issue in self.issues
            ],
        }


def audit_mutation_receipts(
    store: JsonMutationReceiptStore,
    registry: MemoryBackendRegistry,
) -> MutationReceiptAuditReport:
    issues: list[MutationReceiptAuditIssue] = []
    try:
        receipts = store.all()
    except (OSError, ValueError):
        issue = MutationReceiptAuditIssue("receipt_store_corrupt")
        return MutationReceiptAuditReport(False, 0, (), (issue,))

    status_counts = Counter(receipt.status.value for receipt in receipts)
    receipt_ids = [receipt.receipt_id for receipt in receipts]
    mutation_ids = [receipt.mutation_id for receipt in receipts]
    if len(receipt_ids) != len(set(receipt_ids)):
        issues.append(MutationReceiptAuditIssue("duplicate_receipt_id"))
    if len(mutation_ids) != len(set(mutation_ids)):
        issues.append(MutationReceiptAuditIssue("duplicate_mutation_id"))

    committed = tuple(
        receipt for receipt in receipts
        if receipt.status == MutationReceiptStatus.COMMITTED
    )
    owned_ids = {
        receipt.applied_artifact_id
        for receipt in committed
        if receipt.action in {InternalMemoryAction.ADD, InternalMemoryAction.UPDATE}
        and receipt.applied_artifact_id is not None
    }
    retired_targets = {
        receipt.target_artifact_id
        for receipt in committed
        if receipt.action in {InternalMemoryAction.UPDATE, InternalMemoryAction.DELETE}
        and receipt.target_artifact_id is not None
    }
    for receipt in receipts:
        _audit_receipt(receipt, registry, issues, owned_ids, retired_targets)
    return MutationReceiptAuditReport(
        ok=not issues,
        receipt_count=len(receipts),
        status_counts=tuple(sorted(status_counts.items())),
        issues=tuple(issues),
    )


def _issue(
    issues: list[MutationReceiptAuditIssue],
    kind: str,
    receipt: MutationReceipt,
) -> None:
    value = MutationReceiptAuditIssue(kind, receipt.receipt_id, receipt.mutation_id)
    if value not in issues:
        issues.append(value)


def _audit_receipt(
    receipt: MutationReceipt,
    registry: MemoryBackendRegistry,
    issues: list[MutationReceiptAuditIssue],
    owned_ids: set[str],
    retired_targets: set[str],
) -> None:
    try:
        backend = registry.resolve(receipt.kind)
        artifacts = tuple(
            sorted(
                (
                    hit.artifact
                    for hit in backend.query(MemoryQuery(
                        receipt.kind,
                        "",
                        namespace=receipt.namespace,
                        limit=10_000,
                    ))
                ),
                key=lambda artifact: artifact.artifact_id,
            )
        )
    except Exception:
        _issue(issues, "backend_audit_failed", receipt)
        return

    by_id = {artifact.artifact_id: artifact for artifact in artifacts}
    current_ids = set(by_id)
    new_ids = current_ids - set(receipt.pre_artifact_ids)
    if receipt.status == MutationReceiptStatus.PENDING and receipt.target_blocked:
        _issue(issues, "blocked_pending_receipt", receipt)

    if receipt.status == MutationReceiptStatus.COMMITTED:
        if not receipt.verified:
            _issue(issues, "unverified_committed_receipt", receipt)
            return
        if receipt.action in {InternalMemoryAction.ADD, InternalMemoryAction.UPDATE}:
            artifact = by_id.get(receipt.applied_artifact_id or "")
            if artifact is None:
                if receipt.applied_artifact_id not in retired_targets:
                    _issue(issues, "missing_committed_artifact", receipt)
                return
            if artifact.kind != receipt.kind or artifact.namespace != receipt.namespace:
                _issue(issues, "committed_artifact_identity_mismatch", receipt)
            if _sha(artifact.content) != receipt.applied_content_digest:
                _issue(issues, "committed_artifact_digest_mismatch", receipt)
            if artifact.revision != receipt.applied_revision:
                _issue(issues, "committed_artifact_revision_mismatch", receipt)
            actual_bytes = len(artifact.content.encode("utf-8")) + sum(
                len(resource.content) for resource in artifact.resources
            )
            if actual_bytes != receipt.storage_bytes:
                _issue(issues, "committed_storage_bytes_mismatch", receipt)
        elif receipt.action == InternalMemoryAction.DELETE:
            if receipt.target_artifact_id in by_id:
                _issue(issues, "committed_delete_target_present", receipt)
        return

    if new_ids - owned_ids:
        _issue(issues, "orphan_artifact", receipt)
    if (
        receipt.action == InternalMemoryAction.DELETE
        and receipt.target_artifact_id not in by_id
        and receipt.status == MutationReceiptStatus.PENDING
    ):
        _issue(issues, "uncommitted_delete", receipt)
