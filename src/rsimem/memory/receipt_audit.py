"""Content-free audit of mutation receipts against current backend state."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass

from .contracts import MemoryKind, MemoryQuery
from .ingestion import InternalMemoryAction
from .receipts import (
    JsonMutationReceiptStore,
    MutationReceipt,
    MutationReceiptStatus,
    SemanticMutationWriter,
)
from .runtime import MemoryBackendRegistry


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class MutationReceiptAuditIssue:
    kind: str
    receipt_id: str | None = None
    mutation_id: str | None = None
    writer_identity: SemanticMutationWriter | None = None


@dataclass(frozen=True, slots=True)
class SemanticNamespaceState:
    namespace: str
    artifact_ids: tuple[str, ...]
    revisions: tuple[str | None, ...]

    def __post_init__(self) -> None:
        if not self.namespace.strip():
            raise ValueError("semantic namespace state requires a namespace")
        if len(self.artifact_ids) != len(self.revisions):
            raise ValueError("semantic namespace artifacts and revisions must align")
        if len(self.artifact_ids) != len(set(self.artifact_ids)):
            raise ValueError("semantic namespace artifact IDs must be unique")


@dataclass(frozen=True, slots=True)
class SemanticMutationAuditScope:
    backend: str
    namespaces: tuple[SemanticNamespaceState, ...]
    baseline_receipt_ids: tuple[str, ...]
    state_digest: str

    def __post_init__(self) -> None:
        if not self.backend.strip():
            raise ValueError("semantic mutation audit scope requires a backend")
        names = [state.namespace for state in self.namespaces]
        if len(names) != len(set(names)) or not names:
            raise ValueError("semantic mutation audit namespaces must be unique")
        if len(self.baseline_receipt_ids) != len(set(self.baseline_receipt_ids)):
            raise ValueError("semantic mutation audit receipt IDs must be unique")
        if len(self.state_digest) != 64:
            raise ValueError("semantic mutation audit state digest must be sha256")


@dataclass(frozen=True, slots=True)
class MutationReceiptAuditReport:
    ok: bool
    receipt_count: int
    status_counts: tuple[tuple[str, int], ...]
    writer_counts: tuple[tuple[str, int], ...]
    issues: tuple[MutationReceiptAuditIssue, ...]

    def observer_evidence(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "receipt_count": self.receipt_count,
            "status_counts": dict(self.status_counts),
            "writer_counts": dict(self.writer_counts),
            "issues": [
                {
                    "kind": issue.kind,
                    "receipt_id": issue.receipt_id,
                    "mutation_id": issue.mutation_id,
                    "writer_identity": (
                        issue.writer_identity.value
                        if issue.writer_identity is not None
                        else None
                    ),
                }
                for issue in self.issues
            ],
        }


def audit_mutation_receipts(
    store: JsonMutationReceiptStore,
    registry: MemoryBackendRegistry,
    *,
    semantic_scope: SemanticMutationAuditScope | None = None,
    allowed_writers: tuple[SemanticMutationWriter, ...] | None = None,
) -> MutationReceiptAuditReport:
    issues: list[MutationReceiptAuditIssue] = []
    try:
        receipts = store.all()
    except (OSError, ValueError):
        issue = MutationReceiptAuditIssue("receipt_store_corrupt")
        return MutationReceiptAuditReport(False, 0, (), (), (issue,))

    status_counts = Counter(receipt.status.value for receipt in receipts)
    writer_counts = Counter(
        receipt.writer_identity.value
        for receipt in receipts
        if receipt.writer_identity is not None
    )
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
    if semantic_scope is not None:
        _audit_semantic_scope(
            semantic_scope,
            receipts,
            registry,
            issues,
            allowed_writers=allowed_writers,
        )
    return MutationReceiptAuditReport(
        ok=not issues,
        receipt_count=len(receipts),
        status_counts=tuple(sorted(status_counts.items())),
        writer_counts=tuple(sorted(writer_counts.items())),
        issues=tuple(issues),
    )


def capture_semantic_mutation_audit_scope(
    store: JsonMutationReceiptStore,
    registry: MemoryBackendRegistry,
) -> SemanticMutationAuditScope:
    receipts = store.all()
    backend, namespaces, state_digest = _semantic_state(registry)
    return SemanticMutationAuditScope(
        backend=backend,
        namespaces=namespaces,
        baseline_receipt_ids=tuple(sorted(receipt.receipt_id for receipt in receipts)),
        state_digest=state_digest,
    )


def _semantic_state(
    registry: MemoryBackendRegistry,
) -> tuple[str, tuple[SemanticNamespaceState, ...], str]:
    backend = registry.resolve(MemoryKind.SEMANTIC)
    states = []
    for namespace in ("memory", "user"):
        artifacts = tuple(sorted(
            (
                hit.artifact
                for hit in backend.query(MemoryQuery(
                    MemoryKind.SEMANTIC,
                    "",
                    namespace=namespace,
                    limit=10_000,
                ))
            ),
            key=lambda artifact: artifact.artifact_id,
        ))
        states.append(SemanticNamespaceState(
            namespace,
            tuple(artifact.artifact_id for artifact in artifacts),
            tuple(artifact.revision for artifact in artifacts),
        ))
    payload = {
        "backend": backend.descriptor.name,
        "namespaces": [
            {
                "namespace": state.namespace,
                "artifact_ids": list(state.artifact_ids),
                "revisions": list(state.revisions),
            }
            for state in states
        ],
    }
    digest = hashlib.sha256(json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    return backend.descriptor.name, tuple(states), digest


def _audit_semantic_scope(
    scope: SemanticMutationAuditScope,
    receipts: tuple[MutationReceipt, ...],
    registry: MemoryBackendRegistry,
    issues: list[MutationReceiptAuditIssue],
    *,
    allowed_writers: tuple[SemanticMutationWriter, ...] | None,
) -> None:
    try:
        backend, final_states, _ = _semantic_state(registry)
    except Exception:
        issues.append(MutationReceiptAuditIssue("semantic_state_audit_failed"))
        return
    if backend != scope.backend:
        issues.append(MutationReceiptAuditIssue("semantic_backend_changed"))
        return

    baseline_ids = set(scope.baseline_receipt_ids)
    current_receipts = tuple(
        receipt
        for receipt in receipts
        if receipt.receipt_id not in baseline_ids
        and receipt.kind == MemoryKind.SEMANTIC
        and receipt.backend == scope.backend
    )
    allowed = (
        {SemanticMutationWriter(value) for value in allowed_writers}
        if allowed_writers is not None
        else None
    )
    for receipt in current_receipts:
        if (
            receipt.status == MutationReceiptStatus.COMMITTED
            and receipt.action != InternalMemoryAction.NONE
            and allowed is not None
            and receipt.writer_identity not in allowed
        ):
            _issue(
                issues,
                "disallowed_mutation_writer",
                receipt,
                writer_identity=receipt.writer_identity,
            )

    baseline_by_namespace = {state.namespace: state for state in scope.namespaces}
    final_by_namespace = {state.namespace: state for state in final_states}
    for namespace, baseline in baseline_by_namespace.items():
        final = final_by_namespace.get(namespace)
        if final is None:
            issues.append(MutationReceiptAuditIssue("semantic_namespace_missing"))
            continue
        committed = [
            receipt
            for receipt in current_receipts
            if receipt.namespace == namespace
            and receipt.status == MutationReceiptStatus.COMMITTED
            and receipt.action != InternalMemoryAction.NONE
        ]
        expected = tuple(sorted(baseline.artifact_ids))
        while committed:
            candidates = [
                receipt for receipt in committed
                if tuple(sorted(receipt.pre_artifact_ids)) == expected
            ]
            if len(candidates) != 1:
                receipt = candidates[0] if candidates else committed[0]
                _issue(issues, "semantic_receipt_chain_mismatch", receipt)
                break
            receipt = candidates[0]
            values = list(expected)
            if receipt.action == InternalMemoryAction.ADD:
                assert receipt.applied_artifact_id is not None
                values.append(receipt.applied_artifact_id)
            elif receipt.action == InternalMemoryAction.UPDATE:
                assert receipt.target_artifact_id is not None
                assert receipt.applied_artifact_id is not None
                if receipt.target_artifact_id not in values:
                    _issue(issues, "semantic_receipt_target_missing", receipt)
                    break
                values.remove(receipt.target_artifact_id)
                values.append(receipt.applied_artifact_id)
            elif receipt.action == InternalMemoryAction.DELETE:
                assert receipt.target_artifact_id is not None
                if receipt.target_artifact_id not in values:
                    _issue(issues, "semantic_receipt_target_missing", receipt)
                    break
                values.remove(receipt.target_artifact_id)
            expected = tuple(sorted(values))
            committed.remove(receipt)
        if tuple(sorted(final.artifact_ids)) != expected:
            issues.append(MutationReceiptAuditIssue(
                "semantic_state_changed_without_receipt",
                writer_identity=SemanticMutationWriter.NATIVE_HERMES,
            ))


def _issue(
    issues: list[MutationReceiptAuditIssue],
    kind: str,
    receipt: MutationReceipt,
    *,
    writer_identity: SemanticMutationWriter | None = None,
) -> None:
    value = MutationReceiptAuditIssue(
        kind,
        receipt.receipt_id,
        receipt.mutation_id,
        writer_identity,
    )
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
