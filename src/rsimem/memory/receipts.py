"""Crash-safe mutation receipt state and persistent atomic reservation."""

from __future__ import annotations

import fcntl
import json
import os
import re
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from .contracts import MemoryKind
from .ingestion import InternalMemoryAction
from .validation import TrustedTargetBinding, ValidationProvenance


MUTATION_RECEIPT_SCHEMA_VERSION = 1
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$")
_REASON_CODE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


def _identifier(value: object) -> bool:
    return isinstance(value, str) and _IDENTIFIER.fullmatch(value) is not None


class MutationReceiptStatus(StrEnum):
    PENDING = "pending"
    COMMITTED = "committed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class MutationReceiptPhase(StrEnum):
    RESERVED = "reserved"
    APPLYING = "applying"
    APPLIED = "applied"
    VERIFYING = "verifying"
    VERIFIED = "verified"
    TERMINAL = "terminal"


@dataclass(frozen=True, slots=True)
class MutationReceipt:
    receipt_id: str
    idempotency_key: str
    mutation_id: str
    attempt: int
    backend: str
    lock_key: str
    target_artifact_id: str | None
    pre_revision: str | None
    pre_content_digest: str | None
    mutation_digest: str
    candidate_digest: str
    after_content_digest: str | None
    namespace: str
    action: InternalMemoryAction
    kind: MemoryKind
    provenance: ValidationProvenance
    status: MutationReceiptStatus = MutationReceiptStatus.PENDING
    phase: MutationReceiptPhase = MutationReceiptPhase.RESERVED
    applied_artifact_id: str | None = None
    applied_revision: str | None = None
    applied_content_digest: str | None = None
    verified: bool = False
    target_blocked: bool = False
    reason_code: str | None = None
    storage_bytes: int = 0
    store_revision: int = 0
    schema_version: int = MUTATION_RECEIPT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != MUTATION_RECEIPT_SCHEMA_VERSION:
            raise ValueError("unsupported mutation receipt schema version")
        object.__setattr__(self, "action", InternalMemoryAction(self.action))
        object.__setattr__(self, "kind", MemoryKind(self.kind))
        object.__setattr__(self, "status", MutationReceiptStatus(self.status))
        object.__setattr__(self, "phase", MutationReceiptPhase(self.phase))
        for value in (
            self.receipt_id,
            self.idempotency_key,
            self.mutation_id,
            self.backend,
            self.lock_key,
            self.namespace,
        ):
            if not _identifier(value):
                raise ValueError("mutation receipt identity is incomplete")
        for value in (
            self.target_artifact_id,
            self.pre_revision,
            self.applied_artifact_id,
            self.applied_revision,
        ):
            if value is not None and not _identifier(value):
                raise ValueError("mutation receipt target identity is invalid")
        for value in (
            self.pre_content_digest,
            self.mutation_digest,
            self.candidate_digest,
            self.after_content_digest,
            self.applied_content_digest,
        ):
            if value is not None and not _DIGEST.fullmatch(value):
                raise ValueError("mutation receipt content identity must be sha256")
        if (
            type(self.attempt) is not int
            or type(self.store_revision) is not int
            or type(self.storage_bytes) is not int
            or self.attempt < 1
            or self.store_revision < 0
            or self.storage_bytes < 0
        ):
            raise ValueError("mutation receipt counters must not be negative")
        if type(self.verified) is not bool or type(self.target_blocked) is not bool:
            raise TypeError("mutation receipt safety flags must be bool")
        if self.reason_code is not None and not _REASON_CODE.fullmatch(self.reason_code):
            raise ValueError("mutation receipt reason_code must be machine-readable")
        self._validate_action_shape()
        self._validate_state_shape()

    def _validate_action_shape(self) -> None:
        has_target = self.target_artifact_id is not None
        has_pre = self.pre_revision is not None and self.pre_content_digest is not None
        if self.action == InternalMemoryAction.ADD:
            if has_target or self.pre_revision is not None or self.pre_content_digest is not None:
                raise ValueError("ADD receipt cannot carry pre-existing target state")
            if self.after_content_digest is None:
                raise ValueError("ADD receipt requires after content digest")
        elif self.action == InternalMemoryAction.UPDATE:
            if not has_target or not has_pre or self.after_content_digest is None:
                raise ValueError("UPDATE receipt requires target, pre-state, and after digest")
        elif self.action == InternalMemoryAction.DELETE:
            if not has_target or not has_pre or self.after_content_digest is not None:
                raise ValueError("DELETE receipt requires only target and pre-state")
        elif any((
            has_target,
            self.pre_revision,
            self.pre_content_digest,
            self.after_content_digest,
        )):
            raise ValueError("NONE receipt cannot carry mutation state")

    def _validate_state_shape(self) -> None:
        if self.status == MutationReceiptStatus.PENDING:
            if self.phase == MutationReceiptPhase.TERMINAL:
                raise ValueError("pending receipt cannot be terminal")
            if self.verified != (self.phase == MutationReceiptPhase.VERIFIED):
                raise ValueError("pending receipt verified flag does not match phase")
            if self.reason_code is not None and not self.target_blocked:
                raise ValueError("pending receipt reason requires target_blocked")
            return
        if self.phase != MutationReceiptPhase.TERMINAL:
            raise ValueError("terminal receipt status requires terminal phase")
        if self.target_blocked:
            raise ValueError("terminal receipt cannot retain target block")
        if self.status == MutationReceiptStatus.COMMITTED:
            if not self.verified or self.reason_code is not None:
                raise ValueError("committed receipt must be verified without failure")
            if self.action in {InternalMemoryAction.ADD, InternalMemoryAction.UPDATE}:
                if not all((
                    self.applied_artifact_id,
                    self.applied_revision,
                    self.applied_content_digest,
                )):
                    raise ValueError("committed add/update receipt requires applied artifact state")
            elif self.action == InternalMemoryAction.DELETE:
                if self.applied_artifact_id != self.target_artifact_id:
                    raise ValueError("committed delete receipt must identify deleted target")
                if self.applied_revision is not None or self.applied_content_digest is not None:
                    raise ValueError("committed delete receipt cannot carry live artifact state")
            elif any((
                self.applied_artifact_id,
                self.applied_revision,
                self.applied_content_digest,
                self.storage_bytes,
            )):
                raise ValueError("committed NONE receipt cannot carry artifact state")
        elif self.verified:
            raise ValueError("failed/rolled-back receipt cannot be verified")
        elif self.reason_code is None:
            raise ValueError("failed/rolled-back receipt requires reason_code")

    def core_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "receipt_id": self.receipt_id,
            "idempotency_key": self.idempotency_key,
            "mutation_id": self.mutation_id,
            "attempt": self.attempt,
            "backend": self.backend,
            "lock_key": self.lock_key,
            "target_artifact_id": self.target_artifact_id,
            "pre_revision": self.pre_revision,
            "pre_content_digest": self.pre_content_digest,
            "mutation_digest": self.mutation_digest,
            "candidate_digest": self.candidate_digest,
            "after_content_digest": self.after_content_digest,
            "namespace": self.namespace,
            "action": self.action.value,
            "kind": self.kind.value,
            "provenance": _provenance_payload(self.provenance),
        }

    def to_dict(self) -> dict[str, object]:
        return {
            **self.core_payload(),
            "status": self.status.value,
            "phase": self.phase.value,
            "applied_artifact_id": self.applied_artifact_id,
            "applied_revision": self.applied_revision,
            "applied_content_digest": self.applied_content_digest,
            "verified": self.verified,
            "target_blocked": self.target_blocked,
            "reason_code": self.reason_code,
            "storage_bytes": self.storage_bytes,
            "store_revision": self.store_revision,
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> MutationReceipt:
        if set(value) != _RECEIPT_FIELDS:
            raise ValueError("malformed mutation receipt fields")
        provenance = value.get("provenance")
        if not isinstance(provenance, dict):
            raise ValueError("malformed mutation receipt provenance")
        try:
            return cls(
                receipt_id=value["receipt_id"],
                idempotency_key=value["idempotency_key"],
                mutation_id=value["mutation_id"],
                attempt=value["attempt"],
                backend=value["backend"],
                lock_key=value["lock_key"],
                target_artifact_id=value["target_artifact_id"],
                pre_revision=value["pre_revision"],
                pre_content_digest=value["pre_content_digest"],
                mutation_digest=value["mutation_digest"],
                candidate_digest=value["candidate_digest"],
                after_content_digest=value["after_content_digest"],
                namespace=value["namespace"],
                action=value["action"],
                kind=value["kind"],
                provenance=_provenance_from_payload(provenance),
                status=value["status"],
                phase=value["phase"],
                applied_artifact_id=value["applied_artifact_id"],
                applied_revision=value["applied_revision"],
                applied_content_digest=value["applied_content_digest"],
                verified=value["verified"],
                target_blocked=value["target_blocked"],
                reason_code=value["reason_code"],
                storage_bytes=value["storage_bytes"],
                store_revision=value["store_revision"],
                schema_version=value["schema_version"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("malformed mutation receipt value") from exc


_RECEIPT_FIELDS = set(MutationReceipt.__dataclass_fields__)


def _provenance_payload(value: ValidationProvenance) -> dict[str, object]:
    return {
        "schema_version": value.schema_version,
        "run_id": value.run_id,
        "episode_id": value.episode_id,
        "session_id": value.session_id,
        "task_id": value.task_id,
        "snapshot_id": value.snapshot_id,
        "execution_id": value.execution_id,
        "operation_id": value.operation_id,
        "source_digest": value.source_digest,
    }


def _provenance_from_payload(value: dict[str, object]) -> ValidationProvenance:
    expected = set(ValidationProvenance.__dataclass_fields__)
    if set(value) != expected:
        raise ValueError("malformed mutation receipt provenance fields")
    return ValidationProvenance(**value)


_ALLOWED_PENDING_PHASE_TRANSITIONS = {
    MutationReceiptPhase.RESERVED: {
        MutationReceiptPhase.APPLYING,
    },
    MutationReceiptPhase.APPLYING: {
        MutationReceiptPhase.APPLYING,
        MutationReceiptPhase.APPLIED,
    },
    MutationReceiptPhase.APPLIED: {
        MutationReceiptPhase.APPLIED,
        MutationReceiptPhase.VERIFYING,
    },
    MutationReceiptPhase.VERIFYING: {
        MutationReceiptPhase.VERIFYING,
        MutationReceiptPhase.VERIFIED,
    },
    MutationReceiptPhase.VERIFIED: {
        MutationReceiptPhase.VERIFIED,
    },
}


class JsonMutationReceiptStore:
    """Atomic JSON receipt store and durable target ownership resolver."""

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().resolve()

    @contextmanager
    def _lock(self, operation: int):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        with lock_path.open("w", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), operation)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _read_unlocked(self) -> dict[str, MutationReceipt]:
        if not self.path.exists():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError("malformed mutation receipt store JSON") from exc
        if not isinstance(payload, dict) or set(payload) != {"schema_version", "receipts"}:
            raise ValueError("malformed mutation receipt store envelope")
        if payload["schema_version"] != MUTATION_RECEIPT_SCHEMA_VERSION:
            raise ValueError("unsupported mutation receipt store schema")
        raw_receipts = payload["receipts"]
        if not isinstance(raw_receipts, dict):
            raise ValueError("mutation receipt store receipts must be an object")
        receipts: dict[str, MutationReceipt] = {}
        for key, value in raw_receipts.items():
            if not _identifier(key) or not isinstance(value, dict):
                raise ValueError("malformed mutation receipt store entry")
            receipt = MutationReceipt.from_dict(value)
            if receipt.idempotency_key != key:
                raise ValueError("mutation receipt key does not match payload")
            receipts[key] = receipt
        return receipts

    def _write_unlocked(self, receipts: dict[str, MutationReceipt]) -> None:
        payload = {
            "schema_version": MUTATION_RECEIPT_SCHEMA_VERSION,
            "receipts": {
                key: receipt.to_dict()
                for key, receipt in sorted(receipts.items())
            },
        }
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.",
            dir=self.path.parent,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=True, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, self.path)
            directory_fd = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except BaseException:
            try:
                os.unlink(temp_name)
            except OSError:
                pass
            raise

    def get(self, idempotency_key: str) -> MutationReceipt | None:
        with self._lock(fcntl.LOCK_SH):
            return self._read_unlocked().get(idempotency_key)

    def all(self) -> tuple[MutationReceipt, ...]:
        with self._lock(fcntl.LOCK_SH):
            receipts = self._read_unlocked()
        return tuple(receipts[key] for key in sorted(receipts))

    def reserve_pending(
        self,
        receipt: MutationReceipt,
    ) -> tuple[MutationReceipt, bool]:
        if receipt.status != MutationReceiptStatus.PENDING or receipt.phase != MutationReceiptPhase.RESERVED:
            raise ValueError("only a pending reserved receipt can be reserved")
        if receipt.store_revision != 0:
            raise ValueError("new mutation receipt store_revision must be zero")
        with self._lock(fcntl.LOCK_EX):
            receipts = self._read_unlocked()
            existing = receipts.get(receipt.idempotency_key)
            if existing is not None:
                if existing.core_payload() != receipt.core_payload():
                    raise ValueError("idempotency key has conflicting mutation receipt")
                return existing, False
            for other in receipts.values():
                if (
                    other.status == MutationReceiptStatus.PENDING
                    and other.lock_key == receipt.lock_key
                ):
                    raise ValueError("mutation target already has an active receipt")
            receipts[receipt.idempotency_key] = receipt
            self._write_unlocked(receipts)
            return receipt, True

    def transition(
        self,
        receipt: MutationReceipt,
        *,
        expected_store_revision: int,
    ) -> MutationReceipt:
        with self._lock(fcntl.LOCK_EX):
            receipts = self._read_unlocked()
            current = receipts.get(receipt.idempotency_key)
            if current is None:
                raise KeyError("unknown mutation receipt")
            if current.store_revision != expected_store_revision:
                raise ValueError("mutation receipt CAS revision conflict")
            if receipt.store_revision != current.store_revision + 1:
                raise ValueError("mutation receipt transition must increment store_revision")
            if current.core_payload() != receipt.core_payload():
                raise ValueError("mutation receipt immutable identity changed")
            self._validate_transition(current, receipt)
            receipts[receipt.idempotency_key] = receipt
            self._write_unlocked(receipts)
            return receipt

    @staticmethod
    def _validate_transition(current: MutationReceipt, updated: MutationReceipt) -> None:
        if current.status != MutationReceiptStatus.PENDING:
            raise ValueError("terminal mutation receipt cannot transition")
        if updated.status == MutationReceiptStatus.PENDING:
            allowed = _ALLOWED_PENDING_PHASE_TRANSITIONS[current.phase]
            if updated.phase not in allowed:
                raise ValueError("invalid pending mutation receipt phase transition")
            return
        if updated.status not in {
            MutationReceiptStatus.COMMITTED,
            MutationReceiptStatus.FAILED,
            MutationReceiptStatus.ROLLED_BACK,
        }:
            raise ValueError("unknown mutation receipt status transition")
        if updated.phase != MutationReceiptPhase.TERMINAL:
            raise ValueError("terminal receipt transition requires terminal phase")
        if updated.status == MutationReceiptStatus.COMMITTED:
            if current.phase != MutationReceiptPhase.VERIFIED:
                raise ValueError("only verified mutation receipt can commit")
        elif updated.status == MutationReceiptStatus.ROLLED_BACK:
            if current.phase not in {
                MutationReceiptPhase.RESERVED,
                MutationReceiptPhase.APPLYING,
                MutationReceiptPhase.APPLIED,
                MutationReceiptPhase.VERIFYING,
            }:
                raise ValueError("verified mutation receipt cannot roll back")

    def resolve(self, backend: str, artifact_id: str) -> TrustedTargetBinding | None:
        matches = [
            receipt
            for receipt in self.all()
            if receipt.status == MutationReceiptStatus.COMMITTED
            and receipt.verified
            and receipt.backend == backend
            and receipt.applied_artifact_id == artifact_id
            and receipt.action in {InternalMemoryAction.ADD, InternalMemoryAction.UPDATE}
        ]
        if not matches:
            return None
        bindings = {
            (
                receipt.applied_revision,
                receipt.applied_content_digest,
                receipt.kind,
                receipt.namespace,
                receipt.provenance.run_id,
            )
            for receipt in matches
        }
        if len(bindings) != 1:
            raise ValueError("conflicting committed target ownership receipts")
        revision, digest, kind, namespace, owner_run_id = bindings.pop()
        assert revision is not None and digest is not None
        return TrustedTargetBinding(
            backend=backend,
            artifact_id=artifact_id,
            revision=revision,
            kind=kind,
            namespace=namespace,
            content_digest=digest,
            owner_run_id=owner_run_id,
        )
