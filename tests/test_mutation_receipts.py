from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from rsimem.memory import MemoryKind
from rsimem.memory.ingestion import InternalMemoryAction
from rsimem.memory.receipts import (
    MUTATION_RECEIPT_SCHEMA_VERSION,
    JsonMutationReceiptStore,
    MutationReceipt,
    MutationReceiptPhase,
    MutationReceiptStatus,
    SemanticMutationWriter,
)
from rsimem.memory.validation import ValidationProvenance


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _provenance(run_id="run-fixture") -> ValidationProvenance:
    return ValidationProvenance(
        run_id=run_id,
        episode_id="episode-learn",
        session_id="session-learn",
        task_id="SM01_preference_adoption",
        snapshot_id="snapshot-learn",
        execution_id="ingest.fixture",
        operation_id="operation.fixture",
        source_digest=_sha("source"),
    )


def _receipt(
    *,
    key="mutation-key.fixture",
    mutation_id="mutation.fixture",
    lock_key="lock.user.new-content",
    action=InternalMemoryAction.ADD,
    target=None,
    pre_revision=None,
    pre_digest=None,
    after_digest=None,
) -> MutationReceipt:
    if action == InternalMemoryAction.ADD:
        after_digest = after_digest or _sha("new content")
    elif action == InternalMemoryAction.UPDATE:
        target = target or "artifact.target"
        pre_revision = pre_revision or "revision-1"
        pre_digest = pre_digest or _sha("old content")
        after_digest = after_digest or _sha("new content")
    elif action == InternalMemoryAction.DELETE:
        target = target or "artifact.target"
        pre_revision = pre_revision or "revision-1"
        pre_digest = pre_digest or _sha("old content")
        after_digest = None
    else:
        target = pre_revision = pre_digest = after_digest = None
    return MutationReceipt(
        receipt_id=f"receipt.{mutation_id}",
        idempotency_key=key,
        mutation_id=mutation_id,
        attempt=1,
        backend="hermes-native-semantic",
        lock_key=lock_key,
        target_artifact_id=target,
        pre_revision=pre_revision,
        pre_content_digest=pre_digest,
        mutation_digest=_sha(f"mutation:{mutation_id}"),
        candidate_digest=_sha(f"candidate:{mutation_id}"),
        after_content_digest=after_digest,
        namespace="user",
        action=action,
        kind=MemoryKind.SEMANTIC,
        provenance=_provenance(),
    )


def _commit(store: JsonMutationReceiptStore, receipt: MutationReceipt) -> MutationReceipt:
    applying = store.transition(
        replace(receipt, phase=MutationReceiptPhase.APPLYING, store_revision=1),
        expected_store_revision=0,
    )
    applied = store.transition(
        replace(
            applying,
            phase=MutationReceiptPhase.APPLIED,
            applied_artifact_id="artifact.applied",
            applied_revision="1abc2345",
            applied_content_digest=applying.after_content_digest,
            writer_identity=SemanticMutationWriter.RSIMEM_EXECUTOR,
            storage_bytes=11,
            store_revision=2,
        ),
        expected_store_revision=1,
    )
    verifying = store.transition(
        replace(applied, phase=MutationReceiptPhase.VERIFYING, store_revision=3),
        expected_store_revision=2,
    )
    verified = store.transition(
        replace(
            verifying,
            phase=MutationReceiptPhase.VERIFIED,
            verified=True,
            store_revision=4,
        ),
        expected_store_revision=3,
    )
    return store.transition(
        replace(
            verified,
            status=MutationReceiptStatus.COMMITTED,
            phase=MutationReceiptPhase.TERMINAL,
            store_revision=5,
        ),
        expected_store_revision=4,
    )


def test_receipt_state_machine_and_durable_ownership_resolution(tmp_path) -> None:
    assert MUTATION_RECEIPT_SCHEMA_VERSION == 2
    path = tmp_path / "receipts.json"
    store = JsonMutationReceiptStore(path)
    pending = _receipt()
    receipt, reserved = store.reserve_pending(pending)
    assert reserved is True
    assert receipt.status == MutationReceiptStatus.PENDING
    assert receipt.phase == MutationReceiptPhase.RESERVED

    committed = _commit(store, receipt)
    assert committed.status == MutationReceiptStatus.COMMITTED
    assert committed.verified is True
    restarted = JsonMutationReceiptStore(path)
    assert restarted.get(pending.idempotency_key) == committed
    binding = restarted.resolve("hermes-native-semantic", "artifact.applied")
    assert binding is not None
    assert binding.revision == "1abc2345"
    assert binding.content_digest == pending.after_content_digest
    assert binding.owner_run_id == "run-fixture"


def test_two_concurrent_store_instances_only_reserve_once(tmp_path) -> None:
    path = tmp_path / "receipts.json"
    receipt = _receipt()

    def reserve():
        return JsonMutationReceiptStore(path).reserve_pending(receipt)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: reserve(), range(2)))
    assert sorted(reserved for _, reserved in results) == [False, True]
    assert all(value == receipt for value, _ in results)
    assert len(JsonMutationReceiptStore(path).all()) == 1


def test_idempotency_digest_and_target_conflicts_fail_closed(tmp_path) -> None:
    store = JsonMutationReceiptStore(tmp_path / "receipts.json")
    receipt = _receipt()
    store.reserve_pending(receipt)

    conflict = replace(receipt, mutation_digest=_sha("different"))
    with pytest.raises(ValueError, match="conflicting mutation receipt"):
        store.reserve_pending(conflict)

    target_conflict = _receipt(
        key="mutation-key.other",
        mutation_id="mutation.other",
        lock_key=receipt.lock_key,
    )
    with pytest.raises(ValueError, match="active receipt"):
        store.reserve_pending(target_conflict)


def test_transition_is_cas_and_immutable_identity_is_enforced(tmp_path) -> None:
    store = JsonMutationReceiptStore(tmp_path / "receipts.json")
    receipt, _ = store.reserve_pending(_receipt())
    applying = replace(receipt, phase=MutationReceiptPhase.APPLYING, store_revision=1)
    stored = store.transition(applying, expected_store_revision=0)
    assert stored.store_revision == 1

    with pytest.raises(ValueError, match="CAS revision conflict"):
        store.transition(replace(stored, store_revision=2), expected_store_revision=0)
    with pytest.raises(ValueError, match="immutable identity changed"):
        store.transition(
            replace(
                stored,
                mutation_digest=_sha("forged"),
                store_revision=2,
            ),
            expected_store_revision=1,
        )
    with pytest.raises(ValueError, match="only verified"):
        store.transition(
            replace(
                stored,
                status=MutationReceiptStatus.COMMITTED,
                phase=MutationReceiptPhase.TERMINAL,
                verified=True,
                applied_artifact_id="artifact.applied",
                applied_revision="revision-2",
                applied_content_digest=stored.after_content_digest,
                writer_identity=SemanticMutationWriter.RSIMEM_EXECUTOR,
                store_revision=2,
            ),
            expected_store_revision=1,
        )


@pytest.mark.parametrize(
    "payload",
    (
        "not json",
        "[]",
        '{"schema_version": 2, "receipts": []}',
        '{"schema_version": 99, "receipts": {}}',
        '{"schema_version": 1, "receipts": {"key": 7}}',
    ),
)
def test_malformed_and_unknown_receipt_store_fails_closed(tmp_path, payload) -> None:
    path = tmp_path / "receipts.json"
    path.write_text(payload, encoding="utf-8")
    store = JsonMutationReceiptStore(path)
    with pytest.raises(ValueError, match="mutation receipt|unsupported"):
        store.all()


def test_receipt_payload_corruption_and_schema_mismatch_fail_closed(tmp_path) -> None:
    path = tmp_path / "receipts.json"
    store = JsonMutationReceiptStore(path)
    receipt, _ = store.reserve_pending(_receipt())
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["receipts"][receipt.idempotency_key]["mutation_digest"] = "bad"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="malformed mutation receipt"):
        store.get(receipt.idempotency_key)

    with pytest.raises(ValueError, match="unsupported mutation receipt schema"):
        replace(receipt, schema_version=3)
    with pytest.raises(ValueError, match="counters"):
        replace(receipt, attempt=True)


def test_terminal_receipt_is_idempotent_and_cannot_transition(tmp_path) -> None:
    store = JsonMutationReceiptStore(tmp_path / "receipts.json")
    pending, _ = store.reserve_pending(_receipt())
    committed = _commit(store, pending)
    existing, reserved = store.reserve_pending(pending)
    assert existing == committed
    assert reserved is False
    with pytest.raises(ValueError, match="terminal mutation receipt"):
        store.transition(
            replace(committed, store_revision=committed.store_revision + 1),
            expected_store_revision=committed.store_revision,
        )
