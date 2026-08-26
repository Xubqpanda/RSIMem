from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import threading

import pytest

from rsimem.lifecycle import (
    AllowlistedUpdateTargetResolver,
    ContextAction,
    ContextEvaluation,
    DeterministicPreferenceEvaluator,
    EvaluationSignal,
    EvaluationTrigger,
    IdempotencyReceipt,
    HermesMessage,
    HermesSnapshotCollector,
    JsonIdempotencyReceiptStore,
    PlanValidationStatus,
    RawResourceUsage,
    SegmentKind,
    TaskLifecycleState,
    WritebackAction,
    WritebackCoordinator,
    WritebackPlanValidator,
    run_sm01_preference_fixture,
    snapshot_to_evaluation_request,
)
from rsimem.memory import (
    MemoryAccessMode,
    MemoryArtifact,
    MemoryBackendDescriptor,
    MemoryBackendRegistry,
    MemoryKind,
    MemoryKindCapability,
    MemoryMutationResult,
)


class _UpdateBackend:
    def __init__(
        self,
        name: str,
        artifacts: tuple[MemoryArtifact, ...],
        *,
        updatable: bool = True,
    ) -> None:
        self.name = name
        self.artifacts = {artifact.artifact_id: artifact for artifact in artifacts}
        self.updatable = updatable

    @property
    def descriptor(self) -> MemoryBackendDescriptor:
        return MemoryBackendDescriptor(
            self.name,
            (MemoryKindCapability(
                MemoryKind.SEMANTIC,
                MemoryAccessMode.SEARCH,
                updatable=self.updatable,
            ),),
        )

    def get(self, artifact_id: str) -> MemoryArtifact | None:
        return self.artifacts.get(artifact_id)

    def query(self, query):
        return ()

    def mutate(self, mutation) -> MemoryMutationResult:
        return MemoryMutationResult(
            True,
            self.name,
            mutation.action,
            artifact_id=mutation.resolved_artifact_id,
        )

    def close(self) -> None:
        return None


def test_sm01_replay_has_stable_identity_and_content_free_events() -> None:
    first = run_sm01_preference_fixture()
    second = run_sm01_preference_fixture()

    assert [item.segment_id for item in first.snapshot.segments] == [
        item.segment_id for item in second.snapshot.segments
    ]
    assert first.snapshot.context_revision == second.snapshot.context_revision
    assert first.snapshot.provenance == second.snapshot.provenance
    assert first.plans[0].idempotency_key == second.plans[0].idempotency_key
    serialized = repr([event.to_dict() for event in first.events])
    assert "Use TSV with owner" not in serialized
    assert "current task is complete" not in serialized


def test_active_and_current_segments_cannot_be_evicted() -> None:
    result = run_sm01_preference_fixture()
    snapshot = result.snapshot
    request = snapshot_to_evaluation_request(snapshot, evaluation_id="unsafe")
    signals = tuple(
        replace(
            signal,
            context_action=ContextAction.EVICT,
            writeback_action=WritebackAction.ADD,
            memory_kind=MemoryKind.SEMANTIC,
        )
        for signal in result.evaluation.signals
    )
    evaluation = ContextEvaluation(
        evaluation_id=request.evaluation_id,
        evaluator="unsafe",
        trigger=EvaluationTrigger.TASK_COMPLETED,
        signals=signals,
    )
    coordinator = WritebackCoordinator()
    assert coordinator.create_plans(snapshot, evaluation) == ()


def test_tool_call_and_result_are_one_writeback_unit() -> None:
    collector = HermesSnapshotCollector()
    snapshot = collector.collect(
        (
            HermesMessage(
                "call", "assistant", "call search", "turn-1", 2,
                kind=SegmentKind.TOOL_CALL, tool_call_id="tool-1", completed=True,
            ),
            HermesMessage(
                "result", "tool", "result", "turn-1", 1,
                kind=SegmentKind.TOOL_RESULT, tool_call_id="tool-1", completed=True,
            ),
        ),
        run_id="run", episode_id="episode", session_id="session", task_id="task",
        current_turn_id=None, task_state=TaskLifecycleState.COMPLETED,
        lifecycle_state="task_completed", source_ref="fixture:tool",
    )
    request = snapshot_to_evaluation_request(snapshot, evaluation_id="evaluation")
    signals = tuple(
        EvaluationSignal(
            segment_id=segment.segment_id,
            context_action=ContextAction.EVICT,
            writeback_action=WritebackAction.ADD,
            memory_kind=MemoryKind.SEMANTIC,
            utility_estimate=1.0,
            confidence=1.0,
        )
        for segment in request.segments
    )
    evaluation = ContextEvaluation(
        "evaluation", "fixture", EvaluationTrigger.TASK_COMPLETED, signals,
    )
    plan = WritebackCoordinator().create_plans(snapshot, evaluation)
    assert len(plan) == 1
    assert set(plan[0].source_segment_ids) == {
        segment.segment_id for segment in snapshot.segments
    }


def test_changed_snapshot_marks_old_plan_stale() -> None:
    result = run_sm01_preference_fixture()
    plan = result.plans[0]
    changed = HermesSnapshotCollector().collect(
        (
            HermesMessage(
                "sm01-learn-preference", "user", "Use CSV instead.", "learn-turn", 4,
                completed=True,
            ),
            HermesMessage(
                "sm01-preference-ack", "assistant", "Acknowledged.", "learn-turn", 2,
                completed=True,
            ),
            HermesMessage(
                "sm01-task-complete", "assistant", "The current task is complete.",
                "completion-turn", 6, completed=True,
            ),
        ),
        run_id="run-sm01-fixture", episode_id="episode-sm01-learn",
        session_id="session-sm01-learn", task_id="SM01_preference_adoption",
        current_turn_id="completion-turn", task_state=TaskLifecycleState.COMPLETED,
        lifecycle_state="task_completed", source_ref="fixture:sm01_preference_adoption:v1",
        active_message_ids=("sm01-task-complete",),
    )
    receipt = WritebackCoordinator().dry_run(plan, changed)
    assert changed.snapshot_id == result.snapshot.snapshot_id
    assert changed.context_revision != result.snapshot.context_revision
    assert receipt.validation.status == PlanValidationStatus.STALE
    assert receipt.status.value == "stale"


def test_retrying_plan_is_idempotent() -> None:
    result = run_sm01_preference_fixture()
    coordinator = WritebackCoordinator()
    first = coordinator.dry_run(result.plans[0], result.snapshot)
    second = coordinator.dry_run(result.plans[0], result.snapshot)
    assert first.status.value == "accepted"
    assert second.status.value == "duplicate"
    assert first.mutation_id == second.mutation_id
    assert len(coordinator.dry_run_mutations) == 1


def test_incomplete_evaluation_creates_no_plan() -> None:
    result = run_sm01_preference_fixture()
    incomplete = replace(result.evaluation, signals=result.evaluation.signals[:1])
    coordinator = WritebackCoordinator()
    assert coordinator.create_plans(result.snapshot, incomplete) == ()


def _update_evaluation(
    *,
    compiler_version: str = "deterministic-compiler-v2",
    update_hints: tuple[str, ...] = ("replace_preference",),
) -> tuple[object, ContextEvaluation]:
    fixture = run_sm01_preference_fixture()
    preference = fixture.evaluation.signals[0]
    update = replace(
        preference,
        writeback_action=WritebackAction.UPDATE,
        target_backend=None,
        target_artifact_id=None,
        expected_memory_revision=None,
        update_hints=update_hints,
        update_mode="replace",
        compiler_version=compiler_version,
    )
    return fixture, replace(
        fixture.evaluation,
        evaluation_id="evaluation-update",
        signals=(update, *fixture.evaluation.signals[1:]),
    )


def _target_resolver(
    artifact_id: str,
    *,
    expected_revision: str = "memory-revision-7",
    backend: str = "hermes-native-semantic",
    candidates: tuple[str, ...] | None = None,
    artifacts: tuple[MemoryArtifact, ...] | None = None,
    updatable: bool = True,
    allowed_backends: dict[str, frozenset[MemoryKind]] | None = None,
) -> AllowlistedUpdateTargetResolver:
    stored_artifacts = artifacts or (MemoryArtifact(
        artifact_id=artifact_id,
        kind=MemoryKind.SEMANTIC,
        content="Stored preference.",
        revision=expected_revision,
    ),)
    registry = MemoryBackendRegistry()
    registry.register(_UpdateBackend(
        backend,
        stored_artifacts,
        updatable=updatable,
    ))
    return AllowlistedUpdateTargetResolver(
        registry,
        lambda snapshot, signal: candidates or (artifact_id,),
        allowed_backends=(
            {backend: frozenset({MemoryKind.SEMANTIC})}
            if allowed_backends is None
            else allowed_backends
        ),
    )


def test_update_plan_requires_target_revision_and_backend_capability() -> None:
    fixture, evaluation = _update_evaluation()
    validator = WritebackPlanValidator(updatable_backends={
        "hermes-native-semantic": frozenset({MemoryKind.SEMANTIC}),
    })
    plan = WritebackCoordinator(
        validator=validator,
        target_resolver=_target_resolver("artifact-a"),
    ).create_plans(
        fixture.snapshot,
        evaluation,
    )[0]
    assert plan.target_backend == "hermes-native-semantic"
    assert plan.target_artifact_id == "artifact-a"
    assert plan.expected_memory_revision == "memory-revision-7"
    assert plan.update_mode == "replace"
    assert plan.compiler_version == "deterministic-compiler-v2"
    assert plan.exit_evidence.safe_to_evict is True
    assert plan.exit_evidence.update_hints == ("replace_preference",)
    assert plan.exit_evidence.scope == evaluation.signals[0].scope
    assert plan.exit_evidence.temporal_validity == evaluation.signals[0].temporal_validity
    assert plan.exit_evidence.provenance[-1] == plan.source_segment_ids[0]

    rejected_events = []

    class Observer:
        def record(self, event):
            rejected_events.append(event)

    assert WritebackCoordinator(
        target_resolver=_target_resolver("artifact-a"),
        observers=(Observer(),),
    ).create_plans(
        fixture.snapshot,
        evaluation,
    ) == ()
    assert rejected_events[-1].reason_codes == ("backend_update_not_supported",)


@pytest.mark.parametrize(
    ("resolver", "reason"),
    [
        (
            _target_resolver(
                "artifact-a",
                candidates=("made-up-artifact",),
            ),
            "fabricated artifact",
        ),
        (
            _target_resolver(
                "artifact-a",
                artifacts=(MemoryArtifact(
                    "artifact-a",
                    MemoryKind.SEMANTIC,
                    "Stored preference.",
                ),),
            ),
            "revisionless artifact",
        ),
        (
            _target_resolver(
                "artifact-a",
                candidates=([], "artifact-a"),
                artifacts=(MemoryArtifact(
                    "artifact-a",
                    MemoryKind.SEMANTIC,
                    "Stored preference.",
                    revision=7,
                ),),
            ),
            "malformed candidate and revision types",
        ),
        (
            _target_resolver("artifact-a", updatable=False),
            "read-only backend",
        ),
        (
            _target_resolver("artifact-a", allowed_backends={}),
            "backend outside allowlist",
        ),
        (
            _target_resolver(
                "artifact-a",
                candidates=("artifact-a", "artifact-b"),
                artifacts=(
                    MemoryArtifact(
                        "artifact-a",
                        MemoryKind.SEMANTIC,
                        "Stored preference A.",
                        revision="revision-a",
                    ),
                    MemoryArtifact(
                        "artifact-b",
                        MemoryKind.SEMANTIC,
                        "Stored preference B.",
                        revision="revision-b",
                    ),
                ),
            ),
            "ambiguous artifacts",
        ),
    ],
)
def test_update_target_resolver_rejects_untrusted_candidates(
    resolver: AllowlistedUpdateTargetResolver,
    reason: str,
) -> None:
    fixture, evaluation = _update_evaluation()
    coordinator = WritebackCoordinator(
        validator=WritebackPlanValidator(updatable_backends={
            "hermes-native-semantic": frozenset({MemoryKind.SEMANTIC}),
        }),
        target_resolver=resolver,
    )

    assert coordinator.create_plans(fixture.snapshot, evaluation) == (), reason


def test_update_idempotency_distinguishes_target_artifacts() -> None:
    fixture, evaluation = _update_evaluation()
    validator = WritebackPlanValidator(updatable_backends={
        "hermes-native-semantic": frozenset({MemoryKind.SEMANTIC}),
    })
    def plan(
        artifact_id: str,
        *,
        expected_revision: str = "memory-revision-7",
        current_evaluation: ContextEvaluation = evaluation,
    ):
        return WritebackCoordinator(
            validator=validator,
            target_resolver=_target_resolver(
                artifact_id,
                expected_revision=expected_revision,
            ),
        ).create_plans(fixture.snapshot, current_evaluation)[0]

    first = plan("artifact-a")
    second = plan("artifact-b")
    assert first.idempotency_key != second.idempotency_key

    revision_plan = plan("artifact-a", expected_revision="memory-revision-8")
    assert first.idempotency_key != revision_plan.idempotency_key

    _, compiler_evaluation = _update_evaluation(
        compiler_version="deterministic-compiler-v3",
    )
    compiler_plan = plan("artifact-a", current_evaluation=compiler_evaluation)
    assert first.idempotency_key != compiler_plan.idempotency_key

    _, hints_a = _update_evaluation(update_hints=("same_first", "second_a"))
    _, hints_b = _update_evaluation(update_hints=("same_first", "second_b"))
    assert plan("artifact-a", current_evaluation=hints_a).idempotency_key != plan(
        "artifact-a",
        current_evaluation=hints_b,
    ).idempotency_key


@pytest.mark.parametrize(
    "changes",
    [
        {"reusable_facts": ("changed preference",)},
        {"reusable_procedures": ("changed procedure",)},
        {"completion_status": "blocked"},
        {"completion_evidence": ("different completion evidence",)},
        {"scope": "global"},
        {"temporal_validity": "current"},
    ],
)
def test_idempotency_covers_compiler_relevant_exit_evidence(
    changes: dict[str, object],
) -> None:
    fixture = run_sm01_preference_fixture()
    baseline = fixture.plans[0]
    changed_signal = replace(fixture.evaluation.signals[0], **changes)
    changed_evaluation = replace(
        fixture.evaluation,
        signals=(changed_signal, *fixture.evaluation.signals[1:]),
    )
    changed = WritebackCoordinator().create_plans(
        fixture.snapshot,
        changed_evaluation,
    )[0]

    assert baseline.idempotency_key != changed.idempotency_key


def test_idempotency_is_stable_across_equivalent_reevaluation() -> None:
    fixture = run_sm01_preference_fixture()
    reevaluation = replace(fixture.evaluation, evaluation_id="evaluation-retry")
    coordinator = WritebackCoordinator()
    original_plan = coordinator.create_plans(
        fixture.snapshot,
        fixture.evaluation,
    )[0]
    retry_plan = coordinator.create_plans(
        fixture.snapshot,
        reevaluation,
    )[0]
    original_receipt = coordinator.dry_run(original_plan, fixture.snapshot)
    retry_receipt = coordinator.dry_run(retry_plan, fixture.snapshot)

    assert retry_plan.evaluation_id != original_plan.evaluation_id
    assert retry_plan.exit_evidence.provenance != original_plan.exit_evidence.provenance
    assert retry_plan.idempotency_key == original_plan.idempotency_key
    assert retry_plan.plan_id == original_plan.plan_id
    assert original_receipt.status.value == "accepted"
    assert retry_receipt.status.value == "duplicate"
    assert retry_receipt.mutation_id == original_receipt.mutation_id


def test_idempotency_covers_resolved_unresolved_state() -> None:
    fixture = run_sm01_preference_fixture()
    retained_add = replace(
        fixture.evaluation.signals[0],
        context_action=ContextAction.RETAIN,
    )
    evaluation = replace(
        fixture.evaluation,
        signals=(retained_add, *fixture.evaluation.signals[1:]),
    )
    completed_plan = WritebackCoordinator().create_plans(
        fixture.snapshot,
        evaluation,
    )[0]
    unresolved_snapshot = replace(
        fixture.snapshot,
        segments=(
            replace(fixture.snapshot.segments[0], completed=False),
            *fixture.snapshot.segments[1:],
        ),
    )
    unresolved_plan = WritebackCoordinator().create_plans(
        unresolved_snapshot,
        evaluation,
    )[0]

    assert completed_plan.exit_evidence.unresolved_state is None
    assert unresolved_plan.exit_evidence.unresolved_state == "host_unresolved"
    assert completed_plan.idempotency_key != unresolved_plan.idempotency_key


def test_exit_evidence_requires_a_real_boolean_safety_decision() -> None:
    evidence = run_sm01_preference_fixture().plans[0].exit_evidence

    with pytest.raises(TypeError, match="safe_to_evict must be bool"):
        replace(evidence, safe_to_evict="false")


def test_add_and_discard_reject_existing_memory_targets() -> None:
    fixture = run_sm01_preference_fixture()
    add = fixture.evaluation.signals[0]
    with pytest.raises(ValueError, match="add signals"):
        replace(add, target_artifact_id="existing")

    with pytest.raises(ValueError, match="defer/discard"):
        replace(
            add,
            writeback_action=WritebackAction.DISCARD,
            memory_kind=None,
            target_backend="hermes-native-semantic",
        )

    with pytest.raises(ValueError, match="update signals require"):
        replace(add, writeback_action=WritebackAction.UPDATE)

    plan = fixture.plans[0]
    with pytest.raises(ValueError, match="add plans"):
        replace(plan, target_backend="hermes-native-semantic")
    with pytest.raises(ValueError, match="discard plans"):
        replace(
            plan,
            memory_action="discard",
            memory_kind=None,
            target_artifact_id="existing",
        )
    with pytest.raises(ValueError, match="update plans require"):
        replace(plan, memory_action="update")


def test_persistent_idempotency_receipt_survives_coordinator_restart(
    tmp_path,
) -> None:
    fixture, evaluation = _update_evaluation()
    receipt_path = tmp_path / "idempotency-receipts.json"

    def coordinator() -> WritebackCoordinator:
        return WritebackCoordinator(
            validator=WritebackPlanValidator(updatable_backends={
                "hermes-native-semantic": frozenset({MemoryKind.SEMANTIC}),
            }),
            target_resolver=_target_resolver("artifact-a"),
            receipt_store=JsonIdempotencyReceiptStore(receipt_path),
        )

    first_coordinator = coordinator()
    plan = first_coordinator.create_plans(fixture.snapshot, evaluation)[0]
    first = first_coordinator.dry_run(plan, fixture.snapshot)
    second_coordinator = coordinator()
    reevaluation = replace(evaluation, evaluation_id="evaluation-update-retry")
    retry_plan = second_coordinator.create_plans(
        fixture.snapshot,
        reevaluation,
    )[0]
    second = second_coordinator.dry_run(retry_plan, fixture.snapshot)

    assert retry_plan.evaluation_id != plan.evaluation_id
    assert retry_plan.plan_id == plan.plan_id
    assert first.status.value == "accepted"
    assert second.status.value == "duplicate"
    assert first.mutation_id == second.mutation_id
    serialized = receipt_path.read_text(encoding="utf-8")
    assert "Use TSV" not in serialized


@pytest.mark.parametrize(
    "payload",
    [
        '{"idem_bad": "not-an-object"}',
        '{"idem_bad": {"plan_id": "plan-only"}}',
        '{"idem_bad": {"plan_id": 7, "mutation_id": "mutation"}}',
    ],
)
def test_malformed_idempotency_receipt_fails_closed(tmp_path, payload: str) -> None:
    receipt_path = tmp_path / "idempotency-receipts.json"
    receipt_path.write_text(payload, encoding="utf-8")
    store = JsonIdempotencyReceiptStore(receipt_path)
    with pytest.raises(ValueError, match="malformed idempotency receipt"):
        store.get("idem_bad")


def test_json_receipt_reservation_is_atomic_under_concurrency(tmp_path) -> None:
    receipt_path = tmp_path / "idempotency-receipts.json"
    receipt = IdempotencyReceipt("idem-concurrent", "plan-a", "mutation-a")
    workers = 8
    barrier = threading.Barrier(workers)

    def reserve() -> tuple[IdempotencyReceipt, bool]:
        store = JsonIdempotencyReceiptStore(receipt_path)
        barrier.wait()
        return store.reserve_if_absent(receipt)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = tuple(executor.map(lambda _: reserve(), range(workers)))

    assert sum(created for _, created in results) == 1
    assert {stored for stored, _ in results} == {receipt}


def test_lifecycle_usage_preserves_all_raw_request_buckets() -> None:
    usage = RawResourceUsage(
        input_tokens=100,
        output_tokens=20,
        cache_read_tokens=30,
        cache_write_tokens=4,
        reasoning_tokens=7,
        model_requests=2,
        retry_count=1,
        duration_ms=250,
        storage_bytes=64,
    )
    assert usage.to_dict() == {
        "input_tokens": 100,
        "output_tokens": 20,
        "cache_read_tokens": 30,
        "cache_write_tokens": 4,
        "reasoning_tokens": 7,
        "model_requests": 2,
        "retry_count": 1,
        "duration_ms": 250,
        "storage_bytes": 64,
    }
