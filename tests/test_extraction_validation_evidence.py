from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from rsimem.extraction_experiment_manifest import (
    EXTRACTION_METHOD_VARIANTS,
    extraction_execution_order,
    initialize_extraction_batch_manifest,
    record_extraction_attempt,
)
from rsimem.extraction_validation_evidence import (
    ExtractionMatchedEvidenceBatch,
    assemble_extraction_matched_evidence_batch,
    load_extraction_matched_evidence_batch,
)
from rsimem.memory.extraction_projection import (
    ExtractionActivationFingerprint,
    ExtractionSourceRecord,
    JsonExtractionSourceRecordStore,
)
from rsimem.memory.extraction_prompt_validation import (
    ExtractionPromptValidationSplit,
    ExtractionSplitAssignment,
    ExtractionValidationSplitRole,
    ExtractionValidationVariant,
)
from rsimem.memory.prompt_components import (
    MatchedSemanticPolicyManifest,
    SemanticPolicyManifest,
    text_digest,
)
from rsimem.memory.process_corpus import JsonProcessCorpusStore, ProcessCorpus
from rsimem.memory.process_feedback import (
    ProcessEvent,
    ProcessEventKind,
    ProcessEventStatus,
)
from rsimem.memory_systems.mem0_flat import (
    MEM0_FLAT_EXTRACTION_SLOT,
    MEM0_FLAT_EXTRACTION_SLOT_ID,
    Mem0FlatPromptAdapter,
    Mem0FlatSemanticPolicy,
)
from test_extraction_experiment_analysis import _run_evidence
from test_extraction_experiment_manifest import _inputs
from test_extraction_matched_activation import _offline_decision
from test_extraction_offline_validation import _candidate, _criteria, _parent
from extraction_fingerprint_support import extraction_activation_fixture


def _semantic_policy(artifact):
    adapter = Mem0FlatPromptAdapter()
    binding = adapter.bind_policy_artifact(
        MEM0_FLAT_EXTRACTION_SLOT_ID,
        artifact,
    )
    return Mem0FlatSemanticPolicy(
        object(),
        fact_prompt=adapter.bound_template(binding),
        extraction_binding=binding,
    ).semantic_manifest


def _split(task_manifest_digest: str) -> ExtractionPromptValidationSplit:
    return ExtractionPromptValidationSplit(
        "split.formal-matched-v1",
        (
            ExtractionSplitAssignment(
                ExtractionValidationSplitRole.TRAIN,
                "family.formal-train-v1",
                "template.formal-train-v1",
                text_digest("formal train manifest"),
            ),
            ExtractionSplitAssignment(
                ExtractionValidationSplitRole.VALIDATION,
                "SM01_preference_adoption",
                "sm01.formal-matched-v1",
                task_manifest_digest,
            ),
            ExtractionSplitAssignment(
                ExtractionValidationSplitRole.FINAL,
                "family.formal-final-v1",
                "template.formal-final-v1",
                text_digest("formal final manifest"),
            ),
        ),
    )


def _batch(tmp_path: Path):
    parent = _parent()
    candidate = _candidate(parent=parent)
    parent_policy = _semantic_policy(parent)
    candidate_policy = _semantic_policy(candidate)
    root = tmp_path / "validation-batch"
    task_manifest_digest = text_digest("formal matched task manifest")
    inputs = _inputs(tmp_path, phase="validation")
    inputs.update({
        "path": root / "batch_manifest.json",
        "registry_path": tmp_path / "formal-batch-registry.json",
        "batch_id": "batch.formal-matched-v1",
        "replicates": 3,
        "task_template_group_id": "sm01.formal-matched-v1",
        "task_manifest_digest": task_manifest_digest,
        "parent_policy": parent_policy,
        "active_policy": candidate_policy,
        "matched_policy": MatchedSemanticPolicyManifest.create(
            parent_policy,
            candidate_policy,
        ),
        "acceptance_criteria": _criteria(),
        "model_profile_id": parent_policy.model_profile,
    })
    initialize_extraction_batch_manifest(**inputs)
    policies = {
        EXTRACTION_METHOD_VARIANTS[0]: parent_policy,
        EXTRACTION_METHOD_VARIANTS[1]: candidate_policy,
    }
    for replicate in range(1, 4):
        for ordinal, method in enumerate(
            extraction_execution_order(replicate),
            start=1,
        ):
            run_name = f"r{replicate:02d}_{method.replace('-', '_')}"
            record_extraction_attempt(
                root / "batch_manifest.json",
                replicate=replicate,
                ordinal=ordinal,
                method=method,
                run_name=run_name,
                status="running",
            )
            run = root / run_name
            run.mkdir(parents=True)
            adaptive = method == EXTRACTION_METHOD_VARIANTS[1]
            policy = policies[method]
            evidence_key = f"{method}.r{replicate}"
            _run_evidence(
                run,
                method=evidence_key,
                extraction_artifact_id=policy.extraction_component_id,
                extraction_artifact_digest=policy.extraction_component_digest,
                output_digest=text_digest(
                    f"{'candidate' if adaptive else 'parent'} output {replicate}"
                ),
                memory_artifact_id=(
                    f"artifact.candidate-{replicate}"
                    if adaptive
                    else f"artifact.parent-{replicate}"
                ),
                run_id=run_name,
                semantic_policy=policy,
                policy_artifact_id=(
                    candidate.artifact_id if adaptive else parent.artifact_id
                ),
                policy_artifact_digest=(
                    candidate.artifact_digest
                    if adaptive
                    else parent.artifact_digest
                ),
                matched_validation=adaptive,
                policy_artifact=(candidate if adaptive else parent),
            )
            process_event = ProcessEvent.create(
                kind=ProcessEventKind.HOST_LIFECYCLE,
                status=ProcessEventStatus.PENDING,
                run_id=run_name,
                variant="with_persistence",
                trace_id=f"trace.{run_name}",
                episode_id=f"episode.{run_name}",
                session_id=f"session.{run_name}",
                task_id=f"task.{run_name}",
                host_event_id=f"event.{run_name}",
                source_revision="revision.validation",
                input_payload={"boundary": "task_completed"},
                output_payload={"observed": True},
                family_id="SM01_preference_adoption",
                stage="validation",
            )
            process_corpus = ProcessCorpus.create(
                (process_event,),
                split_role="validation",
                family_id="SM01_preference_adoption",
                task_template_group_id="sm01.formal-matched-v1",
                task_manifest_digest=task_manifest_digest,
            )
            JsonProcessCorpusStore(run / "process_corpus.json").put(process_corpus)
            record_extraction_attempt(
                root / "batch_manifest.json",
                replicate=replicate,
                ordinal=ordinal,
                method=method,
                run_name=run_name,
                status="completed",
            )
    return (
        root,
        parent,
        candidate,
        _offline_decision(parent, candidate),
        _split(task_manifest_digest),
    )


def test_formal_validation_evidence_assembles_and_replays_content_free(
    tmp_path,
) -> None:
    root, parent, candidate, offline, split = _batch(tmp_path)
    output = root / "matched-evidence.json"
    batch = assemble_extraction_matched_evidence_batch(
        root,
        parent=parent,
        candidate=candidate,
        offline_decision=offline,
        split=split,
        output_path=output,
    )
    replay = assemble_extraction_matched_evidence_batch(
        root,
        parent=parent,
        candidate=candidate,
        offline_decision=offline,
        split=split,
        output_path=output,
    )
    assert replay == batch
    assert ExtractionMatchedEvidenceBatch.from_payload(batch.payload()) == batch
    assert load_extraction_matched_evidence_batch(output) == batch
    assert len(batch.observations) == 6
    assert len(batch.safety_evidence) == 6
    assert len(batch.evidence_joins) == 6
    assert {value.variant for value in batch.observations} == set(
        ExtractionValidationVariant
    )
    assert all(value.failure_counts == (0, 0, 0, 0) for value in batch.observations)
    assert batch.parent_runtime_artifact_id == parent.to_prompt_component(
        MEM0_FLAT_EXTRACTION_SLOT
    ).artifact_id
    assert batch.candidate_runtime_artifact_id == candidate.to_prompt_component(
        MEM0_FLAT_EXTRACTION_SLOT
    ).artifact_id
    assert batch.decision.quality_decision.changed_extraction_count == 3
    assert batch.decision.quality_decision.parent_metrics.useful_count == 3
    assert batch.decision.quality_decision.proposal_metrics.useful_count == 3
    assert (
        batch.decision.quality_decision.parent_metrics.high_confidence_missed_rate
        == 0
    )
    assert (
        batch.decision.quality_decision.proposal_metrics.high_confidence_missed_rate
        == 0
    )
    assert "useful_rate_not_improved" in batch.decision.reason_codes
    serialized = output.read_text(encoding="utf-8")
    for forbidden in (
        "task_score",
        "official_score",
        "grader",
        "answer_key",
        "lifecycle_cost",
        "input_tokens",
        "output_tokens",
    ):
        assert forbidden not in serialized


def test_run_level_safety_issue_propagates_to_candidate_observation(tmp_path) -> None:
    root, parent, candidate, offline, split = _batch(tmp_path)
    run = root / "r01_adaptive_extraction_rsimem"
    audit_path = run / "audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["issues"].append({"kind": "memory_text_leak", "count": 2})
    audit_path.write_text(json.dumps(audit), encoding="utf-8")

    batch = assemble_extraction_matched_evidence_batch(
        root,
        parent=parent,
        candidate=candidate,
        offline_decision=offline,
        split=split,
    )
    failed = [
        value
        for value in batch.observations
        if value.variant == ExtractionValidationVariant.PROPOSAL
        and value.replicate == 1
    ]
    assert len(failed) == 1
    assert failed[0].prompt_leakage_failure_count == 2
    assert "safety_failure" in batch.decision.reason_codes


def test_validation_evidence_rejects_source_run_and_runtime_artifact_drift(
    tmp_path,
) -> None:
    root, parent, candidate, offline, split = _batch(tmp_path)
    run = root / "r01_static_extraction_rsimem"
    source_path = next(run.rglob("extraction_sources.jsonl"))
    store = JsonExtractionSourceRecordStore(source_path)
    original = store.records()[0]
    drifted = ExtractionSourceRecord.create(
        family_id=original.family_id,
        stage=original.stage,
        run_id="another-run",
        episode_id=original.episode_id,
        session_id=original.session_id,
        task_id=original.task_id,
        compilation_id=original.compilation_id,
        extraction_artifact_id=original.extraction_artifact_id,
        extraction_artifact_digest=original.extraction_artifact_digest,
        extraction_output_digest=original.extraction_output_digest,
        source=original.source,
        activation=original.activation,
    )
    source_path.unlink()
    JsonExtractionSourceRecordStore(source_path).append(drifted)
    with pytest.raises(ValueError, match="source identity differs"):
        assemble_extraction_matched_evidence_batch(
            root,
            parent=parent,
            candidate=candidate,
            offline_decision=offline,
            split=split,
        )

    root, parent, candidate, offline, split = _batch(tmp_path / "artifact-drift")
    run = root / "r01_static_extraction_rsimem"
    source_path = next(run.rglob("extraction_sources.jsonl"))
    store = JsonExtractionSourceRecordStore(source_path)
    original = store.records()[0]
    drifted = ExtractionSourceRecord.create(
        family_id=original.family_id,
        stage=original.stage,
        run_id=original.run_id,
        episode_id=original.episode_id,
        session_id=original.session_id,
        task_id=original.task_id,
        compilation_id=original.compilation_id,
        extraction_artifact_id="prompt-component.runtime-drift-v1",
        extraction_artifact_digest=original.extraction_artifact_digest,
        extraction_output_digest=original.extraction_output_digest,
        source=original.source,
        activation=extraction_activation_fixture(
            compilation_id=original.compilation_id,
            extraction_operation_id=original.source.extraction_set_id,
            component_artifact_id="prompt-component.runtime-drift-v1",
            component_artifact_digest=original.extraction_artifact_digest,
            parsed_output_digest=original.extraction_output_digest,
            persisted_artifact_ids=original.artifact_ids,
            mutation_ids=original.activation.mutation_ids,
            matched_validation=(
                original.activation.runtime_binding.deployment_scope.value
                == "matched_validation"
            ),
        ),
    )
    source_path.unlink()
    JsonExtractionSourceRecordStore(source_path).append(drifted)
    with pytest.raises(ValueError, match="source identity differs"):
        assemble_extraction_matched_evidence_batch(
            root,
            parent=parent,
            candidate=candidate,
            offline_decision=offline,
            split=split,
        )


def test_validation_evidence_rejects_source_without_feedback_closure(tmp_path) -> None:
    root, parent, candidate, offline, split = _batch(tmp_path)
    run = root / "r01_static_extraction_rsimem"
    source_path = next(run.rglob("extraction_sources.jsonl"))
    store = JsonExtractionSourceRecordStore(source_path)
    original = store.records()[0]
    uncovered = ExtractionSourceRecord.create(
        family_id=original.family_id,
        stage="learn_without_future_closure",
        run_id=original.run_id,
        episode_id="episode.uncovered-v1",
        session_id="session.uncovered-v1",
        task_id="task.uncovered-v1",
        compilation_id="compilation.uncovered-v1",
        extraction_artifact_id=original.extraction_artifact_id,
        extraction_artifact_digest=original.extraction_artifact_digest,
        extraction_output_digest=original.extraction_output_digest,
        source=original.source,
        activation=extraction_activation_fixture(
            compilation_id="compilation.uncovered-v1",
            extraction_operation_id=original.source.extraction_set_id,
            component_artifact_id=original.extraction_artifact_id,
            component_artifact_digest=original.extraction_artifact_digest,
            parsed_output_digest=original.extraction_output_digest,
            persisted_artifact_ids=original.artifact_ids,
            mutation_ids=("mutation.uncovered-v1",),
            matched_validation=(
                original.activation.runtime_binding.deployment_scope.value
                == "matched_validation"
            ),
            semantic_policy=original.activation.semantic_policy,
            policy_artifact_id=(
                original.activation.runtime_binding.policy_artifact_id
            ),
            policy_artifact_digest=(
                original.activation.runtime_binding.policy_artifact_digest
            ),
            policy_artifact=parent,
        ),
    )
    store.append(uncovered)

    with pytest.raises(ValueError, match="complete feedback closure"):
        assemble_extraction_matched_evidence_batch(
            root,
            parent=parent,
            candidate=candidate,
            offline_decision=offline,
            split=split,
        )


@pytest.mark.parametrize(
    "drift_field",
    (
        "route",
        "boundary",
        "backend",
        "model_profile",
        "update_component_digest",
        "retrieval_component_digest",
    ),
)
def test_validation_evidence_rejects_frozen_semantic_component_drift(
    tmp_path,
    drift_field: str,
) -> None:
    root, parent, candidate, offline, split = _batch(tmp_path / drift_field)
    run = root / "r01_static_extraction_rsimem"
    source_path = next(run.rglob("extraction_sources.jsonl"))
    original = JsonExtractionSourceRecordStore(source_path).records()[0]
    semantic = original.activation.semantic_policy
    values = {
        field: getattr(semantic, field)
        for field in (
            "route",
            "boundary",
            "backend",
            "framework_version",
            "model_profile",
            "extraction_component_id",
            "extraction_component_digest",
            "update_component_id",
            "update_component_digest",
            "retrieval_component_id",
            "retrieval_component_digest",
        )
    }
    values[drift_field] = (
        "f" * 64 if drift_field.endswith("_digest") else f"drifted-{drift_field}"
    )
    drifted_semantic = SemanticPolicyManifest.create(**values)
    drifted = ExtractionSourceRecord.create(
        family_id=original.family_id,
        stage=original.stage,
        run_id=original.run_id,
        episode_id=original.episode_id,
        session_id=original.session_id,
        task_id=original.task_id,
        compilation_id=original.compilation_id,
        extraction_artifact_id=original.extraction_artifact_id,
        extraction_artifact_digest=original.extraction_artifact_digest,
        extraction_output_digest=original.extraction_output_digest,
        source=original.source,
        activation=extraction_activation_fixture(
            compilation_id=original.compilation_id,
            extraction_operation_id=original.source.extraction_set_id,
            component_artifact_id=original.extraction_artifact_id,
            component_artifact_digest=original.extraction_artifact_digest,
            parsed_output_digest=original.extraction_output_digest,
            persisted_artifact_ids=original.artifact_ids,
            mutation_ids=original.activation.mutation_ids,
            semantic_policy=drifted_semantic,
            policy_artifact=parent,
        ),
    )
    source_path.unlink()
    JsonExtractionSourceRecordStore(source_path).append(drifted)

    with pytest.raises(ValueError, match="run activation"):
        assemble_extraction_matched_evidence_batch(
            root,
            parent=parent,
            candidate=candidate,
            offline_decision=offline,
            split=split,
        )


@pytest.mark.parametrize(
    "drift_field",
    (
        "adapter_id",
        "slot_contract_digest",
        "frozen_wrapper_digest",
        "input_schema_digest",
        "output_schema_digest",
    ),
)
def test_validation_evidence_rejects_runtime_binding_contract_drift(
    tmp_path,
    drift_field: str,
) -> None:
    root, parent, candidate, offline, split = _batch(tmp_path / drift_field)
    run = root / "r01_static_extraction_rsimem"
    source_path = next(run.rglob("extraction_sources.jsonl"))
    original = JsonExtractionSourceRecordStore(source_path).records()[0]
    replacement = (
        "drifted-adapter-v1" if drift_field == "adapter_id" else "f" * 64
    )
    binding = replace(
        original.activation.runtime_binding,
        **{drift_field: replacement},
    )
    activation = ExtractionActivationFingerprint.create(
        compilation_id=original.compilation_id,
        extraction_operation_id=original.source.extraction_set_id,
        runtime_binding=binding,
        semantic_policy=original.activation.semantic_policy,
        invocation=original.activation.invocation,
        parsed_output_digest=original.extraction_output_digest,
        mutation_ids=original.activation.mutation_ids,
        persisted_artifact_ids=original.activation.persisted_artifact_ids,
    )
    drifted = ExtractionSourceRecord.create(
        family_id=original.family_id,
        stage=original.stage,
        run_id=original.run_id,
        episode_id=original.episode_id,
        session_id=original.session_id,
        task_id=original.task_id,
        compilation_id=original.compilation_id,
        extraction_artifact_id=original.extraction_artifact_id,
        extraction_artifact_digest=original.extraction_artifact_digest,
        extraction_output_digest=original.extraction_output_digest,
        source=original.source,
        activation=activation,
    )
    source_path.unlink()
    JsonExtractionSourceRecordStore(source_path).append(drifted)

    with pytest.raises(ValueError, match="run activation"):
        assemble_extraction_matched_evidence_batch(
            root,
            parent=parent,
            candidate=candidate,
            offline_decision=offline,
            split=split,
        )


def test_validation_evidence_rejects_incomplete_slots_and_wrong_split(tmp_path) -> None:
    root, parent, candidate, offline, split = _batch(tmp_path)
    manifest_path = root / "batch_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["attemptHistory"] = [
        event
        for event in manifest["attemptHistory"]
        if not (
            event["replicate"] == 3
            and event["method"] == EXTRACTION_METHOD_VARIANTS[1]
        )
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="one completed run per slot"):
        assemble_extraction_matched_evidence_batch(
            root,
            parent=parent,
            candidate=candidate,
            offline_decision=offline,
            split=split,
        )

    root, parent, candidate, offline, _ = _batch(tmp_path / "other")
    wrong_split = ExtractionPromptValidationSplit(
        "split.wrong-validation-v1",
        tuple(
            value
            for value in split.assignments
            if value.role != ExtractionValidationSplitRole.VALIDATION
        )
        + (ExtractionSplitAssignment(
            ExtractionValidationSplitRole.VALIDATION,
            "SM01_preference_adoption",
            "other-template-v1",
            text_digest("other validation manifest"),
        ),),
    )
    with pytest.raises(ValueError, match="outside the declared split"):
        assemble_extraction_matched_evidence_batch(
            root,
            parent=parent,
            candidate=candidate,
            offline_decision=offline,
            split=wrong_split,
        )


def test_validation_evidence_requires_process_corpus_and_declared_identity(tmp_path) -> None:
    root, parent, candidate, offline, split = _batch(tmp_path)
    process_path = root / "r01_static_extraction_rsimem" / "process_corpus.json"
    process_path.unlink()
    with pytest.raises(ValueError, match="incomplete process evidence"):
        assemble_extraction_matched_evidence_batch(
            root,
            parent=parent,
            candidate=candidate,
            offline_decision=offline,
            split=split,
        )

    root, parent, candidate, offline, split = _batch(tmp_path / "identity")
    process_path = root / "r01_static_extraction_rsimem" / "process_corpus.json"
    original = JsonProcessCorpusStore(process_path).get()
    assert original is not None
    replacement = ProcessCorpus.create(
        original.events,
        split_role="pilot",
        family_id=original.family_id,
        task_template_group_id=original.task_template_group_id,
        task_manifest_digest=original.task_manifest_digest,
    )
    process_path.unlink()
    JsonProcessCorpusStore(process_path).put(replacement)
    with pytest.raises(ValueError, match="process corpus identity differs"):
        assemble_extraction_matched_evidence_batch(
            root,
            parent=parent,
            candidate=candidate,
            offline_decision=offline,
            split=split,
        )
