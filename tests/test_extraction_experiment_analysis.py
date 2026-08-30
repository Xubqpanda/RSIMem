from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from rsimem.extraction_experiment_analysis import (
    analyze_extraction_batch,
    classify_extraction_audit_failure,
)
from rsimem.extraction_experiment_manifest import (
    EXTRACTION_METHOD_VARIANTS,
    extraction_execution_order,
    initialize_extraction_batch_manifest,
    record_extraction_attempt,
)
from rsimem.memory.extraction_feedback import (
    ArtifactSemanticBinding,
    DeploymentObservation,
    ExposureMode,
    ExtractedFactEvidence,
    ExtractionFeedbackBuilder,
    ExtractionSetStatus,
    ExtractionSourceEvidence,
    FactDisposition,
    FutureMemoryEvidence,
    ObservableToolEvent,
    default_feedback_contract_registry,
)
from rsimem.memory.extraction_projection import (
    ExtractionSourceRecord,
    JsonExtractionSourceRecordStore,
    JsonLiveExtractionFeedbackRecordLog,
    LiveExtractionFeedbackRecord,
)
from rsimem.memory.process_signal import JsonProcessSignalCaseStore, ProcessSignalCase
from rsimem.memory.signal_protocol import (
    PROCESS_SIGNAL_PROTOCOL_FILENAME,
    JsonProcessSignalAnalysisProtocolStore,
    protocol_for_extraction_manifest,
)
from test_extraction_experiment_manifest import _inputs
from extraction_fingerprint_support import extraction_activation_fixture


PRIVATE_TEXT = "Do not include this private memory value."
SEMANTIC_KEY = "preference.summary.tsv"


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _source(method: str, artifact_id: str) -> ExtractionSourceEvidence:
    return ExtractionSourceEvidence(
        f"source.{method}",
        _sha(f"source projection {method}"),
        f"extraction-set.{method}",
        ExtractionSetStatus.NONEMPTY,
        (SEMANTIC_KEY,),
        (
            ExtractedFactEvidence(
                f"fact.{method}",
                SEMANTIC_KEY,
                FactDisposition.PERSISTED,
                artifact_id=artifact_id,
            ),
        ),
    )


def _feedback(
    source: ExtractionSourceEvidence,
    *,
    source_record_id: str,
    method: str,
    artifact_id: str,
    run_id: str | None = None,
) -> LiveExtractionFeedbackRecord:
    future = FutureMemoryEvidence(
        f"opportunity.{method}",
        ExposureMode.EAGER_SYSTEM_PROMPT,
        (ArtifactSemanticBinding(artifact_id, SEMANTIC_KEY),),
        f"operation.opportunity.{method}",
        f"operation.injection.{method}",
    )
    observation = DeploymentObservation(
        f"observation.{method}",
        "SM01_preference_adoption",
        "eval_near",
        "task.eval",
        _sha(f"current input {method}"),
        (),
        (SEMANTIC_KEY,),
        "owner\tpriority\ttask\tdue_date\nA\thigh\tShip\t2026/09/01",
        (
            ObservableToolEvent(
                f"tool.share.{method}",
                "notes_share",
                True,
                subject_ids=("note.1",),
                recipient_ids=("owner_a",),
            ),
        ),
        True,
    )
    dataset = ExtractionFeedbackBuilder(
        default_feedback_contract_registry()
    ).build(source, observation, future)
    primary = next(example for example in dataset.examples if example.primary)
    return LiveExtractionFeedbackRecord.create(
        family_id="SM01_preference_adoption",
        stage="eval_near",
        run_id=run_id or f"run.{method}",
        trace_id=f"trace.{method}",
        episode_id=f"episode.{method}",
        session_id=f"session.{method}",
        task_id="task.eval",
        deployment_observation_id=observation.observation_id,
        source_record_id=source_record_id,
        opportunity_operation_id=primary.opportunity_operation_id,
        use_operation_id=primary.use_operation_id,
        outcome_operation_id=primary.outcome_operation_id,
        dataset=dataset,
    )


def _run_evidence(
    run: Path,
    *,
    method: str,
    extraction_artifact_id: str,
    extraction_artifact_digest: str,
    output_digest: str,
    memory_artifact_id: str,
    run_id: str | None = None,
    semantic_policy=None,
    policy_artifact_id: str | None = None,
    policy_artifact_digest: str | None = None,
    matched_validation: bool | None = None,
    policy_artifact=None,
) -> None:
    source = _source(method, memory_artifact_id)
    source_record = ExtractionSourceRecord.create(
        family_id="SM01_preference_adoption",
        stage="learn_a",
        run_id=run_id or f"run.{method}",
        episode_id=f"episode.learn.{method}",
        session_id=f"session.learn.{method}",
        task_id="task.learn",
        compilation_id=f"compilation.{method}",
        extraction_artifact_id=extraction_artifact_id,
        extraction_artifact_digest=extraction_artifact_digest,
        extraction_output_digest=output_digest,
        source=source,
        activation=extraction_activation_fixture(
            compilation_id=f"compilation.{method}",
            extraction_operation_id=source.extraction_set_id,
            component_artifact_id=extraction_artifact_id,
            component_artifact_digest=extraction_artifact_digest,
            parsed_output_digest=output_digest,
            persisted_artifact_ids=(memory_artifact_id,),
            mutation_ids=(f"mutation.{method}",),
            policy_artifact_id=policy_artifact_id,
            policy_artifact_digest=policy_artifact_digest,
            matched_validation=(
                method.startswith(EXTRACTION_METHOD_VARIANTS[1])
                if matched_validation is None
                else matched_validation
            ),
            semantic_policy=semantic_policy,
            policy_artifact=policy_artifact,
        ),
    )
    source_path = (
        run
        / "family_homes"
        / "SM01_preference_adoption"
        / "hermes_home"
        / ".rsimem"
        / "extraction_sources.jsonl"
    )
    JsonExtractionSourceRecordStore(source_path).append(source_record)
    feedback = _feedback(
        source,
        source_record_id=source_record.record_id,
        method=method,
        artifact_id=memory_artifact_id,
        run_id=run_id,
    )
    JsonLiveExtractionFeedbackRecordLog(
        run / "02_eval" / "artifacts" / "rsimem_extraction_feedback.jsonl"
    ).append(feedback)
    (run / "sequence_comparison.json").write_text(json.dumps({
        "with_persistence": {
            "episodes": [{"timing": {"wall_time_s": 1.5}}],
        },
    }), encoding="utf-8")
    audit = {
        "ok": False,
        "issues": [{"kind": "incomplete_model_usage"}],
        "uniquePhysicalUsage": {
            "requests": 2,
            "inputTokens": 100,
            "outputTokens": 20,
            "cacheReadTokens": 0,
            "cacheWriteTokens": 0,
            "reasoningTokens": 0,
            "retries": 0,
        },
        "ingestionUsage": {
            "modelRequests": 2,
            "inputTokens": 50,
            "outputTokens": 10,
            "cacheReadTokens": 0,
            "cacheWriteTokens": 0,
            "reasoningTokens": 0,
            "retries": 0,
            "durationMs": 12,
            "storageBytes": 20,
            "complete": {
                "inputTokens": True,
                "outputTokens": True,
                "cacheReadTokens": False,
                "cacheWriteTokens": False,
                "reasoningTokens": False,
                "durationMs": True,
            },
        },
    }
    (run / "audit.json").write_text(json.dumps(audit), encoding="utf-8")
    ledger = (
        {"kind": "memory_injection", "data": {"contentChars": 120}},
        {"kind": "storage_snapshot", "data": {
            "memoryFilesBytes": 30,
            "skillFilesBytes": 0,
            "stateDbBytes": 20,
        }},
    )
    (run / "ledger.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in ledger),
        encoding="utf-8",
    )


def _batch(tmp_path: Path, *, changed: bool) -> Path:
    root = tmp_path / "batch"
    inputs = _inputs(tmp_path, phase="validation")
    inputs["path"] = root / "batch_manifest.json"
    inputs["registry_path"] = tmp_path / "registry.json"
    inputs["replicates"] = 1
    initialize_extraction_batch_manifest(**inputs)
    manifest = json.loads((root / "batch_manifest.json").read_text(encoding="utf-8"))
    protocol = protocol_for_extraction_manifest(manifest)
    JsonProcessSignalAnalysisProtocolStore(
        root / PROCESS_SIGNAL_PROTOCOL_FILENAME
    ).freeze(protocol)
    active = {
        method: (
            inputs["parent_policy"]
            if method == EXTRACTION_METHOD_VARIANTS[0]
            else inputs["active_policy"]
        )
        for method in EXTRACTION_METHOD_VARIANTS
    }
    for ordinal, method in enumerate(extraction_execution_order(1), start=1):
        if ordinal == 1:
            failed_name = "r01_provider_failed"
            record_extraction_attempt(
                root / "batch_manifest.json",
                replicate=1,
                ordinal=ordinal,
                method=method,
                run_name=failed_name,
                status="running",
            )
            record_extraction_attempt(
                root / "batch_manifest.json",
                replicate=1,
                ordinal=ordinal,
                method=method,
                run_name=failed_name,
                status="failed",
                failure_stage="provider",
            )
        run_name = f"r01_{method.replace('-', '_')}"
        record_extraction_attempt(
            root / "batch_manifest.json",
            replicate=1,
            ordinal=ordinal,
            method=method,
            run_name=run_name,
            status="running",
        )
        run = root / run_name
        run.mkdir(parents=True)
        adaptive = method == EXTRACTION_METHOD_VARIANTS[1]
        _run_evidence(
            run,
            method=method,
            extraction_artifact_id=active[method].extraction_component_id,
            extraction_artifact_digest=active[method].extraction_component_digest,
            output_digest=(
                _sha("adaptive output") if adaptive and changed else _sha("parent output")
            ),
            memory_artifact_id=(
                "artifact.adaptive"
                if adaptive and changed
                else "artifact.parent"
            ),
            run_id=run_name,
            semantic_policy=active[method],
            matched_validation=adaptive,
        )
        record_extraction_attempt(
            root / "batch_manifest.json",
            replicate=1,
            ordinal=ordinal,
            method=method,
            run_name=run_name,
            status="completed",
        )
    return root


def test_analysis_reports_quality_raw_unknown_usage_and_complete_funnel(
    tmp_path: Path,
) -> None:
    report = analyze_extraction_batch(_batch(tmp_path, changed=True))

    assert report["qualityReady"] is True
    assert report["processCorpus"]["evaluationScoreAccessible"] is False
    assert report["evidencePlanes"]["pureProcess"]["familyIdentityPresent"] is False
    assert report["evidencePlanes"]["pureProcess"]["stageIdentityPresent"] is False
    assert report["evidencePlanes"]["benchmarkAudit"][
        "qualityLabelsAreAuditOnly"
    ] is True
    assert report["evidencePlanes"]["finalEvaluation"] == {
        "present": False,
        "consumer": "final_reporter_only",
        "officialScoreAccessible": False,
    }
    assert "task_score" not in report["processCorpus"]
    assert report["usageComplete"] is False
    assert report["failedAttempts"] == [{
        "replicate": 1,
        "method": "static-extraction-rsimem",
        "attemptNumber": 1,
        "runName": "r01_provider_failed",
        "failureStage": "provider",
    }]
    assert report["activationFunnel"] == {
        "eligible": 1,
        "renderedNPlus1": 1,
        "changedExtraction": 1,
        "noIntervention": 0,
        "changedArtifact": 1,
        "futureExposure": 1,
        "attributableUse": 1,
        "attributableOutcome": 1,
    }
    assert report["claimGate"]["operationAttributedExtractionAdaptation"][
        "eligible"
    ] is True
    adaptive = next(
        run for run in report["runs"]
        if run["method"] == "adaptive-extraction-rsimem"
    )
    assert adaptive["quality"]["usefulCount"] == 1
    assert adaptive["quality"]["resolvedUsefulRate"] == 1.0
    assert adaptive["quality"]["nonemptyCoverage"] == 1.0
    assert adaptive["quality"]["highConfidenceMissedRate"] is None
    assert adaptive["rawUsage"]["requests"]["value"] == 2
    assert adaptive["rawUsage"]["inputTokens"] == {
        "value": None,
        "observedValue": 100,
        "complete": False,
    }
    assert adaptive["rawUsage"]["ingestionInputTokens"]["value"] == 50
    assert adaptive["rawUsage"]["ingestionCacheReadTokens"]["value"] is None
    adaptive_summary = report["summaryByMethod"][
        "adaptive-extraction-rsimem"
    ]["quality"]
    assert adaptive_summary["resolvedUsefulRate"] == 1.0
    assert adaptive_summary["nonemptyCoverage"] == 1.0
    assert report["pairedRawUsageDelta"]["requests"]["values"] == [0]
    assert report["pairedRawUsageDelta"]["inputTokens"]["values"] == [None]
    assert report["pairedRawUsageDelta"]["inputTokens"]["mean"] is None
    assert report["providerPricing"] is None
    serialized = json.dumps(report)
    assert "lifecycleCostUnits" not in serialized
    assert "futureUtilityPerCost" not in serialized
    assert PRIVATE_TEXT not in serialized


def test_analysis_reports_persisted_logical_process_signal_census(tmp_path: Path) -> None:
    root = _batch(tmp_path, changed=False)
    completed = next(
        event
        for event in json.loads(
            (root / "batch_manifest.json").read_text(encoding="utf-8")
        )["attemptHistory"]
        if event["status"] == "completed"
    )
    run = root / completed["outputDirectory"]
    manifest = json.loads((root / "batch_manifest.json").read_text(encoding="utf-8"))
    protocol = protocol_for_extraction_manifest(manifest)
    JsonProcessSignalAnalysisProtocolStore(
        root / PROCESS_SIGNAL_PROTOCOL_FILENAME
    ).freeze(protocol)
    case = ProcessSignalCase.create(
        logical_case_id="logical-case.analysis.v1",
        physical_observation_ids=("physical-observation.analysis.v1",),
        source_observed=True,
        extraction_observed=True,
        persistence_observed=True,
        retrieval_observed=True,
        exposure_observed=False,
        outcome_observed=False,
        extraction_attributable=False,
        abstract_hypothesis_digest=None,
        observation_complete=True,
        analysis_protocol_id=protocol.protocol_id,
        replicate_id=f"replicate.{completed['replicate']}",
        observation_window=protocol.observation_window,
    )
    JsonProcessSignalCaseStore(run / "process_signal_cases.jsonl").append(case)
    report = analyze_extraction_batch(root)
    assert report["processSignalCases"]["caseCount"] == 1
    assert report["processSignalCases"]["logicalCaseCount"] == 1
    assert report["processSignalCases"]["physicalObservationCount"] == 1


def test_analysis_rejects_process_signal_cases_without_frozen_protocol(
    tmp_path: Path,
) -> None:
    root = _batch(tmp_path, changed=False)
    completed = next(
        event
        for event in json.loads(
            (root / "batch_manifest.json").read_text(encoding="utf-8")
        )["attemptHistory"]
        if event["status"] == "completed"
    )
    run = root / completed["outputDirectory"]
    manifest = json.loads((root / "batch_manifest.json").read_text(encoding="utf-8"))
    protocol = protocol_for_extraction_manifest(manifest)
    (root / PROCESS_SIGNAL_PROTOCOL_FILENAME).unlink()
    case = ProcessSignalCase.create(
        logical_case_id="logical-case.missing-protocol.v1",
        physical_observation_ids=("physical-observation.missing-protocol.v1",),
        source_observed=True,
        extraction_observed=True,
        persistence_observed=True,
        retrieval_observed=True,
        exposure_observed=False,
        outcome_observed=False,
        extraction_attributable=False,
        abstract_hypothesis_digest=None,
        observation_complete=True,
        analysis_protocol_id=protocol.protocol_id,
        replicate_id=f"replicate.{completed['replicate']}",
        observation_window=protocol.observation_window,
    )
    JsonProcessSignalCaseStore(run / "process_signal_cases.jsonl").append(case)
    with pytest.raises(ValueError, match="process signal protocol is missing"):
        analyze_extraction_batch(root)


def test_analysis_rejects_process_signal_protocol_manifest_drift(
    tmp_path: Path,
) -> None:
    root = _batch(tmp_path, changed=False)
    manifest_path = root / "batch_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    completed = next(
        event for event in manifest["attemptHistory"] if event["status"] == "completed"
    )
    run = root / completed["outputDirectory"]
    protocol = protocol_for_extraction_manifest(manifest)
    case = ProcessSignalCase.create(
        logical_case_id="logical-case.drifted-protocol.v1",
        physical_observation_ids=("physical-observation.drifted-protocol.v1",),
        source_observed=True,
        extraction_observed=True,
        persistence_observed=True,
        retrieval_observed=True,
        exposure_observed=False,
        outcome_observed=False,
        extraction_attributable=False,
        abstract_hypothesis_digest=None,
        observation_complete=True,
        analysis_protocol_id=protocol.protocol_id,
        replicate_id=f"replicate.{completed['replicate']}",
        observation_window=protocol.observation_window,
    )
    JsonProcessSignalCaseStore(run / "process_signal_cases.jsonl").append(case)
    drifted = protocol.__class__.create(
        training_family_ids=protocol.training_family_ids,
        task_template_group_ids=protocol.task_template_group_ids,
        provider_model=protocol.provider_model,
        replicate_count=protocol.replicate_count,
        observation_window="window.drifted.v1",
        case_dedup_rule=protocol.case_dedup_rule,
        no_signal_case_id=protocol.no_signal_case_id,
    )
    (root / PROCESS_SIGNAL_PROTOCOL_FILENAME).write_text(
        json.dumps(drifted.payload()) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="does not match extraction manifest"):
        analyze_extraction_batch(root)


def test_analysis_rejects_adaptation_claim_without_changed_extraction(
    tmp_path: Path,
) -> None:
    report = analyze_extraction_batch(_batch(tmp_path, changed=False))
    claim = report["claimGate"]["operationAttributedExtractionAdaptation"]

    assert report["qualityReady"] is True
    assert report["activationFunnel"]["changedExtraction"] == 0
    assert report["activationFunnel"]["noIntervention"] == 1
    assert claim["eligible"] is False
    assert claim["reason"] == "activation_funnel_incomplete"
    assert "changedExtraction" in claim["missingStages"]


def test_analysis_does_not_report_complete_usage_without_completed_runs(
    tmp_path: Path,
) -> None:
    root = tmp_path / "empty-batch"
    inputs = _inputs(tmp_path, phase="validation")
    inputs["path"] = root / "batch_manifest.json"
    inputs["registry_path"] = tmp_path / "registry.json"
    initialize_extraction_batch_manifest(**inputs)
    manifest = json.loads((root / "batch_manifest.json").read_text(encoding="utf-8"))
    JsonProcessSignalAnalysisProtocolStore(
        root / PROCESS_SIGNAL_PROTOCOL_FILENAME
    ).freeze(protocol_for_extraction_manifest(manifest))

    report = analyze_extraction_batch(root)

    assert report["runs"] == []
    assert report["qualityReady"] is False
    assert report["usageComplete"] is False


def test_analysis_rejects_feedback_joined_to_an_unrelated_existing_source(
    tmp_path: Path,
) -> None:
    root = _batch(tmp_path, changed=True)
    manifest = json.loads(
        (root / "batch_manifest.json").read_text(encoding="utf-8")
    )
    completed = next(
        event
        for event in manifest["attemptHistory"]
        if event["status"] == "completed"
    )
    run = root / completed["outputDirectory"]
    source_path = next(run.rglob("extraction_sources.jsonl"))
    source_store = JsonExtractionSourceRecordStore(source_path)
    original = source_store.records()[0]
    unrelated_source = _source("unrelated", "artifact.unrelated")
    unrelated = ExtractionSourceRecord.create(
        family_id=original.family_id,
        stage=original.stage,
        run_id=original.run_id,
        episode_id="episode.unrelated",
        session_id="session.unrelated",
        task_id="task.unrelated",
        compilation_id="compilation.unrelated",
        extraction_artifact_id=original.extraction_artifact_id,
        extraction_artifact_digest=original.extraction_artifact_digest,
        extraction_output_digest=_sha("unrelated output"),
        source=unrelated_source,
        activation=extraction_activation_fixture(
            compilation_id="compilation.unrelated",
            extraction_operation_id=unrelated_source.extraction_set_id,
            component_artifact_id=original.extraction_artifact_id,
            component_artifact_digest=original.extraction_artifact_digest,
            parsed_output_digest=_sha("unrelated output"),
            persisted_artifact_ids=("artifact.unrelated",),
            mutation_ids=("mutation.unrelated",),
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
        ),
    )
    source_store.append(unrelated)

    feedback_path = next(run.rglob("rsimem_extraction_feedback.jsonl"))
    feedback = JsonLiveExtractionFeedbackRecordLog(feedback_path).records()[0]
    mismatched = LiveExtractionFeedbackRecord.create(
        family_id=feedback.family_id,
        stage=feedback.stage,
        run_id=feedback.run_id,
        trace_id=feedback.trace_id,
        episode_id=feedback.episode_id,
        session_id=feedback.session_id,
        task_id=feedback.task_id,
        deployment_observation_id=feedback.deployment_observation_id,
        source_record_id=unrelated.record_id,
        opportunity_operation_id=feedback.opportunity_operation_id,
        use_operation_id=feedback.use_operation_id,
        outcome_operation_id=feedback.outcome_operation_id,
        dataset=feedback.dataset,
    )
    feedback_path.unlink()
    JsonLiveExtractionFeedbackRecordLog(feedback_path).append(mismatched)

    try:
        analyze_extraction_batch(root)
    except ValueError as exc:
        assert "does not join" in str(exc)
    else:
        raise AssertionError("mismatched source/feedback join was accepted")


def test_audit_failure_classification_requires_all_calls_to_be_provider_errors() -> None:
    assert classify_extraction_audit_failure({
        "modelCallStatuses": {"error": 3},
    }) == "provider"
    assert classify_extraction_audit_failure({
        "modelCallStatuses": {"ok": 2, "error": 1},
    }) == "audit"
    assert classify_extraction_audit_failure({
        "issues": [{"kind": "incomplete_model_usage"}],
    }) == "audit"
