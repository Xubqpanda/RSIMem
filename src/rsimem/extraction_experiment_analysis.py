"""Content-free quality, activation, and raw-usage analysis for extraction batches."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence

from .extraction_experiment_manifest import (
    EXTRACTION_METHOD_VARIANTS,
    load_extraction_manifest,
)
from .memory.extraction_feedback import (
    ExposureMode,
    ExtractionFeedbackLabel,
    ExtractionQualityIssue,
    ExtractionSetStatus,
)
from .memory.extraction_projection import (
    ExtractionSourceRecord,
    JsonExtractionSourceRecordStore,
    JsonLiveExtractionFeedbackRecordLog,
    LiveExtractionFeedbackRecord,
)
from .memory.live_writeback import ExtractionPromptRuntimeScope
from .memory.prompt_components import SemanticPolicyManifest
from .memory.operation_graph import (
    AppendOnlyOperationEvidenceLog,
    OperationKind,
    materialize_operation_graph,
)
from .memory.process_feedback import JsonProcessFeedbackLedger, ProcessEvent


EXTRACTION_ANALYSIS_SCHEMA_VERSION = 1
EXTRACTION_ANALYSIS_SCHEMA = "extraction-prompt-experiment-analysis-v1"
_MODEL_USAGE_FIELDS = (
    "requests",
    "inputTokens",
    "outputTokens",
    "cacheReadTokens",
    "cacheWriteTokens",
    "reasoningTokens",
    "retries",
)
_RAW_USAGE_FIELDS = (
    *_MODEL_USAGE_FIELDS,
    "wallTimeSeconds",
    "peakStoredBytes",
    "injectedChars",
    "recoveryDurationMs",
    "ingestionModelRequests",
    "ingestionInputTokens",
    "ingestionOutputTokens",
    "ingestionCacheReadTokens",
    "ingestionCacheWriteTokens",
    "ingestionReasoningTokens",
    "ingestionRetries",
    "ingestionDurationMs",
    "ingestionStorageBytes",
)


def _read_json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} cannot be read") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"JSONL evidence cannot be read: {path.name}") from exc
    events = []
    for line in lines:
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"malformed JSONL evidence: {path.name}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"JSONL evidence must contain objects: {path.name}")
        events.append(value)
    return tuple(events)


def _deduplicate(
    values: Iterable[Any],
    *,
    identity: str,
    payload,
) -> tuple[Any, ...]:
    result = []
    canonical_by_id: dict[str, str] = {}
    for value in values:
        key = getattr(value, identity)
        canonical = json.dumps(
            payload(value), ensure_ascii=True, sort_keys=True, separators=(",", ":")
        )
        previous = canonical_by_id.get(key)
        if previous is not None and previous != canonical:
            raise ValueError(f"conflicting extraction evidence identity: {key}")
        if previous is None:
            result.append(value)
        canonical_by_id[key] = canonical
    return tuple(result)


def _source_records(run_dir: Path) -> tuple[ExtractionSourceRecord, ...]:
    records = (
        record
        for path in sorted(run_dir.rglob("extraction_sources.jsonl"))
        for record in JsonExtractionSourceRecordStore(path).records()
    )
    return _deduplicate(
        records,
        identity="record_id",
        payload=lambda value: value.payload(),
    )


def _feedback_records(run_dir: Path) -> tuple[LiveExtractionFeedbackRecord, ...]:
    records = (
        record
        for path in sorted(run_dir.rglob("rsimem_extraction_feedback.jsonl"))
        for record in JsonLiveExtractionFeedbackRecordLog(path).records()
    )
    return _deduplicate(
        records,
        identity="record_id",
        payload=lambda value: value.payload(),
    )


def _process_events(run_dir: Path) -> tuple[ProcessEvent, ...]:
    """Read the bridge process corpus without touching evaluation results."""

    records: list[ProcessEvent] = []
    seen: dict[str, str] = {}
    for path in sorted(run_dir.rglob("rsimem_process_feedback.jsonl")):
        for event in JsonProcessFeedbackLedger(path).events:
            canonical = json.dumps(
                event.payload(), ensure_ascii=True, sort_keys=True,
                separators=(",", ":"),
            )
            previous = seen.get(event.event_id)
            if previous is not None and previous != canonical:
                raise ValueError(f"conflicting process feedback identity: {event.event_id}")
            if previous is None:
                records.append(event)
                seen[event.event_id] = canonical
    return tuple(records)


def _terminal_attempts(manifest: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    latest: dict[str, dict[str, Any]] = {}
    for event in manifest["attemptHistory"]:
        latest[event["runName"]] = event
    return tuple(
        event for event in latest.values() if event["status"] != "running"
    )


def _episode_wall_time(comparison: dict[str, Any]) -> float | None:
    if set(comparison) != {"with_persistence"}:
        raise ValueError("formal extraction run has an unexpected persistence variant")
    episodes = comparison["with_persistence"].get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("formal extraction run has no episode results")
    values = []
    for episode in episodes:
        timing = episode.get("timing") if isinstance(episode, dict) else None
        value = timing.get("wall_time_s") if isinstance(timing, dict) else None
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
            return None
        values.append(float(value))
    return sum(values)


def _operation_recovery_duration(run_dir: Path) -> int:
    paths = tuple(sorted(run_dir.rglob("rsimem_semantic_operations.jsonl")))
    if not paths:
        return 0
    merged = AppendOnlyOperationEvidenceLog()
    for path in paths:
        for event in AppendOnlyOperationEvidenceLog(path).events:
            merged.append(event)
    graph = materialize_operation_graph(merged.events)
    return sum(
        operation.latency_ms
        for operation in graph.operations
        if operation.kind == OperationKind.RECOVERY
    )


def _usage_bucket(value: object, *, complete: bool) -> dict[str, object]:
    if value is None:
        observed = None
    elif (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and value >= 0
    ):
        observed = value
    else:
        raise ValueError("raw usage bucket must be non-negative or unknown")
    return {
        "value": observed if complete else None,
        "observedValue": observed,
        "complete": bool(complete and observed is not None),
    }


def _raw_usage(
    comparison: dict[str, Any],
    audit: dict[str, Any],
    ledger: tuple[dict[str, Any], ...],
    run_dir: Path,
) -> dict[str, dict[str, object]]:
    physical = audit.get("uniquePhysicalUsage")
    if not isinstance(physical, dict):
        physical = {}
    issue_kinds = {
        issue.get("kind")
        for issue in audit.get("issues", [])
        if isinstance(issue, dict)
    }
    model_detail_complete = "incomplete_model_usage" not in issue_kinds
    usage = {
        field: _usage_bucket(
            physical.get(field),
            complete=(
                True if field in {"requests", "retries"} else model_detail_complete
            ),
        )
        for field in _MODEL_USAGE_FIELDS
    }
    wall_time = _episode_wall_time(comparison)
    storage = [
        sum(
            int(data.get(field) or 0)
            for field in ("memoryFilesBytes", "skillFilesBytes", "stateDbBytes")
        )
        for event in ledger
        if event.get("kind") == "storage_snapshot"
        and isinstance((data := event.get("data")), dict)
    ]
    injected_chars = sum(
        int(data.get("contentChars") or 0)
        for event in ledger
        if event.get("kind") == "memory_injection"
        and isinstance((data := event.get("data")), dict)
    )
    usage.update({
        "wallTimeSeconds": _usage_bucket(wall_time, complete=wall_time is not None),
        "peakStoredBytes": _usage_bucket(max(storage, default=0), complete=True),
        "injectedChars": _usage_bucket(injected_chars, complete=True),
        "recoveryDurationMs": _usage_bucket(
            _operation_recovery_duration(run_dir), complete=True
        ),
    })
    ingestion = audit.get("ingestionUsage")
    ingestion_values = ingestion if isinstance(ingestion, dict) else {}
    ingestion_complete = ingestion_values.get("complete")
    ingestion_complete = (
        ingestion_complete if isinstance(ingestion_complete, dict) else {}
    )
    fields = {
        "ingestionModelRequests": ("modelRequests", True),
        "ingestionInputTokens": (
            "inputTokens", ingestion_complete.get("inputTokens") is True
        ),
        "ingestionOutputTokens": (
            "outputTokens", ingestion_complete.get("outputTokens") is True
        ),
        "ingestionCacheReadTokens": (
            "cacheReadTokens", ingestion_complete.get("cacheReadTokens") is True
        ),
        "ingestionCacheWriteTokens": (
            "cacheWriteTokens", ingestion_complete.get("cacheWriteTokens") is True
        ),
        "ingestionReasoningTokens": (
            "reasoningTokens", ingestion_complete.get("reasoningTokens") is True
        ),
        "ingestionRetries": ("retries", True),
        "ingestionDurationMs": (
            "durationMs", ingestion_complete.get("durationMs") is True
        ),
        "ingestionStorageBytes": ("storageBytes", True),
    }
    for output, (source, complete) in fields.items():
        usage[output] = _usage_bucket(
            ingestion_values.get(source),
            complete=bool(ingestion_values) and complete,
        )
    return usage


def _primary_feedback(
    records: tuple[LiveExtractionFeedbackRecord, ...],
) -> tuple[Any, ...]:
    examples = []
    identities: dict[str, str] = {}
    for record in records:
        primary = next(example for example in record.dataset.examples if example.primary)
        canonical = json.dumps(
            record.dataset.payload(),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        previous = identities.get(primary.primary_unit_id)
        if previous is not None and previous != canonical:
            raise ValueError("conflicting primary extraction feedback unit")
        if previous is None:
            examples.append(primary)
        identities[primary.primary_unit_id] = canonical
    return tuple(examples)


def _quality(
    sources: tuple[ExtractionSourceRecord, ...],
    feedback: tuple[LiveExtractionFeedbackRecord, ...],
    safety_failures: int,
) -> dict[str, object]:
    status = Counter(record.source.status.value for record in sources)
    primary = _primary_feedback(feedback)
    labels = Counter(example.label.value for example in primary)
    nonempty = status[ExtractionSetStatus.NONEMPTY.value]
    empty = status[ExtractionSetStatus.EMPTY.value]
    resolved = (
        labels[ExtractionFeedbackLabel.USEFUL.value]
        + labels[ExtractionFeedbackLabel.HARMFUL.value]
    )
    harmed_sources = {
        example.source_id
        for example in primary
        if example.label == ExtractionFeedbackLabel.HARMFUL
    }
    quality_issues = Counter(
        fact.quality_issue.value
        for record in sources
        for fact in record.source.facts
        if fact.quality_issue is not None
    )
    return {
        "completedSourceCount": len(sources),
        "eligibleOpportunityCount": len(primary),
        "usefulCount": labels[ExtractionFeedbackLabel.USEFUL.value],
        "harmfulCount": labels[ExtractionFeedbackLabel.HARMFUL.value],
        "harmedSourceCount": len(harmed_sources),
        "missedCount": labels[ExtractionFeedbackLabel.MISSED.value],
        "unresolvedCount": labels[ExtractionFeedbackLabel.UNRESOLVED.value],
        "censoredCount": labels[ExtractionFeedbackLabel.CENSORED.value],
        "resolvedCount": resolved,
        "resolvedUsefulRate": (
            labels[ExtractionFeedbackLabel.USEFUL.value] / resolved
            if resolved else None
        ),
        "observedHarmfulRate": (
            len(harmed_sources) / nonempty if nonempty else None
        ),
        "sourceStatusCounts": {
            value.value: status[value.value] for value in ExtractionSetStatus
        },
        "nonemptyCoverage": nonempty / len(sources) if sources else None,
        "emptyExtractionRate": empty / len(sources) if sources else None,
        "highConfidenceMissedRate": None,
        "missedAssessability": "unknown",
        "qualityIssueCounts": {
            value.value: quality_issues[value.value]
            for value in ExtractionQualityIssue
        },
        "schemaSafetyFailureCount": safety_failures,
    }


def _quality_summary(rows: tuple[dict[str, Any], ...]) -> dict[str, object]:
    counts = {
        field: sum(int(row["quality"][field]) for row in rows)
        for field in (
            "completedSourceCount",
            "eligibleOpportunityCount",
            "usefulCount",
            "harmfulCount",
            "harmedSourceCount",
            "missedCount",
            "unresolvedCount",
            "censoredCount",
            "resolvedCount",
            "schemaSafetyFailureCount",
        )
    }
    statuses = {
        status.value: sum(
            int(row["quality"]["sourceStatusCounts"][status.value])
            for row in rows
        )
        for status in ExtractionSetStatus
    }
    issues = {
        issue.value: sum(
            int(row["quality"]["qualityIssueCounts"][issue.value])
            for row in rows
        )
        for issue in ExtractionQualityIssue
    }
    total = counts["completedSourceCount"]
    resolved = counts["resolvedCount"]
    nonempty = statuses[ExtractionSetStatus.NONEMPTY.value]
    empty = statuses[ExtractionSetStatus.EMPTY.value]
    return {
        **counts,
        "resolvedUsefulRate": (
            counts["usefulCount"] / resolved if resolved else None
        ),
        "observedHarmfulRate": (
            counts["harmedSourceCount"] / nonempty if nonempty else None
        ),
        "sourceStatusCounts": statuses,
        "nonemptyCoverage": nonempty / total if total else None,
        "emptyExtractionRate": empty / total if total else None,
        "highConfidenceMissedRate": None,
        "missedAssessability": "unknown",
        "qualityIssueCounts": issues,
    }


def _safety_failures(audit: dict[str, Any]) -> int:
    issues = audit.get("issues")
    if not isinstance(issues, list):
        return 1
    return sum(
        isinstance(issue, dict)
        and issue.get("kind") != "incomplete_model_usage"
        for issue in issues
    )


def classify_extraction_audit_failure(audit: dict[str, Any]) -> str:
    """Distinguish an all-provider-error run from an evidence/audit failure."""

    statuses = audit.get("modelCallStatuses")
    if isinstance(statuses, dict) and statuses:
        error_count = statuses.get("error")
        non_error = sum(
            value
            for key, value in statuses.items()
            if key != "error" and type(value) is int and value >= 0
        )
        if type(error_count) is int and error_count > 0 and non_error == 0:
            return "provider"
    return "audit"


def _run_evidence(
    batch_root: Path,
    manifest: dict[str, Any],
    attempt: dict[str, Any],
) -> dict[str, Any]:
    run_dir = (batch_root / attempt["outputDirectory"]).resolve()
    if not run_dir.is_relative_to(batch_root):
        raise ValueError("formal extraction run directory escapes its batch")
    comparison = _read_json(run_dir / "sequence_comparison.json", "sequence results")
    audit = _read_json(run_dir / "audit.json", "audit report")
    ledger = _read_jsonl(run_dir / "ledger.jsonl")
    sources = _source_records(run_dir)
    feedback = _feedback_records(run_dir)
    process_events = _process_events(run_dir)
    split = manifest["split"]
    expected_artifact = manifest["semanticPolicy"]["activeArtifactByMethod"][
        attempt["method"]
    ]
    expected_policy = SemanticPolicyManifest.from_payload(
        manifest["semanticPolicy"][
            "parent"
            if attempt["method"] == EXTRACTION_METHOD_VARIANTS[0]
            else "active"
        ]
    )
    expected_scope = (
        ExtractionPromptRuntimeScope.ROOT_STATIC
        if attempt["method"] == EXTRACTION_METHOD_VARIANTS[0]
        else ExtractionPromptRuntimeScope.MATCHED_VALIDATION
    )
    expected_contract = manifest["feedbackContract"]["contractDigest"]
    sources_by_id = {record.record_id: record for record in sources}
    if any(
        record.family_id != split["familyId"]
        or record.extraction_artifact_id != expected_artifact["artifactId"]
        or record.extraction_artifact_digest != expected_artifact["artifactDigest"]
        or record.activation.semantic_policy != expected_policy
        or record.activation.runtime_binding.deployment_scope != expected_scope
        or record.activation.persisted_artifact_ids != record.artifact_ids
        for record in sources
    ):
        raise ValueError("run extraction source identity differs from its manifest")
    for record in feedback:
        source_record = sources_by_id.get(record.source_record_id)
        source = source_record.source if source_record is not None else None
        source_fact_ids = {
            fact.fact_id for fact in source.facts
        } if source is not None else set()
        source_artifact_ids = set(source_record.artifact_ids) if source_record else set()
        if (
            source_record is None
            or source is None
            or record.family_id != split["familyId"]
            or record.family_id != source_record.family_id
            or record.run_id != source_record.run_id
            or record.dataset.contract_digest != expected_contract
            or record.dataset.source_projection_digest
            != source.source_projection_digest
            or any(
                example.source_id != source.source_id
                or example.extraction_set_id != source.extraction_set_id
                or (
                    example.fact_id is not None
                    and example.fact_id not in source_fact_ids
                )
                or not set(example.artifact_ids).issubset(source_artifact_ids)
                for example in record.dataset.examples
            )
        ):
            raise ValueError("run feedback evidence does not join its manifest/source")
    safety = _safety_failures(audit)
    return {
        "replicate": attempt["replicate"],
        "method": attempt["method"],
        "runName": attempt["runName"],
        "sources": sources,
        "feedback": feedback,
        "processEvents": process_events,
        "quality": _quality(sources, feedback, safety),
        "rawUsage": _raw_usage(comparison, audit, ledger, run_dir),
    }


def _source_map(row: dict[str, Any]) -> dict[tuple[str, str], ExtractionSourceRecord]:
    result = {}
    for record in row["sources"]:
        key = (record.stage, record.task_id)
        if key in result:
            raise ValueError("run has duplicate completed-task extraction identity")
        result[key] = record
    return result


def _activation_funnel(rows: tuple[dict[str, Any], ...]) -> dict[str, int]:
    by_method: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_method[row["method"]][row["replicate"]] = row
    if any(method not in by_method for method in EXTRACTION_METHOD_VARIANTS):
        return {
            "eligible": 0,
            "renderedNPlus1": 0,
            "changedExtraction": 0,
            "noIntervention": 0,
            "changedArtifact": 0,
            "futureExposure": 0,
            "attributableUse": 0,
            "attributableOutcome": 0,
        }
    static = by_method[EXTRACTION_METHOD_VARIANTS[0]]
    adaptive = by_method[EXTRACTION_METHOD_VARIANTS[1]]
    if set(static) != set(adaptive):
        raise ValueError("static/adaptive extraction replicates do not match")
    eligible = rendered = changed_output = changed_artifact = 0
    no_intervention = 0
    changed_active_ids: set[str] = set()
    adaptive_feedback: list[LiveExtractionFeedbackRecord] = []
    for replicate in sorted(static):
        parent_sources = _source_map(static[replicate])
        active_sources = _source_map(adaptive[replicate])
        if set(parent_sources) != set(active_sources):
            raise ValueError("static/adaptive completed extraction sources do not match")
        adaptive_feedback.extend(adaptive[replicate]["feedback"])
        for key in sorted(parent_sources):
            parent = parent_sources[key]
            active = active_sources[key]
            eligible += 1
            if (
                active.activation.runtime_binding.deployment_scope
                == ExtractionPromptRuntimeScope.MATCHED_VALIDATION
                and active.activation.invocation.binding_id
                == active.activation.runtime_binding.binding_id
            ):
                rendered += 1
            if (
                parent.activation.parsed_output_digest
                != active.activation.parsed_output_digest
            ):
                changed_output += 1
                if (
                    parent.activation.persisted_artifact_ids
                    != active.activation.persisted_artifact_ids
                    or parent.source.status != active.source.status
                    or tuple(fact.disposition for fact in parent.source.facts)
                    != tuple(fact.disposition for fact in active.source.facts)
                ):
                    changed_artifact += 1
                    changed_active_ids.add(active.record_id)
            else:
                no_intervention += 1
    primary = [
        next(example for example in record.dataset.examples if example.primary)
        for record in adaptive_feedback
        if record.source_record_id in changed_active_ids
    ]
    exposure = sum(
        example.exposure_mode != ExposureMode.NOT_EXPOSED for example in primary
    )
    attributable = [
        example for example in primary
        if example.label == ExtractionFeedbackLabel.USEFUL
        or "memory_use_harmfully_attributed" in example.reason_codes
    ]
    return {
        "eligible": eligible,
        "renderedNPlus1": rendered,
        "changedExtraction": changed_output,
        "noIntervention": no_intervention,
        "changedArtifact": changed_artifact,
        "futureExposure": exposure,
        "attributableUse": len(attributable),
        "attributableOutcome": len(attributable),
    }


def _usage_summary(rows: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    result = {}
    for field in _RAW_USAGE_FIELDS:
        values = [row["rawUsage"][field]["value"] for row in rows]
        known = [
            value for value in values
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ]
        result[field] = {
            "values": values,
            "completeCount": len(known),
            "unknownCount": len(values) - len(known),
            "mean": mean(known) if known else None,
        }
    return result


def _process_corpus_summary(rows: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    """Summarize process observations separately from evaluation quality."""

    events = [event for row in rows for event in row["processEvents"]]
    by_kind = Counter(event.kind.value for event in events)
    by_reason = Counter(reason for event in events for reason in event.reason_codes)
    return {
        "eventCount": len(events),
        "eventIds": sorted(event.event_id for event in events),
        "byKind": dict(sorted(by_kind.items())),
        "byReason": dict(sorted(by_reason.items())),
        "evaluationScoreAccessible": False,
    }


def _paired_usage_delta(rows: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    by_method: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_method[row["method"]][row["replicate"]] = row
    if any(method not in by_method for method in EXTRACTION_METHOD_VARIANTS):
        return {}
    static = by_method[EXTRACTION_METHOD_VARIANTS[0]]
    adaptive = by_method[EXTRACTION_METHOD_VARIANTS[1]]
    if set(static) != set(adaptive):
        raise ValueError("paired raw usage replicates do not match")
    result = {}
    for field in _RAW_USAGE_FIELDS:
        values = []
        for replicate in sorted(static):
            left = static[replicate]["rawUsage"][field]
            right = adaptive[replicate]["rawUsage"][field]
            if left["complete"] is True and right["complete"] is True:
                values.append(right["value"] - left["value"])
            else:
                values.append(None)
        known = [value for value in values if value is not None]
        result[field] = {
            "values": values,
            "completeCount": len(known),
            "unknownCount": len(values) - len(known),
            "mean": mean(known) if known else None,
        }
    return result


def _claim(funnel: dict[str, int], safety_failures: int) -> dict[str, object]:
    required = (
        "eligible",
        "renderedNPlus1",
        "changedExtraction",
        "changedArtifact",
        "futureExposure",
        "attributableUse",
        "attributableOutcome",
    )
    missing = tuple(field for field in required if funnel[field] < 1)
    if safety_failures:
        return {
            "eligible": False,
            "reason": "schema_or_safety_failure",
            "missingStages": list(missing),
        }
    if missing:
        return {
            "eligible": False,
            "reason": "activation_funnel_incomplete",
            "missingStages": list(missing),
        }
    return {"eligible": True, "reason": "complete_activation_funnel", "missingStages": []}


def analyze_extraction_batch(batch_root: Path) -> dict[str, Any]:
    root = batch_root.expanduser().resolve()
    manifest = load_extraction_manifest(root / "batch_manifest.json")
    terminal_attempts = _terminal_attempts(manifest)
    attempts = tuple(
        attempt for attempt in terminal_attempts
        if attempt["status"] == "completed"
    )
    failed_attempts = tuple(
        attempt for attempt in terminal_attempts
        if attempt["status"] == "failed"
    )
    rows = tuple(
        _run_evidence(root, manifest, attempt)
        for attempt in sorted(
            attempts,
            key=lambda value: (value["replicate"], value["ordinal"]),
        )
    )
    by_method = {
        method: tuple(row for row in rows if row["method"] == method)
        for method in manifest["methods"]
    }
    funnel = _activation_funnel(rows)
    safety_failures = sum(
        int(row["quality"]["schemaSafetyFailureCount"]) for row in rows
    )
    completed_slots = {(row["replicate"], row["method"]) for row in rows}
    expected_slots = {
        (replicate, method)
        for replicate in range(1, manifest["replicates"] + 1)
        for method in manifest["methods"]
    }
    return {
        "schemaVersion": EXTRACTION_ANALYSIS_SCHEMA_VERSION,
        "analysisSchema": EXTRACTION_ANALYSIS_SCHEMA,
        "batchId": manifest["batchId"],
        "experimentId": manifest["experimentId"],
        "phase": manifest["phase"],
        "split": manifest["split"],
        "qualityReady": (
            completed_slots == expected_slots
            and safety_failures == 0
            and bool(rows)
            and all(
                row["quality"]["completedSourceCount"] > 0
                and row["quality"]["eligibleOpportunityCount"] > 0
                for row in rows
            )
        ),
        "usageComplete": bool(rows) and all(
            bucket["complete"]
            for row in rows
            for bucket in row["rawUsage"].values()
        ),
        "failedAttempts": [{
            "replicate": attempt["replicate"],
            "method": attempt["method"],
            "attemptNumber": attempt["attemptNumber"],
            "runName": attempt["runName"],
            "failureStage": attempt["failureStage"],
        } for attempt in failed_attempts],
        "runs": [{
            "replicate": row["replicate"],
            "method": row["method"],
            "runName": row["runName"],
            "quality": row["quality"],
            "rawUsage": row["rawUsage"],
            "processFeedback": [event.payload() for event in row["processEvents"]],
        } for row in rows],
        "summaryByMethod": {
            method: {
                "sampleSize": len(method_rows),
                "quality": _quality_summary(method_rows),
                "rawUsage": _usage_summary(method_rows),
            }
            for method, method_rows in by_method.items()
        },
        "activationFunnel": funnel,
        "processCorpus": _process_corpus_summary(rows),
        "pairedRawUsageDelta": _paired_usage_delta(rows),
        "claimGate": {
            "operationAttributedExtractionAdaptation": _claim(
                funnel, safety_failures
            ),
        },
        "providerPricing": None,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze one formal extraction-prompt experiment batch",
    )
    parser.add_argument("batch_root", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = analyze_extraction_batch(args.batch_root)
    serialized = json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(serialized, end="")
    else:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
