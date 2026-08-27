"""Audit one request-accounted RSIMem run without exposing experiment content."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .ledger import resolve_comparison_evidence_path


_SECRET_PATTERNS = {
    "openai_style": re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    "github": re.compile(r"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})"),
    "aws": re.compile(r"AKIA[0-9A-Z]{16}"),
    "bearer": re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/-]{24,}"),
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, dict):
            events.append(value)
    return events


def _call_sum(calls: list[dict[str, Any]], field: str) -> int | None:
    values = [call.get("usage", {}).get(field) for call in calls]
    if any(value is None for value in values):
        return None
    return sum(int(value) for value in values)


def summarize_ingestion_usage(ledger: list[dict[str, Any]]) -> dict[str, Any]:
    """Audit content-free ingestion component usage without double billing it."""

    events = [event for event in ledger if event.get("kind") == "memory_ingestion"]
    by_execution: dict[str, str] = {}
    unique: list[dict[str, Any]] = []
    for event in events:
        data = event.get("data")
        if not isinstance(data, dict):
            raise ValueError("memory ingestion ledger event requires data")
        execution_id = data.get("executionId")
        resources = data.get("resources")
        if not isinstance(execution_id, str) or not execution_id:
            raise ValueError("memory ingestion ledger event requires executionId")
        if not isinstance(resources, dict):
            raise ValueError("memory ingestion ledger event requires resources")
        canonical = json.dumps(data, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        existing = by_execution.get(execution_id)
        if existing is not None:
            if existing != canonical:
                raise ValueError(f"conflicting memory ingestion execution: {execution_id}")
            continue
        by_execution[execution_id] = canonical
        unique.append(data)

    resource_fields = {
        "inputTokens": "inputTokens",
        "outputTokens": "outputTokens",
        "cacheReadTokens": "cacheReadTokens",
        "cacheWriteTokens": "cacheWriteTokens",
        "reasoningTokens": "reasoningTokens",
        "modelRequests": "modelRequests",
        "retryCount": "retries",
        "durationMs": "durationMs",
        "storageBytes": "storageBytes",
    }
    totals = {output: 0 for output in resource_fields.values()}
    completeness = {
        output: True
        for field, output in resource_fields.items()
        if field in {
            "inputTokens",
            "outputTokens",
            "cacheReadTokens",
            "cacheWriteTokens",
            "reasoningTokens",
            "durationMs",
        }
    }
    statuses = Counter()
    outcomes = Counter()
    for data in unique:
        resources = data["resources"]
        statuses.update((str(data.get("status")),))
        outcomes.update((str(data.get("outcome")),))
        for source, output in resource_fields.items():
            value = resources.get(source)
            if value is None:
                if output in completeness:
                    completeness[output] = False
                continue
            if type(value) is not int or value < 0:
                raise ValueError(f"invalid memory ingestion resource: {source}")
            totals[output] += value
    return {
        "events": len(events),
        "uniqueExecutions": len(unique),
        "duplicateViews": len(events) - len(unique),
        **totals,
        "complete": completeness,
        "statuses": dict(statuses),
        "outcomes": dict(outcomes),
    }


def audit_run(run_dir: Path) -> dict[str, Any]:
    """Return a privacy-safe reconciliation report for one completed run."""
    run_dir = run_dir.resolve()
    comparison = _read_json(run_dir / "sequence_comparison.json")
    ledger_path = run_dir / "ledger.jsonl"
    ledger_text = ledger_path.read_text(encoding="utf-8")
    ledger = _read_jsonl(ledger_path)
    issues: list[dict[str, Any]] = []

    episodes_by_variant: dict[str, list[dict[str, Any]]] = {}
    trace_paths: dict[str, Path] = {}
    memory_entries: list[str] = []
    for variant in ("with_persistence", "without_persistence"):
        episodes = [
            episode for episode in comparison.get(variant, {}).get("episodes", [])
            if isinstance(episode, dict)
        ]
        episodes_by_variant[variant] = episodes
        for episode in episodes:
            trace_id = str(episode.get("trace_id") or "")
            if trace_id:
                trace_paths.setdefault(
                    trace_id,
                    resolve_comparison_evidence_path(episode.get("trace"), run_dir),
                )
            artifacts = episode.get("artifacts", {})
            for field in ("memory_entries", "user_entries"):
                memory_entries.extend(
                    entry for entry in artifacts.get(field, [])
                    if isinstance(entry, str) and entry
                )

    totals = Counter()
    statuses = Counter()
    components = Counter()
    purposes = Counter()
    for trace_id, path in trace_paths.items():
        if not path.exists():
            issues.append({"kind": "missing_trace", "traceId": trace_id})
            continue
        events = _read_jsonl(path)
        calls = [event for event in events if event.get("type") == "model_call_usage"]
        ends = [event for event in events if event.get("type") == "trace_end"]
        if not ends:
            issues.append({"kind": "missing_trace_end", "traceId": trace_id})
            continue
        end = ends[-1]
        expected = {
            "model_input_tokens": _call_sum(calls, "input_tokens"),
            "model_output_tokens": _call_sum(calls, "output_tokens"),
            "cache_read_tokens": _call_sum(calls, "cache_read_tokens"),
            "cache_write_tokens": _call_sum(calls, "cache_write_tokens"),
            "reasoning_tokens": _call_sum(calls, "reasoning_tokens"),
            "model_request_count": len(calls),
            "model_retry_count": sum(1 for call in calls if int(call.get("attempt") or 1) > 1),
        }
        for field, value in expected.items():
            if end.get(field) != value:
                issues.append({
                    "kind": "trace_total_mismatch",
                    "traceId": trace_id,
                    "field": field,
                    "requestTotal": value,
                    "traceEnd": end.get(field),
                })
        if not end.get("model_usage_complete", False):
            issues.append({"kind": "incomplete_model_usage", "traceId": trace_id})
        totals.update({
            "inputTokens": expected["model_input_tokens"] or 0,
            "outputTokens": expected["model_output_tokens"] or 0,
            "cacheReadTokens": expected["cache_read_tokens"] or 0,
            "cacheWriteTokens": expected["cache_write_tokens"] or 0,
            "reasoningTokens": expected["reasoning_tokens"] or 0,
            "requests": expected["model_request_count"],
            "retries": expected["model_retry_count"],
        })
        statuses.update(str(call.get("status")) for call in calls)
        components.update(str(call.get("component")) for call in calls)
        purposes.update(str(call.get("purpose")) for call in calls)

    ledger_calls = [event for event in ledger if event.get("kind") == "model_call_usage"]
    ingestion_usage = summarize_ingestion_usage(ledger)
    billing_ids = [event.get("data", {}).get("billingExecutionId") for event in ledger_calls]
    billing_ids = [value for value in billing_ids if isinstance(value, str) and value]
    if len(set(billing_ids)) != totals["requests"]:
        issues.append({
            "kind": "billing_call_count_mismatch",
            "uniqueLedgerCalls": len(set(billing_ids)),
            "traceCalls": totals["requests"],
        })

    unresolved_injections = sum(
        1 for event in ledger if event.get("kind") == "memory_injection_unresolved"
    )
    if unresolved_injections:
        issues.append({
            "kind": "unresolved_memory_injection",
            "count": unresolved_injections,
        })
    projection_checks = [
        event for event in ledger if event.get("kind") == "projection_check"
    ]
    projection_mismatches = sum(
        1
        for event in projection_checks
        if event.get("data", {}).get("attributes", {}).get("equivalent") is not True
    )
    if projection_mismatches:
        issues.append({"kind": "projection_mismatch", "count": projection_mismatches})
    adapter_bypasses = sum(
        1
        for event in ledger
        if event.get("data", {}).get("reasonCode") == "adapter_failure_native_bypass"
    )
    if adapter_bypasses:
        issues.append({"kind": "adapter_native_bypass", "count": adapter_bypasses})

    memory_leaks = sum(1 for entry in memory_entries if entry in ledger_text)
    if memory_leaks:
        issues.append({"kind": "memory_text_leak", "count": memory_leaks})
    secret_hits = {
        name: len(pattern.findall(ledger_text))
        for name, pattern in _SECRET_PATTERNS.items()
    }
    for name, count in secret_hits.items():
        if count:
            issues.append({"kind": "credential_pattern", "pattern": name, "count": count})
    absolute_source_paths = sum(
        1
        for event in ledger
        for key, value in event.get("source", {}).items()
        if key.lower().endswith("path") and isinstance(value, str) and value.startswith("/")
    )
    if absolute_source_paths:
        issues.append({"kind": "absolute_source_path", "count": absolute_source_paths})

    variant_totals = {}
    for variant, episodes in episodes_by_variant.items():
        usage_rows = [episode.get("token_usage", {}) for episode in episodes]
        variant_totals[variant] = {
            "inputTokens": sum(int(row.get("input_tokens") or 0) for row in usage_rows),
            "outputTokens": sum(int(row.get("output_tokens") or 0) for row in usage_rows),
            "cacheReadTokens": sum(int(row.get("cache_read_tokens") or 0) for row in usage_rows),
            "cacheWriteTokens": sum(int(row.get("cache_write_tokens") or 0) for row in usage_rows),
            "reasoningTokens": sum(int(row.get("reasoning_tokens") or 0) for row in usage_rows),
            "requests": sum(int(row.get("model_request_count") or 0) for row in usage_rows),
            "retries": sum(int(row.get("model_retry_count") or 0) for row in usage_rows),
        }

    return {
        "schemaVersion": 1,
        "runId": run_dir.name,
        "ok": not issues,
        "issues": issues,
        "uniqueTraceCount": len(trace_paths),
        "uniquePhysicalUsage": dict(totals),
        "variantUsageIncludingShared": variant_totals,
        "modelCallStatuses": dict(statuses),
        "modelCallComponents": dict(components),
        "modelCallPurposes": dict(purposes),
        "ledgerModelCallViews": len(ledger_calls),
        "ledgerUniqueBillingCalls": len(set(billing_ids)),
        "ledgerDuplicateViews": len(ledger_calls) - len(set(billing_ids)),
        "ingestionUsage": ingestion_usage,
        "projectionChecks": len(projection_checks),
        "projectionMismatches": projection_mismatches,
        "adapterNativeBypasses": adapter_bypasses,
        "unresolvedMemoryInjections": unresolved_injections,
        "privacy": {
            "memoryTextLeaks": memory_leaks,
            "credentialPatternHits": secret_hits,
            "absoluteSourcePaths": absolute_source_paths,
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Completed run directory")
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args(argv)
    report = audit_run(args.run_dir)
    rendered = json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
