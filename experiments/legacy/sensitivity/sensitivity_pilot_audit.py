"""Content-free integrity and usage audit for one sensitivity pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected JSON object event: {path}")
        values.append(value)
    return values


def _usage(events: list[dict[str, Any]], trace_id: str) -> tuple[dict[str, int], list[str]]:
    calls = [item for item in events if item.get("type") == "model_call_usage"]
    ends = [item for item in events if item.get("type") == "trace_end"]
    issues: list[str] = []
    if len(ends) != 1:
        return {}, ["trace_end_missing_or_duplicate"]
    end = ends[0]
    totals: dict[str, int] = {}
    for output, source, trace_end_field in (
        ("input_tokens", "input_tokens", "model_input_tokens"),
        ("output_tokens", "output_tokens", "model_output_tokens"),
        ("cache_read_tokens", "cache_read_tokens", "cache_read_tokens"),
        ("cache_write_tokens", "cache_write_tokens", "cache_write_tokens"),
        ("reasoning_tokens", "reasoning_tokens", "reasoning_tokens"),
    ):
        values = [item.get("usage", {}).get(source) for item in calls]
        if not all(type(value) is int and value >= 0 for value in values):
            issues.append("usage_incomplete")
            continue
        totals[output] = sum(values)
        if end.get(trace_end_field) != totals[output]:
            issues.append("usage_total_mismatch:" + output)
    totals["requests"] = len(calls)
    totals["retries"] = sum(int(item.get("attempt") or 1) - 1 for item in calls)
    if end.get("model_request_count") != totals["requests"]:
        issues.append("usage_total_mismatch:requests")
    if end.get("model_retry_count") != totals["retries"]:
        issues.append("usage_total_mismatch:retries")
    if end.get("model_usage_complete") is not True:
        issues.append("usage_incomplete")
    if end.get("trace_id") != trace_id:
        issues.append("trace_identity_mismatch")
    return totals, sorted(set(issues))


def audit_sensitivity_pilot(output_root: Path) -> dict[str, object]:
    """Audit five registered condition runs without retaining benchmark content."""

    root = Path(output_root).expanduser().resolve()
    plan = _read_json(root / "sensitivity_pilot_plan.json")
    manifest = _read_json(root / "sensitivity_manifest.json")
    probe = _read_json(root / "provider_probe.json")
    events = _read_jsonl(root / "sensitivity_pilot_events.jsonl")
    expected_runs = plan.get("run_ids")
    conditions = plan.get("condition_order")
    if not isinstance(expected_runs, list) or not isinstance(conditions, list) or len(expected_runs) != len(conditions):
        raise ValueError("sensitivity pilot plan is malformed")
    completed = {
        item.get("run_id") for item in events
        if item.get("status") == "completed" and item.get("return_code") == 0
    }
    runs = {item.get("run_id"): item for item in manifest.get("runs", []) if isinstance(item, dict)}
    rows: list[dict[str, object]] = []
    all_issues: list[str] = []
    for run_id, condition in zip(expected_runs, conditions, strict=True):
        row_issues: list[str] = []
        run = runs.get(run_id)
        if not isinstance(run, dict):
            row_issues.append("manifest_run_missing")
            run = {}
        trace_relative = run.get("trace_directory")
        if not isinstance(trace_relative, str):
            row_issues.append("trace_directory_missing")
            trace_root = root
        else:
            trace_root = (root / trace_relative).resolve()
            if not trace_root.is_relative_to(root):
                row_issues.append("trace_directory_escapes_root")
        if run_id not in completed:
            row_issues.append("run_not_completed")
        try:
            result = _read_json(trace_root / "sequence_results.json")
            episodes = result.get("episodes")
            if not isinstance(episodes, list) or not episodes:
                raise ValueError("episode results missing")
        except (OSError, ValueError, json.JSONDecodeError):
            episodes = []
            row_issues.append("sequence_results_missing")
        totals = {key: 0 for key in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens", "reasoning_tokens", "requests", "retries")}
        trace_count = 0
        memory_event_count = 0
        for episode in episodes:
            if not isinstance(episode, dict) or not isinstance(episode.get("trace"), str) or not isinstance(episode.get("trace_id"), str):
                row_issues.append("episode_trace_missing")
                continue
            trace_path = Path(episode["trace"]).resolve()
            if not trace_path.is_relative_to(trace_root):
                row_issues.append("episode_trace_escapes_root")
                continue
            try:
                values, issues = _usage(_read_jsonl(trace_path), episode["trace_id"])
            except (OSError, ValueError, json.JSONDecodeError):
                row_issues.append("trace_unreadable")
                continue
            trace_count += 1
            row_issues.extend(issues)
            for key, value in values.items():
                totals[key] += value
            memory_path = trace_path.parent / "artifacts" / "rsimem_memory_events.jsonl"
            if memory_path.exists():
                memory_events = _read_jsonl(memory_path)
                memory_event_count += len(memory_events)
                if any(item.get("taskId") != run.get("method_task_id") for item in memory_events):
                    row_issues.append("memory_event_method_task_id_mismatch")
        row = {
            "run_id": run_id,
            "condition": condition,
            "trace_count": trace_count,
            "memory_event_count": memory_event_count,
            "usage": totals,
            "issues": sorted(set(row_issues)),
            "ok": not row_issues,
        }
        rows.append(row)
        all_issues.extend(row["issues"])
    report = {
        "schema": "rsimem-sensitivity-pilot-audit-v1",
        "pilot_id": plan.get("pilot_id"),
        "provider_probe_ok": probe.get("ok") is True,
        "run_count": len(rows),
        "runs": rows,
        "issues": sorted(set(all_issues)),
    }
    report["ok"] = report["provider_probe_ok"] and not report["issues"] and all(row["ok"] for row in rows)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = audit_sensitivity_pilot(args.output_root)
    args.output.write_text(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["audit_sensitivity_pilot"]
