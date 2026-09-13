"""Formal full-suite launcher and audit for the three static memory backends."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

from .base_memory_experiment import (
    HISTORICAL_BASE_MEMORY_CONDITIONS,
    BaseMemoryCondition,
    FROZEN_MODEL_ID,
)
from .base_memory_launcher import run_comparison


SCHEMA = "rsimem-base-memory-full-suite-formal-manifest-v1"
# This module owns the frozen historical three-backend comparison. The new
# AllMemoryOff formal baseline has a separate one-condition manifest.
BACKEND_ORDER = HISTORICAL_BASE_MEMORY_CONDITIONS


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"formal evidence must be an object: {path}")
    return value


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build_formal_manifest(*, source_formal_manifest: Path, output_root: Path) -> dict[str, object]:
    """Bind the static-baseline protocol to the already frozen 26-family suite."""
    source = _load(source_formal_manifest)
    if source.get("schema") != "rsimem-adamem-full-suite-formal-manifest-v1":
        raise ValueError("source must be the frozen AdaMem full-suite formal manifest")
    if source.get("family_count") != 26 or source.get("replicate_count") != 3:
        raise ValueError("base-memory formal suite requires 26 families and three replicates")
    if source.get("base_model") != FROZEN_MODEL_ID:
        raise ValueError("base-memory formal suite model drift")
    families = source.get("families")
    if not isinstance(families, list) or len(families) != 26:
        raise ValueError("source formal manifest family list is incomplete")
    frozen: list[dict[str, object]] = []
    for record in families:
        if not isinstance(record, Mapping):
            raise ValueError("source formal family record is invalid")
        family_id, config, digest, category = (record.get(key) for key in ("family_id", "config", "config_digest", "category"))
        if not all(isinstance(value, str) and value for value in (family_id, config, digest, category)):
            raise ValueError("source formal family record lacks identity")
        config_path = Path(config)
        if not config_path.is_file() or _digest(config_path) != digest:
            raise ValueError(f"formal config digest drift: {family_id}")
        frozen.append({"family_id": family_id, "config": str(config_path.resolve()), "config_digest": digest, "category": category})
    if len({str(item["family_id"]) for item in frozen}) != 26:
        raise ValueError("base-memory formal family identities are not unique")
    payload: dict[str, object] = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_formal_manifest": str(source_formal_manifest.resolve()),
        "source_formal_manifest_digest": _digest(source_formal_manifest),
        "source_formal_protocol_digest": source.get("formal_manifest_digest"),
        "output_root": str(output_root.resolve()),
        "base_model": FROZEN_MODEL_ID,
        "temperature": 0.0,
        "token_budget": 4096,
        "replicate_count": 3,
        "backend_order": [item.value for item in BACKEND_ORDER],
        "family_count": 26,
        "families": frozen,
    }
    payload["formal_manifest_digest"] = _canonical_digest(payload)
    return payload


def _family(manifest: Mapping[str, object], family_id: str) -> Mapping[str, object]:
    for item in manifest.get("families", ()):
        if isinstance(item, Mapping) and item.get("family_id") == family_id:
            return item
    raise ValueError(f"family is not in the frozen base-memory manifest: {family_id}")


def run_replicate_batch(
    *, formal_manifest: Path, family_id: str, condition: BaseMemoryCondition,
    output_root: Path, past_bin: Path, past_root: Path, registry: Path,
    past_config: Path, base_url: str, batch_id: str, dry_run: bool = False,
) -> dict[str, object]:
    """Run exactly three isolated replicates for one backend-family batch."""
    manifest = _load(formal_manifest)
    if manifest.get("schema") != SCHEMA or manifest.get("replicate_count") != 3:
        raise ValueError("unsupported base-memory formal manifest")
    condition = BaseMemoryCondition(condition)
    record = _family(manifest, family_id)
    sequence = Path(str(record["config"])).resolve()
    if _digest(sequence) != record["config_digest"]:
        raise ValueError("formal config digest drift before provider execution")
    past_config = Path(past_config).resolve()
    if not past_config.is_file():
        raise ValueError("PAST runtime config is unavailable")
    root = output_root / batch_id
    starts = datetime.now(timezone.utc).isoformat()
    batch = {
        "schema": "rsimem-base-memory-replicate-batch-v1", "batch_id": batch_id,
        "formal_manifest_digest": manifest["formal_manifest_digest"], "family_id": family_id,
        "category": record["category"], "condition": condition.value, "replicate_count": 3,
        "max_workers": 3, "source_sequence": str(sequence), "source_sequence_digest": record["config_digest"],
        "runtime_config": str(past_config), "runtime_config_digest": _digest(past_config),
        "started_at": starts, "retry_reason": None,
    }
    _write(root / "batch_manifest.json", batch)

    def execute(replicate: int) -> dict[str, object]:
        replica_root = root / f"replicate-{replicate:02d}"
        prefix = f"{batch_id}-r{replicate:02d}"
        run_comparison(
            source_sequence=sequence, output_root=replica_root, family_id=family_id,
            past_bin=past_bin, past_root=past_root, config=past_config, registry=registry,
            base_url=base_url, token_budget=int(manifest["token_budget"]), run_prefix=prefix,
            condition=condition, dry_run=dry_run,
            # Family batches never overlap. These three values isolate the
            # concurrent replicate services and remain below PAST's port range.
            port_base=40000 + replicate * 1000,
        )
        run_root = replica_root / f"{prefix}-{condition.value.lower()}"
        return {"replicate": replicate, "run_root": str(run_root.relative_to(root)), "status": "accepted"}

    outcomes: dict[int, dict[str, object]] = {}
    with ThreadPoolExecutor(max_workers=3, thread_name_prefix="base-memory-replicate") as pool:
        futures = {pool.submit(execute, replicate): replicate for replicate in range(1, 4)}
        for future in as_completed(futures):
            replicate = futures[future]
            try:
                outcomes[replicate] = future.result()
            except Exception as exc:
                outcomes[replicate] = {"replicate": replicate, "status": "infrastructure_failure", "error_type": type(exc).__name__}
    accepted = len(outcomes) == 3 and all(item.get("status") == "accepted" for item in outcomes.values())
    report = {
        "schema": "rsimem-base-memory-replicate-batch-result-v1", "batch_id": batch_id,
        "accepted": accepted, "outcomes": [outcomes[index] for index in sorted(outcomes)],
        "started_at": starts, "finished_at": datetime.now(timezone.utc).isoformat(),
        "provider_health": {"status": "healthy" if accepted else "degraded", "evidence": "complete_run_usage" if accepted else "infrastructure_failure"},
        "retry_reason": None,
    }
    _write(root / "batch_result.json", report)
    return report


def _read_events(root: Path) -> dict[str, object]:
    count, kinds = 0, set()
    for path in root.rglob("rsimem_memory_events.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            if isinstance(event, Mapping):
                count += 1
                value = event.get("event_type") or event.get("kind")
                if isinstance(value, str):
                    kinds.add(value)
    return {"event_count": count, "event_kinds": sorted(kinds)}


def _mem0_formal_evidence(root: Path) -> dict[str, object]:
    """Report semantic operation applicability without rejecting cross-workflow cases."""
    kinds: set[str] = set()
    count = 0
    for path in root.rglob("rsimem_semantic_operations.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            payload = event.get("payload") if isinstance(event, Mapping) else None
            if isinstance(payload, Mapping) and isinstance(payload.get("kind"), str):
                count += 1
                kinds.add(payload["kind"])
    required = {"fact_extraction", "mutation", "retrieval", "injection"}
    return {"operation_event_count": count, "operation_kinds": sorted(kinds),
            "semantic_mechanism_applicable": required.issubset(kinds),
            "missing_required_kinds": sorted(required - kinds)}


def audit_replicate_batch(batch_root: Path) -> dict[str, object]:
    """Reject a formal batch unless all three replicates have complete evidence."""
    root = Path(batch_root).resolve()
    batch, result = _load(root / "batch_manifest.json"), _load(root / "batch_result.json")
    condition = BaseMemoryCondition(batch.get("condition"))
    outcomes = result.get("outcomes")
    if result.get("accepted") is not True or not isinstance(outcomes, list) or len(outcomes) != 3:
        raise ValueError("base-memory batch is not accepted")
    rows, identities = [], []
    for outcome in outcomes:
        if not isinstance(outcome, Mapping) or outcome.get("status") != "accepted" or not isinstance(outcome.get("run_root"), str):
            raise ValueError("base-memory batch contains rejected replicate")
        run_root = root / str(outcome["run_root"])
        manifest = _load(run_root / "run_manifest.json")
        if manifest.get("condition") != condition.value or manifest.get("family_id") != batch.get("family_id"):
            raise ValueError("base-memory replicate identity differs from batch")
        if (manifest.get("source_sequence_digest") != batch.get("source_sequence_digest")
                or manifest.get("config_digest") != batch.get("runtime_config_digest")
                or manifest.get("base_model") != FROZEN_MODEL_ID):
            raise ValueError("base-memory replicate config/model drift")
        trace = run_root / str(manifest.get("trace_directory")) / "sequence_results.json"
        sequence = _load(trace)
        episodes = sequence.get("episodes")
        if not isinstance(episodes, list) or not episodes:
            raise ValueError("base-memory replicate has no episode results")
        usage = [item.get("token_usage") for item in episodes if isinstance(item, Mapping)]
        if len(usage) != len(episodes) or any(not isinstance(item, Mapping) or item.get("model_usage_complete") is not True for item in usage):
            raise ValueError("base-memory replicate has incomplete model usage")
        evaluations = {str(item.get("task_id")): item.get("task_score") for item in episodes if isinstance(item, Mapping) and item.get("bucket") == "evaluation"}
        if not evaluations or any(not isinstance(score, (int, float)) or isinstance(score, bool) for score in evaluations.values()):
            raise ValueError("base-memory replicate lacks scored evaluation episode")
        evidence = _mem0_formal_evidence(run_root) if condition is BaseMemoryCondition.MEM0_STATIC else _read_events(run_root)
        if batch.get("category") == "semantic-targeted" and condition is BaseMemoryCondition.MEM0_STATIC and evidence["semantic_mechanism_applicable"] is not True:
            raise ValueError("semantic-targeted Mem0Static replicate lacks semantic operation evidence")
        rows.append({"replicate": outcome.get("replicate"), "run_root": outcome["run_root"], "evaluation_scores": evaluations,
                     "input_tokens": sum(int(item.get("input_tokens") or 0) for item in usage), "memory_evidence": evidence})
        identities.append((manifest, run_root))
    for field in ("state_directory", "trace_directory", "artifact_directory", "hermes_home_directory", "port_offset"):
        values = {
            item.get(field) if field == "port_offset" else str((run_root / str(item.get(field))).resolve())
            for item, run_root in identities
        }
        if len(values) != 3:
            raise ValueError(f"base-memory replicate {field} is not isolated")
    return {"schema": "rsimem-base-memory-replicate-batch-audit-v1", "accepted": True,
            "batch_id": batch.get("batch_id"), "family_id": batch.get("family_id"),
            "category": batch.get("category"), "condition": condition.value, "replicates": rows}


def _mean_sd(values: list[float]) -> dict[str, object]:
    return {"values": values, "mean": statistics.fmean(values),
            "sd": statistics.stdev(values) if len(values) > 1 else 0.0}


def aggregate_family_batches(batch_roots: Mapping[BaseMemoryCondition, Path]) -> dict[str, object]:
    """Aggregate only three independently audited static-backend batches."""
    normalized = {BaseMemoryCondition(key): Path(value) for key, value in batch_roots.items()}
    if set(normalized) != set(BACKEND_ORDER):
        raise ValueError("formal family aggregate requires NoMemory, HermesNative, and Mem0Static")
    audits = {condition: audit_replicate_batch(root) for condition, root in normalized.items()}
    family_ids = {report.get("family_id") for report in audits.values()}
    categories = {report.get("category") for report in audits.values()}
    if len(family_ids) != 1 or len(categories) != 1:
        raise ValueError("formal aggregate batch family/category mismatch")
    summary: dict[str, object] = {"schema": "rsimem-base-memory-family-aggregate-v1", "accepted": True,
                                  "family_id": next(iter(family_ids)), "category": next(iter(categories)),
                                  "conditions": {}}
    for condition, report in audits.items():
        rows = report["replicates"]
        if not isinstance(rows, list) or len(rows) != 3:
            raise ValueError("formal aggregate needs three audited replicates per condition")
        tasks = sorted({task for row in rows for task in row["evaluation_scores"]})
        if any(set(row["evaluation_scores"]) != set(tasks) for row in rows):
            raise ValueError("formal aggregate evaluation task identities differ")
        summary["conditions"][condition.value] = {
            "replicates": rows,
            "evaluation_scores": {task: _mean_sd([float(row["evaluation_scores"][task]) for row in rows]) for task in tasks},
            "input_tokens": _mean_sd([float(row["input_tokens"]) for row in rows]),
            "memory_evidence": [row["memory_evidence"] for row in rows],
        }
    return summary


def _accepted_batch_roots(*, formal_manifest_digest: str, family_id: str,
                         output_root: Path) -> dict[BaseMemoryCondition, Path]:
    """Find only batches that can be re-audited against this frozen manifest.

    A batch without a result is deliberately an error rather than an invitation
    to submit another request: it may still be running after a launcher restart.
    """
    accepted: dict[BaseMemoryCondition, list[Path]] = {condition: [] for condition in BACKEND_ORDER}
    in_progress: list[Path] = []
    for manifest_path in output_root.glob("*/batch_manifest.json"):
        try:
            batch = _load(manifest_path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if (batch.get("formal_manifest_digest") != formal_manifest_digest
                or batch.get("family_id") != family_id):
            continue
        try:
            condition = BaseMemoryCondition(batch.get("condition"))
        except ValueError:
            continue
        root = manifest_path.parent
        if not (root / "batch_result.json").is_file():
            in_progress.append(root)
            continue
        try:
            audit_replicate_batch(root)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        accepted[condition].append(root)
    if in_progress:
        paths = ", ".join(str(path) for path in sorted(in_progress))
        raise RuntimeError(f"matching formal batch is incomplete; resume it before scheduling: {paths}")
    # A valid retry is preferable to an older valid run, and directory mtime is
    # sufficient because batch roots are immutable after their result is written.
    return {condition: max(roots, key=lambda root: root.stat().st_mtime)
            for condition, roots in accepted.items() if roots}


def _new_batch_id(*, family_id: str, condition: BaseMemoryCondition) -> str:
    family = "".join(char.lower() if char.isalnum() else "-" for char in family_id).strip("-")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{family}-{condition.value.lower()}-formal-{stamp}"


def run_formal_suite(
    *, formal_manifest: Path, output_root: Path, past_bin: Path, past_root: Path,
    registry: Path, past_config: Path, base_url: str, progress_output: Path,
    family_ids: Iterable[str] | None = None,
) -> dict[str, object]:
    """Continuously execute the frozen suite, advancing only after each audit.

    Replicates within a batch remain concurrent. Families and backend conditions
    remain serial, giving each accepted batch an audit boundary and making a
    provider or evidence failure a durable stop point.
    """
    manifest = _load(formal_manifest)
    if manifest.get("schema") != SCHEMA:
        raise ValueError("unsupported base-memory formal manifest")
    requested = set(family_ids or ())
    available = [str(item["family_id"]) for item in manifest.get("families", ())
                 if isinstance(item, Mapping) and isinstance(item.get("family_id"), str)]
    if requested - set(available):
        raise ValueError(f"requested family is not in the frozen manifest: {sorted(requested - set(available))}")
    schedule = [family for family in available if not requested or family in requested]
    progress: dict[str, object] = {
        "schema": "rsimem-base-memory-formal-suite-progress-v1",
        "formal_manifest": str(formal_manifest.resolve()),
        "formal_manifest_digest": manifest["formal_manifest_digest"],
        "families": schedule,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "completed_families": [],
        "batches": [],
    }
    _write(progress_output, progress)
    try:
        for family_id in schedule:
            roots = _accepted_batch_roots(formal_manifest_digest=str(manifest["formal_manifest_digest"]),
                                         family_id=family_id, output_root=output_root)
            for condition in BACKEND_ORDER:
                root = roots.get(condition)
                if root is None:
                    batch_id = _new_batch_id(family_id=family_id, condition=condition)
                    run_replicate_batch(formal_manifest=formal_manifest, family_id=family_id,
                                        condition=condition, output_root=output_root, past_bin=past_bin,
                                        past_root=past_root, registry=registry, past_config=past_config,
                                        base_url=base_url, batch_id=batch_id)
                    root = output_root / batch_id
                    audit = audit_replicate_batch(root)
                    _write(root / "batch_audit.json", audit)
                    roots[condition] = root
                    action = "executed"
                else:
                    action = "reused"
                progress["batches"].append({"family_id": family_id, "condition": condition.value,
                                            "batch_root": str(root.resolve()), "action": action})
                _write(progress_output, progress)
            aggregate = aggregate_family_batches(roots)
            aggregate_path = progress_output.parent / f"{family_id.lower().replace('_', '-')}-baseline-aggregate.json"
            _write(aggregate_path, aggregate)
            progress["completed_families"].append({"family_id": family_id,
                                                    "aggregate": str(aggregate_path.resolve())})
            _write(progress_output, progress)
    except Exception as exc:
        progress["status"] = "stopped"
        progress["stopped_at"] = datetime.now(timezone.utc).isoformat()
        progress["error_type"] = type(exc).__name__
        progress["error"] = str(exc)
        _write(progress_output, progress)
        raise
    progress["status"] = "completed"
    progress["finished_at"] = datetime.now(timezone.utc).isoformat()
    _write(progress_output, progress)
    return progress


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--source-formal-manifest", type=Path, required=True)
    freeze.add_argument("--output-root", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("run-batch")
    run.add_argument("--formal-manifest", type=Path, required=True); run.add_argument("--family-id", required=True)
    run.add_argument("--condition", choices=[item.value for item in BaseMemoryCondition], required=True)
    run.add_argument("--output-root", type=Path, required=True); run.add_argument("--batch-id", required=True)
    run.add_argument("--past-bin", type=Path, required=True); run.add_argument("--past-root", type=Path, required=True)
    run.add_argument("--config", type=Path, required=True); run.add_argument("--registry", type=Path, required=True); run.add_argument("--base-url", required=True); run.add_argument("--dry-run", action="store_true")
    audit = sub.add_parser("audit-batch"); audit.add_argument("batch_root", type=Path); audit.add_argument("--output", type=Path, required=True)
    aggregate = sub.add_parser("aggregate-family")
    aggregate.add_argument("--no-memory", type=Path, required=True); aggregate.add_argument("--hermes-native", type=Path, required=True)
    aggregate.add_argument("--mem0-static", type=Path, required=True); aggregate.add_argument("--output", type=Path, required=True)
    suite = sub.add_parser("run-suite")
    suite.add_argument("--formal-manifest", type=Path, required=True)
    suite.add_argument("--output-root", type=Path, required=True)
    suite.add_argument("--past-bin", type=Path, required=True); suite.add_argument("--past-root", type=Path, required=True)
    suite.add_argument("--config", type=Path, required=True); suite.add_argument("--registry", type=Path, required=True)
    suite.add_argument("--base-url", required=True); suite.add_argument("--progress-output", type=Path, required=True)
    suite.add_argument("--family-id", action="append", dest="family_ids")
    args = parser.parse_args(argv)
    if args.command == "freeze":
        report = build_formal_manifest(source_formal_manifest=args.source_formal_manifest.resolve(), output_root=args.output_root.resolve())
        _write(args.output.resolve(), report)
    elif args.command == "run-batch":
        report = run_replicate_batch(formal_manifest=args.formal_manifest.resolve(), family_id=args.family_id,
            condition=BaseMemoryCondition(args.condition), output_root=args.output_root.resolve(), past_bin=args.past_bin.resolve(),
            past_root=args.past_root.resolve(), registry=args.registry.resolve(), past_config=args.config.resolve(),
            base_url=args.base_url, batch_id=args.batch_id, dry_run=args.dry_run)
    elif args.command == "audit-batch":
        report = audit_replicate_batch(args.batch_root.resolve()); _write(args.output.resolve(), report)
    elif args.command == "aggregate-family":
        report = aggregate_family_batches({BaseMemoryCondition.NO_MEMORY: args.no_memory.resolve(),
            BaseMemoryCondition.HERMES_NATIVE: args.hermes_native.resolve(),
            BaseMemoryCondition.MEM0_STATIC: args.mem0_static.resolve()})
        _write(args.output.resolve(), report)
    else:
        report = run_formal_suite(
            formal_manifest=args.formal_manifest.resolve(), output_root=args.output_root.resolve(),
            past_bin=args.past_bin.resolve(), past_root=args.past_root.resolve(), registry=args.registry.resolve(),
            past_config=args.config.resolve(), base_url=args.base_url, progress_output=args.progress_output.resolve(),
            family_ids=args.family_ids,
        )
    print(json.dumps(report, ensure_ascii=True, sort_keys=True))
    return 0 if report.get("accepted", True) else 1


__all__ = ["BACKEND_ORDER", "SCHEMA", "aggregate_family_batches", "audit_replicate_batch", "build_formal_manifest", "run_formal_suite", "run_replicate_batch"]


if __name__ == "__main__":
    raise SystemExit(main())
