"""Freeze, dry-run, execute, and audit the formal AllMemoryOff baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

from .base_memory_experiment import BaseMemoryCondition, FROZEN_MODEL_ID
from .base_memory_launcher import _backend_descriptor, prepare_comparison, run_comparison
from .base_memory_smoke_audit import _all_memory_off_evidence, _complete_usage


SCHEMA = "rsimem-all-memory-off-formal-manifest-v1"
CONDITION = BaseMemoryCondition.ALL_MEMORY_OFF
REPLICATE_COUNT = 3


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"AllMemoryOff evidence must be an object: {path}")
    return value


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build_formal_manifest(*, source_formal_manifest: Path, output_root: Path) -> dict[str, object]:
    """Bind a new 26-family AllMemoryOff protocol to the frozen suite source."""
    source = _load(source_formal_manifest)
    if source.get("schema") != "rsimem-adamem-full-suite-formal-manifest-v1":
        raise ValueError("source must be the frozen AdaMem full-suite formal manifest")
    if source.get("family_count") != 26 or source.get("replicate_count") != REPLICATE_COUNT:
        raise ValueError("AllMemoryOff formal suite requires 26 families and three replicates")
    if source.get("base_model") != FROZEN_MODEL_ID:
        raise ValueError("AllMemoryOff formal suite model drift")
    records = source.get("families")
    if not isinstance(records, list) or len(records) != 26:
        raise ValueError("AllMemoryOff source family list is incomplete")
    families: list[dict[str, object]] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("AllMemoryOff source family record is invalid")
        family_id, config, config_digest, category = (
            record.get(key) for key in ("family_id", "config", "config_digest", "category")
        )
        if not all(isinstance(value, str) and value for value in (family_id, config, config_digest, category)):
            raise ValueError("AllMemoryOff source family record lacks identity")
        config_path = Path(config)
        if not config_path.is_file() or _digest(config_path) != config_digest:
            raise ValueError(f"AllMemoryOff formal config digest drift: {family_id}")
        families.append({
            "family_id": family_id,
            "config": str(config_path.resolve()),
            "config_digest": config_digest,
            "category": category,
        })
    if len({item["family_id"] for item in families}) != 26:
        raise ValueError("AllMemoryOff formal family identities are not unique")
    payload: dict[str, object] = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_formal_manifest": str(source_formal_manifest.resolve()),
        "source_formal_manifest_digest": _digest(source_formal_manifest),
        "source_formal_protocol_digest": source.get("formal_manifest_digest"),
        "output_root": str(output_root.resolve()),
        "condition": CONDITION.value,
        "backend_descriptor": _backend_descriptor(CONDITION),
        "base_model": FROZEN_MODEL_ID,
        "temperature": 0.0,
        "token_budget": 4096,
        "replicate_count": REPLICATE_COUNT,
        "family_count": 26,
        "families": families,
    }
    payload["formal_manifest_digest"] = _canonical_digest(payload)
    return payload


def _family(manifest: Mapping[str, object], family_id: str) -> Mapping[str, object]:
    for record in manifest.get("families", ()):
        if isinstance(record, Mapping) and record.get("family_id") == family_id:
            return record
    raise ValueError(f"family is not in frozen AllMemoryOff manifest: {family_id}")


def dry_run_manifest(*, formal_manifest: Path, output_root: Path, past_config: Path, registry: Path) -> dict[str, object]:
    """Materialize every run receipt without starting a provider-backed process."""
    manifest = _load(formal_manifest)
    if manifest.get("schema") != SCHEMA:
        raise ValueError("unsupported AllMemoryOff formal manifest")
    config = Path(past_config).resolve()
    registry = Path(registry).resolve()
    if not config.is_file() or not registry.is_file():
        raise ValueError("AllMemoryOff dry-run runtime config or registry is unavailable")
    receipts = []
    for index, record in enumerate(manifest.get("families", ())):
        if not isinstance(record, Mapping):
            raise ValueError("AllMemoryOff manifest family record is invalid")
        sequence = Path(str(record["config"])).resolve()
        if _digest(sequence) != record["config_digest"]:
            raise ValueError(f"AllMemoryOff dry-run config digest drift: {record['family_id']}")
        root = output_root / f"{index + 1:02d}-{record['family_id']}"
        comparison = prepare_comparison(
            source_sequence=sequence, output_root=root, family_id=str(record["family_id"]),
            config=config, registry=registry, token_budget=int(manifest["token_budget"]),
            run_prefix=f"dry-{index + 1:02d}", port_base=50000 + index * 100,
            conditions=(CONDITION,),
        )
        run = comparison.runs[0]
        run_manifest = _load(root / run.run_id / "run_manifest.json")
        if run_manifest.get("backend") != manifest.get("backend_descriptor"):
            raise ValueError("AllMemoryOff dry-run backend descriptor drift")
        receipts.append({"family_id": record["family_id"], "run_manifest": str((root / run.run_id / "run_manifest.json").resolve())})
    report = {
        "schema": "rsimem-all-memory-off-dry-run-v1",
        "accepted": True,
        "formal_manifest_digest": manifest["formal_manifest_digest"],
        "family_count": len(receipts),
        "condition": CONDITION.value,
        "receipts": receipts,
    }
    _write(output_root / "dry_run_report.json", report)
    return report


def run_replicate_batch(
    *, formal_manifest: Path, family_id: str, output_root: Path, past_bin: Path,
    past_root: Path, registry: Path, past_config: Path, base_url: str, batch_id: str,
    dry_run: bool = False,
) -> dict[str, object]:
    """Run exactly three isolated AllMemoryOff replicates for one family."""
    manifest = _load(formal_manifest)
    if manifest.get("schema") != SCHEMA or manifest.get("replicate_count") != REPLICATE_COUNT:
        raise ValueError("unsupported AllMemoryOff formal manifest")
    record = _family(manifest, family_id)
    sequence = Path(str(record["config"])).resolve()
    config = Path(past_config).resolve()
    if _digest(sequence) != record["config_digest"]:
        raise ValueError("AllMemoryOff formal config digest drift before execution")
    if not config.is_file():
        raise ValueError("AllMemoryOff runtime config is unavailable")
    root = output_root / batch_id
    started_at = datetime.now(timezone.utc).isoformat()
    batch = {
        "schema": "rsimem-all-memory-off-replicate-batch-v1",
        "batch_id": batch_id,
        "formal_manifest_digest": manifest["formal_manifest_digest"],
        "family_id": family_id,
        "category": record["category"],
        "condition": CONDITION.value,
        "backend_descriptor": manifest["backend_descriptor"],
        "replicate_count": REPLICATE_COUNT,
        "max_workers": REPLICATE_COUNT,
        "source_sequence": str(sequence),
        "source_sequence_digest": record["config_digest"],
        "runtime_config": str(config),
        "runtime_config_digest": _digest(config),
        "started_at": started_at,
        "retry_reason": None,
    }
    _write(root / "batch_manifest.json", batch)

    def execute(replicate: int) -> dict[str, object]:
        replica_root = root / f"replicate-{replicate:02d}"
        prefix = f"{batch_id}-r{replicate:02d}"
        run_comparison(
            source_sequence=sequence, output_root=replica_root, family_id=family_id,
            past_bin=past_bin, past_root=past_root, config=config, registry=registry,
            base_url=base_url, token_budget=int(manifest["token_budget"]), run_prefix=prefix,
            condition=CONDITION, dry_run=dry_run, port_base=40000 + replicate * 1000,
        )
        run_root = replica_root / f"{prefix}-{CONDITION.value.lower()}"
        return {"replicate": replicate, "run_root": str(run_root.relative_to(root)), "status": "accepted"}

    outcomes: dict[int, dict[str, object]] = {}
    with ThreadPoolExecutor(max_workers=REPLICATE_COUNT, thread_name_prefix="all-memory-off-replicate") as pool:
        futures = {pool.submit(execute, replica): replica for replica in range(1, REPLICATE_COUNT + 1)}
        for future in as_completed(futures):
            replica = futures[future]
            try:
                outcomes[replica] = future.result()
            except Exception as exc:
                outcomes[replica] = {"replicate": replica, "status": "infrastructure_failure", "error_type": type(exc).__name__}
    accepted = len(outcomes) == REPLICATE_COUNT and all(item.get("status") == "accepted" for item in outcomes.values())
    report = {
        "schema": "rsimem-all-memory-off-replicate-batch-result-v1",
        "batch_id": batch_id,
        "accepted": accepted,
        "outcomes": [outcomes[index] for index in sorted(outcomes)],
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "provider_health": {"status": "healthy" if accepted else "degraded", "evidence": "complete_run_usage" if accepted else "infrastructure_failure"},
        "retry_reason": None,
    }
    _write(root / "batch_result.json", report)
    return report


def audit_replicate_batch(batch_root: Path) -> dict[str, object]:
    root = Path(batch_root).resolve()
    batch, result = _load(root / "batch_manifest.json"), _load(root / "batch_result.json")
    outcomes = result.get("outcomes")
    if batch.get("condition") != CONDITION.value or result.get("accepted") is not True:
        raise ValueError("AllMemoryOff batch is not accepted")
    if not isinstance(outcomes, list) or len(outcomes) != REPLICATE_COUNT:
        raise ValueError("AllMemoryOff batch requires exactly three outcomes")
    rows, identities = [], []
    for outcome in outcomes:
        if not isinstance(outcome, Mapping) or outcome.get("status") != "accepted" or not isinstance(outcome.get("run_root"), str):
            raise ValueError("AllMemoryOff batch contains rejected replicate")
        run_root = root / outcome["run_root"]
        manifest = _load(run_root / "run_manifest.json")
        if manifest.get("condition") != CONDITION.value or manifest.get("family_id") != batch.get("family_id"):
            raise ValueError("AllMemoryOff replicate identity differs from batch")
        if (manifest.get("source_sequence_digest") != batch.get("source_sequence_digest")
                or manifest.get("config_digest") != batch.get("runtime_config_digest")
                or manifest.get("base_model") != FROZEN_MODEL_ID
                or manifest.get("backend") != batch.get("backend_descriptor")):
            raise ValueError("AllMemoryOff replicate config/model/backend drift")
        usage = _complete_usage(run_root, manifest)
        evidence = _all_memory_off_evidence(run_root, manifest)
        rows.append({"replicate": outcome["replicate"], "run_root": outcome["run_root"], "usage": usage, "all_memory_off_evidence": evidence})
        identities.append((manifest, run_root))
    for field in ("state_directory", "trace_directory", "artifact_directory", "hermes_home_directory", "port_offset"):
        values = {item.get(field) if field == "port_offset" else str((run_root / str(item.get(field))).resolve()) for item, run_root in identities}
        if len(values) != REPLICATE_COUNT:
            raise ValueError(f"AllMemoryOff replicate {field} is not isolated")
    return {"schema": "rsimem-all-memory-off-replicate-batch-audit-v1", "accepted": True, "batch_id": batch["batch_id"], "family_id": batch["family_id"], "condition": CONDITION.value, "replicates": rows}


def _new_batch_id(family_id: str) -> str:
    normalized = "".join(char.lower() if char.isalnum() else "-" for char in family_id).strip("-")
    return f"{normalized}-all-memory-off-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def _accepted_batch_roots(*, formal_manifest_digest: str, output_root: Path) -> dict[str, Path]:
    """Return re-audited accepted batches, never treating partial roots as reusable."""
    accepted: dict[str, Path] = {}
    for manifest_path in output_root.glob("*/batch_manifest.json"):
        try:
            batch = _load(manifest_path)
            if batch.get("formal_manifest_digest") != formal_manifest_digest:
                continue
            family_id = batch.get("family_id")
            if not isinstance(family_id, str) or not family_id:
                continue
            audit_replicate_batch(manifest_path.parent)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        previous = accepted.get(family_id)
        if previous is None or manifest_path.parent.stat().st_mtime > previous.stat().st_mtime:
            accepted[family_id] = manifest_path.parent
    return accepted


def run_formal_suite(*, formal_manifest: Path, output_root: Path, past_bin: Path, past_root: Path, registry: Path, past_config: Path, base_url: str, progress_output: Path, family_ids: Iterable[str] | None = None) -> dict[str, object]:
    manifest = _load(formal_manifest)
    if manifest.get("schema") != SCHEMA:
        raise ValueError("unsupported AllMemoryOff formal manifest")
    requested = set(family_ids or ())
    families = [str(record["family_id"]) for record in manifest.get("families", ()) if isinstance(record, Mapping)]
    if requested - set(families):
        raise ValueError(f"requested family is not in frozen AllMemoryOff manifest: {sorted(requested - set(families))}")
    schedule = [family for family in families if not requested or family in requested]
    progress: dict[str, object] = {"schema": "rsimem-all-memory-off-suite-progress-v1", "formal_manifest": str(formal_manifest.resolve()), "formal_manifest_digest": manifest["formal_manifest_digest"], "families": schedule, "status": "running", "started_at": datetime.now(timezone.utc).isoformat(), "completed_families": [], "batches": []}
    _write(progress_output, progress)
    try:
        for family_id in schedule:
            batch_id = _new_batch_id(family_id)
            report = run_replicate_batch(formal_manifest=formal_manifest, family_id=family_id, output_root=output_root, past_bin=past_bin, past_root=past_root, registry=registry, past_config=past_config, base_url=base_url, batch_id=batch_id)
            root = output_root / batch_id
            audit = audit_replicate_batch(root)
            _write(root / "batch_audit.json", audit)
            progress["batches"].append({"family_id": family_id, "batch_root": str(root.resolve()), "accepted": report["accepted"]})
            progress["completed_families"].append(family_id)
            _write(progress_output, progress)
    except Exception as exc:
        progress.update({"status": "stopped", "stopped_at": datetime.now(timezone.utc).isoformat(), "error_type": type(exc).__name__, "error": str(exc)})
        _write(progress_output, progress)
        raise
    progress.update({"status": "completed", "finished_at": datetime.now(timezone.utc).isoformat()})
    _write(progress_output, progress)
    return progress


def resume_formal_suite(*, formal_manifest: Path, output_root: Path, past_bin: Path, past_root: Path, registry: Path, past_config: Path, base_url: str, progress_output: Path, family_ids: Iterable[str] | None = None) -> dict[str, object]:
    """Continue only families without a currently re-auditable accepted batch."""
    manifest = _load(formal_manifest)
    if manifest.get("schema") != SCHEMA:
        raise ValueError("unsupported AllMemoryOff formal manifest")
    requested = set(family_ids or ())
    families = [str(record["family_id"]) for record in manifest.get("families", ()) if isinstance(record, Mapping)]
    if requested - set(families):
        raise ValueError(f"requested family is not in frozen AllMemoryOff manifest: {sorted(requested - set(families))}")
    accepted = _accepted_batch_roots(
        formal_manifest_digest=str(manifest["formal_manifest_digest"]),
        output_root=output_root,
    )
    schedule = [family for family in families if (not requested or family in requested) and family not in accepted]
    if progress_output.is_file():
        progress = _load(progress_output)
        if progress.get("formal_manifest_digest") != manifest["formal_manifest_digest"]:
            raise ValueError("resume progress formal manifest differs")
    else:
        progress = {
            "schema": "rsimem-all-memory-off-suite-progress-v1",
            "formal_manifest": str(formal_manifest.resolve()),
            "formal_manifest_digest": manifest["formal_manifest_digest"],
            "families": families,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "completed_families": [],
            "batches": [],
        }
    progress["status"] = "running"
    progress.pop("error_type", None)
    progress.pop("error", None)
    progress.pop("stopped_at", None)
    completed = [family for family in families if family in accepted]
    progress["completed_families"] = completed
    progress["batches"] = [
        {"family_id": family, "batch_root": str(root.resolve()), "accepted": True, "action": "reused"}
        for family, root in accepted.items()
    ]
    _write(progress_output, progress)
    try:
        for family_id in schedule:
            batch_id = _new_batch_id(family_id)
            report = run_replicate_batch(
                formal_manifest=formal_manifest, family_id=family_id, output_root=output_root,
                past_bin=past_bin, past_root=past_root, registry=registry, past_config=past_config,
                base_url=base_url, batch_id=batch_id,
            )
            root = output_root / batch_id
            audit = audit_replicate_batch(root)
            _write(root / "batch_audit.json", audit)
            progress["batches"].append({"family_id": family_id, "batch_root": str(root.resolve()), "accepted": report["accepted"], "action": "executed"})
            progress["completed_families"].append(family_id)
            _write(progress_output, progress)
    except Exception as exc:
        progress.update({"status": "stopped", "stopped_at": datetime.now(timezone.utc).isoformat(), "error_type": type(exc).__name__, "error": str(exc)})
        _write(progress_output, progress)
        raise
    progress.update({"status": "completed" if not schedule or len(progress["completed_families"]) == len(families) else "paused", "finished_at": datetime.now(timezone.utc).isoformat()})
    _write(progress_output, progress)
    return progress


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze"); freeze.add_argument("--source-formal-manifest", type=Path, required=True); freeze.add_argument("--output-root", type=Path, required=True); freeze.add_argument("--output", type=Path, required=True)
    dry = sub.add_parser("dry-run"); dry.add_argument("--formal-manifest", type=Path, required=True); dry.add_argument("--output-root", type=Path, required=True); dry.add_argument("--config", type=Path, required=True); dry.add_argument("--registry", type=Path, required=True)
    run = sub.add_parser("run-batch"); run.add_argument("--formal-manifest", type=Path, required=True); run.add_argument("--family-id", required=True); run.add_argument("--output-root", type=Path, required=True); run.add_argument("--batch-id", required=True); run.add_argument("--past-bin", type=Path, required=True); run.add_argument("--past-root", type=Path, required=True); run.add_argument("--config", type=Path, required=True); run.add_argument("--registry", type=Path, required=True); run.add_argument("--base-url", required=True); run.add_argument("--dry-run", action="store_true")
    audit = sub.add_parser("audit-batch"); audit.add_argument("batch_root", type=Path); audit.add_argument("--output", type=Path, required=True)
    suite = sub.add_parser("run-suite"); suite.add_argument("--formal-manifest", type=Path, required=True); suite.add_argument("--output-root", type=Path, required=True); suite.add_argument("--past-bin", type=Path, required=True); suite.add_argument("--past-root", type=Path, required=True); suite.add_argument("--config", type=Path, required=True); suite.add_argument("--registry", type=Path, required=True); suite.add_argument("--base-url", required=True); suite.add_argument("--progress-output", type=Path, required=True); suite.add_argument("--family-id", action="append", dest="family_ids")
    resume = sub.add_parser("resume-suite"); resume.add_argument("--formal-manifest", type=Path, required=True); resume.add_argument("--output-root", type=Path, required=True); resume.add_argument("--past-bin", type=Path, required=True); resume.add_argument("--past-root", type=Path, required=True); resume.add_argument("--config", type=Path, required=True); resume.add_argument("--registry", type=Path, required=True); resume.add_argument("--base-url", required=True); resume.add_argument("--progress-output", type=Path, required=True); resume.add_argument("--family-id", action="append", dest="family_ids")
    args = parser.parse_args(argv)
    if args.command == "freeze":
        report = build_formal_manifest(source_formal_manifest=args.source_formal_manifest.resolve(), output_root=args.output_root.resolve()); _write(args.output.resolve(), report)
    elif args.command == "dry-run":
        report = dry_run_manifest(formal_manifest=args.formal_manifest.resolve(), output_root=args.output_root.resolve(), past_config=args.config.resolve(), registry=args.registry.resolve())
    elif args.command == "run-batch":
        report = run_replicate_batch(formal_manifest=args.formal_manifest.resolve(), family_id=args.family_id, output_root=args.output_root.resolve(), past_bin=args.past_bin.resolve(), past_root=args.past_root.resolve(), registry=args.registry.resolve(), past_config=args.config.resolve(), base_url=args.base_url, batch_id=args.batch_id, dry_run=args.dry_run)
    elif args.command == "audit-batch":
        report = audit_replicate_batch(args.batch_root.resolve()); _write(args.output.resolve(), report)
    elif args.command == "run-suite":
        report = run_formal_suite(formal_manifest=args.formal_manifest.resolve(), output_root=args.output_root.resolve(), past_bin=args.past_bin.resolve(), past_root=args.past_root.resolve(), registry=args.registry.resolve(), past_config=args.config.resolve(), base_url=args.base_url, progress_output=args.progress_output.resolve(), family_ids=args.family_ids)
    else:
        report = resume_formal_suite(formal_manifest=args.formal_manifest.resolve(), output_root=args.output_root.resolve(), past_bin=args.past_bin.resolve(), past_root=args.past_root.resolve(), registry=args.registry.resolve(), past_config=args.config.resolve(), base_url=args.base_url, progress_output=args.progress_output.resolve(), family_ids=args.family_ids)
    print(json.dumps(report, ensure_ascii=True, sort_keys=True))
    return 0 if report.get("accepted", True) else 1


__all__ = ["CONDITION", "REPLICATE_COUNT", "SCHEMA", "audit_replicate_batch", "build_formal_manifest", "dry_run_manifest", "resume_formal_suite", "run_formal_suite", "run_replicate_batch"]


if __name__ == "__main__":
    raise SystemExit(main())
