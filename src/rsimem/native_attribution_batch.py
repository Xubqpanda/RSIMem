"""Assemble a content-free native attribution corpus from a frozen manifest."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping

from .native_attribution import attribute_native_observation, expectation_from_benchmark_contract
from .native_attribution_corpus import NativeAttributionCorpus, NativeAttributionCorpusStore
from .native_attribution_run import NativeAttributionRunManifestStore
from .native_observation import extract_native_observations


AUDIT_SCHEMA = "rsimem-native-attribution-batch-audit-v1"


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _write_audit(path: Path, *, manifest_id: str, protocol_id: str,
                 accepted_ids: list[str], excluded: list[dict[str, object]]) -> None:
    identity = {
        "schema": AUDIT_SCHEMA,
        "manifest_id": manifest_id,
        "protocol_id": protocol_id,
        "accepted_run_ids": sorted(accepted_ids),
        "excluded_runs": excluded,
    }
    payload = {"audit_id": "native-batch-audit." + _digest(identity)[:40], **identity}
    target = Path(path).expanduser().resolve()
    serialized = _canonical(payload) + "\n"
    target.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target.with_name(target.name + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if target.is_symlink() or lock_path.is_symlink():
            raise ValueError("native attribution batch audit cannot be symlinked")
        if target.exists():
            if target.read_text(encoding="utf-8") != serialized:
                raise ValueError("native attribution batch audit conflicts with existing audit")
            return
        temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
        temporary.write_text(serialized, encoding="utf-8")
        os.replace(temporary, target)


def load_native_attribution_batch_audit(path: Path) -> dict[str, object]:
    """Reload and verify a content-free batch audit sidecar."""

    target = Path(path).expanduser().resolve()
    if target.is_symlink() or not target.is_file():
        raise ValueError("native attribution batch audit is missing or symlinked")
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("native attribution batch audit is unreadable") from exc
    expected = {"audit_id", "schema", "manifest_id", "protocol_id", "accepted_run_ids", "excluded_runs"}
    if not isinstance(payload, Mapping) or set(payload) != expected:
        raise ValueError("native attribution batch audit fields are invalid")
    if payload["schema"] != AUDIT_SCHEMA:
        raise ValueError("native attribution batch audit schema is invalid")
    if any(not isinstance(payload[field], str) or not payload[field]
           for field in ("audit_id", "manifest_id", "protocol_id")):
        raise ValueError("native attribution batch audit identity is invalid")
    accepted = payload["accepted_run_ids"]
    excluded = payload["excluded_runs"]
    if (not isinstance(accepted, list) or any(not isinstance(item, str) for item in accepted)
            or accepted != sorted(set(accepted)) or not isinstance(excluded, list)):
        raise ValueError("native attribution batch audit collections are invalid")
    for item in excluded:
        if not isinstance(item, Mapping) or set(item) != {"run_id", "reason"}:
            raise ValueError("native attribution batch audit exclusion is invalid")
    identity = {key: payload[key] for key in expected if key != "audit_id"}
    if payload["audit_id"] != "native-batch-audit." + _digest(identity)[:40]:
        raise ValueError("native attribution batch audit ID mismatch")
    if target.read_text(encoding="utf-8") != _canonical(dict(payload)) + "\n":
        raise ValueError("native attribution batch audit is not canonical")
    return dict(payload)


def _read_sequence_results(*, output_root: Path, trace_directory: str) -> tuple[Mapping[str, object], ...]:
    path = output_root / trace_directory / "sequence_results.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("native sequence results are unreadable") from exc
    if not isinstance(value, Mapping) or not isinstance(value.get("episodes"), list):
        raise ValueError("native sequence results are malformed")
    episodes = tuple(item for item in value["episodes"] if isinstance(item, Mapping))
    if len(episodes) != len(value["episodes"]):
        raise ValueError("native sequence episode is malformed")
    return episodes


def assemble_native_attribution_corpus(
    *, manifest_path: Path, output_root: Path, corpus_path: Path | None = None,
    audit_path: Path | None = None,
) -> NativeAttributionCorpus:
    """Audit every manifest run and build a deterministic attribution corpus.

    A run that cannot pass the complete native audit is retained only as a
    content-free exclusion. No final output, score, grader, or provider payload
    is copied into the corpus.
    """

    manifest = NativeAttributionRunManifestStore(manifest_path).get()
    root = Path(output_root).expanduser().resolve()
    accepted_ids: list[str] = []
    observations = []
    candidates = []
    excluded: list[dict[str, object]] = []
    for run in manifest.runs:
        try:
            episodes = _read_sequence_results(output_root=root, trace_directory=run.trace_directory)
            by_task = {
                str(item["task_id"]): item
                for item in episodes
                if isinstance(item.get("task_id"), str)
            }
            extracted = extract_native_observations(run=run, output_root=root)
            expectations = {
                task_id: expectation_from_benchmark_contract(episode)
                for task_id, episode in by_task.items()
            }
        except (OSError, TypeError, ValueError) as exc:
            excluded.append({"run_id": run.run_id, "reason": _classify_exclusion(str(exc))})
            continue
        accepted_ids.append(run.run_id)
        observations.extend(extracted)
        candidates.extend(
            attribute_native_observation(value, expectations.get(value.task_id))
            for value in extracted
        )
    if audit_path is not None:
        _write_audit(audit_path, manifest_id=manifest.manifest_id,
                     protocol_id=manifest.protocol_id, accepted_ids=accepted_ids,
                     excluded=excluded)
    if not accepted_ids:
        # A corpus without an accepted observation cannot satisfy the frozen
        # corpus contract. Keep this fail-closed instead of emitting a fake
        # empty corpus that could be mistaken for evidence.
        raise ValueError("no accepted native runs; no attribution corpus written")
    corpus = NativeAttributionCorpus.create(
        protocol_id=manifest.protocol_id,
        accepted_run_ids=accepted_ids,
        observations=observations,
        candidates=candidates,
        excluded_runs=excluded,
    )
    if corpus_path is not None:
        NativeAttributionCorpusStore(corpus_path).put(corpus)
    return corpus


def _classify_exclusion(message: str) -> str:
    lowered = message.lower()
    if "usage" in lowered:
        return "usage_incomplete"
    if "service" in lowered or "fixture" in lowered:
        return "service_identity_invalid"
    if "trace" in lowered or "sequence" in lowered:
        return "trace_invalid"
    return "native_audit_failed"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--audit", type=Path, default=None)
    args = parser.parse_args(argv)
    corpus = assemble_native_attribution_corpus(
        manifest_path=args.manifest, output_root=args.output_root, corpus_path=args.corpus,
        audit_path=args.audit,
    )
    print(json.dumps({
        "corpus_id": corpus.corpus_id,
        "accepted_run_count": len(corpus.accepted_run_ids),
        "observation_count": len(corpus.observations),
        "excluded_run_count": len(corpus.excluded_runs),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["AUDIT_SCHEMA", "assemble_native_attribution_corpus", "load_native_attribution_batch_audit", "main"]
