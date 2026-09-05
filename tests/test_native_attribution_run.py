from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from rsimem.native_attribution_protocol import NativeAttributionRepairProtocol
from rsimem.native_attribution_run import (
    NativeAttributionRunManifest,
    NativeAttributionRunManifestStore,
    NativeAttributionRunSpec,
    build_native_attribution_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
PAST_ROOT = ROOT / "benchmarks" / "past-bench"


def _manifest(replicates: int = 1):
    return build_native_attribution_manifest(
        batch_id="native-attribution-fixture",
        protocol=NativeAttributionRepairProtocol.create(),
        past_bench_root=PAST_ROOT,
        rsimem_commit="commit.rsimem.fixture",
        past_bench_commit="commit.past.fixture",
        replicate_count=replicates,
        port_base_offset=100,
    )


def test_manifest_freezes_all_26_families_with_isolated_runtime_identity() -> None:
    manifest = _manifest(3)
    assert len(manifest.runs) == 26 * 3
    assert {run.condition for run in manifest.runs} == {"native_static"}
    assert len({run.run_id for run in manifest.runs}) == 78
    assert len({run.method_case_id for run in manifest.runs}) == 26
    assert all(run.family_id not in run.method_case_id for run in manifest.runs)
    ports = [port for run in manifest.runs for port in run.service_ports]
    assert len(ports) == len(set(ports))
    directories = [
        value for run in manifest.runs for value in (
            run.state_directory, run.hermes_home_directory, run.session_directory,
            run.artifact_directory, run.trace_directory,
        )
    ]
    assert len(directories) == len(set(directories))
    assert all(run.fixture_digest and run.family_source_digest for run in manifest.runs)
    assert all(len(run.native_task_ids) == len(run.native_episode_ids) for run in manifest.runs)
    assert all(not any("control" in episode.lower() for episode in run.native_episode_ids) for run in manifest.runs)


def test_manifest_is_replay_stable_and_changes_with_batch_seed() -> None:
    first = _manifest()
    second = _manifest()
    assert first.payload() == second.payload()
    changed = build_native_attribution_manifest(
        batch_id="native-attribution-other",
        protocol=NativeAttributionRepairProtocol.create(),
        past_bench_root=PAST_ROOT,
        rsimem_commit="commit.rsimem.fixture",
        past_bench_commit="commit.past.fixture",
        replicate_count=1,
        port_base_offset=100,
    )
    assert [run.seed for run in first.runs] != [run.seed for run in changed.runs]


def test_manifest_rejects_port_and_directory_collisions() -> None:
    manifest = _manifest()
    first, second, *rest = manifest.runs

    def changed_run(**changes):
        values = second.identity_payload()
        values.update(changes)
        run_id = "native-run." + hashlib.sha256(
            json.dumps(values, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:40]
        return NativeAttributionRunSpec(run_id=run_id, **values)

    colliding_ports = [first.service_ports[0], *second.service_ports[1:]]
    colliding_services = second.identity_payload()["service_identities"]
    old_port = second.service_ports[0]
    for value in colliding_services:
        if value["port"] == old_port:
            value["port"] = first.service_ports[0]
    with pytest.raises(ValueError, match="ports collide"):
        NativeAttributionRunManifest(
            manifest_id=manifest.manifest_id,
            batch_id=manifest.batch_id,
            runs=(first, changed_run(
                service_ports=colliding_ports,
                service_identities=colliding_services,
            ), *rest),
            protocol_id=manifest.protocol_id,
            protocol_digest=manifest.protocol_digest,
        )
    with pytest.raises(ValueError, match="directories collide"):
        NativeAttributionRunManifest(
            manifest_id=manifest.manifest_id,
            batch_id=manifest.batch_id,
            runs=(first, changed_run(trace_directory=first.trace_directory), *rest),
            protocol_id=manifest.protocol_id,
            protocol_digest=manifest.protocol_digest,
        )


def test_manifest_store_is_append_once(tmp_path) -> None:
    store = NativeAttributionRunManifestStore(tmp_path / "manifest.json")
    manifest = _manifest()
    assert store.initialize(manifest) is True
    assert store.initialize(manifest) is False
    assert store.get() == manifest
    with pytest.raises(ValueError, match="conflicts"):
        store.initialize(build_native_attribution_manifest(
            batch_id="native-attribution-conflict",
            protocol=NativeAttributionRepairProtocol.create(),
            past_bench_root=PAST_ROOT,
            rsimem_commit="commit.rsimem.fixture",
            past_bench_commit="commit.past.fixture",
            replicate_count=1,
            port_base_offset=100,
        ))


def test_manifest_store_rejects_tampered_fixture_digest(tmp_path) -> None:
    import json

    path = tmp_path / "manifest.json"
    store = NativeAttributionRunManifestStore(path)
    store.initialize(_manifest())
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["runs"][0]["fixture_digest"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="malformed native attribution run manifest"):
        store.get()
