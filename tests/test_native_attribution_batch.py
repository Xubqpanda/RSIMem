from __future__ import annotations

from rsimem.native_attribution_batch import assemble_native_attribution_corpus
from rsimem.native_attribution_run import (
    NativeAttributionRunManifest,
    NativeAttributionRunManifestStore,
    _digest,
)
from test_native_execution_audit import _fixture


def test_assembler_accepts_audited_run_and_writes_content_free_corpus(tmp_path) -> None:
    run = _fixture(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    values = {
        "schema": "rsimem-native-attribution-run-v1", "schema_version": 1,
        "batch_id": "native-batch.fixture", "protocol_id": run.protocol_id,
        "protocol_digest": run.protocol_digest, "runs": [run.payload()],
    }
    NativeAttributionRunManifestStore(manifest_path).initialize(NativeAttributionRunManifest(
        manifest_id="native-manifest." + _digest(values)[:40],
        batch_id=values["batch_id"], runs=(run,), protocol_id=run.protocol_id,
        protocol_digest=run.protocol_digest,
    ))
    corpus = assemble_native_attribution_corpus(
        manifest_path=manifest_path,
        output_root=tmp_path,
        corpus_path=tmp_path / "corpus.json",
    )
    assert corpus.accepted_run_ids == (run.run_id,)
    assert len(corpus.observations) == 5
    assert corpus.actionable_count == 0
    assert (tmp_path / "corpus.json").is_file()


def test_batch_audit_is_append_once_and_rejects_conflict(tmp_path) -> None:
    run = _fixture(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    values = {
        "schema": "rsimem-native-attribution-run-v1", "schema_version": 1,
        "batch_id": "native-batch.fixture", "protocol_id": run.protocol_id,
        "protocol_digest": run.protocol_digest, "runs": [run.payload()],
    }
    NativeAttributionRunManifestStore(manifest_path).initialize(NativeAttributionRunManifest(
        manifest_id="native-manifest." + _digest(values)[:40],
        batch_id=values["batch_id"], runs=(run,), protocol_id=run.protocol_id,
        protocol_digest=run.protocol_digest,
    ))
    audit_path = tmp_path / "audit.json"
    assemble_native_attribution_corpus(
        manifest_path=manifest_path, output_root=tmp_path, audit_path=audit_path
    )
    first = audit_path.read_text()
    assemble_native_attribution_corpus(
        manifest_path=manifest_path, output_root=tmp_path, audit_path=audit_path
    )
    assert audit_path.read_text() == first
    audit_path.write_text(first.replace(run.run_id, "native-run.conflict"))
    import pytest
    with pytest.raises(ValueError, match="conflicts with existing audit"):
        assemble_native_attribution_corpus(
            manifest_path=manifest_path, output_root=tmp_path, audit_path=audit_path
        )


def test_assembler_records_incomplete_run_without_crashing(tmp_path) -> None:
    run = _fixture(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    values = {
        "schema": "rsimem-native-attribution-run-v1", "schema_version": 1,
        "batch_id": "native-batch.fixture", "protocol_id": run.protocol_id,
        "protocol_digest": run.protocol_digest, "runs": [run.payload()],
    }
    NativeAttributionRunManifestStore(manifest_path).initialize(NativeAttributionRunManifest(
        manifest_id="native-manifest." + _digest(values)[:40],
        batch_id=values["batch_id"], runs=(run,), protocol_id=run.protocol_id,
        protocol_digest=run.protocol_digest,
    ))
    (tmp_path / run.trace_directory / "sequence_results.json").unlink()
    import pytest
    audit_path = tmp_path / "batch-audit.json"
    with pytest.raises(ValueError, match="no accepted native runs"):
        assemble_native_attribution_corpus(
            manifest_path=manifest_path, output_root=tmp_path, audit_path=audit_path
        )
    payload = __import__("json").loads(audit_path.read_text())
    assert payload["schema"] == "rsimem-native-attribution-batch-audit-v1"
    assert payload["accepted_run_ids"] == []
    assert payload["excluded_runs"][0]["run_id"] == run.run_id
