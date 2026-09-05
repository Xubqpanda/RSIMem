from __future__ import annotations

import json

import pytest

from rsimem.native_attribution_protocol import (
    HISTORICAL_STATUS,
    NativeAttributionProtocolStore,
    NativeAttributionRepairProtocol,
)


def test_protocol_freezes_new_execution_line_and_26_families() -> None:
    protocol = NativeAttributionRepairProtocol.create()
    assert protocol.protocol_id.startswith("native-attribution-repair-v1.")
    assert protocol.baseline_condition == "native_static"
    assert protocol.background_lower_bound == "no_persistence"
    assert protocol.execution_flow == (
        "native_static_execution",
        "lifecycle_evidence",
        "failure_attribution",
        "one_axis_repair",
        "same_downstream_task",
        "paired_headroom",
        "feedback_to_update",
    )
    assert sum(len(values) for values in protocol.panels.values()) == 26
    assert set(protocol.repair_axes) == {
        "formation", "persistence", "maintenance", "retrieval", "application"
    }


def test_old_five_condition_matrix_is_historical_only() -> None:
    protocol = NativeAttributionRepairProtocol.create()
    assert set(protocol.historical_experiments.values()) == {HISTORICAL_STATUS}
    assert not {
        "shortcut_current_input", "wrong_mechanism", "type_matched_oracle"
    }.intersection(protocol.execution_flow)


def test_updater_plane_excludes_audit_and_resource_fields() -> None:
    protocol = NativeAttributionRepairProtocol.create()
    assert protocol.updater_allowed_planes == ("pure_process",)
    assert {"official_score", "grader", "hidden_answer"}.issubset(protocol.audit_only_fields)
    assert "input_tokens" in protocol.raw_resource_fields
    assert not set(protocol.audit_only_fields).intersection(protocol.updater_allowed_planes)


def test_protocol_store_is_immutable_and_detects_tampering(tmp_path) -> None:
    path = tmp_path / "protocol.json"
    store = NativeAttributionProtocolStore(path)
    protocol = NativeAttributionRepairProtocol.create()
    assert store.freeze(protocol) is True
    assert store.freeze(protocol) is False
    assert store.get() == protocol

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["baseline_condition"] = "shortcut_current_input"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="malformed native attribution protocol"):
        store.get()
