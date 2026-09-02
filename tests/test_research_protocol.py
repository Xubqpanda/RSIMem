from __future__ import annotations

import json

import pytest

from rsimem.memory import MemoryKind
from rsimem.memory.family_matrix import PastFamilyMatrix
from rsimem.memory.taxonomy import MemoryControlKind
from rsimem.research_protocol import (
    ComparisonLevel,
    ExperimentSplit,
    JsonResearchProtocolStore,
    SensitivityCondition,
    default_research_protocol,
)


def test_default_protocol_freezes_taxonomy_split_conditions_and_raw_metrics() -> None:
    protocol = default_research_protocol()
    assert protocol.protocol_id.startswith("research-protocol.")
    assert protocol.family_matrix_id.startswith("past-family-matrix.")
    assert protocol.lifecycle_surface_schema == "rsimem-memory-lifecycle-v1"
    assert set(protocol.memory_control_kinds) == set(MemoryControlKind)
    assert set(protocol.comparison_levels) == {
        ComparisonLevel.VANILLA_NO_PERSISTENCE,
        ComparisonLevel.HERMES_NATIVE_STATIC,
        ComparisonLevel.SENSITIVITY,
    }
    assert {item.condition_id for item in protocol.conditions} == set(SensitivityCondition)
    assert protocol.metric.updater_evidence_plane.value == "pure_process"
    assert "past_bench.official_task_metric.v1" == protocol.metric.primary_metric
    assert "input_tokens" in protocol.metric.raw_resource_fields
    assert "score" not in protocol.metric.raw_resource_fields


def test_protocol_can_be_instantiated_for_each_target_kind() -> None:
    from rsimem.research_protocol import ResearchProtocol

    matrix = PastFamilyMatrix.create_default()
    split = ExperimentSplit(
        split_id="split.fixture.v1",
        train_family_ids=("SM01_preference_adoption",),
        validation_family_ids=("SM03_fact_correction",),
        final_family_ids=("SM04_rule_migration",),
        task_template_group_ids=("train.v1", "validation.v1", "final.v1"),
        leakage_rules=("no_cross_condition_state",),
    )
    protocol = ResearchProtocol.create(
        memory_units=default_research_protocol().memory_units,
        family_matrix=matrix,
        split=split,
        sensitivity_target_kind=MemoryKind.PROCEDURAL,
    )
    assert protocol.method_visible_condition(SensitivityCondition.TYPE_MATCHED_ORACLE)["target_kind"] == "procedural"


def test_method_visible_condition_does_not_expose_family_or_final_score() -> None:
    protocol = default_research_protocol()
    view = protocol.method_visible_condition(SensitivityCondition.NATIVE_STATIC)
    assert set(view) == {"condition", "target_kind", "persistence_mode", "mechanism"}
    assert "family_id" not in json.dumps(view)
    assert "official_score" not in json.dumps(view)
    assert "grader" not in json.dumps(view)


def test_protocol_store_is_immutable_and_restart_safe(tmp_path) -> None:
    protocol = default_research_protocol()
    store = JsonResearchProtocolStore(tmp_path / "protocol.json")
    assert store.freeze(protocol) is True
    assert store.freeze(protocol) is False
    assert store.get() == protocol

    changed_base = default_research_protocol()
    # Constructing a different protocol ID is intentional: protocol changes
    # must be a new manifest rather than an in-place edit.
    from rsimem.research_protocol import ExperimentSplit, ResearchProtocol
    changed = ResearchProtocol.create(
        memory_units=changed_base.memory_units,
        family_matrix=PastFamilyMatrix.create_default(),
        split=changed_base.split,
        max_turns=changed_base.max_turns + 1,
    )
    with pytest.raises(ValueError, match="conflicts"):
        store.freeze(changed)


def test_protocol_store_rejects_tampered_payload(tmp_path) -> None:
    protocol = default_research_protocol()
    path = tmp_path / "protocol.json"
    JsonResearchProtocolStore(path).freeze(protocol)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["provider_id"] = "provider.tampered.v1"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="malformed research protocol"):
        JsonResearchProtocolStore(path).get()
