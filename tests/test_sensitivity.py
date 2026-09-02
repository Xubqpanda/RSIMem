from __future__ import annotations

import pytest

from rsimem.memory import MemoryKind
from rsimem.memory.evidence_planes import EvidencePlane
from rsimem.memory.family_matrix import PastFamilyMatrix
from rsimem.research_protocol import ResearchProtocol, default_research_protocol
from rsimem.sensitivity import (
    OracleArtifact,
    SensitivityMatrix,
    SensitivityPanel,
)


def _protocol(kind: MemoryKind) -> ResearchProtocol:
    base = default_research_protocol()
    return ResearchProtocol.create(
        memory_units=base.memory_units,
        family_matrix=PastFamilyMatrix.create_default(),
        split=base.split,
        sensitivity_target_kind=kind,
    )


def test_semantic_matrix_has_five_conditions_per_target_family() -> None:
    protocol = default_research_protocol()
    matrix = SensitivityMatrix.create_for_panel(
        panel=SensitivityPanel.SEMANTIC,
        protocol=protocol,
        family_matrix=PastFamilyMatrix.create_default(),
    )
    assert len(matrix.family_ids) == 7
    assert len(matrix.cases) == 35
    assert len(matrix.oracle_artifacts) == 1
    assert matrix.oracle_artifacts[0].oracle_only is True
    assert matrix.oracle_artifacts[0].evidence_plane is EvidencePlane.BENCHMARK_AUDIT
    view = matrix.method_visible_case(matrix.family_ids[0], "type_matched_oracle")
    assert set(view) == {"condition", "target_kind", "mechanism", "oracle_available"}
    assert "family_id" not in str(view)
    assert "self-evolve-tasks-v2" not in str(view)


def test_each_panel_is_type_isolated() -> None:
    expected = {
        SensitivityPanel.SEMANTIC: (7, MemoryKind.SEMANTIC),
        SensitivityPanel.EPISODIC: (3, MemoryKind.EPISODIC),
        SensitivityPanel.PROCEDURAL: (10, MemoryKind.PROCEDURAL),
    }
    for panel, (family_count, kind) in expected.items():
        matrix = SensitivityMatrix.create_for_panel(
            panel=panel,
            protocol=_protocol(kind),
            family_matrix=PastFamilyMatrix.create_default(),
        )
        assert len(matrix.family_ids) == family_count
        assert len(matrix.cases) == family_count * 5
        assert matrix.target_kind is kind
        assert all(case.target_kind is kind for case in matrix.cases)


def test_sensitivity_matrix_rejects_protocol_oracle_kind_drift() -> None:
    with pytest.raises(ValueError, match="oracle target kind"):
        SensitivityMatrix.create_for_panel(
            panel=SensitivityPanel.EPISODIC,
            protocol=default_research_protocol(),
            family_matrix=PastFamilyMatrix.create_default(),
        )


def test_oracle_artifact_cannot_be_final_or_non_oracle() -> None:
    with pytest.raises(ValueError, match="plane and source"):
        OracleArtifact(
            artifact_id="oracle.semantic.invalid",
            panel=SensitivityPanel.SEMANTIC,
            target_kind=MemoryKind.SEMANTIC,
            mechanism="type_matched",
            minimal_field_ids=("fact",),
            content_digest="0" * 64,
            evidence_plane=EvidencePlane.FINAL_EVALUATION,
        )
    with pytest.raises(ValueError, match="oracle_only"):
        OracleArtifact(
            artifact_id="oracle.semantic.invalid",
            panel=SensitivityPanel.SEMANTIC,
            target_kind=MemoryKind.SEMANTIC,
            mechanism="type_matched",
            minimal_field_ids=("fact",),
            content_digest="0" * 64,
            oracle_only=False,
        )
