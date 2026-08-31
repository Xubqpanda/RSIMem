from __future__ import annotations

import json
from dataclasses import replace

import pytest

from rsimem.memory.opportunity import (
    ApplicationOpportunitySchema,
    JsonApplicationOpportunitySchemaRegistry,
    JsonOpportunityEvidenceLog,
    OpportunityEvidence,
    OpportunityResolutionStatus,
    OpportunitySurface,
    resolve_opportunity,
)
from rsimem.memory.evidence_planes import EvidenceSourceKind
from rsimem.memory import OpportunityEvidence as ExportedOpportunityEvidence


def _evidence() -> OpportunityEvidence:
    return OpportunityEvidence.create(
        source_surface=OpportunitySurface.TOOL_SCHEMA,
        semantic_requirement="resource.share.recipient_policy",
        observation_time="2026-08-30T01:02:03Z",
        operation_id="op.tool-schema.v1",
        provenance_id="provenance.run.v1",
        source_payload={"tool_name_digest": "a" * 64, "success_field": True},
    )


def test_opportunity_evidence_is_stable_and_has_no_benchmark_identity() -> None:
    first = _evidence()
    assert ExportedOpportunityEvidence is OpportunityEvidence
    second = OpportunityEvidence.from_payload(json.loads(json.dumps(first.payload())))
    assert first == second
    serialized = json.dumps(first.payload(), ensure_ascii=True)
    for forbidden in ("family_id", "stage", "grader", "answer_key", "official_score"):
        assert forbidden not in serialized
    assert first.evidence_plane.value == "pure_process"
    assert first.evidence_source.value == "runtime_observation"


def test_application_schema_is_frozen_and_versioned() -> None:
    schema = ApplicationOpportunitySchema.create(
        schema_id="resource-policy",
        version="v1",
        requirement_ids=("resource.share.recipient_policy",),
    )
    assert ApplicationOpportunitySchema.from_payload(
        json.loads(json.dumps(schema.payload()))
    ) == schema
    tampered_schema = schema.payload()
    tampered_schema["requirement_ids"] = ["resource.other_policy"]
    with pytest.raises(ValueError, match="malformed application opportunity schema"):
        ApplicationOpportunitySchema.from_payload(tampered_schema)
    evidence = OpportunityEvidence.create(
        source_surface=OpportunitySurface.APPLICATION_SCHEMA,
        semantic_requirement="resource.share.recipient_policy",
        observation_time="2026-08-30T01:02:03Z",
        operation_id="op.application-schema.v1",
        provenance_id="provenance.application.v1",
        source_payload={"schema_event": "published"},
        application_schema=schema,
    )
    assert evidence.application_schema_digest == schema.schema_digest
    assert evidence.evidence_source is EvidenceSourceKind.APPLICATION_CONTRACT
    with pytest.raises(ValueError, match="not in the frozen schema"):
        OpportunityEvidence.create(
            source_surface=OpportunitySurface.APPLICATION_SCHEMA,
            semantic_requirement="unregistered.requirement",
            observation_time="2026-08-30T01:02:03Z",
            operation_id="op.application-schema.v2",
            provenance_id="provenance.application.v2",
            source_payload={},
            application_schema=schema,
        )


def test_application_schema_cannot_be_relabelled_as_runtime_observation() -> None:
    schema = ApplicationOpportunitySchema.create(
        schema_id="resource-policy",
        version="v1",
        requirement_ids=("resource.share.recipient_policy",),
    )
    evidence = OpportunityEvidence.create(
        source_surface=OpportunitySurface.APPLICATION_SCHEMA,
        semantic_requirement="resource.share.recipient_policy",
        observation_time="2026-08-30T01:02:03Z",
        operation_id="op.application-schema.v3",
        provenance_id="provenance.application.v3",
        source_payload={"schema_event": "published"},
        application_schema=schema,
    )
    with pytest.raises(ValueError, match="application-schema opportunity"):
        replace(evidence, evidence_source=EvidenceSourceKind.RUNTIME_OBSERVATION)


def test_application_schema_replay_requires_trusted_registry(tmp_path) -> None:
    schema = ApplicationOpportunitySchema.create(
        schema_id="resource-policy",
        version="v1",
        requirement_ids=("resource.share.recipient_policy",),
    )
    evidence = OpportunityEvidence.create(
        source_surface=OpportunitySurface.APPLICATION_SCHEMA,
        semantic_requirement="resource.share.recipient_policy",
        observation_time="2026-08-30T01:02:03Z",
        operation_id="op.application-schema.registry",
        provenance_id="provenance.application.registry",
        source_payload={"schema_event": "published"},
        application_schema=schema,
    )
    payload = json.loads(json.dumps(evidence.payload()))
    with pytest.raises(ValueError, match="malformed opportunity evidence"):
        OpportunityEvidence.from_payload(payload)
    assert OpportunityEvidence.from_payload(
        payload,
        application_schema=schema,
    ) == evidence

    registry = JsonApplicationOpportunitySchemaRegistry(
        tmp_path / "application-schemas.jsonl"
    )
    assert registry.register(schema) is True
    assert registry.register(schema) is False
    assert JsonApplicationOpportunitySchemaRegistry(
        tmp_path / "application-schemas.jsonl"
    ).records() == (schema,)
    conflicting = ApplicationOpportunitySchema.create(
        schema_id="resource-policy",
        version="v1",
        requirement_ids=("resource.delete.confirmation",),
    )
    with pytest.raises(ValueError, match="conflicting application opportunity schema"):
        registry.register(conflicting)

    evidence_log_path = tmp_path / "opportunities.jsonl"
    evidence_log = JsonOpportunityEvidenceLog(
        evidence_log_path,
        schema_registry=registry,
    )
    assert evidence_log.append(evidence) is True
    assert JsonOpportunityEvidenceLog(
        evidence_log_path,
        schema_registry=JsonApplicationOpportunitySchemaRegistry(
            tmp_path / "application-schemas.jsonl"
        ),
    ).records() == (evidence,)
    with pytest.raises(ValueError, match="malformed opportunity evidence"):
        JsonOpportunityEvidenceLog(evidence_log_path)


def test_application_schema_create_rejects_untyped_contract() -> None:
    with pytest.raises(TypeError, match="wrong type"):
        OpportunityEvidence.create(
            source_surface=OpportunitySurface.APPLICATION_SCHEMA,
            semantic_requirement="resource.share.recipient_policy",
            observation_time="2026-08-30T01:02:03Z",
            operation_id="op.application-schema.wrong-type",
            provenance_id="provenance.application.wrong-type",
            source_payload={"schema_event": "published"},
            application_schema="resource-policy",  # type: ignore[arg-type]
        )


def test_opportunity_resolution_distinguishes_confounded_censored_and_absent() -> None:
    evidence = _evidence()
    observed = resolve_opportunity(evidence)
    assert observed.status == OpportunityResolutionStatus.OBSERVED
    confounded = resolve_opportunity(
        evidence,
        current_input_requirements=("resource.share.recipient_policy",),
    )
    assert confounded.status == OpportunityResolutionStatus.CURRENT_INPUT_CONFOUNDED
    censored = resolve_opportunity(evidence, observation_complete=False)
    assert censored.status == OpportunityResolutionStatus.CENSORED
    absent = resolve_opportunity(None)
    assert absent.status == OpportunityResolutionStatus.UNRESOLVED
    assert absent.reason_code == "opportunity_not_observed"


def test_benchmark_or_final_plane_cannot_be_used_for_opportunity() -> None:
    evidence = _evidence()
    with pytest.raises(ValueError, match="plane and source identity"):
        replace(evidence, evidence_plane="benchmark_audit")


def test_hidden_evaluation_payload_cannot_create_opportunity() -> None:
    with pytest.raises(ValueError, match="forbidden evaluation fields"):
        OpportunityEvidence.create(
            source_surface=OpportunitySurface.TOOL_SCHEMA,
            semantic_requirement="resource.share.recipient_policy",
            observation_time="2026-08-30T01:02:03Z",
            operation_id="op.hidden.v1",
            provenance_id="provenance.hidden.v1",
            source_payload={"hidden_expectation": "recipient"},
        )


@pytest.mark.parametrize("payload", (None, "", {}, [], ()))
def test_empty_source_cannot_create_visible_opportunity(payload) -> None:
    with pytest.raises(ValueError, match="source payload"):
        OpportunityEvidence.create(
            source_surface=OpportunitySurface.TOOL_SCHEMA,
            semantic_requirement="resource.share.recipient_policy",
            observation_time="2026-08-30T01:02:03Z",
            operation_id="op.empty.v1",
            provenance_id="provenance.empty.v1",
            source_payload=payload,
        )


def test_opportunity_identity_does_not_depend_on_family_or_stage() -> None:
    first = _evidence()
    second = OpportunityEvidence.create(
        source_surface=OpportunitySurface.TOOL_SCHEMA,
        semantic_requirement="resource.share.recipient_policy",
        observation_time="2026-08-30T01:02:03Z",
        operation_id="op.tool-schema.v1",
        provenance_id="provenance.run.v1",
        source_payload={"tool_name_digest": "a" * 64, "success_field": True},
    )
    assert first.evidence_id == second.evidence_id
    assert "SM02" not in json.dumps(first.payload())


def test_opportunity_log_is_restart_safe_and_rejects_conflicts(tmp_path) -> None:
    path = tmp_path / "opportunities.jsonl"
    first = _evidence()
    log = JsonOpportunityEvidenceLog(path)
    assert log.append(first) is True
    assert log.append(first) is False
    assert JsonOpportunityEvidenceLog(path).records() == (first,)

    conflicting = first.payload()
    conflicting["source_digest"] = "b" * 64
    path.write_text(
        path.read_text(encoding="utf-8")
        + json.dumps(conflicting, ensure_ascii=True, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="malformed opportunity evidence"):
        JsonOpportunityEvidenceLog(path)


def test_opportunity_log_failed_commit_keeps_previous_complete_records(
    tmp_path,
    monkeypatch,
) -> None:
    first = _evidence()
    second = OpportunityEvidence.create(
        source_surface=OpportunitySurface.TOOL_SCHEMA,
        semantic_requirement="resource.share.recipient_policy",
        observation_time="2026-08-30T01:02:04Z",
        operation_id="op.tool-schema.v2",
        provenance_id="provenance.run.v2",
        source_payload={"tool_name_digest": "b" * 64, "success_field": True},
    )
    path = tmp_path / "opportunities-atomic.jsonl"
    log = JsonOpportunityEvidenceLog(path)
    assert log.append(first) is True

    def fail_replace(*args, **kwargs):
        raise OSError("simulated archive interruption")

    monkeypatch.setattr("rsimem.memory._atomic_jsonl.os.replace", fail_replace)
    with pytest.raises(OSError, match="simulated archive interruption"):
        log.append(second)
    assert log.records() == (first,)
    assert path.read_text(encoding="utf-8").endswith("\n")


@pytest.mark.parametrize(
    "store_factory",
    (
        lambda path: JsonOpportunityEvidenceLog(path),
        lambda path: JsonApplicationOpportunitySchemaRegistry(path),
    ),
)
def test_opportunity_stores_reject_symlinked_paths(tmp_path, store_factory) -> None:
    target = tmp_path / "target"
    target.write_text("", encoding="utf-8")
    path = tmp_path / "store"
    path.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        store_factory(path)


@pytest.mark.parametrize(
    "store_factory",
    (
        lambda path: JsonOpportunityEvidenceLog(path),
        lambda path: JsonApplicationOpportunitySchemaRegistry(path),
    ),
)
def test_opportunity_stores_reject_symlinked_locks(tmp_path, store_factory) -> None:
    path = tmp_path / "store"
    lock_target = tmp_path / "lock-target"
    lock_target.write_text("", encoding="utf-8")
    path.with_name(path.name + ".lock").symlink_to(lock_target)
    with pytest.raises(ValueError, match="lock.*symlink"):
        store_factory(path).records()
