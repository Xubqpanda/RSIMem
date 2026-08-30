from __future__ import annotations

import sys
from pathlib import Path

_PAST_BENCH_SRC = Path(__file__).parents[1] / "benchmarks" / "past-bench" / "src"
if str(_PAST_BENCH_SRC) not in sys.path:
    sys.path.insert(0, str(_PAST_BENCH_SRC))

from past_bench.runtime.adapters.hermes import (
    _past_bench_artifact_set_provider,
    _past_bench_opportunity_provider,
)
from past_bench.models.task import TaskDefinition
from past_bench.runner.self_evolve import build_past_bench_application_opportunity_schema
from past_bench.runner.self_evolve import build_hermes_extra_body
from rsimem.memory.extraction_feedback import (
    ExtractedFactEvidence,
    ExtractionSetStatus,
    ExtractionSourceEvidence,
    FactDisposition,
)


def test_past_opportunity_provider_uses_source_provenance_and_visible_surface() -> None:
    values = _past_bench_opportunity_provider({
        "messages": [
            {
                "role": "user",
                    "content": "Save a durable default: use TSV with owner, priority, task, due_date. Share only with approved recipients.",
            },
            {
                "role": "assistant",
                "tool_calls": [{"function": {"name": "notes_share"}}],
            },
        ],
        "rsimem_source_provenance_id": "pure-extraction-provenance.test",
    })
    assert {value.semantic_requirement for value in values} == {
        "application.notes.output.tsv",
        "application.notes.share.recipient_policy",
    }
    assert all(
        value.provenance_id == "pure-extraction-provenance.test"
        and value.source_surface.value == "current_input"
        for value in values
    )


def test_past_opportunity_provider_binds_only_retrieved_source_keys() -> None:
    values = _past_bench_opportunity_provider({
        "messages": [
            {
                "role": "user",
                "content": "Extract action items and do not share with external recipients.",
            },
            {
                "role": "assistant",
                "tool_calls": [{"function": {"name": "notes_share"}}],
            },
        ],
        "rsimem_source_records": [
            {
                "provenance_id": "pure-extraction-provenance.one",
                "semantic_keys": ["application.notes.share.recipient_policy"],
            },
            {
                "provenance_id": "pure-extraction-provenance.two",
                "semantic_keys": ["application.notes.output.tsv"],
            },
        ],
    })
    assert [value.provenance_id for value in values] == [
        "pure-extraction-provenance.one"
    ]
    assert values[0].source_surface.value == "tool_schema"


def test_past_opportunity_provider_uses_frozen_public_schema_for_generic_future_request() -> None:
    schema = {
        "schema_id": "past-bench.notes.application.v1",
        "schema_version": 1,
        "application_contract": {
            "schema_id": "past-bench.notes.application.v1",
            "schema_version": 2,
            "version": "v1",
            "requirement_ids": ["application.notes.share.recipient_policy"],
            "schema_digest": "122c4d36732dd4a2824d1b48944eed2aa80e9bf10ad51f596ba143673051797f",
        },
        "opportunities": [{
            "semantic_key": "application.notes.share.recipient_policy",
            "surface": "tool_schema",
            "tool_name": "notes_share",
            "required_parameter": "recipients",
        }],
    }
    values = _past_bench_opportunity_provider({
        "messages": [{
            "role": "user",
            "content": "Review today's note and share it with the people who should receive it.",
        }, {
            "role": "assistant",
            "tool_calls": [{"function": {"name": "notes_share"}}],
        }],
        "rsimem_application_schema": schema,
        "rsimem_source_records": [{
            "provenance_id": "pure-extraction-provenance.future",
            "semantic_keys": ["application.notes.share.recipient_policy"],
        }],
    })
    assert len(values) == 1
    assert values[0].semantic_requirement == "application.notes.share.recipient_policy"
    assert values[0].source_surface.value == "application_schema"


def test_past_opportunity_provider_treats_supplied_schema_as_authoritative() -> None:
    """Legacy notes text must not manufacture pure opportunities in a formal run."""

    schema = {
        "schema_id": "past-bench.notes.application.v1",
        "schema_version": 1,
        "application_contract": {
            "schema_id": "past-bench.notes.application.v1",
            "schema_version": 2,
            "version": "v1",
            "requirement_ids": ["application.notes.share.recipient_policy"],
            "schema_digest": "122c4d36732dd4a2824d1b48944eed2aa80e9bf10ad51f596ba143673051797f",
        },
        "opportunities": [{
            "semantic_key": "application.notes.share.recipient_policy",
            "surface": "tool_schema",
            "tool_name": "notes_share",
            "required_parameter": "recipients",
        }],
    }
    values = _past_bench_opportunity_provider({
        "messages": [{
            "role": "user",
            "content": "Use TSV with owner, priority, task, and due_date columns.",
        }],
        "rsimem_application_schema": schema,
        "rsimem_source_provenance_id": "pure-extraction-provenance.schema-authority",
    })
    assert values == ()


def test_past_opportunity_provider_supports_non_notes_schema_tools() -> None:
    """Application-owned tool schemas are not coupled to the notes prefix."""

    from rsimem.memory.opportunity import ApplicationOpportunitySchema

    contract = ApplicationOpportunitySchema.create(
        schema_id="calendar.application.v1",
        version="v1",
        requirement_ids=("application.calendar.read.policy",),
    )
    schema = {
        "schema_id": "calendar.application.v1",
        "schema_version": 1,
        "application_contract": contract.payload(),
        "opportunities": [{
            "semantic_key": "application.calendar.read.policy",
            "surface": "tool_schema",
            "tool_name": "calendar_read",
            "required_parameter": "calendar_id",
        }],
    }
    values = _past_bench_opportunity_provider({
        "messages": [{
            "role": "assistant",
            "tool_calls": [{"function": {"name": "calendar_read"}}],
        }],
        "rsimem_application_schema": schema,
        "rsimem_source_records": [{
            "provenance_id": "pure-extraction-provenance.calendar",
            "semantic_keys": ["application.calendar.read.policy"],
        }],
    })
    assert len(values) == 1
    assert values[0].semantic_requirement == "application.calendar.read.policy"
    assert values[0].source_surface.value == "application_schema"


def test_application_schema_is_derived_from_visible_tool_contract() -> None:
    task = TaskDefinition.model_validate({
        "task_id": "schema-fixture",
        "task_name": "schema fixture",
        "prompt": {"text": "share a note", "language": "en"},
        "tools": [{
            "name": "notes_share",
            "description": "Share a meeting note",
            "input_schema": {
                "type": "object",
                "properties": {
                    "note_id": {"type": "string"},
                    "recipients": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["note_id", "recipients"],
            },
        }],
    })
    schema = build_past_bench_application_opportunity_schema(task)
    assert schema["schema_id"] == "past-bench.notes.application.v1"
    assert schema["schema_version"] == 1
    assert schema["opportunities"] == [{
        "semantic_key": "application.notes.share.recipient_policy",
        "surface": "tool_schema",
        "tool_name": "notes_share",
        "required_parameter": "recipients",
    }]


def test_hermes_extra_body_carries_application_schema_to_runtime() -> None:
    schema = {
        "schema_id": "past-bench.notes.application.v1",
        "schema_version": 1,
        "opportunities": [],
    }
    body = build_hermes_extra_body(
        home_dir=Path("/tmp/hermes-schema-home"),
        artifacts_dir=Path("/tmp/hermes-schema-artifacts"),
        persistence_enabled=True,
        memory_enabled=False,
        user_profile_enabled=False,
        skills_enabled=False,
        session_search_enabled=False,
        memory_nudge_interval=0,
        memory_flush_min_turns=0,
        skill_creation_nudge_interval=0,
        background_review_wait_s=0,
        rsimem_mode="native+ledger",
        rsimem_semantic_writeback_mode="static",
        rsimem_application_opportunity_schema=schema,
    )
    assert body["hermes"]["rsimem"]["application_opportunity_schema"] == schema


def test_past_artifact_set_provider_requires_complete_multi_fact_source() -> None:
    source = ExtractionSourceEvidence(
        "source.provider-set",
        "a" * 64,
        "extraction-set.provider-set",
        ExtractionSetStatus.NONEMPTY,
        ("application.notes.output.tsv",),
        (
            ExtractedFactEvidence(
                "fact.provider-set.a",
                ("application.notes.output.tsv",),
                FactDisposition.PERSISTED,
                artifact_id="artifact.provider-set.a",
            ),
            ExtractedFactEvidence(
                "fact.provider-set.b",
                ("application.notes.output.tsv",),
                FactDisposition.PERSISTED,
                artifact_id="artifact.provider-set.b",
            ),
        ),
    )
    values = _past_bench_artifact_set_provider(
        source,
        provenance_id="pure-extraction-provenance.provider-set",
    )
    assert len(values) == 1
    assert values[0].complete is True
    assert values[0].provenance_id == "pure-extraction-provenance.provider-set"


def test_past_artifact_set_provider_does_not_copy_source_key_to_unbound_facts() -> None:
    source = ExtractionSourceEvidence(
        "source.provider-unbound",
        "a" * 64,
        "extraction-set.provider-unbound",
        ExtractionSetStatus.NONEMPTY,
        ("application.notes.output.tsv",),
        (
            ExtractedFactEvidence(
                "fact.provider-unbound.a",
                ("application.notes.output.tsv",),
                FactDisposition.PERSISTED,
                artifact_id="artifact.provider-unbound.a",
            ),
            # The source advertises one key, but this member has no explicit
            # per-fact binding.  A provider must remain unresolved rather than
            # copying the source key to the unbound fact.
            ExtractedFactEvidence(
                "fact.provider-unbound.b",
                (),
                FactDisposition.PERSISTED,
                artifact_id="artifact.provider-unbound.b",
            ),
        ),
    )
    assert _past_bench_artifact_set_provider(
        source,
        provenance_id="pure-extraction-provenance.provider-unbound",
    ) == ()


def test_past_artifact_set_provider_rejects_filtered_members() -> None:
    source = ExtractionSourceEvidence(
        "source.provider-filtered",
        "a" * 64,
        "extraction-set.provider-filtered",
        ExtractionSetStatus.NONEMPTY,
        ("application.notes.output.tsv",),
        (
            ExtractedFactEvidence(
                "fact.provider-filtered.a",
                ("application.notes.output.tsv",),
                FactDisposition.PERSISTED,
                artifact_id="artifact.provider-filtered.a",
            ),
            ExtractedFactEvidence(
                "fact.provider-filtered.b",
                ("application.notes.output.tsv",),
                FactDisposition.FILTERED,
            ),
        ),
    )
    assert _past_bench_artifact_set_provider(
        source,
        provenance_id="pure-extraction-provenance.provider-filtered",
    ) == ()
