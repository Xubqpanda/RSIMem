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
