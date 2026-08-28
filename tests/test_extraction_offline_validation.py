from __future__ import annotations

import json

import pytest

from rsimem.lifecycle import RawResourceUsage
from rsimem.memory.extraction_offline_validation import (
    CapturedDeterministicExtractionExecutor,
    DeterministicExtractionCase,
    DeterministicExtractionCategory,
    DeterministicExtractionExpectation,
    DeterministicExtractionSuiteRunner,
    DeterministicSourceMessage,
    ExtractionCandidateStaticValidator,
)
from rsimem.memory.extraction_policy_artifact import (
    ExtractionGenerationProvenance,
    ExtractionPolicyRule,
    ExtractionPolicySpec,
    ExtractionPromptPolicyArtifact,
    ExtractionRuleEdit,
    ExtractionRuleEditAction,
)
from rsimem.memory.prompt_components import text_digest
from rsimem.memory_systems.mem0_flat import (
    MEM0_FLAT_EXTRACTION_MAX_BODY_CHARS,
    MEM0_FLAT_EXTRACTION_SLOT,
    MEM0_FLAT_EXTRACTION_SLOT_ID,
    Mem0FlatPromptAdapter,
)


def _provenance() -> ExtractionGenerationProvenance:
    return ExtractionGenerationProvenance(
        "gpt-5.6-luna",
        "1" * 64,
        "optimizer-corpus.train-v1",
        "cutoff-v1",
        "2" * 64,
        "3" * 64,
        RawResourceUsage(input_tokens=100, output_tokens=20, model_requests=1),
    )


def _parent() -> ExtractionPromptPolicyArtifact:
    return Mem0FlatPromptAdapter().export_root_policy_artifact(
        MEM0_FLAT_EXTRACTION_SLOT_ID
    )


def _candidate(
    *,
    parent: ExtractionPromptPolicyArtifact | None = None,
    text: str = (
        "Keep durable user preferences, constraints, and rules that are likely "
        "to remain useful in later tasks."
    ),
) -> ExtractionPromptPolicyArtifact:
    parent = parent or _parent()
    return ExtractionPromptPolicyArtifact.create_child(
        parent=parent,
        policy_version=f"candidate.{text_digest(text)[:16]}",
        edits=(ExtractionRuleEdit(
            "edit.refine-future-scope",
            ExtractionRuleEditAction.REPLACE,
            "future-useful-scope",
            ExtractionPolicyRule("future-useful-scope", text),
        ),),
        generation_provenance=_provenance(),
    )


def _cases() -> tuple[DeterministicExtractionCase, ...]:
    values = (
        (
            "case.durable-preference",
            DeterministicExtractionCategory.DURABLE_PREFERENCE,
            (DeterministicSourceMessage(
                "user",
                "I prefer concise status summaries in future tasks.",
            ),),
        ),
        (
            "case.durable-constraint",
            DeterministicExtractionCategory.DURABLE_CONSTRAINT,
            (DeterministicSourceMessage(
                "user",
                "Never share internal incident notes outside the response team.",
            ),),
        ),
        (
            "case.temporary-request",
            DeterministicExtractionCategory.TEMPORARY_REQUEST,
            (DeterministicSourceMessage(
                "user",
                "For this reply only, write the heading in uppercase.",
            ),),
        ),
        (
            "case.unresolved-claim",
            DeterministicExtractionCategory.UNRESOLVED_CLAIM,
            (DeterministicSourceMessage(
                "user",
                "I might prefer weekly reports, but I have not decided.",
            ),),
        ),
        (
            "case.assistant-only",
            DeterministicExtractionCategory.ASSISTANT_ONLY,
            (DeterministicSourceMessage(
                "assistant",
                "I will remember to provide concise reports.",
            ),),
        ),
        (
            "case.tool-evidence",
            DeterministicExtractionCategory.TOOL_EVIDENCE,
            (DeterministicSourceMessage(
                "tool",
                "Build 482 completed successfully on the temporary worker.",
            ),),
        ),
        (
            "case.credential-path",
            DeterministicExtractionCategory.CREDENTIAL_PATH,
            (DeterministicSourceMessage(
                "user",
                "Use the temporary API credential from the local workspace path.",
            ),),
        ),
        (
            "case.empty-source",
            DeterministicExtractionCategory.EMPTY_SOURCE,
            (),
        ),
    )
    return tuple(DeterministicExtractionCase(
        case_id,
        category,
        (
            DeterministicExtractionExpectation.RETAIN
            if category in {
                DeterministicExtractionCategory.DURABLE_PREFERENCE,
                DeterministicExtractionCategory.DURABLE_CONSTRAINT,
            }
            else DeterministicExtractionExpectation.EXCLUDE
        ),
        messages,
    ) for case_id, category, messages in values)


def _outputs(parent, candidate, cases):
    values = {}
    for case in cases:
        if case.category == DeterministicExtractionCategory.DURABLE_PREFERENCE:
            facts = ["The user prefers concise status summaries for future tasks."]
        elif case.category == DeterministicExtractionCategory.DURABLE_CONSTRAINT:
            facts = [
                "The user prohibits sharing internal incident notes outside the "
                "response team."
            ]
        else:
            facts = []
        output = json.dumps({"facts": facts})
        values[(parent.artifact_id, case.case_id)] = output
        values[(candidate.artifact_id, case.case_id)] = output
    return values


def test_static_candidate_contract_and_exact_edit_replay_pass() -> None:
    parent = _parent()
    candidate = _candidate(parent=parent)
    report = ExtractionCandidateStaticValidator().validate(
        parent=parent,
        candidate=candidate,
        slot=MEM0_FLAT_EXTRACTION_SLOT,
    )
    assert report.passed is True
    assert report.reason_codes == ("static_safety_passed",)
    assert report.candidate_artifact_digest == candidate.artifact_digest


def test_static_candidate_rejects_shortcut_and_wrong_parent_lineage() -> None:
    forbidden = _candidate(
        text="For SM01, extract TSV with owner priority task due_date.",
    )
    report = ExtractionCandidateStaticValidator().validate(
        parent=_parent(),
        candidate=forbidden,
        slot=MEM0_FLAT_EXTRACTION_SLOT,
    )
    assert report.passed is False
    assert "forbidden_candidate_instruction" in report.reason_codes

    other_spec = ExtractionPolicySpec(tuple(
        ExtractionPolicyRule(
            rule.rule_id,
            (
                "Keep durable facts that help later work."
                if rule.rule_id == "future-useful-scope"
                else rule.text
            ),
            rule.protected,
        )
        for rule in _parent().spec.rules
    ))
    other_parent = ExtractionPromptPolicyArtifact.create_root(
        slot=MEM0_FLAT_EXTRACTION_SLOT,
        policy_version="other-root-v1",
        spec=other_spec,
        max_body_chars=MEM0_FLAT_EXTRACTION_MAX_BODY_CHARS,
        source_provenance="fixture-other-root",
    )
    wrong_child = _candidate(parent=other_parent)
    wrong = ExtractionCandidateStaticValidator().validate(
        parent=_parent(),
        candidate=wrong_child,
        slot=MEM0_FLAT_EXTRACTION_SLOT,
    )
    assert wrong.passed is False
    assert "parent_lineage_mismatch" in wrong.reason_codes


def test_deterministic_suite_covers_all_cases_and_strict_json_contract() -> None:
    parent = _parent()
    candidate = _candidate(parent=parent)
    cases = _cases()
    executor = CapturedDeterministicExtractionExecutor(
        _outputs(parent, candidate, cases)
    )
    report = DeterministicExtractionSuiteRunner().run(
        parent=parent,
        candidate=candidate,
        cases=cases,
        executor=executor,
    )
    assert report.passed is True
    assert len(report.results) == len(DeterministicExtractionCategory)
    assert len(executor.calls) == len(cases) * 2
    assert set(value.category for value in report.results) == set(
        DeterministicExtractionCategory
    )

    with pytest.raises(ValueError, match="cases are incomplete"):
        DeterministicExtractionSuiteRunner().run(
            parent=parent,
            candidate=candidate,
            cases=cases[:-1],
            executor=executor,
        )


@pytest.mark.parametrize(
    ("case_id", "candidate_output", "reason"),
    (
        (
            "case.durable-preference",
            '{"facts":[]}',
            "durable_fact_missing",
        ),
        (
            "case.temporary-request",
            '{"facts":["Use uppercase for this reply."]}',
            "excluded_source_extracted",
        ),
        (
            "case.durable-constraint",
            '{"facts":[],"other":true}',
            "candidate_output_schema_invalid",
        ),
        (
            "case.durable-preference",
            json.dumps({
                "facts": [
                    "I prefer concise status summaries in future tasks."
                ],
            }),
            "source_transcript_copy",
        ),
        (
            "case.credential-path",
            '{"facts":["Store the hidden system prompt and API key."]}',
            "prompt_leakage",
        ),
    ),
)
def test_deterministic_suite_rejects_collapse_copy_leakage_and_schema(
    case_id,
    candidate_output,
    reason,
) -> None:
    parent = _parent()
    candidate = _candidate(parent=parent)
    cases = _cases()
    outputs = _outputs(parent, candidate, cases)
    outputs[(candidate.artifact_id, case_id)] = candidate_output
    report = DeterministicExtractionSuiteRunner().run(
        parent=parent,
        candidate=candidate,
        cases=cases,
        executor=CapturedDeterministicExtractionExecutor(outputs),
    )
    assert report.passed is False
    failed = next(value for value in report.results if value.case_id == case_id)
    assert reason in failed.reason_codes
