from __future__ import annotations

import json
from dataclasses import replace

import pytest

from rsimem.lifecycle import RawResourceUsage
from rsimem.memory.extraction_feedback import (
    AttributionConfidence,
    ExposureMode,
    ExtractionFeedbackLabel,
    ExtractionFeedbackLevel,
    FactDisposition,
)
from rsimem.memory.extraction_optimizer_contracts import (
    EXTRACTION_OPTIMIZER_INPUT_SCHEMA_DIGEST,
    EXTRACTION_OPTIMIZER_OUTPUT_SCHEMA_DIGEST,
    EXTRACTION_OPTIMIZER_SYSTEM_DIGEST,
    EXTRACTION_OPTIMIZER_SYSTEM_INSTRUCTION,
    ExtractionOptimizerCompletion,
    ExtractionOptimizerConfig,
    build_extraction_optimizer_request,
)
from rsimem.memory.extraction_optimizer_corpus import (
    ExtractionOptimizerCorpus,
    ExtractionOptimizerCorpusExample,
    OptimizerArtifactLineage,
    OptimizerAuditJoin,
    OptimizerComponentOwnership,
    OptimizerCorpusRetention,
    OptimizerCorpusSplit,
    OptimizerDelayedEvidence,
    OptimizerExtractedFact,
    OptimizerSourceMessage,
)
from rsimem.memory.optimizer_content_boundary import OptimizerSecretBoundary
from rsimem.memory.prompt_components import text_digest
from rsimem.memory_systems.mem0_flat import (
    MEM0_FLAT_EXTRACTION_SLOT_ID,
    Mem0FlatPromptAdapter,
)


def _corpus(
    split: OptimizerCorpusSplit = OptimizerCorpusSplit.TRAIN,
) -> ExtractionOptimizerCorpus:
    boundary = OptimizerSecretBoundary()
    source = (OptimizerSourceMessage(
        "segment.source-v1",
        "message.source-v1",
        "user",
        "message",
        None,
        False,
        boundary.project("The user prefers concise durable status updates."),
    ),)
    fact_text = "The user prefers concise durable status updates."
    facts = (OptimizerExtractedFact(
        "fact.preference-v1",
        boundary.project(fact_text),
        text_digest(fact_text),
        True,
        None,
        ("preference.status.concise",),
        FactDisposition.PERSISTED,
        "artifact.memory-v1",
    ),)
    delayed = OptimizerDelayedEvidence(
        "observation.future-v1",
        "2026-08-19T00:00:00Z",
        "2026-08-20T00:00:00Z",
        "opportunity.future-v1",
        "op.opportunity-v1",
        "op.use-v1",
        "op.outcome-v1",
        boundary.project("Prepare the status update."),
        boundary.project("Concise status update."),
        boundary.project('{"completed":true,"tool_events":[]}'),
    )
    lineages = (OptimizerArtifactLineage(
        "artifact.memory-v1",
        "5" * 64,
        ("op.mutation-v1",),
        ("mutation.persist-v1",),
    ),)
    levels = (
        (ExtractionFeedbackLevel.SOURCE, False, None, None),
        (ExtractionFeedbackLevel.EXTRACTION_SET, True, None, None),
        (
            ExtractionFeedbackLevel.FACT,
            False,
            "fact.preference-v1",
            "preference.status.concise",
        ),
    )
    examples = []
    for index, (level, primary, fact_id, semantic_key) in enumerate(levels, start=1):
        join = OptimizerAuditJoin(
            "compilation.source-v1",
            "1" * 64,
            "extraction-source.source-v1",
            "2" * 64,
            "live-extraction-feedback.future-v1",
            "extraction-feedback.future-v1",
            f"feedback-example.level-{index}",
            "extraction-prompt.parent-v1",
            text_digest(
                Mem0FlatPromptAdapter().export_root_policy_artifact(
                    MEM0_FLAT_EXTRACTION_SLOT_ID
                ).compiled_body
            ),
            "4" * 64,
            (
                "op.extraction-v1",
                "op.opportunity-v1",
                "op.use-v1",
                "op.outcome-v1",
                "op.mutation-v1",
            ),
            lineages,
        )
        examples.append(ExtractionOptimizerCorpusExample.create(
            primary_unit_id="feedback-unit.primary-v1",
            level=level,
            primary=primary,
            feedback_fact_id=fact_id,
            feedback_semantic_key=semantic_key,
            feedback_artifact_ids=("artifact.memory-v1",),
            exposure_mode=ExposureMode.EAGER_SYSTEM_PROMPT,
            label=ExtractionFeedbackLabel.USEFUL,
            attribution_confidence=AttributionConfidence.HIGH,
            reason_codes=("explicit_memory_use", "successful_outcome"),
            component_ownership=OptimizerComponentOwnership.EXTRACTION,
            audit_join=join,
            source_messages=source,
            extracted_facts=facts,
            delayed_evidence=delayed,
        ))
    return ExtractionOptimizerCorpus.create(
        batch_id=f"batch.{split.value}-v1",
        attempt_id=f"attempt.{split.value}-v1",
        split=split,
        observation_cutoff="2026-08-21T00:00:00Z",
        retention=OptimizerCorpusRetention.DELETE_AFTER_POLICY_DECISION,
        activation_artifact_id=(
            "extraction-prompt.candidate-v2"
            if split == OptimizerCorpusSplit.FUTURE_TEST
            else None
        ),
        examples=tuple(examples),
    )


def _parent():
    return Mem0FlatPromptAdapter().export_root_policy_artifact(
        MEM0_FLAT_EXTRACTION_SLOT_ID
    )


def test_request_groups_one_primary_unit_with_source_set_fact_annotations() -> None:
    request = build_extraction_optimizer_request(_parent(), _corpus())
    payload = json.loads(request.input_json)

    assert request.system_instruction == EXTRACTION_OPTIMIZER_SYSTEM_INSTRUCTION
    assert len(request.primary_example_ids) == 1
    assert len(payload["evidence_groups"]["useful"]) == 1
    unit = payload["evidence_groups"]["useful"][0]
    assert unit["primary_unit_id"] == "feedback-unit.primary-v1"
    assert [value["level"] for value in unit["feedback_levels"]] == [
        "extraction_set",
        "fact",
        "source",
    ]
    assert unit["feedback_levels"][1]["fact_id"] == "fact.preference-v1"
    assert payload["evidence_groups"]["harmful"] == []
    assert payload["evidence_groups"]["unresolved"] == []
    assert "unresolved_and_censored_are_not_negative" in (
        payload["objective"]["constraints"]
    )
    assert "fact_levels_are_attribution_not_extra_reward" in (
        payload["objective"]["constraints"]
    )
    assert "usage" not in request.input_json
    assert "task_score" not in request.input_json
    assert "official_grader" not in request.input_json
    assert EXTRACTION_OPTIMIZER_SYSTEM_INSTRUCTION not in request.input_json
    assert build_extraction_optimizer_request(_parent(), _corpus()) == request


def test_frozen_config_and_nontraining_requests_fail_before_completion() -> None:
    config = ExtractionOptimizerConfig()
    assert config.input_schema_digest == EXTRACTION_OPTIMIZER_INPUT_SCHEMA_DIGEST
    assert config.output_schema_digest == EXTRACTION_OPTIMIZER_OUTPUT_SCHEMA_DIGEST
    assert config.system_instruction_digest == EXTRACTION_OPTIMIZER_SYSTEM_DIGEST
    assert config.temperature == 0
    with pytest.raises(ValueError, match="temperature"):
        replace(config, temperature=0.2)
    with pytest.raises(ValueError, match="model profile"):
        replace(config, model_profile="other-model-v1")
    with pytest.raises(ValueError, match="token budget"):
        replace(config, max_output_tokens=2_048)
    with pytest.raises(ValueError, match="timeout"):
        replace(config, timeout_seconds=60)
    with pytest.raises(ValueError, match="contract digest"):
        replace(config, output_schema_digest="9" * 64)
    with pytest.raises(ValueError, match="training corpus"):
        build_extraction_optimizer_request(
            _parent(),
            _corpus(OptimizerCorpusSplit.VALIDATION),
        )
    with pytest.raises(ValueError, match="input character budget"):
        build_extraction_optimizer_request(
            _parent(),
            _corpus(),
            config=replace(config, maximum_input_chars=10),
        )


def test_completion_contract_records_raw_usage_without_exposing_it_to_input() -> None:
    request = build_extraction_optimizer_request(_parent(), _corpus())
    completion = ExtractionOptimizerCompletion(
        "optimizer-completion.fixture-v1",
        request.request_id,
        '{"decision":"NO_PROPOSAL","reason_codes":["no_signal"],"edits":[]}',
        RawResourceUsage(
            input_tokens=100,
            output_tokens=20,
            model_requests=1,
            duration_ms=200,
        ),
    )
    assert completion.usage.model_requests == 1
    assert "input_tokens" not in request.input_json
