from __future__ import annotations

from dataclasses import replace

import pytest

from rsimem.lifecycle import RawResourceUsage, SegmentKind
from rsimem.memory.extraction_feedback import (
    ArtifactSemanticBinding,
    DeploymentObservation,
    ExposureMode,
    ExtractedFactEvidence,
    ExtractionFeedbackBuilder,
    ExtractionSetStatus,
    ExtractionSourceEvidence,
    FactDisposition,
    FeedbackOperationJoin,
    FutureMemoryEvidence,
    default_feedback_contract_registry,
)
from rsimem.memory.extraction_optimizer_builder import (
    DelayedEvidenceContent,
    ExtractionFactContent,
    ExtractionOptimizerCorpusBuilder,
)
from rsimem.memory.extraction_optimizer_audit import audit_optimizer_corpus_isolation
from rsimem.memory.extraction_optimizer_corpus import (
    ExtractionOptimizerCorpus,
    OptimizerCorpusRetention,
    OptimizerCorpusSplit,
)
from rsimem.memory.extraction_projection import (
    ExtractionSourceRecord,
    LiveExtractionFeedbackRecord,
)
from rsimem.memory.extraction_source import (
    ExtractionSourceMessage,
    ExtractionSourceProjection,
)
from rsimem.memory.ingestion import InternalMemoryAction
from rsimem.memory.operation_graph import (
    ArtifactKind,
    ArtifactNode,
    MutationEdge,
    OperationContext,
    OperationGraph,
    OperationKind,
    OperationRecord,
    OperationStatus,
)
from rsimem.memory.prompt_components import content_digest, text_digest


TSV_KEY = "preference.summary.tsv"
FACT_TEXT = "The user prefers four-column TSV task summaries."


def _projection(content: str = "Always provide my task summary as TSV.") -> ExtractionSourceProjection:
    message = ExtractionSourceMessage(
        "segment.learn-v1",
        "message.learn-v1",
        "user",
        content,
        SegmentKind.MESSAGE,
    )
    values = {
        "schema_version": 1,
        "schema": "completed-task-extraction-source-v1",
        "snapshot_id": "snapshot.learn-v1",
        "task_id": "task.learn-v1",
        "context_revision": "revision.learn-v1",
        "messages": [message.prompt_payload()],
        "source_message_ids": [message.source_message_id],
        "source_segment_ids": [message.segment_id],
        "omitted_segment_ids": [],
        "truncated_segment_ids": [],
        "max_content_chars": 1_000,
        "projected_content_chars": len(content),
    }
    digest = content_digest(values)
    return ExtractionSourceProjection(
        f"extraction-source.{digest[:40]}",
        "snapshot.learn-v1",
        "task.learn-v1",
        "revision.learn-v1",
        (message,),
        (message.source_message_id,),
        (message.segment_id,),
        (),
        (),
        1_000,
        len(content),
        digest,
    )


def _fixture():
    projection = _projection()
    source = ExtractionSourceEvidence(
        "artifact.source-v1",
        projection.projection_digest,
        "op.extraction-v1",
        ExtractionSetStatus.NONEMPTY,
        (TSV_KEY,),
        (ExtractedFactEvidence(
            "fact.preference-v1",
            (TSV_KEY,),
            FactDisposition.PERSISTED,
            artifact_id="artifact.memory-v1",
        ),),
    )
    traces = (ExtractionFactContent(
        "fact.preference-v1",
        FACT_TEXT,
        True,
        None,
    ),)
    source_record = ExtractionSourceRecord.create(
        family_id="SM01_preference_adoption",
        stage="learn_a",
        run_id="run.learn-v1",
        episode_id="episode.learn-v1",
        session_id="session.learn-v1",
        task_id="task.learn-v1",
        compilation_id="compilation.learn-v1",
        extraction_artifact_id="prompt-component.root-v1",
        extraction_artifact_digest="1" * 64,
        extraction_output_digest=content_digest([
            value.trace_payload() for value in traces
        ]),
        source=source,
    )
    join = FeedbackOperationJoin(
        "op.opportunity-v1",
        "op.use-v1",
        "op.outcome-v1",
    )
    current_input = "Prepare the action-item report."
    observation = DeploymentObservation(
        "observation.eval-v1",
        "SM01_preference_adoption",
        "eval_near",
        "task.eval-v1",
        text_digest(current_input),
        (),
        (TSV_KEY,),
        "owner\tpriority\ttask\tdue_date\nA\thigh\tShip\t2026/09/01",
        (),
        True,
    )
    feedback_dataset = ExtractionFeedbackBuilder(
        default_feedback_contract_registry()
    ).build(
        source,
        observation,
        FutureMemoryEvidence(
            "opportunity.eval-v1",
            ExposureMode.EAGER_SYSTEM_PROMPT,
            (ArtifactSemanticBinding("artifact.memory-v1", (TSV_KEY,)),),
            "op.opportunity-v1",
            "op.injection-v1",
        ),
        operation_join=join,
    )
    feedback = LiveExtractionFeedbackRecord.create(
        family_id="SM01_preference_adoption",
        stage="eval_near",
        run_id="run.eval-v1",
        trace_id="trace.eval-v1",
        episode_id="episode.eval-v1",
        session_id="session.eval-v1",
        task_id="task.eval-v1",
        deployment_observation_id="observation.eval-v1",
        source_record_id=source_record.record_id,
        opportunity_operation_id=join.opportunity_operation_id,
        use_operation_id=join.use_operation_id,
        outcome_operation_id=join.outcome_operation_id,
        dataset=feedback_dataset,
    )
    source_context = OperationContext(
        source_record.run_id,
        source_record.episode_id,
        source_record.session_id,
        source_record.task_id,
        "policy.root-v1",
        "prompt.root-v1",
        "framework-v1",
    )
    future_context = OperationContext(
        feedback.run_id,
        feedback.episode_id,
        feedback.session_id,
        feedback.task_id,
        "policy.root-v1",
        "prompt.root-v1",
        "framework-v1",
    )
    usage = RawResourceUsage()

    def operation(
        operation_id: str,
        kind: OperationKind,
        context: OperationContext,
        *,
        inputs: tuple[str, ...] = (),
        outputs: tuple[str, ...] = (),
    ) -> OperationRecord:
        return OperationRecord(
            operation_id,
            kind,
            context,
            (),
            inputs,
            outputs,
            "attempt-0",
            OperationStatus.SUCCESS,
            None,
            0,
            usage,
        )

    graph = OperationGraph(
        (
            ArtifactNode(
                source.source_id,
                ArtifactKind.SOURCE_OBSERVATION,
                "source-v1",
                projection.projection_digest,
                len(projection.messages[0].content),
                None,
                None,
                "snapshot.learn-v1",
            ),
            ArtifactNode(
                traces[0].fact_id,
                ArtifactKind.EXTRACTED_FACT,
                "fact-v1",
                text_digest(FACT_TEXT),
                len(FACT_TEXT),
                None,
                None,
                "snapshot.learn-v1",
            ),
        ),
        (
            operation(
                source.extraction_set_id,
                OperationKind.FACT_EXTRACTION,
                source_context,
                inputs=(source.source_id,),
                outputs=(traces[0].fact_id,),
            ),
            operation("op.proposal-v1", OperationKind.INTERNAL_OPERATION_DECISION, source_context),
            operation("op.mutation-v1", OperationKind.MUTATION, source_context),
            operation(join.opportunity_operation_id, OperationKind.FUTURE_QUERY, future_context),
            operation(join.use_operation_id, OperationKind.USE, future_context),
            operation(join.outcome_operation_id, OperationKind.DOWNSTREAM_OUTCOME, future_context),
        ),
        (MutationEdge(
            "mutation.persist-v1",
            "op.mutation-v1",
            ("op.proposal-v1",),
            InternalMemoryAction.ADD,
            "artifact.memory-v1",
            None,
            None,
            text_digest(FACT_TEXT),
            "receipt.persist-v1",
        ),),
    )
    delayed = DelayedEvidenceContent(
        "2026-08-20T00:00:00Z",
        "2026-08-21T00:00:00Z",
        current_input,
    )
    return projection, source_record, feedback, observation, graph, traces, delayed


def test_builder_exactly_joins_content_and_content_free_evidence() -> None:
    projection, source, feedback, observation, graph, facts, delayed = _fixture()
    examples = ExtractionOptimizerCorpusBuilder().build_examples(
        projection=projection,
        source_record=source,
        feedback_record=feedback,
        observation=observation,
        operation_graph=graph,
        fact_contents=facts,
        delayed_content=delayed,
    )

    assert len(examples) == 3
    assert sum(value.primary for value in examples) == 1
    assert {value.audit_join.feedback_example_id for value in examples} == {
        value.example_id for value in feedback.dataset.examples
    }
    assert examples[0].audit_join.source_record_digest == source.content_digest
    assert examples[0].audit_join.source_projection_id == projection.projection_id
    assert examples[0].extracted_facts[0].content.text == FACT_TEXT
    assert examples[0].audit_join.artifacts[0].mutation_ids == (
        "mutation.persist-v1",
    )
    fact_example = next(
        value for value in examples
        if value.level.value == "fact"
    )
    assert fact_example.feedback_fact_id == "fact.preference-v1"
    assert fact_example.feedback_semantic_key == TSV_KEY
    assert fact_example.feedback_artifact_ids == ("artifact.memory-v1",)
    replay = ExtractionOptimizerCorpusBuilder().build_examples(
        projection=projection,
        source_record=source,
        feedback_record=feedback,
        observation=observation,
        operation_graph=graph,
        fact_contents=facts,
        delayed_content=delayed,
    )
    first = ExtractionOptimizerCorpus.create(
        batch_id="batch.train-v1",
        attempt_id="attempt.train-v1",
        split=OptimizerCorpusSplit.TRAIN,
        observation_cutoff="2026-08-22T00:00:00Z",
        retention=OptimizerCorpusRetention.DELETE_AFTER_POLICY_DECISION,
        examples=examples,
    )
    second = ExtractionOptimizerCorpus.create(
        batch_id="batch.train-v1",
        attempt_id="attempt.train-v1",
        split=OptimizerCorpusSplit.TRAIN,
        observation_cutoff="2026-08-22T00:00:00Z",
        retention=OptimizerCorpusRetention.DELETE_AFTER_POLICY_DECISION,
        examples=tuple(reversed(replay)),
    )
    assert first == second
    assert audit_optimizer_corpus_isolation(first, {
        "source-record": source.payload(),
        "feedback-record": feedback.payload(),
        "operation-graph": {
            "artifacts": [value.to_payload() for value in graph.artifacts],
            "operations": [value.to_payload() for value in graph.operations],
            "mutations": [value.to_payload() for value in graph.mutations],
        },
    }) == ()
    assert audit_optimizer_corpus_isolation(first, {
        "manifest": {"debug": FACT_TEXT},
    }) == ("corpus_content_leak:manifest",)


@pytest.mark.parametrize("failure", ("source", "fact", "operation", "mutation"))
def test_builder_rejects_any_broken_exact_join(failure: str) -> None:
    projection, source, feedback, observation, graph, facts, delayed = _fixture()
    if failure == "source":
        projection = _projection("A different bounded source.")
    elif failure == "fact":
        facts = (replace(facts[0], content="Changed extracted fact."),)
    elif failure == "operation":
        graph = OperationGraph(graph.artifacts, graph.operations[:-1], graph.mutations)
    else:
        graph = OperationGraph(graph.artifacts, graph.operations, ())

    with pytest.raises(ValueError):
        ExtractionOptimizerCorpusBuilder().build_examples(
            projection=projection,
            source_record=source,
            feedback_record=feedback,
            observation=observation,
            operation_graph=graph,
            fact_contents=facts,
            delayed_content=delayed,
        )


def test_builder_rejects_forbidden_evaluation_content() -> None:
    projection, source, feedback, observation, graph, facts, delayed = _fixture()
    with pytest.raises(ValueError, match="forbidden evaluation evidence"):
        ExtractionOptimizerCorpusBuilder().build_examples(
            projection=projection,
            source_record=source,
            feedback_record=feedback,
            observation=replace(
                observation,
                final_response="The official grader answer key says this passed.",
            ),
            operation_graph=graph,
            fact_contents=facts,
            delayed_content=delayed,
        )
