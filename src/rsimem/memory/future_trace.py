"""Content-free future semantic retrieval, injection, use, and outcome evidence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from .contracts import MemoryKind, MemoryQuery
from .operation_graph import (
    ArtifactKind,
    ArtifactNode,
    AtomicOperationRecorder,
    OperationContext,
    OperationKind,
    OperationSpec,
    OperationStatus,
    build_artifact_id,
    build_operation_id,
)
from .runtime import MemoryBackendRegistry


SEMANTIC_FUTURE_TRACE_SCHEMA_VERSION = 1


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class SemanticFutureEvidence:
    query_operation_id: str
    retrieval_operation_id: str
    injection_operation_id: str
    memory_artifact_ids: tuple[str, ...]
    memory_revisions: tuple[str, ...]
    injection_artifact_id: str | None
    schema_version: int = SEMANTIC_FUTURE_TRACE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SEMANTIC_FUTURE_TRACE_SCHEMA_VERSION:
            raise ValueError("unsupported semantic future evidence schema version")
        if len(self.memory_artifact_ids) != len(self.memory_revisions):
            raise ValueError("future semantic artifacts and revisions must align")


@dataclass(frozen=True, slots=True)
class SemanticOutcomeEvidence:
    use_operation_id: str
    outcome_operation_id: str
    used_artifact_ids: tuple[str, ...]
    outcome_status: OperationStatus
    schema_version: int = SEMANTIC_FUTURE_TRACE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SEMANTIC_FUTURE_TRACE_SCHEMA_VERSION:
            raise ValueError("unsupported semantic outcome evidence schema version")
        object.__setattr__(self, "outcome_status", OperationStatus(self.outcome_status))


class SemanticFutureTraceRecorder:
    """Record future eager semantic exposure without retaining model-visible text."""

    def __init__(
        self,
        recorder: AtomicOperationRecorder,
        context: OperationContext,
    ) -> None:
        self.recorder = recorder
        self.context = context

    def _artifact(
        self,
        kind: ArtifactKind,
        logical_name: str,
        payload: object,
        *,
        provenance_ref: str,
    ) -> ArtifactNode:
        serialized = _canonical_json(payload)
        digest = _sha(serialized)
        return ArtifactNode(
            build_artifact_id(
                kind,
                self.context,
                logical_name=logical_name,
                content_digest=digest,
            ),
            kind,
            "semantic-future-evidence-v1",
            digest,
            len(serialized.encode("utf-8")),
            None,
            None,
            provenance_ref,
        )

    def _spec(
        self,
        kind: OperationKind,
        step_id: str,
        *,
        parents: tuple[str, ...] = (),
        inputs: tuple[str, ...] = (),
    ) -> OperationSpec:
        return OperationSpec(
            build_operation_id(
                kind,
                self.context,
                step_id=step_id,
                parent_operation_ids=parents,
                input_artifact_ids=inputs,
            ),
            kind,
            self.context,
            parents,
            inputs,
        )

    def record_prompt_injection(
        self,
        registry: MemoryBackendRegistry,
        model_visible_prompt: str,
        *,
        namespace: str,
        parent_operation_ids: tuple[str, ...],
        step_id: str = "future-semantic",
    ) -> SemanticFutureEvidence:
        if not model_visible_prompt:
            raise ValueError("future semantic trace requires a model-visible prompt")
        backend = registry.resolve(MemoryKind.SEMANTIC)
        query_artifact = self._artifact(
            ArtifactKind.QUERY,
            f"{step_id}.query",
            {"kind": "semantic", "namespace": namespace, "limit": 10_000},
            provenance_ref=backend.descriptor.name,
        )
        self.recorder.record_artifact(query_artifact)
        query_spec = self._spec(
            OperationKind.FUTURE_QUERY,
            f"{step_id}.query",
            parents=parent_operation_ids,
            inputs=(query_artifact.artifact_id,),
        )
        with self.recorder.operation_scope(query_spec) as operation:
            operation.complete()

        hits = tuple(backend.query(MemoryQuery(
            MemoryKind.SEMANTIC,
            "",
            namespace=namespace,
            limit=10_000,
        )))
        memory_ids = []
        revisions = []
        for hit in hits:
            artifact = hit.artifact
            if artifact.revision is None:
                raise ValueError("future semantic artifact requires a revision")
            node = ArtifactNode(
                artifact.artifact_id,
                ArtifactKind.MEMORY_ARTIFACT,
                "hermes-semantic-artifact-v1",
                _sha(artifact.content),
                len(artifact.content.encode("utf-8")),
                None,
                artifact.revision,
                backend.descriptor.name,
            )
            self.recorder.record_artifact(node)
            memory_ids.append(node.artifact_id)
            revisions.append(artifact.revision)

        retrieval_payload = {
            "artifact_ids": memory_ids,
            "revisions": revisions,
            "count": len(memory_ids),
        }
        retrieval_artifact = self._artifact(
            ArtifactKind.RETRIEVAL_RESULT,
            f"{step_id}.retrieval",
            retrieval_payload,
            provenance_ref=backend.descriptor.name,
        )
        self.recorder.record_artifact(retrieval_artifact)
        retrieval_spec = self._spec(
            OperationKind.RETRIEVAL,
            f"{step_id}.retrieval",
            parents=(query_spec.operation_id,),
            inputs=(query_artifact.artifact_id, *memory_ids),
        )
        with self.recorder.operation_scope(retrieval_spec) as operation:
            operation.complete(
                output_artifact_ids=(retrieval_artifact.artifact_id,),
                status=OperationStatus.SUCCESS if hits else OperationStatus.NONE,
                reason_code=None if hits else "retrieval_miss",
            )

        injected = tuple(
            hit for hit in hits if hit.artifact.content in model_visible_prompt
        )
        injection_artifact = None
        if injected:
            injection_artifact = self._artifact(
                ArtifactKind.INJECTION,
                f"{step_id}.injection",
                {
                    "artifact_ids": [hit.artifact.artifact_id for hit in injected],
                    "revisions": [hit.artifact.revision for hit in injected],
                    "surface": "system_prompt",
                },
                provenance_ref=backend.descriptor.name,
            )
            self.recorder.record_artifact(injection_artifact)
        injection_spec = self._spec(
            OperationKind.INJECTION,
            f"{step_id}.injection",
            parents=(retrieval_spec.operation_id,),
            inputs=(retrieval_artifact.artifact_id, *memory_ids),
        )
        with self.recorder.operation_scope(injection_spec) as operation:
            operation.complete(
                output_artifact_ids=(
                    (injection_artifact.artifact_id,)
                    if injection_artifact is not None
                    else ()
                ),
                status=(
                    OperationStatus.SUCCESS if injected else OperationStatus.NONE
                ),
                reason_code=(
                    None
                    if injected
                    else "retrieved_not_injected"
                    if hits
                    else "retrieval_miss"
                ),
            )
        return SemanticFutureEvidence(
            query_spec.operation_id,
            retrieval_spec.operation_id,
            injection_spec.operation_id,
            tuple(memory_ids),
            tuple(revisions),
            injection_artifact.artifact_id if injection_artifact is not None else None,
        )

    def record_use_and_outcome(
        self,
        future: SemanticFutureEvidence,
        *,
        used_artifact_ids: tuple[str, ...],
        outcome_status: OperationStatus,
        outcome_reason_code: str | None = None,
        step_id: str = "future-semantic",
    ) -> SemanticOutcomeEvidence:
        if not set(used_artifact_ids).issubset(future.memory_artifact_ids):
            raise ValueError("used semantic artifacts must come from future retrieval")
        use_artifact = None
        if used_artifact_ids:
            use_artifact = self._artifact(
                ArtifactKind.USE_EVIDENCE,
                f"{step_id}.use",
                {"artifact_ids": used_artifact_ids, "used": True},
                provenance_ref=future.injection_operation_id,
            )
            self.recorder.record_artifact(use_artifact)
        use_inputs = tuple(dict.fromkeys((
            *future.memory_artifact_ids,
            *((future.injection_artifact_id,) if future.injection_artifact_id else ()),
        )))
        use_spec = self._spec(
            OperationKind.USE,
            f"{step_id}.use",
            parents=(future.injection_operation_id,),
            inputs=use_inputs,
        )
        with self.recorder.operation_scope(use_spec) as operation:
            operation.complete(
                output_artifact_ids=(
                    (use_artifact.artifact_id,) if use_artifact is not None else ()
                ),
                status=(
                    OperationStatus.SUCCESS
                    if used_artifact_ids
                    else OperationStatus.NONE
                ),
                reason_code=(
                    None
                    if used_artifact_ids
                    else "retrieved_but_unused"
                    if future.injection_artifact_id is not None
                    else "not_exposed"
                ),
            )

        normalized_status = OperationStatus(outcome_status)
        if normalized_status != OperationStatus.SUCCESS and outcome_reason_code is None:
            raise ValueError("non-success future outcome requires a reason code")
        outcome_artifact = self._artifact(
            ArtifactKind.OUTCOME,
            f"{step_id}.outcome",
            {
                "status": normalized_status.value,
                "reason_code": outcome_reason_code,
            },
            provenance_ref=use_spec.operation_id,
        )
        self.recorder.record_artifact(outcome_artifact)
        outcome_spec = self._spec(
            OperationKind.DOWNSTREAM_OUTCOME,
            f"{step_id}.outcome",
            parents=(use_spec.operation_id,),
            inputs=(
                (use_artifact.artifact_id,)
                if use_artifact is not None
                else future.memory_artifact_ids
            ),
        )
        with self.recorder.operation_scope(outcome_spec) as operation:
            operation.complete(
                output_artifact_ids=(outcome_artifact.artifact_id,),
                status=normalized_status,
                reason_code=outcome_reason_code,
            )
        return SemanticOutcomeEvidence(
            use_spec.operation_id,
            outcome_spec.operation_id,
            used_artifact_ids,
            normalized_status,
        )
