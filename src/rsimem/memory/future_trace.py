"""Content-free future semantic retrieval, injection, use, and outcome evidence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Sequence
from .contracts import MemoryHit, MemoryKind, MemoryQuery
from .extraction_feedback import (
    ArtifactSemanticBinding,
    DeploymentObservation,
    ExposureMode,
    FeedbackContractRegistry,
    FutureMemoryEvidence,
    default_feedback_contract_registry,
)
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


class SemanticFeedbackContract(StrEnum):
    DISABLED = "disabled"
    SM01_TSV_V1 = "sm01_tsv_v1"
    SM02_BOUNDARY_V1 = "sm02_boundary_v1"
    SM03_FACT_CORRECTION_V1 = "sm03_fact_correction_v1"
    SM05_NORMALIZED_TSV_V1 = "sm05_normalized_tsv_v1"


_SEMANTIC_FEEDBACK_FAMILIES = {
    SemanticFeedbackContract.SM01_TSV_V1: "SM01_preference_adoption",
    SemanticFeedbackContract.SM02_BOUNDARY_V1: "SM02_constraint_retention",
    SemanticFeedbackContract.SM03_FACT_CORRECTION_V1: "SM03_fact_correction",
    SemanticFeedbackContract.SM05_NORMALIZED_TSV_V1: (
        "SM05_weak_trigger_preference_adoption"
    ),
}


@dataclass(frozen=True, slots=True)
class SemanticFeedbackResolution:
    used_artifact_ids: tuple[str, ...]
    outcome_status: OperationStatus
    outcome_reason_code: str | None
    reuse_signal_observed: bool
    eligible: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "outcome_status", OperationStatus(self.outcome_status))
        if type(self.reuse_signal_observed) is not bool or type(self.eligible) is not bool:
            raise TypeError("semantic feedback flags must be bool")


class SemanticFeedbackResolver:
    """Resolve pre-registered deployment signals without grader evidence."""

    def __init__(
        self,
        contract: SemanticFeedbackContract,
        *,
        family_id: str,
        stage: str,
        registry: FeedbackContractRegistry | None = None,
    ) -> None:
        self.contract = SemanticFeedbackContract(contract)
        self.family_id = family_id
        self.stage = stage
        self.registry = registry or default_feedback_contract_registry()
        expected_family = _SEMANTIC_FEEDBACK_FAMILIES.get(self.contract)
        if expected_family is not None and family_id != expected_family:
            raise ValueError(
                f"{self.contract.value} feedback contract requires family "
                f"{expected_family}"
            )

    def resolve(
        self,
        future: "SemanticFutureEvidence",
        observation: DeploymentObservation,
    ) -> SemanticFeedbackResolution:
        if self.contract == SemanticFeedbackContract.DISABLED:
            raise ValueError("disabled semantic feedback contract cannot resolve")
        if observation.family_id != self.family_id or observation.stage != self.stage:
            raise ValueError("semantic feedback observation identity mismatch")
        resolver = self.registry.resolver(self.family_id)
        eligible = self.stage in resolver.contract.opportunity.eligible_stages
        if not eligible:
            return SemanticFeedbackResolution(
                (),
                OperationStatus.NONE,
                "observation_censored",
                False,
                False,
            )
        # The benchmark contract is an audit-time allowlist, not an
        # opportunity generator.  Runtime observations must carry a visible
        # semantic requirement; otherwise a family/stage match alone would
        # manufacture demand and leak benchmark semantics into the process
        # signal.
        scope = tuple(
            key
            for key in observation.task_semantic_keys
            if key in resolver.contract.opportunity.memory_scope_keys
        )
        if not scope:
            return SemanticFeedbackResolution(
                (),
                OperationStatus.NONE,
                "opportunity_not_observed",
                False,
                True,
            )
        if len(scope) == 1:
            semantic_keys = tuple((scope[0],) for _ in future.memory_artifact_ids)
        elif len(scope) == len(future.memory_artifact_ids):
            semantic_keys = tuple((value,) for value in scope)
        else:
            semantic_keys = tuple(scope for _ in future.memory_artifact_ids)
        bindings = tuple(
            ArtifactSemanticBinding(artifact_id, keys)
            for artifact_id, keys in zip(
                future.memory_artifact_ids,
                semantic_keys,
            )
        )
        future_contract = FutureMemoryEvidence(
            future_opportunity_id=f"opportunity.{future.query_operation_id}",
            exposure_mode=(
                ExposureMode.EAGER_SYSTEM_PROMPT
                if future.injection_artifact_id is not None
                else ExposureMode.NOT_EXPOSED
            ),
            artifact_bindings=bindings,
            opportunity_operation_id=future.query_operation_id,
            injection_operation_id=(
                future.injection_operation_id
                if future.injection_artifact_id is not None
                else None
            ),
        )
        resolution = resolver.resolve(observation, future_contract)
        if not observation.observation_complete:
            status = OperationStatus.NONE
            reason = observation.censor_reason or "observation_censored"
        elif resolution.current_input_confounded:
            status = OperationStatus.NONE
            reason = "current_input_confounded"
        elif not resolution.opportunity_observed:
            status = OperationStatus.NONE
            reason = "opportunity_not_observed"
        elif not resolution.explicit_use:
            status = OperationStatus.NONE
            reason = (
                "injected_not_used"
                if future.injection_artifact_id is not None
                else "not_exposed"
            )
        elif not resolution.used_artifact_ids:
            status = OperationStatus.NONE
            reason = "use_not_bound_to_memory"
        elif resolution.successful_outcome is True:
            status = OperationStatus.SUCCESS
            reason = None
        elif resolution.harmful_outcome:
            status = OperationStatus.FAILED
            reason = "memory_use_harmfully_attributed"
        else:
            status = OperationStatus.NONE
            reason = "outcome_not_attributable"
        return SemanticFeedbackResolution(
            resolution.used_artifact_ids,
            status,
            reason,
            resolution.explicit_use,
            resolution.opportunity_observed and not resolution.current_input_confounded,
        )


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
    injected_artifact_ids: tuple[str, ...] = ()
    schema_version: int = SEMANTIC_FUTURE_TRACE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SEMANTIC_FUTURE_TRACE_SCHEMA_VERSION:
            raise ValueError("unsupported semantic future evidence schema version")
        if len(self.memory_artifact_ids) != len(self.memory_revisions):
            raise ValueError("future semantic artifacts and revisions must align")
        if len(self.memory_artifact_ids) != len(set(self.memory_artifact_ids)):
            raise ValueError("future semantic artifacts must be unique")
        if len(self.injected_artifact_ids) != len(set(self.injected_artifact_ids)):
            raise ValueError("future injected artifacts must be unique")
        if not set(self.injected_artifact_ids).issubset(set(self.memory_artifact_ids)):
            raise ValueError("injected artifacts must come from future retrieval")
        if self.injected_artifact_ids and self.injection_artifact_id is None:
            raise ValueError("injected artifacts require an injection operation")


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
        retrieved_hits: Sequence[MemoryHit] | None = None,
        retrieval_limit: int = 10_000,
    ) -> SemanticFutureEvidence:
        if not isinstance(model_visible_prompt, str):
            raise TypeError("future semantic trace prompt must be a string")
        if type(retrieval_limit) is not int or retrieval_limit < 1:
            raise ValueError("future semantic trace retrieval_limit must be positive")
        backend = registry.resolve(MemoryKind.SEMANTIC)
        query_artifact = self._artifact(
            ArtifactKind.QUERY,
            f"{step_id}.query",
            {"kind": "semantic", "namespace": namespace, "limit": retrieval_limit},
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

        if retrieved_hits is None:
            hits = tuple(backend.query(MemoryQuery(
                MemoryKind.SEMANTIC,
                "",
                namespace=namespace,
                limit=retrieval_limit,
            )))
        else:
            # Host adapters may already have performed the authoritative
            # retrieval.  Reusing those hits keeps the future trace tied to
            # the exact model-visible read and avoids a second query that can
            # diverge after a concurrent memory mutation.  Validate the
            # ownership/type boundary just as ``MemoryBackendRuntime.query``
            # does for backend-produced hits.
            hits = tuple(retrieved_hits)
            for hit in hits:
                if not isinstance(hit, MemoryHit):
                    raise TypeError("retrieved semantic hits must be MemoryHit values")
                if hit.artifact.kind is not MemoryKind.SEMANTIC:
                    raise ValueError("future semantic trace received a non-semantic hit")
                if hit.backend != backend.descriptor.name:
                    raise ValueError(
                        "future semantic trace received a hit owned by another backend"
                    )
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
            tuple(hit.artifact.artifact_id for hit in injected),
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
