"""Mem0-flat semantic construction over fixed Hermes semantic storage."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ...lifecycle import MemoryScope, RawResourceUsage, TemporalValidity
from ...memory.contracts import MemoryKind, MemoryQuery
from ...memory.ingestion import (
    ExistingMemoryCandidate,
    ExistingMemoryCandidateReader,
    InternalMemoryAction,
    InternalOperationProposal,
    InvalidPolicyOutputError,
    MemoryIngestResult,
    MemoryIngestStatus,
    PolicyCapability,
    SemanticIngestRequest,
    SemanticMemoryPolicy,
    SemanticPolicyDecision,
    SemanticPolicyDescriptor,
)
from ...memory.runtime import MemoryBackendRegistry
from ...memory.utility import (
    MEM0_CONSOLIDATION_UPDATE_PARAMETER_ID,
    MEM0_UTILITY_PARAMETER_IDS,
    UtilityTarget,
)
from ...memory.operation_graph import (
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
from ...memory.prompt_components import (
    PromptBindingFingerprint,
    SemanticPolicyManifest,
)
from ...memory.validation import (
    SemanticMemoryCategory,
    TargetOwnershipResolver,
    UntrustedMemoryCandidate,
    ValidationProvenance,
)
from .prompts import (
    CompletionClient,
    POLICY_FACT_EXTRACTION_PROMPT,
    POLICY_INTERNAL_OPERATION_PROMPT,
    PromptTemplate,
    RenderedPrompt,
)
from .utility_gate import FrozenMem0UtilityGate


MEM0_FLAT_POLICY_SCHEMA_VERSION = 1
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_TOKEN = re.compile(r"[\w]+", re.UNICODE)
_TEMPORARY = re.compile(
    r"(?i)\b(?:temporary|temporarily|today only|for this task|for now|right now|currently)\b"
)
_TRANSCRIPT = re.compile(r"(?im)^\s*(?:user|assistant|system|tool)\s*:")
_TOOL_NOISE = re.compile(r'(?i)"(?:tool_calls?|tool_call_id|arguments)"\s*:')


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _stable_id(prefix: str, value: object) -> str:
    return f"{prefix}.{_sha(_canonical_json(value))[:40]}"


def _sum_optional(values: Sequence[int | None]) -> int | None:
    return (
        None
        if not values or any(value is None for value in values)
        else sum(value or 0 for value in values)
    )


def _combine_usage(*values: RawResourceUsage) -> RawResourceUsage:
    model_values = tuple(value for value in values if value.model_requests > 0)
    return RawResourceUsage(
        input_tokens=_sum_optional([value.input_tokens for value in model_values]),
        output_tokens=_sum_optional([value.output_tokens for value in model_values]),
        cache_read_tokens=_sum_optional([value.cache_read_tokens for value in model_values]),
        cache_write_tokens=_sum_optional([value.cache_write_tokens for value in model_values]),
        reasoning_tokens=_sum_optional([value.reasoning_tokens for value in model_values]),
        model_requests=sum(value.model_requests for value in values),
        retry_count=sum(value.retry_count for value in values),
        duration_ms=_sum_optional([value.duration_ms for value in values]),
        storage_bytes=sum(value.storage_bytes for value in values),
    )


@dataclass(frozen=True, slots=True)
class FlatRetrievalConfig:
    embedding_model: str = "rsimem-token-hash-cosine-v1"
    dimensions: int = 256
    top_k: int = 5
    threshold: float = 0.12
    namespaces: tuple[str, ...] = ("memory", "user")
    rebuild_semantics: str = "snapshot-rebuild-per-ingest-v1"
    version: str = "flat-retrieval-v1"

    def __post_init__(self) -> None:
        if not self.embedding_model.strip() or not self.rebuild_semantics.strip() or not self.version.strip():
            raise ValueError("flat retrieval identity is incomplete")
        if self.dimensions < 8 or self.top_k < 1:
            raise ValueError("flat retrieval dimensions and top_k must be positive")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("flat retrieval threshold must be between zero and one")
        if not self.namespaces or len(self.namespaces) != len(set(self.namespaces)):
            raise ValueError("flat retrieval namespaces must be unique")
        if any(namespace not in {"memory", "user"} for namespace in self.namespaces):
            raise ValueError("flat retrieval supports only Hermes semantic namespaces")

    @property
    def digest(self) -> str:
        return _sha(_canonical_json(self.manifest_record()))

    def manifest_record(self) -> dict[str, object]:
        return {
            "embedding_model": self.embedding_model,
            "dimensions": self.dimensions,
            "top_k": self.top_k,
            "threshold": self.threshold,
            "namespaces": list(self.namespaces),
            "rebuild_semantics": self.rebuild_semantics,
            "version": self.version,
        }


@dataclass(frozen=True, slots=True)
class RelatedMemoryView:
    candidate: ExistingMemoryCandidate
    content: str
    namespace: str
    score: float
    mutable: bool
    index_revision: str

    def prompt_record(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate.candidate_id,
            "content": self.content,
            "namespace": self.namespace,
            "score": round(self.score, 8),
            "mutable": self.mutable,
        }


class FlatSemanticCandidateReader(ExistingMemoryCandidateReader):
    """Controlled add-time flat index; it never changes Hermes injection."""

    def __init__(
        self,
        registry: MemoryBackendRegistry,
        *,
        ownership: TargetOwnershipResolver | None = None,
        config: FlatRetrievalConfig = FlatRetrievalConfig(),
    ) -> None:
        self.registry = registry
        self.ownership = ownership
        self.config = config
        self._request_key: str | None = None
        self._views: dict[str, RelatedMemoryView] = {}
        self._exposed: dict[str, ExistingMemoryCandidate] = {}
        self._usage = RawResourceUsage(duration_ms=0)
        self._index_revision = _sha("unbuilt")

    @property
    def usage(self) -> RawResourceUsage:
        return self._usage

    @property
    def index_revision(self) -> str:
        return self._index_revision

    def _ensure(self, request: SemanticIngestRequest) -> None:
        if self._request_key == request.idempotency_key:
            return
        started = time.perf_counter_ns()
        backend = self.registry.resolve(MemoryKind.SEMANTIC)
        artifacts = []
        for namespace in self.config.namespaces:
            hits = backend.query(MemoryQuery(
                MemoryKind.SEMANTIC,
                "",
                namespace=namespace,
                limit=10_000,
            ))
            artifacts.extend(hit.artifact for hit in hits)
        artifacts.sort(key=lambda artifact: artifact.artifact_id)
        index_identity = {
            "config_digest": self.config.digest,
            "backend": backend.descriptor.name,
            "artifacts": [
                {
                    "artifact_id": artifact.artifact_id,
                    "revision": artifact.revision,
                    "content_digest": _sha(artifact.content),
                    "namespace": artifact.namespace,
                }
                for artifact in artifacts
            ],
        }
        self._index_revision = _sha(_canonical_json(index_identity))
        self._views = {}
        self._exposed = {}
        for artifact in artifacts:
            if not isinstance(artifact.revision, str) or not artifact.revision:
                continue
            candidate_id = _stable_id("candidate", {
                "backend": backend.descriptor.name,
                "artifact_id": artifact.artifact_id,
                "revision": artifact.revision,
                "index_revision": self._index_revision,
                "retrieval_version": self.config.version,
            })
            candidate = ExistingMemoryCandidate(
                candidate_id,
                artifact.artifact_id,
                artifact.revision,
                _sha(artifact.content),
            )
            binding = None
            if self.ownership is not None:
                binding = self.ownership.resolve(
                    backend.descriptor.name,
                    artifact.artifact_id,
                )
            mutable = bool(
                binding is not None
                and binding.revision == artifact.revision
                and binding.content_digest == candidate.content_digest
                and binding.namespace == artifact.namespace
                and binding.kind == artifact.kind
                and binding.owner_run_id == request.provenance.source.run_id
            )
            self._views[candidate_id] = RelatedMemoryView(
                candidate,
                artifact.content,
                artifact.namespace,
                0.0,
                mutable,
                self._index_revision,
            )
        duration = max(0, (time.perf_counter_ns() - started) // 1_000_000)
        self._usage = RawResourceUsage(duration_ms=duration)
        self._request_key = request.idempotency_key

    def search(
        self,
        request: SemanticIngestRequest,
        text: str,
    ) -> tuple[RelatedMemoryView, ...]:
        self._ensure(request)
        query_vector = _embed(text, self.config.dimensions)
        ranked = []
        for view in self._views.values():
            score = _cosine(query_vector, _embed(view.content, self.config.dimensions))
            if score >= self.config.threshold:
                ranked.append((score, view.candidate.candidate_id, view))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        selected = tuple(
            RelatedMemoryView(
                view.candidate,
                view.content,
                view.namespace,
                score,
                view.mutable,
                view.index_revision,
            )
            for score, _, view in ranked[: self.config.top_k]
        )
        for view in selected:
            self._exposed[view.candidate.candidate_id] = view.candidate
        return selected

    def candidates(self, request: SemanticIngestRequest) -> Sequence[ExistingMemoryCandidate]:
        self._ensure(request)
        return tuple(self._exposed[key] for key in sorted(self._exposed))

    def resolve(self, candidate_id: str) -> ExistingMemoryCandidate | None:
        return self._exposed.get(candidate_id)


def _embed(value: str, dimensions: int) -> tuple[float, ...]:
    vector = [0.0] * dimensions
    for token in _TOKEN.findall(value.casefold()):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(item * item for item in vector))
    if norm:
        vector = [item / norm for item in vector]
    return tuple(vector)


def _cosine(first: Sequence[float], second: Sequence[float]) -> float:
    return max(0.0, min(1.0, sum(left * right for left, right in zip(first, second))))


@dataclass(frozen=True, slots=True)
class ExtractedSemanticFact:
    fact_id: str
    content: str
    content_digest: str
    category: SemanticMemoryCategory
    namespace: str
    scope: MemoryScope
    temporal_validity: TemporalValidity


@dataclass(frozen=True, slots=True)
class FactExtractionTrace:
    fact_id: str
    content_digest: str
    accepted: bool
    reason_code: str | None

    def __post_init__(self) -> None:
        if not self.fact_id.strip() or not _DIGEST.fullmatch(self.content_digest):
            raise ValueError("fact extraction trace identity is invalid")
        if type(self.accepted) is not bool:
            raise TypeError("fact extraction accepted flag must be bool")
        if self.accepted == (self.reason_code is not None):
            raise ValueError("fact extraction reason must describe only rejected facts")


@dataclass(frozen=True, slots=True)
class ExtractionInvocationFingerprint:
    render_id: str
    render_input_digest: str
    rendered_template_digest: str
    model_output_digest: str
    binding_id: str | None

    @classmethod
    def create(
        cls,
        rendered: RenderedPrompt,
        output_text: str,
    ) -> "ExtractionInvocationFingerprint":
        return cls(
            rendered.render_id,
            rendered.input_digest,
            rendered.artifact.template_digest,
            _sha(output_text),
            rendered.binding_fingerprint,
        )

    def __post_init__(self) -> None:
        if not self.render_id.strip():
            raise ValueError("extraction render identity is incomplete")
        if any(not _DIGEST.fullmatch(value) for value in (
            self.render_input_digest,
            self.rendered_template_digest,
            self.model_output_digest,
        )):
            raise ValueError("extraction invocation digest is invalid")
        if self.binding_id is not None and not self.binding_id.strip():
            raise ValueError("extraction invocation binding is invalid")

    def payload(self) -> dict[str, object]:
        return {
            "render_id": self.render_id,
            "render_input_digest": self.render_input_digest,
            "rendered_template_digest": self.rendered_template_digest,
            "model_output_digest": self.model_output_digest,
            "binding_id": self.binding_id,
        }


@dataclass(frozen=True, slots=True)
class Mem0FlatOperationTrace:
    context: OperationContext
    source_operation_id: str
    source_artifact_id: str
    extraction_operation_id: str
    fact_artifact_ids: tuple[str, ...]
    fact_extractions: tuple[FactExtractionTrace, ...]
    related_operation_ids: tuple[str, ...]
    related_artifact_ids: tuple[str, ...]
    decision_operation_id: str | None
    proposal_artifact_ids: tuple[str, ...]


class _RejectedDecision(Exception):
    def __init__(self, reason_code: str) -> None:
        self.reason_code = reason_code
        super().__init__(reason_code)


class Mem0FlatSemanticPolicy(SemanticMemoryPolicy):
    """Two-prompt Mem0-flat policy with deterministic trusted projections."""

    def __init__(
        self,
        completion_client: CompletionClient,
        *,
        retrieval: FlatRetrievalConfig = FlatRetrievalConfig(),
        fact_prompt: PromptTemplate = POLICY_FACT_EXTRACTION_PROMPT,
        operation_prompt: PromptTemplate = POLICY_INTERNAL_OPERATION_PROMPT,
        policy_version: str | None = None,
        descriptor_policy_version: str | None = None,
        framework_version: str = "mem0-flat-framework-v1",
        feature_schema_version: str = "semantic-fact-features-v1",
        operation_recorder: AtomicOperationRecorder | None = None,
        utility_gate: FrozenMem0UtilityGate | None = None,
        extraction_binding: PromptBindingFingerprint | None = None,
    ) -> None:
        if fact_prompt.artifact.model_profile != operation_prompt.artifact.model_profile:
            raise ValueError("Mem0 prompt model profiles must match")
        if extraction_binding is not None and (
            fact_prompt.binding_fingerprint != extraction_binding.binding_id
            or fact_prompt.artifact.template_digest
            != extraction_binding.rendered_template_digest
        ):
            raise ValueError("Mem0 extraction binding does not match fact prompt")
        self.completion_client = completion_client
        self.retrieval = retrieval
        self.fact_prompt = fact_prompt
        self.operation_prompt = operation_prompt
        self.operation_recorder = operation_recorder
        self.utility_gate = utility_gate
        self.extraction_binding = extraction_binding
        extraction_component_id = (
            extraction_binding.artifact_id
            if extraction_binding is not None
            else f"prompt-component.extraction.{fact_prompt.artifact.template_digest[:24]}"
        )
        extraction_component_digest = (
            extraction_binding.artifact_body_digest
            if extraction_binding is not None
            else fact_prompt.artifact.template_digest
        )
        self.semantic_manifest = SemanticPolicyManifest.create(
            route="hermes-native-semantic",
            boundary="task-completed-v1",
            backend="hermes-native-semantic",
            framework_version=framework_version,
            model_profile=fact_prompt.artifact.model_profile,
            extraction_component_id=extraction_component_id,
            extraction_component_digest=extraction_component_digest,
            update_component_id=(
                "prompt-component.update."
                f"{operation_prompt.artifact.template_digest[:24]}"
            ),
            update_component_digest=operation_prompt.artifact.template_digest,
            retrieval_component_id=(
                f"retrieval-config.{retrieval.version}.{retrieval.digest[:16]}"
            ),
            retrieval_component_digest=retrieval.digest,
        )
        bound_policy_version = self.semantic_manifest.composite_policy_version
        if policy_version is not None:
            bound_policy_version = (
                f"{bound_policy_version}.variant.{_sha(policy_version)[:16]}"
            )
        if utility_gate is not None:
            bound_policy_version = (
                f"{bound_policy_version}.utility.{utility_gate.digest[:16]}"
            )
        if descriptor_policy_version is not None:
            bound_policy_version = descriptor_policy_version
        self._descriptor = SemanticPolicyDescriptor(
            provider="mem0_flat",
            policy_version=bound_policy_version,
            framework_version=framework_version,
            prompt_version=(
                f"semantic-components.{self.semantic_manifest.composite_digest[:24]}"
            ),
            feature_schema_version=(
                utility_gate.feature_schema
                if utility_gate is not None
                else feature_schema_version
            ),
            capability=PolicyCapability(
                frozenset(InternalMemoryAction),
                add_time_update=True,
            ),
        )
        self._facts: dict[str, ExtractedSemanticFact] = {}
        self._target_namespaces: dict[str, str] = {}
        self._operation_traces: dict[str, Mem0FlatOperationTrace] = {}
        self._extraction_invocations: dict[
            str, ExtractionInvocationFingerprint
        ] = {}

    @property
    def descriptor(self) -> SemanticPolicyDescriptor:
        return self._descriptor

    def fact_for_digest(self, content_digest: str) -> ExtractedSemanticFact | None:
        return self._facts.get(content_digest)

    def namespace_for_target(self, artifact_id: str | None) -> str | None:
        return self._target_namespaces.get(artifact_id or "")

    def operation_trace(self, request_id: str) -> Mem0FlatOperationTrace | None:
        return self._operation_traces.get(request_id)

    def extraction_invocation(
        self,
        request_id: str,
    ) -> ExtractionInvocationFingerprint | None:
        return self._extraction_invocations.get(request_id)

    def _operation_context(self, request: SemanticIngestRequest) -> OperationContext:
        source = request.provenance.source
        return OperationContext(
            source.run_id,
            source.episode_id,
            source.session_id,
            source.task_id,
            self.descriptor.policy_version,
            self.descriptor.prompt_version,
            self.descriptor.framework_version,
        )

    @staticmethod
    def _artifact_node(
        context: OperationContext,
        kind: ArtifactKind,
        logical_name: str,
        content_digest: str,
        *,
        byte_size: int,
        revision: str | None,
        provenance_ref: str,
        artifact_id: str | None = None,
    ) -> ArtifactNode:
        return ArtifactNode(
            artifact_id=(
                artifact_id
                or build_artifact_id(
                    kind,
                    context,
                    logical_name=logical_name,
                    content_digest=content_digest,
                )
            ),
            kind=kind,
            artifact_schema_version=(
                "hermes-semantic-artifact-v1"
                if kind == ArtifactKind.MEMORY_ARTIFACT
                else "semantic-operation-artifact-v1"
            ),
            content_digest=content_digest,
            byte_size=byte_size,
            token_size=None,
            revision=revision,
            provenance_ref=provenance_ref,
        )

    @staticmethod
    def _operation_spec(
        context: OperationContext,
        kind: OperationKind,
        step_id: str,
        *,
        parents: tuple[str, ...] = (),
        inputs: tuple[str, ...] = (),
    ) -> OperationSpec:
        return OperationSpec(
            build_operation_id(
                kind,
                context,
                step_id=step_id,
                parent_operation_ids=parents,
                input_artifact_ids=inputs,
            ),
            kind,
            context,
            parents,
            inputs,
        )

    def _record_parameter_artifact(
        self,
        context: OperationContext,
        *,
        logical_name: str,
        content_digest: str,
        revision: str,
        provenance_ref: str,
        artifact_id: str | None = None,
    ) -> str:
        assert self.operation_recorder is not None
        artifact = self._artifact_node(
            context,
            ArtifactKind.POLICY_PARAMETER,
            logical_name,
            content_digest,
            byte_size=0,
            revision=revision,
            provenance_ref=provenance_ref,
            artifact_id=artifact_id,
        )
        self.operation_recorder.record_artifact(artifact)
        return artifact.artifact_id

    def _record_utility_parameter(
        self,
        context: OperationContext,
        target: UtilityTarget,
        *,
        update: bool = False,
    ) -> str | None:
        if self.operation_recorder is None or self.utility_gate is None:
            return None
        policy = (
            self.utility_gate.update_policy
            if update and self.utility_gate.update_policy is not None
            else self.utility_gate.policy_for(target)
        )
        parameter_id = (
            MEM0_CONSOLIDATION_UPDATE_PARAMETER_ID
            if update
            else MEM0_UTILITY_PARAMETER_IDS[target]
        )
        return self._record_parameter_artifact(
            context,
            logical_name=parameter_id,
            content_digest=policy.digest,
            revision=policy.policy_version,
            provenance_ref=parameter_id,
            artifact_id=parameter_id,
        )

    def _begin_trace(
        self,
        request: SemanticIngestRequest,
    ) -> tuple[OperationContext, str, str, OperationSpec] | None:
        if self.operation_recorder is None:
            return None
        context = self._operation_context(request)
        source_payload = _canonical_json(
            request.source_projection.identity_payload()
        )
        source_digest = request.source_projection.projection_digest
        source = self._artifact_node(
            context,
            ArtifactKind.SOURCE_OBSERVATION,
            f"{request.idempotency_key}.source",
            source_digest,
            byte_size=len(source_payload.encode("utf-8")),
            revision=request.provenance.base_revision,
            provenance_ref=request.provenance.source.snapshot_id,
        )
        self.operation_recorder.record_artifact(source)
        source_spec = self._operation_spec(
            context,
            OperationKind.SOURCE_OBSERVATION,
            f"{request.idempotency_key}.source",
        )
        with self.operation_recorder.operation_scope(source_spec) as operation:
            operation.complete(output_artifact_ids=(source.artifact_id,))
        prompt_parameter = self._record_parameter_artifact(
            context,
            logical_name=self.semantic_manifest.extraction_component_id,
            content_digest=self.semantic_manifest.extraction_component_digest,
            revision=self.fact_prompt.artifact.version,
            provenance_ref=(
                self.extraction_binding.binding_id
                if self.extraction_binding is not None
                else self.fact_prompt.artifact.prompt_id
            ),
            artifact_id=self.semantic_manifest.extraction_component_id,
        )
        generation_parameter = self._record_utility_parameter(
            context,
            UtilityTarget.GENERATION,
        )
        extraction_spec = self._operation_spec(
            context,
            OperationKind.FACT_EXTRACTION,
            f"{request.idempotency_key}.extraction",
            parents=(source_spec.operation_id,),
            inputs=tuple(filter(None, (
                source.artifact_id,
                prompt_parameter,
                generation_parameter,
            ))),
        )
        return context, source_spec.operation_id, source.artifact_id, extraction_spec

    def _store_trace(
        self,
        request: SemanticIngestRequest,
        trace_state: tuple[OperationContext, str, str, OperationSpec] | None,
        *,
        fact_artifact_ids: tuple[str, ...] = (),
        fact_extractions: tuple[FactExtractionTrace, ...] = (),
        related_operation_ids: tuple[str, ...] = (),
        related_artifact_ids: tuple[str, ...] = (),
        decision_operation_id: str | None = None,
        proposal_artifact_ids: tuple[str, ...] = (),
    ) -> None:
        if trace_state is None:
            return
        context, source_operation_id, source_artifact_id, extraction_spec = trace_state
        self._operation_traces[request.idempotency_key] = Mem0FlatOperationTrace(
            context,
            source_operation_id,
            source_artifact_id,
            extraction_spec.operation_id,
            fact_artifact_ids,
            fact_extractions,
            related_operation_ids,
            related_artifact_ids,
            decision_operation_id,
            proposal_artifact_ids,
        )

    def ingest(
        self,
        request: SemanticIngestRequest,
        candidates: ExistingMemoryCandidateReader,
    ) -> SemanticPolicyDecision:
        if not isinstance(candidates, FlatSemanticCandidateReader):
            return SemanticPolicyDecision(
                MemoryIngestStatus.REJECTED,
                (),
                reason_codes=("invalid_candidate_reader",),
            )
        if self.utility_gate is not None:
            self.utility_gate.begin_request(request.idempotency_key)
        trace_state = self._begin_trace(request)
        self._store_trace(request, trace_state)
        extraction_spec = trace_state[3] if trace_state is not None else None
        if request.exit_evidence.unresolved_state is not None:
            if extraction_spec is not None:
                assert self.operation_recorder is not None
                with self.operation_recorder.operation_scope(extraction_spec) as operation:
                    operation.complete(
                        status=OperationStatus.REJECTED,
                        reason_code="unresolved_source",
                    )
            return SemanticPolicyDecision(
                MemoryIngestStatus.REJECTED,
                (),
                reason_codes=("unresolved_source",),
            )
        if request.validity != TemporalValidity.DURABLE:
            if extraction_spec is not None:
                assert self.operation_recorder is not None
                with self.operation_recorder.operation_scope(extraction_spec) as operation:
                    operation.complete(
                        status=OperationStatus.REJECTED,
                        reason_code="non_durable_source",
                    )
            return SemanticPolicyDecision(
                MemoryIngestStatus.REJECTED,
                (),
                reason_codes=("non_durable_source",),
            )

        extraction_scope = (
            self.operation_recorder.operation_scope(extraction_spec)
            if self.operation_recorder is not None and extraction_spec is not None
            else nullcontext(None)
        )
        with extraction_scope as extraction_operation:
            extraction = self.fact_prompt.render({
                "source_messages": request.source_projection.prompt_messages(),
                "source_projection_digest": (
                    request.source_projection.projection_digest
                ),
                "exit_evidence": request.exit_evidence.compiler_input_payload(),
            })
            extraction_result = self.completion_client.complete(extraction)
            invocation = ExtractionInvocationFingerprint.create(
                extraction,
                extraction_result.output_text,
            )
            existing_invocation = self._extraction_invocations.get(
                request.idempotency_key
            )
            if existing_invocation is not None and existing_invocation != invocation:
                raise ValueError("extraction invocation identity conflict")
            self._extraction_invocations[request.idempotency_key] = invocation
            facts, fact_extractions = self._parse_facts(
                extraction_result.output_text,
                request,
            )
            filtered_count = sum(not value.accepted for value in fact_extractions)
            fact_artifacts = ()
            if self.operation_recorder is not None and trace_state is not None:
                context = trace_state[0]
                values = []
                for fact in facts:
                    artifact = self._artifact_node(
                        context,
                        ArtifactKind.EXTRACTED_FACT,
                        fact.fact_id,
                        fact.content_digest,
                        byte_size=len(fact.content.encode("utf-8")),
                        revision=self.descriptor.feature_schema_version,
                        provenance_ref=request.provenance.source.snapshot_id,
                        artifact_id=fact.fact_id,
                    )
                    self.operation_recorder.record_artifact(artifact)
                    values.append(artifact.artifact_id)
                fact_artifacts = tuple(values)
                assert extraction_operation is not None
                extraction_operation.complete(
                    output_artifact_ids=fact_artifacts,
                    status=(
                        OperationStatus.SUCCESS
                        if facts
                        else OperationStatus.REJECTED
                        if filtered_count
                        else OperationStatus.NONE
                    ),
                    reason_code=(
                        None
                        if facts
                        else "non_durable_fact"
                        if filtered_count
                        else "no_durable_fact"
                    ),
                    usage=extraction_result.usage,
                )
        self._store_trace(
            request,
            trace_state,
            fact_artifact_ids=fact_artifacts,
            fact_extractions=fact_extractions,
        )
        if not facts:
            if filtered_count:
                return SemanticPolicyDecision(
                    MemoryIngestStatus.REJECTED,
                    (),
                    usage=extraction_result.usage,
                    reason_codes=("non_durable_fact",),
                )
            return SemanticPolicyDecision(
                MemoryIngestStatus.SUCCESS,
                (InternalOperationProposal(
                    InternalMemoryAction.NONE,
                    "no_durable_fact",
                ),),
                usage=extraction_result.usage,
            )

        generation_utility = (
            self.utility_gate.generation_decisions(request, len(facts))
            if self.utility_gate is not None
            else ()
        )

        related_by_fact: dict[int, tuple[RelatedMemoryView, ...]] = {}
        flattened: dict[str, RelatedMemoryView] = {}
        related_operation_ids: list[str] = []
        related_artifact_ids: list[str] = []
        retrieval_parameter = None
        retrieval_utility_parameter = None
        if self.operation_recorder is not None and trace_state is not None:
            retrieval_parameter = self._record_parameter_artifact(
                trace_state[0],
                logical_name=self.semantic_manifest.retrieval_component_id,
                content_digest=self.semantic_manifest.retrieval_component_digest,
                revision=self.retrieval.version,
                provenance_ref=self.retrieval.version,
                artifact_id=self.semantic_manifest.retrieval_component_id,
            )
            retrieval_utility_parameter = self._record_utility_parameter(
                trace_state[0],
                UtilityTarget.RETRIEVAL,
            )
        for index, fact in enumerate(facts):
            related_spec = None
            if trace_state is not None and retrieval_parameter is not None:
                related_spec = self._operation_spec(
                    trace_state[0],
                    OperationKind.RELATED_MEMORY_RETRIEVAL,
                    f"{request.idempotency_key}.related.{index}",
                    parents=(trace_state[3].operation_id,),
                    inputs=tuple(filter(None, (
                        fact_artifacts[index],
                        retrieval_parameter,
                        retrieval_utility_parameter,
                    ))),
                )
            related_scope = (
                self.operation_recorder.operation_scope(related_spec)
                if self.operation_recorder is not None and related_spec is not None
                else nullcontext(None)
            )
            with related_scope as related_operation:
                views = candidates.search(request, fact.content)
                if self.utility_gate is not None:
                    views = self.utility_gate.rank_related(request, views)
                output_ids = []
                if self.operation_recorder is not None:
                    for view in views:
                        artifact = self._artifact_node(
                            trace_state[0],
                            ArtifactKind.MEMORY_ARTIFACT,
                            view.candidate.artifact_id,
                            view.candidate.content_digest,
                            byte_size=len(view.content.encode("utf-8")),
                            revision=view.candidate.revision,
                            provenance_ref=request.fixed_route.backend,
                            artifact_id=view.candidate.artifact_id,
                        )
                        self.operation_recorder.record_artifact(artifact)
                        output_ids.append(artifact.artifact_id)
                    assert related_operation is not None
                    related_operation.complete(
                        output_artifact_ids=tuple(output_ids),
                        status=(
                            OperationStatus.SUCCESS
                            if output_ids
                            else OperationStatus.NONE
                        ),
                        reason_code=None if output_ids else "no_related_memory",
                        usage=(candidates.usage if index == 0 else RawResourceUsage()),
                    )
            if related_spec is not None:
                related_operation_ids.append(related_spec.operation_id)
            for artifact_id in output_ids:
                if artifact_id not in related_artifact_ids:
                    related_artifact_ids.append(artifact_id)
            related_by_fact[index] = views
            for view in views:
                flattened[view.candidate.candidate_id] = view

        decision_prompt = self.operation_prompt.render({
            "new_facts": [
                {
                    "fact_index": index,
                    "content": fact.content,
                    "content_digest": fact.content_digest,
                    "category": fact.category.value,
                    "namespace": fact.namespace,
                }
                for index, fact in enumerate(facts)
            ],
            "related_memories": [
                {
                    **view.prompt_record(),
                    "fact_indexes": [
                        index
                        for index, views in related_by_fact.items()
                        if any(
                            item.candidate.candidate_id == candidate_id
                            for item in views
                        )
                    ],
                }
                for candidate_id, view in sorted(flattened.items())
            ],
        })
        decision_spec = None
        if self.operation_recorder is not None and trace_state is not None:
            decision_parameter = self._record_parameter_artifact(
                trace_state[0],
                logical_name=self.semantic_manifest.update_component_id,
                content_digest=self.semantic_manifest.update_component_digest,
                revision=self.operation_prompt.artifact.version,
                provenance_ref=self.operation_prompt.artifact.prompt_id,
                artifact_id=self.semantic_manifest.update_component_id,
            )
            internal_utility_parameter = self._record_utility_parameter(
                trace_state[0],
                UtilityTarget.INTERNAL_OPERATION,
            )
            update_utility_parameter = self._record_utility_parameter(
                trace_state[0],
                UtilityTarget.INTERNAL_OPERATION,
                update=True,
            )
            decision_spec = self._operation_spec(
                trace_state[0],
                OperationKind.INTERNAL_OPERATION_DECISION,
                f"{request.idempotency_key}.decision",
                parents=(trace_state[3].operation_id, *related_operation_ids),
                inputs=(
                    *fact_artifacts,
                    *related_artifact_ids,
                    decision_parameter,
                    *(
                        (internal_utility_parameter,)
                        if internal_utility_parameter is not None
                        else ()
                    ),
                    *(
                        (update_utility_parameter,)
                        if update_utility_parameter is not None
                        else ()
                    ),
                ),
            )
        decision_scope = (
            self.operation_recorder.operation_scope(decision_spec)
            if self.operation_recorder is not None and decision_spec is not None
            else nullcontext(None)
        )
        proposal_artifact_ids = ()
        try:
            with decision_scope as decision_operation:
                operation_result = self.completion_client.complete(decision_prompt)
                try:
                    operations = self._parse_operations(
                        operation_result.output_text,
                        facts,
                        flattened,
                    )
                    if self.utility_gate is not None:
                        operations = self.utility_gate.apply_operations(
                            request,
                            operations,
                            generation_utility,
                        )
                except _RejectedDecision as exc:
                    if decision_operation is not None:
                        decision_operation.complete(
                            status=OperationStatus.REJECTED,
                            reason_code=exc.reason_code,
                            usage=operation_result.usage,
                        )
                    raise
                if self.operation_recorder is not None and trace_state is not None:
                    values = []
                    for index, proposal in enumerate(operations):
                        payload = _canonical_json({
                            "ordinal": index,
                            "action": proposal.action.value,
                            "reason_code": proposal.reason_code,
                            "candidate_id": proposal.candidate_id,
                            "new_content_digest": proposal.new_content_digest,
                        })
                        artifact = self._artifact_node(
                            trace_state[0],
                            ArtifactKind.OPERATION_PROPOSAL,
                            f"{request.idempotency_key}.proposal.{index}",
                            _sha(payload),
                            byte_size=len(payload.encode("utf-8")),
                            revision=self.descriptor.policy_version,
                            provenance_ref=self.operation_prompt.artifact.prompt_id,
                        )
                        self.operation_recorder.record_artifact(artifact)
                        values.append(artifact.artifact_id)
                    proposal_artifact_ids = tuple(values)
                    assert decision_operation is not None
                    decision_operation.complete(
                        output_artifact_ids=proposal_artifact_ids,
                        usage=operation_result.usage,
                    )
        except _RejectedDecision as exc:
            self._store_trace(
                request,
                trace_state,
                fact_artifact_ids=fact_artifacts,
                fact_extractions=fact_extractions,
                related_operation_ids=tuple(related_operation_ids),
                related_artifact_ids=tuple(related_artifact_ids),
                decision_operation_id=(
                    decision_spec.operation_id if decision_spec is not None else None
                ),
            )
            return SemanticPolicyDecision(
                MemoryIngestStatus.REJECTED,
                (),
                usage=_combine_usage(
                    extraction_result.usage,
                    candidates.usage,
                    operation_result.usage,
                ),
                reason_codes=(exc.reason_code,),
            )
        self._store_trace(
            request,
            trace_state,
            fact_artifact_ids=fact_artifacts,
            fact_extractions=fact_extractions,
            related_operation_ids=tuple(related_operation_ids),
            related_artifact_ids=tuple(related_artifact_ids),
            decision_operation_id=(
                decision_spec.operation_id if decision_spec is not None else None
            ),
            proposal_artifact_ids=proposal_artifact_ids,
        )
        return SemanticPolicyDecision(
            MemoryIngestStatus.SUCCESS,
            operations,
            usage=_combine_usage(
                extraction_result.usage,
                candidates.usage,
                operation_result.usage,
            ),
        )

    def _parse_facts(
        self,
        raw: str,
        request: SemanticIngestRequest,
    ) -> tuple[tuple[ExtractedSemanticFact, ...], tuple[FactExtractionTrace, ...]]:
        value = _strict_object(raw, {"facts"})
        raw_facts = value["facts"]
        if not isinstance(raw_facts, list):
            raise InvalidPolicyOutputError("facts must be a list")
        facts = []
        traces = []
        seen = set()
        for item in raw_facts:
            if not isinstance(item, str):
                raise InvalidPolicyOutputError("fact must be a string")
            content = " ".join(item.split())
            if not content or len(content) > 2_000:
                raise InvalidPolicyOutputError("fact length is invalid")
            digest = _sha(content)
            category = _classify(content)
            scope = request.scope
            namespace = "user" if scope == MemoryScope.USER else "memory"
            fact = ExtractedSemanticFact(
                _stable_id("fact", {
                    "request_id": request.idempotency_key,
                    "content_digest": digest,
                    "category": category.value,
                    "namespace": namespace,
                }),
                content,
                digest,
                category,
                namespace,
                scope,
                request.validity,
            )
            self._facts[digest] = fact
            if (
                _TEMPORARY.search(content)
                or _TRANSCRIPT.search(content)
                or _TOOL_NOISE.search(content)
            ):
                traces.append(FactExtractionTrace(
                    fact.fact_id,
                    digest,
                    False,
                    "non_durable_fact",
                ))
                continue
            if digest in seen:
                continue
            seen.add(digest)
            facts.append(fact)
            traces.append(FactExtractionTrace(fact.fact_id, digest, True, None))
        return tuple(facts), tuple(traces)

    def _parse_operations(
        self,
        raw: str,
        facts: tuple[ExtractedSemanticFact, ...],
        related: Mapping[str, RelatedMemoryView],
    ) -> tuple[InternalOperationProposal, ...]:
        value = _strict_object(raw, {"operations"})
        raw_operations = value["operations"]
        if not isinstance(raw_operations, list) or not raw_operations:
            raise InvalidPolicyOutputError("operations must be a non-empty list")
        proposals = []
        fact_indexes = set()
        targets = set()
        for item in raw_operations:
            if not isinstance(item, dict) or set(item) != {
                "fact_index",
                "action",
                "candidate_id",
            }:
                raise InvalidPolicyOutputError("operation fields are invalid")
            fact_index = item["fact_index"]
            if type(fact_index) is not int or not 0 <= fact_index < len(facts):
                raise InvalidPolicyOutputError("operation fact_index is invalid")
            if fact_index in fact_indexes:
                raise InvalidPolicyOutputError("fact has duplicate operations")
            fact_indexes.add(fact_index)
            try:
                action = InternalMemoryAction(str(item["action"]).lower())
            except ValueError as exc:
                raise InvalidPolicyOutputError("operation action is invalid") from exc
            candidate_id = item["candidate_id"]
            fact = facts[fact_index]
            if action == InternalMemoryAction.ADD:
                if candidate_id is not None:
                    raise InvalidPolicyOutputError("ADD cannot carry candidate")
                proposal = InternalOperationProposal(
                    action,
                    "new_semantic_fact",
                    new_content_digest=fact.content_digest,
                )
            elif action in {InternalMemoryAction.UPDATE, InternalMemoryAction.DELETE}:
                if not isinstance(candidate_id, str) or candidate_id not in related:
                    raise _RejectedDecision("hallucinated_candidate_target")
                view = related[candidate_id]
                if not view.mutable:
                    raise _RejectedDecision("unknown_owner_target")
                if view.namespace != fact.namespace:
                    raise _RejectedDecision("cross_namespace_target")
                if candidate_id in targets:
                    raise InvalidPolicyOutputError("candidate has duplicate operations")
                targets.add(candidate_id)
                self._target_namespaces[view.candidate.artifact_id] = view.namespace
                proposal = InternalOperationProposal(
                    action,
                    (
                        "superseded_semantic_fact"
                        if action == InternalMemoryAction.UPDATE
                        else "expired_semantic_fact"
                    ),
                    candidate_id=candidate_id,
                    new_content_digest=(
                        fact.content_digest
                        if action == InternalMemoryAction.UPDATE
                        else None
                    ),
                )
            else:
                if candidate_id is not None:
                    raise InvalidPolicyOutputError("NONE cannot carry candidate")
                proposal = InternalOperationProposal(action, "duplicate_semantic_fact")
            proposals.append(proposal)
        if fact_indexes != set(range(len(facts))):
            raise InvalidPolicyOutputError("every fact requires one operation")
        return tuple(proposals)


def build_validation_candidate(
    result: MemoryIngestResult,
    operation_index: int,
    policy: Mem0FlatSemanticPolicy,
    provenance: ValidationProvenance,
) -> UntrustedMemoryCandidate:
    if not 0 <= operation_index < len(result.operations):
        raise IndexError("memory operation index is out of range")
    operation = result.operations[operation_index]
    if provenance.execution_id != result.execution_id or provenance.operation_id != operation.operation_id:
        raise ValueError("validation provenance does not match operation")
    fact = (
        policy.fact_for_digest(operation.new_content_digest)
        if operation.new_content_digest is not None
        else None
    )
    if operation.action in {InternalMemoryAction.ADD, InternalMemoryAction.UPDATE}:
        if fact is None:
            raise ValueError("policy content owner has no fact for operation")
        metadata = {
            "category": fact.category.value,
            "scope": fact.scope.value,
            "temporal_validity": fact.temporal_validity.value,
            "source_execution_id": result.execution_id,
            "source_operation_id": operation.operation_id,
        }
        if operation.action == InternalMemoryAction.UPDATE:
            metadata["replaces_artifact_id"] = operation.target_artifact_id
        content = fact.content
        namespace = fact.namespace
        category = fact.category
        scope = fact.scope
        validity = fact.temporal_validity
    else:
        metadata = {}
        content = None
        namespace = policy.namespace_for_target(operation.target_artifact_id) or "user"
        category = scope = validity = None
    return UntrustedMemoryCandidate(
        candidate_id=_stable_id("mutation-candidate", {
            "execution_id": result.execution_id,
            "operation_id": operation.operation_id,
            "content_digest": operation.new_content_digest,
        }),
        action=operation.action,
        kind=MemoryKind.SEMANTIC,
        backend=result.fixed_route.backend,
        namespace=namespace,
        content=content,
        metadata=metadata,
        target_artifact_id=operation.target_artifact_id,
        expected_revision=operation.expected_revision,
        category=category,
        scope=scope,
        temporal_validity=validity,
        provenance=provenance,
    )


def _strict_object(raw: str, fields: set[str]) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise InvalidPolicyOutputError("completion is not valid JSON") from exc
    if not isinstance(value, dict) or set(value) != fields:
        raise InvalidPolicyOutputError("completion object fields are invalid")
    return value


def _classify(content: str) -> SemanticMemoryCategory:
    lowered = content.casefold()
    if re.search(r"\b(?:never|must not|cannot|avoid)\b", lowered):
        return SemanticMemoryCategory.CONSTRAINT
    if re.search(
        r"(?:^(?:please\s+)?use\b|\b(?:prefer|preference|favorite|always use|likes?|dislikes?)\b)",
        lowered,
    ):
        return SemanticMemoryCategory.PREFERENCE
    if re.search(r"\b(?:must|should|required|rule)\b", lowered):
        return SemanticMemoryCategory.RULE
    return SemanticMemoryCategory.FACT
