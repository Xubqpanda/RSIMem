"""Mem0-flat semantic construction over fixed Hermes semantic storage."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
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
)


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
        framework_version: str = "mem0-flat-framework-v1",
        feature_schema_version: str = "semantic-fact-features-v1",
    ) -> None:
        if fact_prompt.artifact.policy_version != operation_prompt.artifact.policy_version:
            raise ValueError("Mem0 prompt policy versions must match")
        self.completion_client = completion_client
        self.retrieval = retrieval
        self.fact_prompt = fact_prompt
        self.operation_prompt = operation_prompt
        base_policy_version = policy_version or fact_prompt.artifact.policy_version
        bound_policy_version = f"{base_policy_version}.{retrieval.digest[:16]}"
        self._descriptor = SemanticPolicyDescriptor(
            provider="mem0_flat",
            policy_version=bound_policy_version,
            framework_version=framework_version,
            prompt_version=(
                f"{fact_prompt.artifact.version}+{operation_prompt.artifact.version}."
                f"{fact_prompt.artifact.template_digest[:8]}."
                f"{operation_prompt.artifact.template_digest[:8]}"
            ),
            feature_schema_version=feature_schema_version,
            capability=PolicyCapability(
                frozenset(InternalMemoryAction),
                add_time_update=True,
            ),
        )
        self._facts: dict[str, ExtractedSemanticFact] = {}
        self._target_namespaces: dict[str, str] = {}

    @property
    def descriptor(self) -> SemanticPolicyDescriptor:
        return self._descriptor

    def fact_for_digest(self, content_digest: str) -> ExtractedSemanticFact | None:
        return self._facts.get(content_digest)

    def namespace_for_target(self, artifact_id: str | None) -> str | None:
        return self._target_namespaces.get(artifact_id or "")

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
        if request.exit_evidence.unresolved_state is not None:
            return SemanticPolicyDecision(
                MemoryIngestStatus.REJECTED,
                (),
                reason_codes=("unresolved_source",),
            )
        if request.validity != TemporalValidity.DURABLE:
            return SemanticPolicyDecision(
                MemoryIngestStatus.REJECTED,
                (),
                reason_codes=("non_durable_source",),
            )

        extraction = self.fact_prompt.render({
            "source_messages": [
                {
                    "role": message.role,
                    "content": message.content,
                    "name": message.name,
                }
                for message in request.source_experience.messages
            ],
            "exit_evidence": request.exit_evidence.compiler_input_payload(),
        })
        extraction_result = self.completion_client.complete(extraction)
        facts, filtered_count = self._parse_facts(extraction_result.output_text, request)
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

        related_by_fact: dict[int, tuple[RelatedMemoryView, ...]] = {}
        flattened: dict[str, RelatedMemoryView] = {}
        for index, fact in enumerate(facts):
            views = candidates.search(request, fact.content)
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
        operation_result = self.completion_client.complete(decision_prompt)
        try:
            operations = self._parse_operations(
                operation_result.output_text,
                facts,
                flattened,
            )
        except _RejectedDecision as exc:
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
    ) -> tuple[tuple[ExtractedSemanticFact, ...], int]:
        value = _strict_object(raw, {"facts"})
        raw_facts = value["facts"]
        if not isinstance(raw_facts, list):
            raise InvalidPolicyOutputError("facts must be a list")
        facts = []
        seen = set()
        filtered_count = 0
        for item in raw_facts:
            if not isinstance(item, str):
                raise InvalidPolicyOutputError("fact must be a string")
            content = " ".join(item.split())
            if not content or len(content) > 2_000:
                raise InvalidPolicyOutputError("fact length is invalid")
            if (
                _TEMPORARY.search(content)
                or _TRANSCRIPT.search(content)
                or _TOOL_NOISE.search(content)
            ):
                filtered_count += 1
                continue
            digest = _sha(content)
            if digest in seen:
                continue
            seen.add(digest)
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
            facts.append(fact)
        return tuple(facts), filtered_count

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
    if re.search(r"\b(?:prefer|preference|favorite|always use|likes?|dislikes?)\b", lowered):
        return SemanticMemoryCategory.PREFERENCE
    if re.search(r"\b(?:must|should|required|rule)\b", lowered):
        return SemanticMemoryCategory.RULE
    return SemanticMemoryCategory.FACT
