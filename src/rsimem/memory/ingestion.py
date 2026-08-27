"""Fixed-route semantic ingestion contracts for Phase 2 planning."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Callable, Mapping, Protocol, Sequence, runtime_checkable

from ..lifecycle import (
    ContextSnapshot,
    EvaluationTrigger,
    ExitEvidence,
    LIFECYCLE_CONTRACT_SCHEMA_VERSION,
    MemoryScope,
    ProvenanceRef,
    RawResourceUsage,
    TaskLifecycleState,
    TemporalValidity,
    WritebackPlan,
    WritebackPlanValidator,
)
from .contracts import MemoryAccessMode, MemoryExperience, MemoryKind


INGESTION_CONTRACT_SCHEMA_VERSION = 1
_REASON_CODE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


def _require_schema(version: int, name: str) -> None:
    if type(version) is not int or version != INGESTION_CONTRACT_SCHEMA_VERSION:
        raise ValueError(f"unsupported {name} schema version")


def _digest(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _stable_id(prefix: str, value: object) -> str:
    return f"{prefix}_{_digest(value)[:40]}"


class ContextExitSemantics(StrEnum):
    NATURAL = "natural"
    LOGICAL = "logical"
    PHYSICAL = "physical"


class InternalMemoryAction(StrEnum):
    ADD = "add"
    UPDATE = "update"
    DELETE = "delete"
    NONE = "none"


class MemoryIngestStatus(StrEnum):
    SUCCESS = "success"
    FAILED = "failed"
    REJECTED = "rejected"


class MemoryIngestOutcome(StrEnum):
    PLANNED_MUTATION = "planned_mutation"
    NO_CHANGE = "no_change"
    FAILED = "failed"
    REJECTED = "rejected"


class InvalidPolicyOutputError(ValueError):
    """An untrusted completion could not be parsed into the policy contract."""


class PolicyExecutionError(RuntimeError):
    """Structured policy failure with content-free usage and reason evidence."""

    def __init__(self, reason_code: str, usage: RawResourceUsage = RawResourceUsage()) -> None:
        if not _REASON_CODE.fullmatch(reason_code):
            raise ValueError("policy execution failure reason must be machine-readable")
        super().__init__(reason_code)
        self.reason_code = reason_code
        self.usage = usage


@dataclass(frozen=True, slots=True)
class FixedMemoryRoute:
    kind: MemoryKind
    backend: str
    access_mode: MemoryAccessMode
    policy_enabled: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", MemoryKind(self.kind))
        object.__setattr__(self, "access_mode", MemoryAccessMode(self.access_mode))
        if not self.backend.strip():
            raise ValueError("fixed memory route backend must not be empty")
        if type(self.policy_enabled) is not bool:
            raise TypeError("fixed memory route policy_enabled must be bool")


HERMES_NATIVE_ROUTES = {
    MemoryKind.SEMANTIC: FixedMemoryRoute(
        MemoryKind.SEMANTIC,
        "hermes-native-semantic",
        MemoryAccessMode.EAGER,
        True,
    ),
    MemoryKind.EPISODIC: FixedMemoryRoute(
        MemoryKind.EPISODIC,
        "hermes-native-episodic",
        MemoryAccessMode.SEARCH,
        False,
    ),
    MemoryKind.PROCEDURAL: FixedMemoryRoute(
        MemoryKind.PROCEDURAL,
        "hermes-native-procedural",
        MemoryAccessMode.PROGRESSIVE,
        False,
    ),
}


class FixedMemoryRouter:
    """Expose the three Hermes routes without a memory-form classifier."""

    def __init__(self, routes: Mapping[MemoryKind, FixedMemoryRoute] | None = None) -> None:
        self._routes = dict(routes or HERMES_NATIVE_ROUTES)
        if set(self._routes) != set(MemoryKind):
            raise ValueError("fixed router requires semantic, episodic, and procedural routes")
        for kind, route in self._routes.items():
            if route.kind != kind:
                raise ValueError("fixed route kind does not match its registry key")

    def resolve(self, kind: MemoryKind) -> FixedMemoryRoute:
        return self._routes[MemoryKind(kind)]

    @property
    def semantic(self) -> FixedMemoryRoute:
        return self.resolve(MemoryKind.SEMANTIC)


@dataclass(frozen=True, slots=True)
class IngestionProvenance:
    source: ProvenanceRef
    plan_id: str
    base_revision: str
    schema_version: int = INGESTION_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_schema(self.schema_version, "ingestion provenance")
        if self.source.schema_version != LIFECYCLE_CONTRACT_SCHEMA_VERSION:
            raise ValueError("ingestion provenance requires lifecycle schema v1")
        if not self.plan_id.strip() or not self.base_revision.strip():
            raise ValueError("ingestion provenance requires plan_id and base_revision")
        if not self.source.segment_ids or not self.source.evaluation_id:
            raise ValueError("ingestion provenance requires source segments and evaluation")


def _semantic_request_identity(
    *,
    source_experience: MemoryExperience,
    fixed_route: FixedMemoryRoute,
    exit_evidence: ExitEvidence,
    scope: MemoryScope,
    validity: TemporalValidity,
    policy_version: str,
    framework_version: str,
    provenance: IngestionProvenance,
    trigger: EvaluationTrigger,
    schema_version: int,
) -> dict[str, object]:
    source = source_experience
    return {
        "schema_version": schema_version,
        "experience": {
            "experience_id": source.experience_id,
            "session_id": source.session_id,
            "task_id": source.task_id,
            "outcome": source.outcome,
            "messages": [
                {
                    "role": message.role,
                    "content": message.content,
                    "name": message.name,
                    "metadata": dict(message.metadata),
                }
                for message in source.messages
            ],
            "metadata": dict(source.metadata),
        },
        "route": {
            "kind": fixed_route.kind.value,
            "backend": fixed_route.backend,
            "access_mode": fixed_route.access_mode.value,
        },
        "exit_evidence": {
            **exit_evidence.compiler_input_payload(),
            "safe_to_evict": exit_evidence.safe_to_evict,
            "provenance": exit_evidence.provenance,
        },
        "scope": scope.value,
        "validity": validity.value,
        "trigger": trigger.value,
        "policy_version": policy_version,
        "framework_version": framework_version,
        "provenance": {
            "run_id": provenance.source.run_id,
            "episode_id": provenance.source.episode_id,
            "session_id": provenance.source.session_id,
            "task_id": provenance.source.task_id,
            "snapshot_id": provenance.source.snapshot_id,
            "source_ref": provenance.source.source_ref,
            "segment_ids": provenance.source.segment_ids,
            "evaluation_id": provenance.source.evaluation_id,
            "plan_id": provenance.plan_id,
            "base_revision": provenance.base_revision,
        },
    }


@dataclass(frozen=True, slots=True)
class SemanticIngestRequest:
    source_experience: MemoryExperience
    fixed_route: FixedMemoryRoute
    exit_evidence: ExitEvidence
    scope: MemoryScope
    validity: TemporalValidity
    policy_version: str
    framework_version: str
    provenance: IngestionProvenance
    idempotency_key: str
    trigger: EvaluationTrigger
    schema_version: int = INGESTION_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_schema(self.schema_version, "semantic ingest request")
        object.__setattr__(self, "trigger", EvaluationTrigger(self.trigger))
        object.__setattr__(self, "scope", MemoryScope(self.scope))
        object.__setattr__(self, "validity", TemporalValidity(self.validity))
        if self.fixed_route != HERMES_NATIVE_ROUTES[MemoryKind.SEMANTIC]:
            raise ValueError("semantic ingestion requires the fixed Hermes semantic route")
        if self.trigger not in {
            EvaluationTrigger.TASK_COMPLETED,
            EvaluationTrigger.SESSION_END,
        }:
            raise ValueError("semantic ingestion requires a natural lifecycle boundary")
        if any(not value.strip() for value in (
            self.policy_version,
            self.framework_version,
            self.idempotency_key,
        )):
            raise ValueError("semantic ingestion policy, framework, and idempotency identity are required")
        if not self.source_experience.messages:
            raise ValueError("semantic ingestion requires source messages")
        if self.source_experience.score is not None:
            raise ValueError("hidden or evaluation score cannot enter semantic ingestion")
        forbidden = {
            "action", "operation", "backend", "target", "artifact_id", "revision",
            "policy_version", "framework_version", "grader", "score",
        }
        if forbidden.intersection(str(key).lower() for key in self.source_experience.metadata):
            raise ValueError("source metadata cannot predeclare routing, operation, or policy")

        source = self.provenance.source
        experience = self.source_experience
        if (
            experience.session_id != source.session_id
            or experience.task_id != source.task_id
        ):
            raise ValueError("semantic ingestion source identity must match provenance")
        if self.exit_evidence.scope != self.scope:
            raise ValueError("semantic ingestion scope must match exit evidence")
        if self.exit_evidence.temporal_validity != self.validity:
            raise ValueError("semantic ingestion validity must match exit evidence")
        expected_key = _stable_id("ingest_request", self.identity_payload())
        if self.idempotency_key != expected_key:
            raise ValueError("semantic ingestion idempotency identity is not canonical")

    @classmethod
    def create(
        cls,
        *,
        source_experience: MemoryExperience,
        fixed_route: FixedMemoryRoute,
        exit_evidence: ExitEvidence,
        scope: MemoryScope,
        validity: TemporalValidity,
        policy_version: str,
        framework_version: str,
        provenance: IngestionProvenance,
        trigger: EvaluationTrigger,
        schema_version: int = INGESTION_CONTRACT_SCHEMA_VERSION,
    ) -> SemanticIngestRequest:
        normalized_scope = MemoryScope(scope)
        normalized_validity = TemporalValidity(validity)
        normalized_trigger = EvaluationTrigger(trigger)
        values = {
            "source_experience": source_experience,
            "fixed_route": fixed_route,
            "exit_evidence": exit_evidence,
            "scope": normalized_scope,
            "validity": normalized_validity,
            "policy_version": policy_version,
            "framework_version": framework_version,
            "provenance": provenance,
            "trigger": normalized_trigger,
            "schema_version": schema_version,
        }
        identity = _semantic_request_identity(**values)
        return cls(
            **values,
            idempotency_key=_stable_id("ingest_request", identity),
        )

    def identity_payload(self) -> dict[str, object]:
        return _semantic_request_identity(
            source_experience=self.source_experience,
            fixed_route=self.fixed_route,
            exit_evidence=self.exit_evidence,
            scope=self.scope,
            validity=self.validity,
            policy_version=self.policy_version,
            framework_version=self.framework_version,
            provenance=self.provenance,
            trigger=self.trigger,
            schema_version=self.schema_version,
        )

    def canonical_payload(self) -> dict[str, object]:
        return {
            **self.identity_payload(),
            "idempotency_key": self.idempotency_key,
        }


def build_semantic_ingest_request(
    snapshot: ContextSnapshot,
    plan: WritebackPlan,
    experience: MemoryExperience,
    *,
    policy_version: str,
    framework_version: str,
    router: FixedMemoryRouter | None = None,
) -> SemanticIngestRequest:
    """Project a validated completed source into operation-free ingestion."""

    router = router or FixedMemoryRouter()
    if snapshot.task_state != TaskLifecycleState.COMPLETED:
        raise ValueError("completed semantic ingestion requires completed task state")
    if snapshot.current_turn_id is not None or snapshot.active_segment_ids:
        raise ValueError("active/current context cannot enter completed semantic ingestion")
    if any(not closure.closed for closure in snapshot.tool_closures):
        raise ValueError("open tool closure cannot enter completed semantic ingestion")
    by_id = {segment.segment_id: segment for segment in snapshot.segments}
    if any(
        segment_id not in by_id or not by_id[segment_id].completed
        for segment_id in plan.source_segment_ids
    ):
        raise ValueError("unresolved source cannot enter completed semantic ingestion")
    if plan.memory_kind != MemoryKind.SEMANTIC:
        raise ValueError("only the fixed semantic route is enabled for ingestion")
    validation = WritebackPlanValidator().validate_plan(plan, snapshot)
    if not validation.valid:
        raise ValueError("semantic ingestion requires a currently valid source plan")
    if experience.session_id != snapshot.session_id or experience.task_id != snapshot.task_id:
        raise ValueError("semantic experience identity must match the source snapshot")
    if plan.base_revision != snapshot.context_revision:
        raise ValueError("semantic ingestion source plan is stale")
    if plan.exit_evidence.scope is None or plan.exit_evidence.temporal_validity is None:
        raise ValueError("semantic ingestion requires resolved scope and validity")
    provenance = IngestionProvenance(
        plan.provenance,
        plan.plan_id,
        plan.base_revision,
    )
    return SemanticIngestRequest.create(
        source_experience=experience,
        fixed_route=router.semantic,
        exit_evidence=plan.exit_evidence,
        scope=plan.exit_evidence.scope,
        validity=plan.exit_evidence.temporal_validity,
        policy_version=policy_version,
        framework_version=framework_version,
        provenance=provenance,
        trigger=EvaluationTrigger(snapshot.lifecycle_state),
    )


@dataclass(frozen=True, slots=True)
class ExistingMemoryCandidate:
    candidate_id: str
    artifact_id: str
    revision: str
    content_digest: str
    kind: MemoryKind = MemoryKind.SEMANTIC

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", MemoryKind(self.kind))
        if self.kind != MemoryKind.SEMANTIC:
            raise ValueError("semantic candidate reader cannot expose another memory kind")
        if any(not value.strip() for value in (
            self.candidate_id, self.artifact_id, self.revision, self.content_digest,
        )):
            raise ValueError("existing memory candidate identity is incomplete")
        if not _DIGEST.fullmatch(self.content_digest):
            raise ValueError("candidate content_digest must be sha256")


@runtime_checkable
class ExistingMemoryCandidateReader(Protocol):
    def candidates(self, request: SemanticIngestRequest) -> Sequence[ExistingMemoryCandidate]: ...

    def resolve(self, candidate_id: str) -> ExistingMemoryCandidate | None: ...


@dataclass(frozen=True, slots=True)
class PolicyCapability:
    operations: frozenset[InternalMemoryAction]
    add_time_update: bool

    def __post_init__(self) -> None:
        normalized = frozenset(InternalMemoryAction(item) for item in self.operations)
        object.__setattr__(self, "operations", normalized)
        if InternalMemoryAction.ADD not in normalized or InternalMemoryAction.NONE not in normalized:
            raise ValueError("semantic policy must support ADD and NONE")
        if type(self.add_time_update) is not bool:
            raise TypeError("add_time_update capability must be bool")
        if self.add_time_update and InternalMemoryAction.UPDATE not in normalized:
            raise ValueError("add_time_update requires UPDATE capability")


@dataclass(frozen=True, slots=True)
class SemanticPolicyDescriptor:
    provider: str
    policy_version: str
    framework_version: str
    prompt_version: str
    feature_schema_version: str
    capability: PolicyCapability

    def __post_init__(self) -> None:
        if any(not value.strip() for value in (
            self.provider, self.policy_version, self.framework_version,
            self.prompt_version, self.feature_schema_version,
        )):
            raise ValueError("semantic policy descriptor versions must not be empty")


@dataclass(frozen=True, slots=True)
class InternalOperationProposal:
    action: InternalMemoryAction
    reason_code: str
    candidate_id: str | None = None
    new_content_digest: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "action", InternalMemoryAction(self.action))
        if not _REASON_CODE.fullmatch(self.reason_code):
            raise ValueError("internal proposal reason_code must be machine-readable")
        if self.action == InternalMemoryAction.ADD:
            if self.candidate_id is not None or self.new_content_digest is None:
                raise ValueError("ADD proposal requires only a new content digest")
        elif self.action == InternalMemoryAction.UPDATE:
            if not self.candidate_id or self.new_content_digest is None:
                raise ValueError("UPDATE proposal requires candidate and new digest")
        elif self.action == InternalMemoryAction.DELETE:
            if not self.candidate_id or self.new_content_digest is not None:
                raise ValueError("DELETE proposal requires only a candidate")
        elif self.candidate_id is not None or self.new_content_digest is not None:
            raise ValueError("NONE proposal cannot carry candidate or content")
        if self.new_content_digest is not None and not _DIGEST.fullmatch(
            self.new_content_digest
        ):
            raise ValueError("proposal new_content_digest must be sha256")


@dataclass(frozen=True, slots=True)
class SemanticPolicyDecision:
    status: MemoryIngestStatus
    operations: tuple[InternalOperationProposal, ...]
    usage: RawResourceUsage = RawResourceUsage()
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", MemoryIngestStatus(self.status))
        if any(not _REASON_CODE.fullmatch(code) for code in self.reason_codes):
            raise ValueError("policy decision reason codes must be machine-readable")
        if self.status == MemoryIngestStatus.SUCCESS:
            if not self.operations:
                raise ValueError("successful decision requires an internal operation")
        elif self.operations:
            raise ValueError("failed/rejected decision cannot carry operations")


@runtime_checkable
class SemanticMemoryPolicy(Protocol):
    @property
    def descriptor(self) -> SemanticPolicyDescriptor: ...

    def ingest(
        self,
        request: SemanticIngestRequest,
        candidates: ExistingMemoryCandidateReader,
    ) -> SemanticPolicyDecision: ...


@dataclass(frozen=True, slots=True)
class InternalMemoryOperation:
    operation_id: str
    action: InternalMemoryAction
    reason_code: str
    target_artifact_id: str | None
    expected_revision: str | None
    old_content_digest: str | None
    new_content_digest: str | None
    transaction_required: bool
    recovery_receipt_required: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "action", InternalMemoryAction(self.action))
        if not self.operation_id.strip() or not _REASON_CODE.fullmatch(self.reason_code):
            raise ValueError("resolved internal operation identity is invalid")
        if type(self.transaction_required) is not bool or type(
            self.recovery_receipt_required
        ) is not bool:
            raise TypeError("resolved mutation safety flags must be bool")
        mutating = self.action != InternalMemoryAction.NONE
        if self.transaction_required != mutating or self.recovery_receipt_required != mutating:
            raise ValueError("resolved mutation safety requirements do not match action")
        for digest in (self.old_content_digest, self.new_content_digest):
            if digest is not None and not _DIGEST.fullmatch(digest):
                raise ValueError("resolved internal operation digest must be sha256")
        target_values = (self.target_artifact_id, self.expected_revision)
        if any(target_values) and not all(target_values):
            raise ValueError("resolved target identity and revision must be complete")
        if self.action == InternalMemoryAction.ADD:
            if any(target_values) or self.old_content_digest is not None or self.new_content_digest is None:
                raise ValueError("resolved ADD operation shape is invalid")
        elif self.action == InternalMemoryAction.UPDATE:
            if not all(target_values) or self.old_content_digest is None or self.new_content_digest is None:
                raise ValueError("resolved UPDATE operation shape is invalid")
        elif self.action == InternalMemoryAction.DELETE:
            if not all(target_values) or self.old_content_digest is None or self.new_content_digest is not None:
                raise ValueError("resolved DELETE operation shape is invalid")
        elif any(target_values) or self.old_content_digest is not None or self.new_content_digest is not None:
            raise ValueError("resolved NONE operation shape is invalid")


@dataclass(frozen=True, slots=True)
class MemoryIngestResult:
    execution_id: str
    status: MemoryIngestStatus
    outcome: MemoryIngestOutcome
    operations: tuple[InternalMemoryOperation, ...]
    usage: RawResourceUsage
    reason_codes: tuple[str, ...]
    source_digest: str
    content_digests: tuple[str, ...]
    fixed_route: FixedMemoryRoute
    policy_provider: str
    policy_version: str
    framework_version: str
    prompt_version: str
    feature_schema_version: str
    idempotency_key: str
    schema_version: int = INGESTION_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_schema(self.schema_version, "memory ingest result")
        object.__setattr__(self, "status", MemoryIngestStatus(self.status))
        object.__setattr__(self, "outcome", MemoryIngestOutcome(self.outcome))
        if any(not value.strip() for value in (
            self.execution_id, self.source_digest, self.policy_provider,
            self.policy_version, self.framework_version, self.prompt_version,
            self.feature_schema_version, self.idempotency_key,
        )):
            raise ValueError("memory ingest result identity is incomplete")
        if not _DIGEST.fullmatch(self.source_digest):
            raise ValueError("memory ingest source_digest must be sha256")
        if any(not _DIGEST.fullmatch(value) for value in self.content_digests):
            raise ValueError("memory ingest content digests must be sha256")
        if len(self.content_digests) != len(set(self.content_digests)):
            raise ValueError("memory ingest content digests must be unique")
        if self.fixed_route != HERMES_NATIVE_ROUTES[MemoryKind.SEMANTIC]:
            raise ValueError("memory ingest result requires the fixed semantic route")
        if any(not _REASON_CODE.fullmatch(code) for code in self.reason_codes):
            raise ValueError("memory ingest result reason codes must be machine-readable")
        if self.status == MemoryIngestStatus.SUCCESS and not self.operations:
            raise ValueError("successful memory ingest result requires operations")
        if self.status != MemoryIngestStatus.SUCCESS and self.operations:
            raise ValueError("failed/rejected memory ingest result cannot carry operations")
        expected_outcome = (
            MemoryIngestOutcome.PLANNED_MUTATION
            if self.status == MemoryIngestStatus.SUCCESS
            and any(item.action != InternalMemoryAction.NONE for item in self.operations)
            else MemoryIngestOutcome.NO_CHANGE
            if self.status == MemoryIngestStatus.SUCCESS
            else MemoryIngestOutcome.FAILED
            if self.status == MemoryIngestStatus.FAILED
            else MemoryIngestOutcome.REJECTED
        )
        if self.outcome != expected_outcome:
            raise ValueError("memory ingest outcome does not match status and operations")
        expected_digests: list[str] = []
        for operation in self.operations:
            for value in (operation.old_content_digest, operation.new_content_digest):
                if value is not None and value not in expected_digests:
                    expected_digests.append(value)
        if self.content_digests != tuple(expected_digests):
            raise ValueError("memory ingest content digests do not match ordered operations")

    def observer_evidence(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "execution_id": self.execution_id,
            "status": self.status.value,
            "outcome": self.outcome.value,
            "operation_ids": [item.operation_id for item in self.operations],
            "operation_actions": [item.action.value for item in self.operations],
            "source_digest": self.source_digest,
            "content_digests": list(self.content_digests),
            "route_backend": self.fixed_route.backend,
            "memory_kind": self.fixed_route.kind.value,
            "policy_provider": self.policy_provider,
            "policy_version": self.policy_version,
            "framework_version": self.framework_version,
            "prompt_version": self.prompt_version,
            "feature_schema_version": self.feature_schema_version,
            "reason_codes": list(self.reason_codes),
            "resources": self.usage.to_dict(),
        }


class SemanticPolicyRegistry:
    def __init__(self) -> None:
        self._policies: dict[str, SemanticMemoryPolicy] = {}

    def register(self, policy: SemanticMemoryPolicy) -> None:
        provider = policy.descriptor.provider
        if provider in self._policies:
            raise ValueError(f"semantic policy already registered: {provider}")
        self._policies[provider] = policy

    def resolve(self, provider: str) -> SemanticMemoryPolicy:
        try:
            return self._policies[provider]
        except KeyError as exc:
            raise KeyError(f"unknown semantic policy provider: {provider}") from exc


class BoundSemanticPolicy:
    """Provider shell shared by deterministic and future Mem0 policies."""

    def __init__(
        self,
        descriptor: SemanticPolicyDescriptor,
        decide: Callable[
            [SemanticIngestRequest, ExistingMemoryCandidateReader],
            SemanticPolicyDecision,
        ],
    ) -> None:
        self._descriptor = descriptor
        self._decide = decide

    @property
    def descriptor(self) -> SemanticPolicyDescriptor:
        return self._descriptor

    def ingest(
        self,
        request: SemanticIngestRequest,
        candidates: ExistingMemoryCandidateReader,
    ) -> SemanticPolicyDecision:
        return self._decide(request, candidates)


def mem0_flat_policy(
    decide: Callable[
        [SemanticIngestRequest, ExistingMemoryCandidateReader],
        SemanticPolicyDecision,
    ],
    *,
    policy_version: str = "mem0-flat-v0",
    framework_version: str = "mem0-flat-contract-v1",
    prompt_version: str = "unbound-v0",
    feature_schema_version: str = "semantic-features-v0",
) -> BoundSemanticPolicy:
    return BoundSemanticPolicy(
        SemanticPolicyDescriptor(
            provider="mem0_flat",
            policy_version=policy_version,
            framework_version=framework_version,
            prompt_version=prompt_version,
            feature_schema_version=feature_schema_version,
            capability=PolicyCapability(
                frozenset(InternalMemoryAction),
                add_time_update=True,
            ),
        ),
        decide,
    )


class DeterministicPassThroughIngestor:
    """Fixture-only policy that returns one predeclared host-neutral decision."""

    fixture_only = True

    def __init__(
        self,
        operations: Sequence[InternalOperationProposal],
        *,
        status: MemoryIngestStatus = MemoryIngestStatus.SUCCESS,
        usage: RawResourceUsage = RawResourceUsage(),
        reason_codes: tuple[str, ...] = (),
        policy_version: str = "deterministic-pass-through-v1",
        framework_version: str = "deterministic-fixture-v1",
        prompt_version: str = "no-prompt-v1",
        feature_schema_version: str = "fixture-features-v1",
    ) -> None:
        self._descriptor = SemanticPolicyDescriptor(
            provider="deterministic_fixture",
            policy_version=policy_version,
            framework_version=framework_version,
            prompt_version=prompt_version,
            feature_schema_version=feature_schema_version,
            capability=PolicyCapability(
                frozenset(InternalMemoryAction),
                add_time_update=True,
            ),
        )
        self._decision = SemanticPolicyDecision(
            status,
            tuple(operations),
            usage=usage,
            reason_codes=reason_codes,
        )

    @property
    def descriptor(self) -> SemanticPolicyDescriptor:
        return self._descriptor

    def ingest(
        self,
        request: SemanticIngestRequest,
        candidates: ExistingMemoryCandidateReader,
    ) -> SemanticPolicyDecision:
        del request, candidates
        return self._decision


class SemanticIngestionCoordinator:
    """Validate policy proposals and bind trusted targets without mutation."""

    def __init__(
        self,
        registry: SemanticPolicyRegistry,
        *,
        provider: str,
        router: FixedMemoryRouter | None = None,
        enabled: bool = True,
        exit_semantics: ContextExitSemantics = ContextExitSemantics.NATURAL,
    ) -> None:
        self.registry = registry
        self.provider = provider
        self.router = router or FixedMemoryRouter()
        self.enabled = enabled
        self.exit_semantics = ContextExitSemantics(exit_semantics)
        if self.exit_semantics != ContextExitSemantics.NATURAL:
            raise ValueError("Phase 2A supports natural context exit only")
        self._results: dict[str, tuple[str, MemoryIngestResult]] = {}

    def ingest(
        self,
        request: SemanticIngestRequest,
        candidates: ExistingMemoryCandidateReader,
        *,
        current_source_revision: str | None = None,
    ) -> MemoryIngestResult | None:
        if not self.enabled:
            return None
        if request.fixed_route != self.router.semantic:
            raise ValueError("semantic request route differs from the fixed runtime route")
        if current_source_revision is not None:
            if not current_source_revision.strip():
                raise ValueError("current source revision must not be empty")
            if current_source_revision != request.provenance.base_revision:
                raise ValueError("semantic ingestion source snapshot is stale")
        policy = self.registry.resolve(self.provider)
        descriptor = policy.descriptor
        if request.policy_version != descriptor.policy_version:
            raise ValueError("semantic request policy version differs from runtime binding")
        if request.framework_version != descriptor.framework_version:
            raise ValueError("semantic request framework version differs from runtime binding")

        request_digest = _digest(request.canonical_payload())
        previous = self._results.get(request.idempotency_key)
        if previous is not None:
            if previous[0] != request_digest:
                raise ValueError("semantic idempotency key conflicts with another request")
            return previous[1]

        execution_id = _stable_id("ingest", {
            "schema_version": INGESTION_CONTRACT_SCHEMA_VERSION,
            "request_digest": request_digest,
            "provider": descriptor.provider,
            "policy_version": descriptor.policy_version,
            "framework_version": descriptor.framework_version,
            "prompt_version": descriptor.prompt_version,
            "feature_schema_version": descriptor.feature_schema_version,
        })
        try:
            decision = policy.ingest(request, candidates)
        except PolicyExecutionError as exc:
            result = self._failure_result(
                request,
                descriptor,
                execution_id,
                request_digest,
                status=MemoryIngestStatus.FAILED,
                reason_code=exc.reason_code,
                usage=exc.usage,
            )
            self._results[request.idempotency_key] = (request_digest, result)
            return result
        except TimeoutError:
            result = self._failure_result(
                request,
                descriptor,
                execution_id,
                request_digest,
                status=MemoryIngestStatus.FAILED,
                reason_code="policy_timeout",
            )
            self._results[request.idempotency_key] = (request_digest, result)
            return result
        except (InvalidPolicyOutputError, json.JSONDecodeError):
            result = self._failure_result(
                request,
                descriptor,
                execution_id,
                request_digest,
                status=MemoryIngestStatus.REJECTED,
                reason_code="invalid_policy_json",
            )
            self._results[request.idempotency_key] = (request_digest, result)
            return result
        except Exception:
            result = self._failure_result(
                request,
                descriptor,
                execution_id,
                request_digest,
                status=MemoryIngestStatus.FAILED,
                reason_code="policy_exception",
            )
            self._results[request.idempotency_key] = (request_digest, result)
            return result
        if not isinstance(decision, SemanticPolicyDecision):
            result = self._failure_result(
                request,
                descriptor,
                execution_id,
                request_digest,
                status=MemoryIngestStatus.REJECTED,
                reason_code="invalid_policy_result",
            )
            self._results[request.idempotency_key] = (request_digest, result)
            return result
        result = self._resolve_decision(
            request, descriptor, decision, candidates, execution_id, request_digest,
        )
        self._results[request.idempotency_key] = (request_digest, result)
        return result

    @staticmethod
    def _failure_result(
        request: SemanticIngestRequest,
        descriptor: SemanticPolicyDescriptor,
        execution_id: str,
        source_digest: str,
        *,
        status: MemoryIngestStatus,
        reason_code: str,
        usage: RawResourceUsage = RawResourceUsage(),
    ) -> MemoryIngestResult:
        outcome = (
            MemoryIngestOutcome.REJECTED
            if status == MemoryIngestStatus.REJECTED
            else MemoryIngestOutcome.FAILED
        )
        return MemoryIngestResult(
            execution_id=execution_id,
            status=status,
            outcome=outcome,
            operations=(),
            usage=usage,
            reason_codes=(reason_code,),
            source_digest=source_digest,
            content_digests=(),
            fixed_route=request.fixed_route,
            policy_provider=descriptor.provider,
            policy_version=descriptor.policy_version,
            framework_version=descriptor.framework_version,
            prompt_version=descriptor.prompt_version,
            feature_schema_version=descriptor.feature_schema_version,
            idempotency_key=request.idempotency_key,
        )

    @staticmethod
    def _resolve_decision(
        request: SemanticIngestRequest,
        descriptor: SemanticPolicyDescriptor,
        decision: SemanticPolicyDecision,
        candidates: ExistingMemoryCandidateReader,
        execution_id: str,
        source_digest: str,
    ) -> MemoryIngestResult:
        resolved: list[InternalMemoryOperation] = []
        seen: set[tuple[InternalMemoryAction, str | None]] = set()
        candidate_values = tuple(candidates.candidates(request))
        candidate_ids = [item.candidate_id for item in candidate_values]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("semantic candidate reader returned an ambiguous candidate")
        available = {item.candidate_id: item for item in candidate_values}
        for index, proposal in enumerate(decision.operations):
            if proposal.action not in descriptor.capability.operations:
                raise ValueError("semantic policy proposed an unsupported operation")
            if (
                proposal.action == InternalMemoryAction.UPDATE
                and not descriptor.capability.add_time_update
            ):
                raise ValueError("semantic policy does not support add-time update")
            target = None
            if proposal.candidate_id is not None:
                target = candidates.resolve(proposal.candidate_id)
                if target is None:
                    raise ValueError("semantic policy proposed an unknown candidate")
                if available.get(proposal.candidate_id) != target:
                    raise ValueError("semantic candidate ownership or revision is stale")
            identity = (proposal.action, target.artifact_id if target else None)
            if identity in seen:
                raise ValueError("semantic policy proposed a duplicate operation")
            seen.add(identity)
            operation_payload = {
                "execution_id": execution_id,
                "ordinal": index,
                "action": proposal.action.value,
                "target_artifact_id": target.artifact_id if target else None,
                "expected_revision": target.revision if target else None,
                "new_content_digest": proposal.new_content_digest,
            }
            resolved.append(InternalMemoryOperation(
                operation_id=_stable_id("operation", operation_payload),
                action=proposal.action,
                reason_code=proposal.reason_code,
                target_artifact_id=target.artifact_id if target else None,
                expected_revision=target.revision if target else None,
                old_content_digest=target.content_digest if target else None,
                new_content_digest=proposal.new_content_digest,
                transaction_required=proposal.action != InternalMemoryAction.NONE,
                recovery_receipt_required=proposal.action != InternalMemoryAction.NONE,
            ))
        content_digests: list[str] = []
        for operation in resolved:
            for value in (operation.old_content_digest, operation.new_content_digest):
                if value is not None and value not in content_digests:
                    content_digests.append(value)
        outcome = (
            MemoryIngestOutcome.PLANNED_MUTATION
            if decision.status == MemoryIngestStatus.SUCCESS
            and any(item.action != InternalMemoryAction.NONE for item in resolved)
            else MemoryIngestOutcome.NO_CHANGE
            if decision.status == MemoryIngestStatus.SUCCESS
            else MemoryIngestOutcome.FAILED
            if decision.status == MemoryIngestStatus.FAILED
            else MemoryIngestOutcome.REJECTED
        )
        return MemoryIngestResult(
            execution_id=execution_id,
            status=decision.status,
            outcome=outcome,
            operations=tuple(resolved),
            usage=decision.usage,
            reason_codes=decision.reason_codes,
            source_digest=source_digest,
            content_digests=tuple(content_digests),
            fixed_route=request.fixed_route,
            policy_provider=descriptor.provider,
            policy_version=descriptor.policy_version,
            framework_version=descriptor.framework_version,
            prompt_version=descriptor.prompt_version,
            feature_schema_version=descriptor.feature_schema_version,
            idempotency_key=request.idempotency_key,
        )
