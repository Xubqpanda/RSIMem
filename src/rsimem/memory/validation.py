"""Host-neutral mutation validation before any memory backend write."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping, Protocol, runtime_checkable

from ..lifecycle import MemoryScope, TemporalValidity
from .contracts import MemoryKind, MemoryQuery
from .ingestion import (
    HERMES_NATIVE_ROUTES,
    InternalMemoryAction,
    MemoryIngestResult,
    MemoryIngestStatus,
)
from .runtime import MemoryBackendRegistry


VALIDATION_CONTRACT_SCHEMA_VERSION = 1
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$")
_REASON_CODE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_SEMANTIC_NAMESPACE_LIMITS = {"memory": 2_200, "user": 1_375}
_SEMANTIC_ENTRY_DELIMITER = "\n§\n"
_SEMANTIC_ENTRY_MARKER = "§"
_SEMANTIC_METADATA_FIELDS = {
    "category",
    "scope",
    "temporal_validity",
    "source_execution_id",
    "source_operation_id",
}
_PROMPT_INJECTION_PATTERNS = (
    re.compile(r"(?i)ignore\s+(?:all\s+)?(?:previous|prior)\s+instructions?"),
    re.compile(r"(?i)system\s+prompt"),
    re.compile(r"(?i)developer\s+message"),
    re.compile(r"(?i)do\s+not\s+follow\s+(?:the\s+)?instructions?"),
    re.compile(r"(?i)you\s+are\s+(?:chatgpt|an?\s+assistant)"),
    re.compile(r"<\|(?:system|assistant|developer)\|>"),
)
_TRANSCRIPT_PATTERNS = (
    re.compile(r"(?im)^\s*(?:user|assistant|system|developer|tool)\s*:"),
    re.compile(r'(?i)"role"\s*:\s*"(?:user|assistant|system|tool)"'),
)
_TOOL_PAYLOAD_PATTERNS = (
    re.compile(r'(?i)"tool_calls?"\s*:'),
    re.compile(r'(?i)"tool_call_id"\s*:'),
    re.compile(r'(?i)"arguments"\s*:\s*[\[{]'),
)
_CREDENTIAL_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/-]{24,}"),
)
_MACHINE_PATH_PATTERNS = (
    re.compile(r"(?:^|\s)/(?:home|mnt|Users|tmp|var|etc)/\S+"),
    re.compile(r"(?:^|\s)[A-Za-z]:\\(?:Users|Windows|Program Files)\\\S+"),
)


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _sha(value: str | bytes) -> str:
    encoded = value if isinstance(value, bytes) else value.encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _valid_identifier(value: object) -> bool:
    return isinstance(value, str) and _IDENTIFIER.fullmatch(value) is not None


def _append_reason(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


class SemanticMemoryCategory(StrEnum):
    FACT = "fact"
    PREFERENCE = "preference"
    RULE = "rule"
    CONSTRAINT = "constraint"


class ValidationStatus(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class ValidationProvenance:
    run_id: str
    episode_id: str
    session_id: str
    task_id: str
    snapshot_id: str
    execution_id: str
    operation_id: str
    source_digest: str
    schema_version: int = VALIDATION_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != VALIDATION_CONTRACT_SCHEMA_VERSION:
            raise ValueError("unsupported validation provenance schema version")
        values = (
            self.run_id,
            self.episode_id,
            self.session_id,
            self.task_id,
            self.snapshot_id,
            self.execution_id,
            self.operation_id,
        )
        if any(not _valid_identifier(value) for value in values):
            raise ValueError("validation provenance identity is incomplete")
        if not _DIGEST.fullmatch(self.source_digest):
            raise ValueError("validation provenance source_digest must be sha256")


@dataclass(frozen=True, slots=True)
class TrustedValidationContext:
    provenance: ValidationProvenance
    scope: MemoryScope
    temporal_validity: TemporalValidity

    def __post_init__(self) -> None:
        object.__setattr__(self, "scope", MemoryScope(self.scope))
        object.__setattr__(
            self,
            "temporal_validity",
            TemporalValidity(self.temporal_validity),
        )


@dataclass(frozen=True, slots=True)
class UntrustedMemoryResource:
    path: object
    content: object
    media_type: object = "application/octet-stream"


@dataclass(frozen=True, slots=True)
class UntrustedMemoryCandidate:
    candidate_id: object
    action: object
    kind: object
    backend: object
    namespace: object
    content: object
    metadata: object = field(default_factory=dict)
    resources: tuple[object, ...] = ()
    target_artifact_id: object = None
    expected_revision: object = None
    category: object = None
    scope: object = None
    temporal_validity: object = None
    provenance: ValidationProvenance | None = None

    def __post_init__(self) -> None:
        if isinstance(self.metadata, Mapping):
            object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        object.__setattr__(self, "resources", tuple(self.resources))


@dataclass(frozen=True, slots=True)
class TrustedTargetBinding:
    backend: str
    artifact_id: str
    revision: str
    kind: MemoryKind
    namespace: str
    content_digest: str
    owner_run_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", MemoryKind(self.kind))
        values = (
            self.backend,
            self.artifact_id,
            self.revision,
            self.namespace,
            self.owner_run_id,
        )
        if any(not _valid_identifier(value) for value in values):
            raise ValueError("trusted target binding identity is incomplete")
        if not _DIGEST.fullmatch(self.content_digest):
            raise ValueError("trusted target content_digest must be sha256")


@runtime_checkable
class TargetOwnershipResolver(Protocol):
    def resolve(self, backend: str, artifact_id: str) -> TrustedTargetBinding | None: ...


class InMemoryTargetOwnershipRegistry:
    """Fixture registry; Phase 2D will back this contract with durable receipts."""

    def __init__(self) -> None:
        self._bindings: dict[tuple[str, str], TrustedTargetBinding] = {}

    def register(self, binding: TrustedTargetBinding) -> None:
        key = (binding.backend, binding.artifact_id)
        existing = self._bindings.get(key)
        if existing is not None and existing != binding:
            raise ValueError("conflicting trusted target ownership binding")
        self._bindings[key] = binding

    def resolve(self, backend: str, artifact_id: str) -> TrustedTargetBinding | None:
        return self._bindings.get((backend, artifact_id))


@dataclass(frozen=True, slots=True)
class SemanticValidationPolicy:
    max_entry_chars: int = 2_000
    max_entry_bytes: int = 8_000
    max_lines: int = 8
    max_metadata_bytes: int = 2_048
    max_resources: int = 16
    max_resource_bytes: int = 1_000_000
    max_total_resource_bytes: int = 4_000_000

    def __post_init__(self) -> None:
        if any(value < 0 for value in (
            self.max_entry_chars,
            self.max_entry_bytes,
            self.max_lines,
            self.max_metadata_bytes,
            self.max_resources,
            self.max_resource_bytes,
            self.max_total_resource_bytes,
        )):
            raise ValueError("semantic validation budgets must not be negative")


@dataclass(frozen=True, slots=True)
class ValidationResult:
    status: ValidationStatus
    operation_id: str
    reason_codes: tuple[str, ...]
    candidate_digest: str
    content_digest: str | None
    content_bytes: int
    metadata_bytes: int
    resource_count: int
    resource_bytes: int
    schema_version: int = VALIDATION_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != VALIDATION_CONTRACT_SCHEMA_VERSION:
            raise ValueError("unsupported validation result schema version")
        object.__setattr__(self, "status", ValidationStatus(self.status))
        if not _valid_identifier(self.operation_id):
            raise ValueError("validation result operation_id is invalid")
        if not _DIGEST.fullmatch(self.candidate_digest):
            raise ValueError("validation candidate_digest must be sha256")
        if self.content_digest is not None and not _DIGEST.fullmatch(self.content_digest):
            raise ValueError("validation content_digest must be sha256")
        if any(value < 0 for value in (
            self.content_bytes,
            self.metadata_bytes,
            self.resource_count,
            self.resource_bytes,
        )):
            raise ValueError("validation sizes must not be negative")
        if any(not _REASON_CODE.fullmatch(reason) for reason in self.reason_codes):
            raise ValueError("validation reason codes must be machine-readable")
        if self.status == ValidationStatus.ACCEPTED and self.reason_codes:
            raise ValueError("accepted validation cannot carry rejection reasons")
        if self.status == ValidationStatus.REJECTED and not self.reason_codes:
            raise ValueError("rejected validation requires reason codes")

    @property
    def accepted(self) -> bool:
        return self.status == ValidationStatus.ACCEPTED

    def observer_evidence(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "status": self.status.value,
            "operation_id": self.operation_id,
            "reason_codes": list(self.reason_codes),
            "candidate_digest": self.candidate_digest,
            "content_digest": self.content_digest,
            "content_bytes": self.content_bytes,
            "metadata_bytes": self.metadata_bytes,
            "resource_count": self.resource_count,
            "resource_bytes": self.resource_bytes,
        }


def _resource_fingerprint(resource: object) -> dict[str, object]:
    if not isinstance(resource, UntrustedMemoryResource):
        return {
            "path": type(resource).__name__,
            "content_digest": _sha(type(resource).__name__),
            "content_bytes": 0,
            "media_type": type(resource).__name__,
        }
    content = resource.content
    return {
        "path": resource.path if isinstance(resource.path, str) else type(resource.path).__name__,
        "content_digest": _sha(content) if isinstance(content, bytes) else _sha(type(content).__name__),
        "content_bytes": len(content) if isinstance(content, bytes) else 0,
        "media_type": (
            resource.media_type
            if isinstance(resource.media_type, str)
            else type(resource.media_type).__name__
        ),
    }


def _metadata_fingerprint(metadata: object) -> tuple[str, int, bool]:
    if not isinstance(metadata, Mapping):
        encoded = _canonical_json({"type": type(metadata).__name__}).encode("utf-8")
        return _sha(encoded), len(encoded), False
    try:
        plain = {str(key): value for key, value in metadata.items()}
        encoded = _canonical_json(plain).encode("utf-8")
    except (TypeError, ValueError):
        shape = sorted((str(key), type(value).__name__) for key, value in metadata.items())
        encoded = _canonical_json(shape).encode("utf-8")
        return _sha(encoded), len(encoded), False
    return _sha(encoded), len(encoded), True


def _candidate_fingerprint(
    candidate: UntrustedMemoryCandidate,
) -> tuple[str, str | None, int, int, int]:
    content = candidate.content
    content_digest = _sha(content) if isinstance(content, str) else None
    content_bytes = len(content.encode("utf-8")) if isinstance(content, str) else 0
    metadata_digest, metadata_bytes, _ = _metadata_fingerprint(candidate.metadata)
    resources = [_resource_fingerprint(resource) for resource in candidate.resources]
    resource_bytes = sum(int(item["content_bytes"]) for item in resources)
    identity = {
        "candidate_id": (
            candidate.candidate_id
            if isinstance(candidate.candidate_id, str)
            else type(candidate.candidate_id).__name__
        ),
        "action": str(candidate.action),
        "kind": str(candidate.kind),
        "backend": str(candidate.backend),
        "namespace": str(candidate.namespace),
        "content_digest": content_digest,
        "metadata_digest": metadata_digest,
        "resources": resources,
        "target_artifact_id": str(candidate.target_artifact_id),
        "expected_revision": str(candidate.expected_revision),
        "category": str(candidate.category),
        "scope": str(candidate.scope),
        "temporal_validity": str(candidate.temporal_validity),
        "provenance": (
            {
                "run_id": candidate.provenance.run_id,
                "episode_id": candidate.provenance.episode_id,
                "session_id": candidate.provenance.session_id,
                "task_id": candidate.provenance.task_id,
                "snapshot_id": candidate.provenance.snapshot_id,
                "execution_id": candidate.provenance.execution_id,
                "operation_id": candidate.provenance.operation_id,
                "source_digest": candidate.provenance.source_digest,
            }
            if isinstance(candidate.provenance, ValidationProvenance)
            else (
                None
                if candidate.provenance is None
                else {"type": type(candidate.provenance).__name__}
            )
        ),
    }
    return _sha(_canonical_json(identity)), content_digest, content_bytes, metadata_bytes, resource_bytes


def fingerprint_memory_candidate(candidate: UntrustedMemoryCandidate) -> str:
    """Return the canonical digest used to bind validation and recovery."""

    return _candidate_fingerprint(candidate)[0]


def _semantic_signature(content: str) -> tuple[tuple[str, ...], bool]:
    tokens = re.findall(r"[a-z0-9]+", content.casefold())
    negative_words = {"not", "never", "no", "avoid", "dislike", "dislikes", "cannot"}
    neutral_modifiers = negative_words | {"always", "must", "should"}
    return tuple(token for token in tokens if token not in neutral_modifiers), bool(
        negative_words.intersection(tokens)
    )


class MutationValidator:
    """Validate an untrusted candidate using only trusted host/runtime state."""

    def __init__(
        self,
        registry: MemoryBackendRegistry,
        *,
        policy: SemanticValidationPolicy = SemanticValidationPolicy(),
        target_resolver: TargetOwnershipResolver | None = None,
    ) -> None:
        self.registry = registry
        self.policy = policy
        self.target_resolver = target_resolver

    def validate(
        self,
        candidate: UntrustedMemoryCandidate,
        ingest_result: MemoryIngestResult,
        *,
        current_source_digest: str,
        trusted_context: TrustedValidationContext,
    ) -> ValidationResult:
        reasons: list[str] = []
        (
            candidate_digest,
            content_digest,
            content_bytes,
            metadata_bytes,
            resource_bytes,
        ) = _candidate_fingerprint(candidate)
        trusted_provenance = trusted_context.provenance
        operation_id = trusted_provenance.operation_id

        action: InternalMemoryAction | None = None
        try:
            action = InternalMemoryAction(candidate.action)
        except (ValueError, TypeError):
            _append_reason(reasons, "invalid_action")
        try:
            kind = MemoryKind(candidate.kind)
        except (ValueError, TypeError):
            kind = None
            _append_reason(reasons, "invalid_memory_kind")
        if not _valid_identifier(candidate.candidate_id):
            _append_reason(reasons, "invalid_candidate_id")
        if not isinstance(candidate.backend, str) or not candidate.backend.strip():
            _append_reason(reasons, "invalid_backend")
        if not isinstance(candidate.namespace, str):
            _append_reason(reasons, "invalid_namespace")

        provenance = candidate.provenance
        if not isinstance(current_source_digest, str) or not _DIGEST.fullmatch(
            current_source_digest
        ):
            _append_reason(reasons, "invalid_current_source_digest")
        if provenance is None:
            _append_reason(reasons, "missing_provenance")
        elif not isinstance(provenance, ValidationProvenance):
            _append_reason(reasons, "invalid_provenance")
        else:
            if provenance != trusted_provenance:
                _append_reason(reasons, "provenance_mismatch")
        if trusted_provenance.execution_id != ingest_result.execution_id:
            _append_reason(reasons, "execution_provenance_mismatch")
        if trusted_provenance.source_digest != ingest_result.source_digest:
            _append_reason(reasons, "source_provenance_mismatch")
        if trusted_provenance.source_digest != current_source_digest:
            _append_reason(reasons, "changed_source")

        if ingest_result.status != MemoryIngestStatus.SUCCESS:
            _append_reason(reasons, "ingest_not_successful")
        matching = [
            operation
            for operation in ingest_result.operations
            if operation.operation_id == operation_id
        ]
        operation = matching[0] if len(matching) == 1 else None
        if operation is None:
            _append_reason(reasons, "unknown_ingest_operation")
        elif action is not None:
            if operation.action != action:
                _append_reason(reasons, "operation_action_mismatch")
            if operation.new_content_digest != content_digest:
                _append_reason(reasons, "operation_content_mismatch")
            if operation.target_artifact_id != candidate.target_artifact_id:
                _append_reason(reasons, "operation_target_mismatch")
            if operation.expected_revision != candidate.expected_revision:
                _append_reason(reasons, "operation_revision_mismatch")

        self._validate_resources(candidate, reasons)
        _, _, metadata_valid = _metadata_fingerprint(candidate.metadata)
        if not metadata_valid:
            _append_reason(reasons, "invalid_metadata")
        if metadata_bytes > self.policy.max_metadata_bytes:
            _append_reason(reasons, "metadata_too_large")

        backend = None
        capability = None
        if kind is not None:
            if kind != MemoryKind.SEMANTIC:
                _append_reason(reasons, "memory_kind_disabled")
            else:
                try:
                    backend = self.registry.resolve(kind)
                except KeyError:
                    _append_reason(reasons, "missing_backend_route")
                if backend is not None:
                    if backend.descriptor.name != candidate.backend:
                        _append_reason(reasons, "backend_ownership_mismatch")
                    if backend.descriptor.name != HERMES_NATIVE_ROUTES[kind].backend:
                        _append_reason(reasons, "non_native_backend_disabled")
                    capability = backend.descriptor.capability_for(kind)
                    if capability is None:
                        _append_reason(reasons, "backend_kind_unsupported")

        if action is not None:
            self._validate_shape(candidate, action, reasons)
            if capability is not None:
                supported = {
                    InternalMemoryAction.ADD: capability.writable,
                    InternalMemoryAction.UPDATE: capability.updatable,
                    InternalMemoryAction.DELETE: capability.deletable,
                    InternalMemoryAction.NONE: True,
                }[action]
                if not supported:
                    _append_reason(reasons, "backend_action_unsupported")

        actual_target = None
        if action in {InternalMemoryAction.UPDATE, InternalMemoryAction.DELETE}:
            target = None
            if (
                self.target_resolver is not None
                and isinstance(candidate.backend, str)
                and isinstance(candidate.target_artifact_id, str)
            ):
                try:
                    target = self.target_resolver.resolve(
                        candidate.backend,
                        candidate.target_artifact_id,
                    )
                except Exception:
                    _append_reason(reasons, "target_ownership_lookup_failed")
            if target is None:
                _append_reason(reasons, "missing_trusted_target")
            if backend is not None and isinstance(candidate.target_artifact_id, str):
                try:
                    actual_target = backend.get(candidate.target_artifact_id)
                except Exception:
                    actual_target = None
                    _append_reason(reasons, "backend_read_failed")
                if actual_target is None:
                    _append_reason(reasons, "fabricated_target")
            if target is not None:
                self._validate_target(
                    candidate,
                    trusted_provenance,
                    target,
                    backend.descriptor.name if backend is not None else None,
                    actual_target,
                    reasons,
                )
        if kind == MemoryKind.SEMANTIC and action is not None:
            self._validate_semantic(
                candidate,
                action,
                ingest_result,
                backend,
                actual_target,
                operation,
                trusted_context,
                content_digest,
                content_bytes,
                reasons,
            )

        return ValidationResult(
            status=(ValidationStatus.REJECTED if reasons else ValidationStatus.ACCEPTED),
            operation_id=operation_id,
            reason_codes=tuple(reasons),
            candidate_digest=candidate_digest,
            content_digest=content_digest,
            content_bytes=content_bytes,
            metadata_bytes=metadata_bytes,
            resource_count=len(candidate.resources),
            resource_bytes=resource_bytes,
        )

    def _validate_resources(
        self,
        candidate: UntrustedMemoryCandidate,
        reasons: list[str],
    ) -> None:
        if len(candidate.resources) > self.policy.max_resources:
            _append_reason(reasons, "too_many_resources")
        paths: list[str] = []
        total = 0
        for resource in candidate.resources:
            if not isinstance(resource, UntrustedMemoryResource):
                _append_reason(reasons, "invalid_resource")
                continue
            if not isinstance(resource.path, str) or not resource.path:
                _append_reason(reasons, "invalid_resource_path")
            else:
                path = PurePosixPath(resource.path)
                paths.append(str(path))
                if path.is_absolute() or re.match(r"^[A-Za-z]:[\\/]", resource.path):
                    _append_reason(reasons, "absolute_resource_path")
                if "\\" in resource.path:
                    _append_reason(reasons, "invalid_resource_path")
                if ".." in path.parts:
                    _append_reason(reasons, "resource_path_traversal")
                if str(path) in {".", "SKILL.md"}:
                    _append_reason(reasons, "invalid_resource_path")
            if not isinstance(resource.content, bytes):
                _append_reason(reasons, "invalid_resource_content")
            else:
                total += len(resource.content)
                if len(resource.content) > self.policy.max_resource_bytes:
                    _append_reason(reasons, "resource_too_large")
            if not isinstance(resource.media_type, str) or not resource.media_type.strip():
                _append_reason(reasons, "invalid_resource_media_type")
        if len(paths) != len(set(paths)):
            _append_reason(reasons, "duplicate_resource_path")
        if total > self.policy.max_total_resource_bytes:
            _append_reason(reasons, "total_resources_too_large")

    @staticmethod
    def _validate_shape(
        candidate: UntrustedMemoryCandidate,
        action: InternalMemoryAction,
        reasons: list[str],
    ) -> None:
        has_content = isinstance(candidate.content, str) and bool(candidate.content.strip())
        has_target = _valid_identifier(candidate.target_artifact_id)
        has_revision = _valid_identifier(candidate.expected_revision)
        if action == InternalMemoryAction.ADD:
            valid = has_content and candidate.target_artifact_id is None and candidate.expected_revision is None
        elif action == InternalMemoryAction.UPDATE:
            valid = has_content and has_target and has_revision
        elif action == InternalMemoryAction.DELETE:
            valid = candidate.content is None and has_target and has_revision
        else:
            valid = (
                candidate.content is None
                and candidate.target_artifact_id is None
                and candidate.expected_revision is None
            )
        if not valid:
            _append_reason(reasons, "invalid_candidate_shape")

    @staticmethod
    def _validate_target(
        candidate: UntrustedMemoryCandidate,
        provenance: ValidationProvenance | None,
        target: TrustedTargetBinding,
        selected_backend: str | None,
        actual_target: Any,
        reasons: list[str],
    ) -> None:
        if target.backend != selected_backend or target.backend != candidate.backend:
            _append_reason(reasons, "target_backend_mismatch")
        if target.artifact_id != candidate.target_artifact_id:
            _append_reason(reasons, "target_artifact_mismatch")
        if target.revision != candidate.expected_revision:
            _append_reason(reasons, "stale_revision")
        if target.kind != MemoryKind.SEMANTIC:
            _append_reason(reasons, "target_kind_mismatch")
        if target.namespace != candidate.namespace:
            _append_reason(reasons, "target_namespace_mismatch")
        if provenance is None or target.owner_run_id != provenance.run_id:
            _append_reason(reasons, "cross_run_target")
        if actual_target is not None:
            if actual_target.revision != target.revision:
                _append_reason(reasons, "stale_revision")
            if actual_target.kind != target.kind:
                _append_reason(reasons, "target_kind_mismatch")
            if actual_target.namespace != target.namespace:
                _append_reason(reasons, "target_namespace_mismatch")
            if _sha(actual_target.content) != target.content_digest:
                _append_reason(reasons, "target_content_mismatch")

    def _validate_semantic(
        self,
        candidate: UntrustedMemoryCandidate,
        action: InternalMemoryAction,
        ingest_result: MemoryIngestResult,
        backend: Any,
        actual_target: Any,
        operation: Any,
        trusted_context: TrustedValidationContext,
        content_digest: str | None,
        content_bytes: int,
        reasons: list[str],
    ) -> None:
        namespace = candidate.namespace
        if namespace not in _SEMANTIC_NAMESPACE_LIMITS:
            _append_reason(reasons, "invalid_semantic_namespace")
        if candidate.resources:
            _append_reason(reasons, "semantic_resources_forbidden")
        metadata = candidate.metadata if isinstance(candidate.metadata, Mapping) else {}
        if action in {InternalMemoryAction.DELETE, InternalMemoryAction.NONE}:
            if metadata:
                _append_reason(reasons, "semantic_metadata_forbidden")
            if candidate.category is not None or candidate.scope is not None or candidate.temporal_validity is not None:
                _append_reason(reasons, "semantic_classification_forbidden")
            return

        try:
            category = SemanticMemoryCategory(candidate.category)
        except (ValueError, TypeError):
            category = None
            _append_reason(reasons, "invalid_semantic_category")
        try:
            scope = MemoryScope(candidate.scope)
        except (ValueError, TypeError):
            scope = None
            _append_reason(reasons, "invalid_semantic_scope")
        try:
            validity = TemporalValidity(candidate.temporal_validity)
        except (ValueError, TypeError):
            validity = None
            _append_reason(reasons, "invalid_semantic_validity")
        if validity is not None and validity != TemporalValidity.DURABLE:
            _append_reason(reasons, "semantic_not_durable")
        if scope is not None and scope != trusted_context.scope:
            _append_reason(reasons, "semantic_scope_mismatch")
        if validity is not None and validity != trusted_context.temporal_validity:
            _append_reason(reasons, "semantic_validity_mismatch")
        if trusted_context.temporal_validity != TemporalValidity.DURABLE:
            _append_reason(reasons, "semantic_not_durable")
        if set(str(key) for key in metadata) - _SEMANTIC_METADATA_FIELDS:
            _append_reason(reasons, "semantic_metadata_not_allowed")
        expected_metadata = {
            "category": category.value if category is not None else None,
            "scope": scope.value if scope is not None else None,
            "temporal_validity": validity.value if validity is not None else None,
            "source_execution_id": ingest_result.execution_id,
            "source_operation_id": (
                trusted_context.provenance.operation_id
            ),
        }
        for key, value in expected_metadata.items():
            if metadata.get(key) != value:
                _append_reason(reasons, "semantic_metadata_mismatch")
                break

        if not isinstance(candidate.content, str):
            _append_reason(reasons, "invalid_semantic_content")
            return
        content = candidate.content.strip()
        if len(content) > self.policy.max_entry_chars or content_bytes > self.policy.max_entry_bytes:
            _append_reason(reasons, "semantic_entry_too_large")
        if len(content.splitlines()) > self.policy.max_lines:
            _append_reason(reasons, "semantic_transcript")
        if _SEMANTIC_ENTRY_MARKER in content:
            _append_reason(reasons, "semantic_entry_delimiter")
        if any(pattern.search(content) for pattern in _TRANSCRIPT_PATTERNS):
            _append_reason(reasons, "semantic_transcript")
        if any(pattern.search(content) for pattern in _TOOL_PAYLOAD_PATTERNS):
            _append_reason(reasons, "semantic_tool_payload")
        if any(pattern.search(content) for pattern in _PROMPT_INJECTION_PATTERNS):
            _append_reason(reasons, "semantic_prompt_injection")
        if any(pattern.search(content) for pattern in _CREDENTIAL_PATTERNS):
            _append_reason(reasons, "semantic_credential")
        if any(pattern.search(content) for pattern in _MACHINE_PATH_PATTERNS):
            _append_reason(reasons, "semantic_machine_path")

        for value in metadata.values():
            if isinstance(value, str):
                if any(pattern.search(value) for pattern in _CREDENTIAL_PATTERNS):
                    _append_reason(reasons, "semantic_credential")
                if any(pattern.search(value) for pattern in _MACHINE_PATH_PATTERNS):
                    _append_reason(reasons, "semantic_machine_path")

        if backend is None or namespace not in _SEMANTIC_NAMESPACE_LIMITS:
            return
        if actual_target is not None and operation is not None:
            if _sha(actual_target.content) != operation.old_content_digest:
                _append_reason(reasons, "operation_target_content_mismatch")
        try:
            hits = tuple(backend.query(MemoryQuery(
                MemoryKind.SEMANTIC,
                "",
                namespace=namespace,
                limit=10_000,
            )))
        except Exception:
            _append_reason(reasons, "backend_read_failed")
            return
        normalized = " ".join(content.split()).casefold()
        new_signature, new_negative = _semantic_signature(content)
        existing_chars = 0
        for hit in hits:
            artifact = hit.artifact
            existing_chars += len(artifact.content)
            existing = " ".join(artifact.content.split()).casefold()
            if artifact.artifact_id == getattr(actual_target, "artifact_id", None):
                if existing == normalized:
                    _append_reason(reasons, "no_change_update")
                continue
            if existing == normalized:
                _append_reason(reasons, "duplicate_semantic_entry")
            signature, negative = _semantic_signature(artifact.content)
            if signature and signature == new_signature and negative != new_negative:
                _append_reason(reasons, "conflicting_semantic_entry")
        if actual_target is not None:
            existing_chars -= len(actual_target.content)
        final_entry_count = (
            len(hits) + 1
            if action == InternalMemoryAction.ADD
            else len(hits)
        )
        projected_chars = (
            existing_chars
            + len(content)
            + len(_SEMANTIC_ENTRY_DELIMITER) * max(0, final_entry_count - 1)
        )
        if projected_chars > _SEMANTIC_NAMESPACE_LIMITS[namespace]:
            _append_reason(reasons, "semantic_namespace_budget_exceeded")
