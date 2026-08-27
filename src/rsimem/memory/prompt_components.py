"""Host-neutral prompt component slots, artifacts, and binding contracts."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from .contracts import MemoryKind


PROMPT_COMPONENT_SCHEMA_VERSION = 1
PROMPT_SLOT_SCHEMA = "memory-prompt-slot-v1"
PROMPT_COMPONENT_ARTIFACT_SCHEMA = "memory-prompt-component-artifact-v1"
PROMPT_BINDING_SCHEMA = "memory-prompt-binding-v1"
SEMANTIC_POLICY_MANIFEST_SCHEMA = "semantic-policy-manifest-v1"
MATCHED_SEMANTIC_POLICY_MANIFEST_SCHEMA = "matched-semantic-policy-manifest-v1"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def content_digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def text_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _require_identifier(value: str, name: str) -> None:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{name} must be a stable identifier")


class PromptPolicyStage(StrEnum):
    EXTRACTION = "extraction"
    UPDATE = "update"


@dataclass(frozen=True, slots=True)
class PromptSlotDescriptor:
    slot_id: str
    memory_kind: MemoryKind
    policy_stage: PromptPolicyStage
    input_schema_digest: str
    output_schema_digest: str
    frozen_wrapper_digest: str
    model_profile: str
    owner_adapter_id: str
    required_placeholders: tuple[str, ...]
    slot_schema: str = PROMPT_SLOT_SCHEMA
    schema_version: int = PROMPT_COMPONENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "memory_kind", MemoryKind(self.memory_kind))
        object.__setattr__(self, "policy_stage", PromptPolicyStage(self.policy_stage))
        if (
            self.schema_version != PROMPT_COMPONENT_SCHEMA_VERSION
            or self.slot_schema != PROMPT_SLOT_SCHEMA
        ):
            raise ValueError("unsupported prompt slot schema")
        for value, name in (
            (self.slot_id, "prompt slot ID"),
            (self.model_profile, "prompt slot model profile"),
            (self.owner_adapter_id, "prompt slot owner adapter"),
        ):
            _require_identifier(value, name)
        for digest in (
            self.input_schema_digest,
            self.output_schema_digest,
            self.frozen_wrapper_digest,
        ):
            if _DIGEST.fullmatch(digest) is None:
                raise ValueError("prompt slot digest must be sha256")
        if (
            not self.required_placeholders
            or len(self.required_placeholders) != len(set(self.required_placeholders))
            or tuple(sorted(self.required_placeholders)) != self.required_placeholders
            or any(not value.strip() for value in self.required_placeholders)
        ):
            raise ValueError("prompt slot placeholders must be sorted and unique")
        if self.memory_kind != MemoryKind.SEMANTIC:
            raise ValueError("the current prompt slot contract is semantic-only")

    @property
    def contract_digest(self) -> str:
        return content_digest(self.payload())

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "slot_schema": self.slot_schema,
            "slot_id": self.slot_id,
            "memory_kind": self.memory_kind.value,
            "policy_stage": self.policy_stage.value,
            "input_schema_digest": self.input_schema_digest,
            "output_schema_digest": self.output_schema_digest,
            "frozen_wrapper_digest": self.frozen_wrapper_digest,
            "model_profile": self.model_profile,
            "owner_adapter_id": self.owner_adapter_id,
            "required_placeholders": list(self.required_placeholders),
        }


@dataclass(frozen=True, slots=True)
class PromptComponentArtifact:
    artifact_id: str
    slot_id: str
    slot_contract_digest: str
    version: str
    policy_body: str
    body_digest: str
    parent_artifact_id: str | None
    owner_adapter_id: str
    source_provenance: str | None = None
    artifact_schema: str = PROMPT_COMPONENT_ARTIFACT_SCHEMA
    schema_version: int = PROMPT_COMPONENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            self.schema_version != PROMPT_COMPONENT_SCHEMA_VERSION
            or self.artifact_schema != PROMPT_COMPONENT_ARTIFACT_SCHEMA
        ):
            raise ValueError("unsupported prompt component artifact schema")
        for value, name in (
            (self.artifact_id, "prompt component artifact ID"),
            (self.slot_id, "prompt component slot ID"),
            (self.version, "prompt component version"),
            (self.owner_adapter_id, "prompt component owner adapter"),
        ):
            _require_identifier(value, name)
        if self.parent_artifact_id is not None:
            _require_identifier(self.parent_artifact_id, "prompt component parent")
        if self.source_provenance is not None and not self.source_provenance.strip():
            raise ValueError("prompt component provenance must be non-empty")
        if not self.policy_body.strip():
            raise ValueError("prompt component policy body must not be empty")
        if "$" in self.policy_body:
            raise ValueError("prompt component policy body cannot contain placeholders")
        if self.body_digest != text_digest(self.policy_body):
            raise ValueError("prompt component body digest mismatch")
        if _DIGEST.fullmatch(self.slot_contract_digest) is None:
            raise ValueError("prompt component slot digest must be sha256")
        expected_id = f"prompt-component.{content_digest(self.identity_payload())[:40]}"
        if self.artifact_id != expected_id:
            raise ValueError("prompt component artifact ID mismatch")

    @classmethod
    def create(
        cls,
        *,
        slot: PromptSlotDescriptor,
        version: str,
        policy_body: str,
        parent_artifact_id: str | None = None,
        source_provenance: str | None = None,
    ) -> "PromptComponentArtifact":
        core = {
            "schema_version": PROMPT_COMPONENT_SCHEMA_VERSION,
            "artifact_schema": PROMPT_COMPONENT_ARTIFACT_SCHEMA,
            "slot_id": slot.slot_id,
            "slot_contract_digest": slot.contract_digest,
            "version": version,
            "body_digest": text_digest(policy_body),
            "parent_artifact_id": parent_artifact_id,
            "owner_adapter_id": slot.owner_adapter_id,
            "source_provenance": source_provenance,
        }
        return cls(
            artifact_id=f"prompt-component.{content_digest(core)[:40]}",
            policy_body=policy_body,
            **core,
        )

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "artifact_schema": self.artifact_schema,
            "slot_id": self.slot_id,
            "slot_contract_digest": self.slot_contract_digest,
            "version": self.version,
            "body_digest": self.body_digest,
            "parent_artifact_id": self.parent_artifact_id,
            "owner_adapter_id": self.owner_adapter_id,
            "source_provenance": self.source_provenance,
        }

    def payload(self) -> dict[str, object]:
        return {
            **self.identity_payload(),
            "artifact_id": self.artifact_id,
            "policy_body": self.policy_body,
        }


@dataclass(frozen=True, slots=True)
class PromptBindingFingerprint:
    binding_id: str
    adapter_id: str
    slot_id: str
    slot_contract_digest: str
    artifact_id: str
    artifact_body_digest: str
    rendered_template_digest: str
    binding_schema: str = PROMPT_BINDING_SCHEMA
    schema_version: int = PROMPT_COMPONENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            self.schema_version != PROMPT_COMPONENT_SCHEMA_VERSION
            or self.binding_schema != PROMPT_BINDING_SCHEMA
        ):
            raise ValueError("unsupported prompt binding schema")
        for value in (self.adapter_id, self.slot_id, self.artifact_id):
            _require_identifier(value, "prompt binding identity")
        for digest in (
            self.slot_contract_digest,
            self.artifact_body_digest,
            self.rendered_template_digest,
        ):
            if _DIGEST.fullmatch(digest) is None:
                raise ValueError("prompt binding digest must be sha256")
        expected = f"prompt-binding.{content_digest(self.identity_payload())[:40]}"
        if self.binding_id != expected:
            raise ValueError("prompt binding ID mismatch")

    @classmethod
    def create(
        cls,
        *,
        adapter_id: str,
        slot_id: str,
        slot_contract_digest: str,
        artifact_id: str,
        artifact_body_digest: str,
        rendered_template_digest: str,
    ) -> "PromptBindingFingerprint":
        core = {
            "schema_version": PROMPT_COMPONENT_SCHEMA_VERSION,
            "binding_schema": PROMPT_BINDING_SCHEMA,
            "adapter_id": adapter_id,
            "slot_id": slot_id,
            "slot_contract_digest": slot_contract_digest,
            "artifact_id": artifact_id,
            "artifact_body_digest": artifact_body_digest,
            "rendered_template_digest": rendered_template_digest,
        }
        return cls(
            binding_id=f"prompt-binding.{content_digest(core)[:40]}",
            **core,
        )

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "binding_schema": self.binding_schema,
            "adapter_id": self.adapter_id,
            "slot_id": self.slot_id,
            "slot_contract_digest": self.slot_contract_digest,
            "artifact_id": self.artifact_id,
            "artifact_body_digest": self.artifact_body_digest,
            "rendered_template_digest": self.rendered_template_digest,
        }


@runtime_checkable
class MemoryPromptAdapter(Protocol):
    @property
    def adapter_id(self) -> str: ...

    def list_slots(self) -> tuple[PromptSlotDescriptor, ...]: ...

    def root_artifact(self, slot_id: str) -> PromptComponentArtifact: ...

    def validate_replacement(
        self,
        slot_id: str,
        artifact: PromptComponentArtifact,
    ) -> None: ...

    def bind(
        self,
        slot_id: str,
        artifact: PromptComponentArtifact,
    ) -> PromptBindingFingerprint: ...


class PromptAdapterRegistry:
    def __init__(self) -> None:
        self._slots: dict[str, tuple[MemoryPromptAdapter, PromptSlotDescriptor]] = {}

    def register(self, adapter: MemoryPromptAdapter) -> None:
        slots = adapter.list_slots()
        if not slots:
            raise ValueError("prompt adapter must expose at least one slot")
        if len({slot.slot_id for slot in slots}) != len(slots):
            raise ValueError("prompt adapter exposes duplicate slots")
        for slot in slots:
            if slot.owner_adapter_id != adapter.adapter_id:
                raise ValueError("prompt slot owner does not match adapter")
            if slot.slot_id in self._slots:
                raise ValueError(f"prompt slot already registered: {slot.slot_id}")
        for slot in slots:
            self._slots[slot.slot_id] = (adapter, slot)

    def descriptor(self, slot_id: str) -> PromptSlotDescriptor:
        try:
            return self._slots[slot_id][1]
        except KeyError as exc:
            raise KeyError(f"unregistered prompt slot: {slot_id}") from exc

    def root_artifact(self, slot_id: str) -> PromptComponentArtifact:
        adapter, _ = self._resolve(slot_id)
        return adapter.root_artifact(slot_id)

    def bind(
        self,
        slot_id: str,
        artifact: PromptComponentArtifact,
    ) -> PromptBindingFingerprint:
        adapter, slot = self._resolve(slot_id)
        if artifact.slot_id != slot_id:
            raise ValueError("prompt artifact belongs to another slot")
        if artifact.owner_adapter_id != adapter.adapter_id:
            raise ValueError("prompt artifact belongs to another adapter")
        if artifact.slot_contract_digest != slot.contract_digest:
            raise ValueError("prompt artifact slot contract differs")
        adapter.validate_replacement(slot_id, artifact)
        fingerprint = adapter.bind(slot_id, artifact)
        if (
            fingerprint.adapter_id != adapter.adapter_id
            or fingerprint.slot_id != slot_id
            or fingerprint.slot_contract_digest != slot.contract_digest
            or fingerprint.artifact_id != artifact.artifact_id
            or fingerprint.artifact_body_digest != artifact.body_digest
        ):
            raise ValueError("prompt adapter returned a mismatched binding")
        return fingerprint

    def _resolve(
        self,
        slot_id: str,
    ) -> tuple[MemoryPromptAdapter, PromptSlotDescriptor]:
        try:
            return self._slots[slot_id]
        except KeyError as exc:
            raise KeyError(f"unregistered prompt slot: {slot_id}") from exc


@dataclass(frozen=True, slots=True)
class SemanticPolicyManifest:
    route: str
    boundary: str
    backend: str
    framework_version: str
    model_profile: str
    extraction_component_id: str
    extraction_component_digest: str
    update_component_id: str
    update_component_digest: str
    retrieval_component_id: str
    retrieval_component_digest: str
    composite_digest: str
    composite_policy_version: str
    manifest_schema: str = SEMANTIC_POLICY_MANIFEST_SCHEMA
    schema_version: int = PROMPT_COMPONENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            self.schema_version != PROMPT_COMPONENT_SCHEMA_VERSION
            or self.manifest_schema != SEMANTIC_POLICY_MANIFEST_SCHEMA
        ):
            raise ValueError("unsupported semantic policy manifest schema")
        for value in (
            self.route,
            self.boundary,
            self.backend,
            self.framework_version,
            self.model_profile,
            self.extraction_component_id,
            self.update_component_id,
            self.retrieval_component_id,
        ):
            _require_identifier(value, "semantic policy component identity")
        for digest in (
            self.extraction_component_digest,
            self.update_component_digest,
            self.retrieval_component_digest,
            self.composite_digest,
        ):
            if _DIGEST.fullmatch(digest) is None:
                raise ValueError("semantic policy component digest must be sha256")
        if self.composite_digest != content_digest(self.identity_payload()):
            raise ValueError("semantic policy composite digest mismatch")
        if self.composite_policy_version != (
            f"semantic-composite.{self.composite_digest[:24]}"
        ):
            raise ValueError("semantic policy composite version mismatch")

    @classmethod
    def create(
        cls,
        **values: str,
    ) -> "SemanticPolicyManifest":
        core = {
            "schema_version": PROMPT_COMPONENT_SCHEMA_VERSION,
            "manifest_schema": SEMANTIC_POLICY_MANIFEST_SCHEMA,
            **values,
        }
        digest = content_digest(core)
        return cls(
            **values,
            composite_digest=digest,
            composite_policy_version=f"semantic-composite.{digest[:24]}",
        )

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "manifest_schema": self.manifest_schema,
            "route": self.route,
            "boundary": self.boundary,
            "backend": self.backend,
            "framework_version": self.framework_version,
            "model_profile": self.model_profile,
            "extraction_component_id": self.extraction_component_id,
            "extraction_component_digest": self.extraction_component_digest,
            "update_component_id": self.update_component_id,
            "update_component_digest": self.update_component_digest,
            "retrieval_component_id": self.retrieval_component_id,
            "retrieval_component_digest": self.retrieval_component_digest,
        }

    def payload(self) -> dict[str, object]:
        return {
            **self.identity_payload(),
            "composite_digest": self.composite_digest,
            "composite_policy_version": self.composite_policy_version,
        }


@dataclass(frozen=True, slots=True)
class MatchedSemanticPolicyManifest:
    parent: SemanticPolicyManifest
    candidate: SemanticPolicyManifest
    intervention_component: PromptPolicyStage
    matched_digest: str
    manifest_schema: str = MATCHED_SEMANTIC_POLICY_MANIFEST_SCHEMA
    schema_version: int = PROMPT_COMPONENT_SCHEMA_VERSION

    _FROZEN_FIELDS = (
        "route",
        "boundary",
        "backend",
        "framework_version",
        "model_profile",
        "update_component_id",
        "update_component_digest",
        "retrieval_component_id",
        "retrieval_component_digest",
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "intervention_component",
            PromptPolicyStage(self.intervention_component),
        )
        if (
            self.schema_version != PROMPT_COMPONENT_SCHEMA_VERSION
            or self.manifest_schema != MATCHED_SEMANTIC_POLICY_MANIFEST_SCHEMA
        ):
            raise ValueError("unsupported matched semantic policy manifest schema")
        if self.intervention_component != PromptPolicyStage.EXTRACTION:
            raise ValueError("the current matched intervention must be extraction")
        drift = tuple(
            field
            for field in self._FROZEN_FIELDS
            if getattr(self.parent, field) != getattr(self.candidate, field)
        )
        if drift:
            raise ValueError(
                "matched semantic policy drift outside extraction: "
                + ", ".join(drift)
            )
        if (
            self.parent.extraction_component_id
            == self.candidate.extraction_component_id
            or self.parent.extraction_component_digest
            == self.candidate.extraction_component_digest
        ):
            raise ValueError("matched extraction intervention must change its artifact")
        if self.parent.composite_digest == self.candidate.composite_digest:
            raise ValueError("matched extraction intervention must change composite identity")
        if self.matched_digest != content_digest(self.identity_payload()):
            raise ValueError("matched semantic policy digest mismatch")

    @classmethod
    def create(
        cls,
        parent: SemanticPolicyManifest,
        candidate: SemanticPolicyManifest,
    ) -> "MatchedSemanticPolicyManifest":
        core = {
            "schema_version": PROMPT_COMPONENT_SCHEMA_VERSION,
            "manifest_schema": MATCHED_SEMANTIC_POLICY_MANIFEST_SCHEMA,
            "intervention_component": PromptPolicyStage.EXTRACTION.value,
            "parent": parent.payload(),
            "candidate": candidate.payload(),
        }
        return cls(
            parent=parent,
            candidate=candidate,
            intervention_component=PromptPolicyStage.EXTRACTION,
            matched_digest=content_digest(core),
        )

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "manifest_schema": self.manifest_schema,
            "intervention_component": self.intervention_component.value,
            "parent": self.parent.payload(),
            "candidate": self.candidate.payload(),
        }

    def payload(self) -> dict[str, object]:
        return {
            **self.identity_payload(),
            "matched_digest": self.matched_digest,
        }
