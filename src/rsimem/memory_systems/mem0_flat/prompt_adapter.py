"""Mem0-flat binding for the host-neutral semantic extraction prompt slot."""

from __future__ import annotations

from dataclasses import replace

from ...memory.prompt_components import (
    MemoryPromptAdapter,
    PromptBindingFingerprint,
    PromptComponentArtifact,
    PromptPolicyStage,
    PromptSlotDescriptor,
    content_digest,
    text_digest,
)
from ...memory.contracts import MemoryKind
from .prompts import (
    POLICY_FACT_EXTRACTION_FROZEN_WRAPPER,
    POLICY_FACT_EXTRACTION_PROMPT,
    POLICY_FACT_EXTRACTION_ROOT_BODY,
    PromptTemplate,
    compile_policy_fact_extraction_template,
)


MEM0_FLAT_PROMPT_ADAPTER_ID = "mem0-flat-prompt-adapter-v1"
MEM0_FLAT_EXTRACTION_SLOT_ID = "mem0-flat.semantic.extraction"
MEM0_FLAT_EXTRACTION_MAX_BODY_CHARS = 8_000


def _schema_digest(name: str) -> str:
    record = POLICY_FACT_EXTRACTION_PROMPT.artifact.manifest_record()
    return content_digest(record[name])


MEM0_FLAT_EXTRACTION_SLOT = PromptSlotDescriptor(
    slot_id=MEM0_FLAT_EXTRACTION_SLOT_ID,
    memory_kind=MemoryKind.SEMANTIC,
    policy_stage=PromptPolicyStage.EXTRACTION,
    input_schema_digest=_schema_digest("input_schema"),
    output_schema_digest=_schema_digest("output_schema"),
    frozen_wrapper_digest=text_digest(POLICY_FACT_EXTRACTION_FROZEN_WRAPPER),
    model_profile=POLICY_FACT_EXTRACTION_PROMPT.artifact.model_profile,
    owner_adapter_id=MEM0_FLAT_PROMPT_ADAPTER_ID,
    required_placeholders=tuple(sorted((
        "exit_evidence",
        "policy_body",
        "source_messages",
        "source_projection_digest",
    ))),
)


class Mem0FlatPromptAdapter(MemoryPromptAdapter):
    adapter_id = MEM0_FLAT_PROMPT_ADAPTER_ID

    def __init__(self) -> None:
        self._root = PromptComponentArtifact.create(
            slot=MEM0_FLAT_EXTRACTION_SLOT,
            version="root-v1",
            policy_body=POLICY_FACT_EXTRACTION_ROOT_BODY,
            source_provenance="pinned-membase-mem0-root",
        )
        self._templates: dict[str, PromptTemplate] = {}

    def list_slots(self) -> tuple[PromptSlotDescriptor, ...]:
        return (MEM0_FLAT_EXTRACTION_SLOT,)

    def root_artifact(self, slot_id: str) -> PromptComponentArtifact:
        self._require_slot(slot_id)
        return self._root

    def validate_replacement(
        self,
        slot_id: str,
        artifact: PromptComponentArtifact,
    ) -> None:
        self._require_slot(slot_id)
        if artifact.slot_id != slot_id:
            raise ValueError("Mem0-flat prompt artifact belongs to another slot")
        if artifact.owner_adapter_id != self.adapter_id:
            raise ValueError("Mem0-flat prompt artifact owner differs")
        if artifact.slot_contract_digest != MEM0_FLAT_EXTRACTION_SLOT.contract_digest:
            raise ValueError("Mem0-flat prompt artifact contract differs")
        if len(artifact.policy_body) > MEM0_FLAT_EXTRACTION_MAX_BODY_CHARS:
            raise ValueError("Mem0-flat extraction policy body is oversized")
        compile_policy_fact_extraction_template(artifact.policy_body)

    def bind(
        self,
        slot_id: str,
        artifact: PromptComponentArtifact,
    ) -> PromptBindingFingerprint:
        self.validate_replacement(slot_id, artifact)
        template_body = compile_policy_fact_extraction_template(
            artifact.policy_body
        )
        prompt_artifact = replace(
            POLICY_FACT_EXTRACTION_PROMPT.artifact,
            version=artifact.version,
            template_digest=text_digest(template_body),
            policy_version=artifact.version,
        )
        fingerprint = PromptBindingFingerprint.create(
            adapter_id=self.adapter_id,
            slot_id=slot_id,
            slot_contract_digest=MEM0_FLAT_EXTRACTION_SLOT.contract_digest,
            artifact_id=artifact.artifact_id,
            artifact_body_digest=artifact.body_digest,
            rendered_template_digest=prompt_artifact.template_digest,
        )
        self._templates[fingerprint.binding_id] = PromptTemplate(
            prompt_artifact,
            template_body,
            binding_fingerprint=fingerprint.binding_id,
        )
        return fingerprint

    def bound_template(
        self,
        fingerprint: PromptBindingFingerprint,
    ) -> PromptTemplate:
        if fingerprint.adapter_id != self.adapter_id:
            raise ValueError("prompt binding belongs to another adapter")
        try:
            template = self._templates[fingerprint.binding_id]
        except KeyError as exc:
            raise KeyError("Mem0-flat prompt binding is not active") from exc
        if template.artifact.template_digest != fingerprint.rendered_template_digest:
            raise ValueError("Mem0-flat bound template fingerprint differs")
        return template

    @staticmethod
    def _require_slot(slot_id: str) -> None:
        if slot_id != MEM0_FLAT_EXTRACTION_SLOT_ID:
            raise KeyError(f"unknown Mem0-flat prompt slot: {slot_id}")
