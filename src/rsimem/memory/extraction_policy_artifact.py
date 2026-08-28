"""Host-neutral extraction policy specs and immutable prompt artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping

from ..lifecycle import RawResourceUsage
from .prompt_components import (
    PromptComponentArtifact,
    PromptSlotDescriptor,
    canonical_json,
    content_digest,
    text_digest,
)


EXTRACTION_POLICY_SCHEMA_VERSION = 1
EXTRACTION_POLICY_SPEC_SCHEMA = "extraction-policy-spec-v1"
EXTRACTION_RULE_EDIT_SCHEMA = "extraction-policy-rule-edit-v1"
EXTRACTION_GENERATION_PROVENANCE_SCHEMA = "extraction-generation-provenance-v1"
EXTRACTION_PROMPT_ARTIFACT_SCHEMA = "extraction-prompt-policy-artifact-v1"
EXTRACTION_POLICY_COMPILER_ID = "extraction-policy-compiler-v1"
EXTRACTION_POLICY_COMPILER_DIGEST = content_digest({
    "compiler_id": EXTRACTION_POLICY_COMPILER_ID,
    "spec_schema": EXTRACTION_POLICY_SPEC_SCHEMA,
    "algorithm": "ordered-rule-text-newline-join-v1",
})


def _require_identifier(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 256
        or not value[0].isalnum()
        or any(not (character.isalnum() or character in "_.:-") for character in value)
    ):
        raise ValueError(f"{name} must be a stable identifier")
    return value


def _require_digest(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be sha256")
    return value


def _strict_mapping(
    value: object,
    fields: set[str],
    name: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"malformed {name}")
    return value


@dataclass(frozen=True, slots=True)
class ExtractionPolicyRule:
    rule_id: str
    text: str
    protected: bool = False

    def __post_init__(self) -> None:
        _require_identifier(self.rule_id, "extraction rule ID")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("extraction rule text must not be empty")
        if "$" in self.text:
            raise ValueError("extraction rule text cannot contain template controls")
        if type(self.protected) is not bool:
            raise TypeError("extraction rule protection flag must be bool")

    def payload(self) -> dict[str, object]:
        return {
            "rule_id": self.rule_id,
            "text": self.text,
            "protected": self.protected,
        }

    @classmethod
    def from_payload(cls, value: object) -> "ExtractionPolicyRule":
        payload = _strict_mapping(
            value,
            {"rule_id", "text", "protected"},
            "extraction policy rule",
        )
        try:
            return cls(**payload)
        except (TypeError, ValueError) as exc:
            raise ValueError("malformed extraction policy rule") from exc


@dataclass(frozen=True, slots=True)
class ExtractionPolicySpec:
    rules: tuple[ExtractionPolicyRule, ...]
    spec_schema: str = EXTRACTION_POLICY_SPEC_SCHEMA
    schema_version: int = EXTRACTION_POLICY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            self.schema_version != EXTRACTION_POLICY_SCHEMA_VERSION
            or self.spec_schema != EXTRACTION_POLICY_SPEC_SCHEMA
        ):
            raise ValueError("unsupported extraction policy spec schema")
        if not isinstance(self.rules, tuple) or not self.rules:
            raise ValueError("extraction policy spec requires ordered rules")
        identifiers = tuple(rule.rule_id for rule in self.rules)
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("extraction policy spec has duplicate rule IDs")

    @property
    def spec_digest(self) -> str:
        return content_digest(self.payload())

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "spec_schema": self.spec_schema,
            "rules": [rule.payload() for rule in self.rules],
        }

    @classmethod
    def from_payload(cls, value: object) -> "ExtractionPolicySpec":
        payload = _strict_mapping(
            value,
            {"schema_version", "spec_schema", "rules"},
            "extraction policy spec",
        )
        rules = payload["rules"]
        if not isinstance(rules, list):
            raise ValueError("extraction policy rules must be a list")
        try:
            return cls(
                tuple(ExtractionPolicyRule.from_payload(rule) for rule in rules),
                spec_schema=payload["spec_schema"],
                schema_version=payload["schema_version"],
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("malformed extraction policy spec") from exc


class ExtractionRuleEditAction(StrEnum):
    ADD = "add"
    REPLACE = "replace"
    DELETE = "delete"


@dataclass(frozen=True, slots=True)
class ExtractionRuleEdit:
    edit_id: str
    action: ExtractionRuleEditAction
    target_rule_id: str | None
    rule: ExtractionPolicyRule | None
    after_rule_id: str | None = None
    edit_schema: str = EXTRACTION_RULE_EDIT_SCHEMA
    schema_version: int = EXTRACTION_POLICY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "action", ExtractionRuleEditAction(self.action))
        if (
            self.schema_version != EXTRACTION_POLICY_SCHEMA_VERSION
            or self.edit_schema != EXTRACTION_RULE_EDIT_SCHEMA
        ):
            raise ValueError("unsupported extraction rule edit schema")
        _require_identifier(self.edit_id, "extraction rule edit ID")
        if self.target_rule_id is not None:
            _require_identifier(self.target_rule_id, "extraction edit target")
        if self.after_rule_id is not None:
            _require_identifier(self.after_rule_id, "extraction edit insertion anchor")
        if self.action == ExtractionRuleEditAction.ADD:
            if self.target_rule_id is not None or self.rule is None:
                raise ValueError("ADD edit requires only a new rule")
            if self.rule.protected:
                raise ValueError("generated edits cannot add protected rules")
        elif self.action == ExtractionRuleEditAction.REPLACE:
            if (
                self.target_rule_id is None
                or self.rule is None
                or self.after_rule_id is not None
                or self.rule.rule_id != self.target_rule_id
            ):
                raise ValueError("REPLACE edit must preserve its target rule ID")
        elif (
            self.target_rule_id is None
            or self.rule is not None
            or self.after_rule_id is not None
        ):
            raise ValueError("DELETE edit requires only a target rule")

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "edit_schema": self.edit_schema,
            "edit_id": self.edit_id,
            "action": self.action.value,
            "target_rule_id": self.target_rule_id,
            "rule": self.rule.payload() if self.rule is not None else None,
            "after_rule_id": self.after_rule_id,
        }

    @classmethod
    def from_payload(cls, value: object) -> "ExtractionRuleEdit":
        payload = _strict_mapping(
            value,
            {
                "schema_version",
                "edit_schema",
                "edit_id",
                "action",
                "target_rule_id",
                "rule",
                "after_rule_id",
            },
            "extraction rule edit",
        )
        try:
            return cls(
                edit_id=payload["edit_id"],
                action=ExtractionRuleEditAction(payload["action"]),
                target_rule_id=payload["target_rule_id"],
                rule=(
                    ExtractionPolicyRule.from_payload(payload["rule"])
                    if payload["rule"] is not None
                    else None
                ),
                after_rule_id=payload["after_rule_id"],
                edit_schema=payload["edit_schema"],
                schema_version=payload["schema_version"],
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("malformed extraction rule edit") from exc


def apply_extraction_rule_edits(
    parent: ExtractionPolicySpec,
    edits: tuple[ExtractionRuleEdit, ...],
) -> ExtractionPolicySpec:
    if not edits:
        raise ValueError("child extraction policy requires rule edits")
    edit_ids = tuple(edit.edit_id for edit in edits)
    if len(edit_ids) != len(set(edit_ids)):
        raise ValueError("extraction policy has duplicate edit IDs")
    rules = list(parent.rules)
    touched: set[str] = set()
    for edit in edits:
        by_id = {rule.rule_id: index for index, rule in enumerate(rules)}
        if edit.action == ExtractionRuleEditAction.ADD:
            rule = edit.rule
            if rule is None:
                raise ValueError("extraction ADD edit has no rule")
            if rule.rule_id in by_id:
                raise ValueError("extraction ADD edit duplicates a rule ID")
            if edit.after_rule_id is None:
                rules.append(rule)
            else:
                if edit.after_rule_id not in by_id:
                    raise ValueError("extraction ADD edit has an unknown anchor")
                rules.insert(by_id[edit.after_rule_id] + 1, rule)
            continue
        target_rule_id = edit.target_rule_id
        if target_rule_id is None:
            raise ValueError("extraction edit has no target")
        if target_rule_id in touched:
            raise ValueError("extraction policy edits one target more than once")
        touched.add(target_rule_id)
        index = by_id.get(target_rule_id)
        if index is None:
            raise ValueError("extraction edit has an unknown target")
        current = rules[index]
        if current.protected:
            raise ValueError("protected extraction rule cannot be edited")
        if edit.action == ExtractionRuleEditAction.REPLACE:
            rule = edit.rule
            if rule is None:
                raise ValueError("extraction replacement edit has no rule")
            if rule.protected != current.protected:
                raise ValueError("extraction edit cannot change rule protection")
            if rule.text == current.text:
                raise ValueError("extraction replacement edit is a no-op")
            rules[index] = rule
        else:
            del rules[index]
    result = ExtractionPolicySpec(tuple(rules))
    if result.spec_digest == parent.spec_digest:
        raise ValueError("extraction policy edits are a no-op")
    return result


def compile_extraction_policy_spec(spec: ExtractionPolicySpec) -> str:
    body = "\n".join(rule.text for rule in spec.rules)
    if not body.strip() or "$" in body:
        raise ValueError("compiled extraction policy body is invalid")
    return body


@dataclass(frozen=True, slots=True)
class ExtractionGenerationProvenance:
    optimizer_model: str
    optimizer_config_digest: str
    training_corpus_id: str
    training_cutoff: str
    proposal_request_digest: str
    completion_digest: str
    usage: RawResourceUsage
    provenance_schema: str = EXTRACTION_GENERATION_PROVENANCE_SCHEMA
    schema_version: int = EXTRACTION_POLICY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            self.schema_version != EXTRACTION_POLICY_SCHEMA_VERSION
            or self.provenance_schema != EXTRACTION_GENERATION_PROVENANCE_SCHEMA
        ):
            raise ValueError("unsupported extraction generation provenance schema")
        for value, name in (
            (self.optimizer_model, "optimizer model"),
            (self.training_corpus_id, "training corpus ID"),
            (self.training_cutoff, "training cutoff"),
        ):
            _require_identifier(value, name)
        for value, name in (
            (self.optimizer_config_digest, "optimizer config digest"),
            (self.proposal_request_digest, "proposal request digest"),
            (self.completion_digest, "optimizer completion digest"),
        ):
            _require_digest(value, name)
        if not isinstance(self.usage, RawResourceUsage):
            raise TypeError("extraction generation usage must be raw resource usage")

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "provenance_schema": self.provenance_schema,
            "optimizer_model": self.optimizer_model,
            "optimizer_config_digest": self.optimizer_config_digest,
            "training_corpus_id": self.training_corpus_id,
            "training_cutoff": self.training_cutoff,
            "proposal_request_digest": self.proposal_request_digest,
            "completion_digest": self.completion_digest,
            "usage": self.usage.to_dict(),
        }

    @classmethod
    def from_payload(cls, value: object) -> "ExtractionGenerationProvenance":
        payload = _strict_mapping(
            value,
            {
                "schema_version",
                "provenance_schema",
                "optimizer_model",
                "optimizer_config_digest",
                "training_corpus_id",
                "training_cutoff",
                "proposal_request_digest",
                "completion_digest",
                "usage",
            },
            "extraction generation provenance",
        )
        usage = _strict_mapping(
            payload["usage"],
            {
                "schema_version",
                "input_tokens",
                "output_tokens",
                "cache_read_tokens",
                "cache_write_tokens",
                "reasoning_tokens",
                "model_requests",
                "retry_count",
                "duration_ms",
                "storage_bytes",
            },
            "extraction generation usage",
        )
        try:
            return cls(
                optimizer_model=payload["optimizer_model"],
                optimizer_config_digest=payload["optimizer_config_digest"],
                training_corpus_id=payload["training_corpus_id"],
                training_cutoff=payload["training_cutoff"],
                proposal_request_digest=payload["proposal_request_digest"],
                completion_digest=payload["completion_digest"],
                usage=RawResourceUsage(**usage),
                provenance_schema=payload["provenance_schema"],
                schema_version=payload["schema_version"],
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("malformed extraction generation provenance") from exc


@dataclass(frozen=True, slots=True)
class ExtractionPromptPolicyArtifact:
    artifact_id: str
    artifact_digest: str
    policy_version: str
    slot_id: str
    slot_contract_digest: str
    parent_artifact_id: str | None
    parent_spec_digest: str | None
    spec: ExtractionPolicySpec
    edits: tuple[ExtractionRuleEdit, ...]
    compiled_body: str
    body_digest: str
    compiler_id: str
    compiler_digest: str
    frozen_wrapper_digest: str
    input_schema_digest: str
    output_schema_digest: str
    required_placeholders: tuple[str, ...]
    model_profile: str
    max_body_chars: int
    source_provenance: str | None
    generation_provenance: ExtractionGenerationProvenance | None
    artifact_schema: str = EXTRACTION_PROMPT_ARTIFACT_SCHEMA
    schema_version: int = EXTRACTION_POLICY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            self.schema_version != EXTRACTION_POLICY_SCHEMA_VERSION
            or self.artifact_schema != EXTRACTION_PROMPT_ARTIFACT_SCHEMA
        ):
            raise ValueError("unsupported extraction prompt artifact schema")
        for value, name in (
            (self.artifact_id, "extraction prompt artifact ID"),
            (self.policy_version, "extraction prompt policy version"),
            (self.slot_id, "extraction prompt slot ID"),
            (self.compiler_id, "extraction prompt compiler ID"),
            (self.model_profile, "extraction prompt model profile"),
        ):
            _require_identifier(value, name)
        if self.parent_artifact_id is not None:
            _require_identifier(self.parent_artifact_id, "extraction prompt parent")
        for value, name in (
            (self.artifact_digest, "extraction prompt artifact digest"),
            (self.slot_contract_digest, "extraction prompt slot digest"),
            (self.body_digest, "extraction prompt body digest"),
            (self.compiler_digest, "extraction prompt compiler digest"),
            (self.frozen_wrapper_digest, "extraction prompt wrapper digest"),
            (self.input_schema_digest, "extraction prompt input schema digest"),
            (self.output_schema_digest, "extraction prompt output schema digest"),
        ):
            _require_digest(value, name)
        if self.parent_spec_digest is not None:
            _require_digest(self.parent_spec_digest, "extraction prompt parent spec")
        if not isinstance(self.spec, ExtractionPolicySpec):
            raise TypeError("extraction prompt spec has the wrong type")
        if not isinstance(self.edits, tuple) or any(
            not isinstance(edit, ExtractionRuleEdit) for edit in self.edits
        ):
            raise TypeError("extraction prompt edits must be a tuple")
        if self.generation_provenance is not None and not isinstance(
            self.generation_provenance,
            ExtractionGenerationProvenance,
        ):
            raise TypeError("extraction prompt generation provenance has the wrong type")
        if (
            not isinstance(self.required_placeholders, tuple)
            or not self.required_placeholders
            or self.required_placeholders != tuple(sorted(set(self.required_placeholders)))
            or any(not isinstance(value, str) or not value for value in self.required_placeholders)
        ):
            raise ValueError("extraction prompt placeholders must be sorted and unique")
        if type(self.max_body_chars) is not int or self.max_body_chars < 1:
            raise ValueError("extraction prompt body limit must be positive")
        if self.compiler_id != EXTRACTION_POLICY_COMPILER_ID or (
            self.compiler_digest != EXTRACTION_POLICY_COMPILER_DIGEST
        ):
            raise ValueError("unknown extraction policy compiler")
        expected_body = compile_extraction_policy_spec(self.spec)
        if self.compiled_body != expected_body:
            raise ValueError("extraction prompt body does not replay from its spec")
        if self.body_digest != text_digest(self.compiled_body):
            raise ValueError("extraction prompt body digest mismatch")
        if len(self.compiled_body) > self.max_body_chars:
            raise ValueError("extraction prompt body is oversized")
        if "$" in self.compiled_body:
            raise ValueError("extraction prompt body cannot contain template controls")
        if self.parent_artifact_id is None:
            if (
                self.parent_spec_digest is not None
                or self.edits
                or self.generation_provenance is not None
                or not isinstance(self.source_provenance, str)
                or not self.source_provenance.strip()
            ):
                raise ValueError("root extraction prompt provenance is invalid")
        elif (
            self.parent_spec_digest is None
            or not self.edits
            or self.generation_provenance is None
            or self.source_provenance is not None
        ):
            raise ValueError("child extraction prompt lineage is incomplete")
        digest = content_digest(self.identity_payload())
        if self.artifact_digest != digest:
            raise ValueError("extraction prompt artifact digest mismatch")
        if self.artifact_id != f"extraction-prompt.{digest[:40]}":
            raise ValueError("extraction prompt artifact ID mismatch")

    @classmethod
    def create_root(
        cls,
        *,
        slot: PromptSlotDescriptor,
        policy_version: str,
        spec: ExtractionPolicySpec,
        max_body_chars: int,
        source_provenance: str,
    ) -> "ExtractionPromptPolicyArtifact":
        return cls._create(
            slot=slot,
            policy_version=policy_version,
            parent_artifact_id=None,
            parent_spec_digest=None,
            spec=spec,
            edits=(),
            max_body_chars=max_body_chars,
            source_provenance=source_provenance,
            generation_provenance=None,
        )

    @classmethod
    def create_child(
        cls,
        *,
        parent: "ExtractionPromptPolicyArtifact",
        policy_version: str,
        edits: tuple[ExtractionRuleEdit, ...],
        generation_provenance: ExtractionGenerationProvenance,
    ) -> "ExtractionPromptPolicyArtifact":
        spec = apply_extraction_rule_edits(parent.spec, edits)
        return cls._create(
            slot=None,
            policy_version=policy_version,
            parent_artifact_id=parent.artifact_id,
            parent_spec_digest=parent.spec.spec_digest,
            spec=spec,
            edits=edits,
            max_body_chars=parent.max_body_chars,
            source_provenance=None,
            generation_provenance=generation_provenance,
            inherited=parent,
        )

    @classmethod
    def _create(
        cls,
        *,
        slot: PromptSlotDescriptor | None,
        policy_version: str,
        parent_artifact_id: str | None,
        parent_spec_digest: str | None,
        spec: ExtractionPolicySpec,
        edits: tuple[ExtractionRuleEdit, ...],
        max_body_chars: int,
        source_provenance: str | None,
        generation_provenance: ExtractionGenerationProvenance | None,
        inherited: "ExtractionPromptPolicyArtifact | None" = None,
    ) -> "ExtractionPromptPolicyArtifact":
        if (slot is None) == (inherited is None):
            raise ValueError("extraction artifact creation requires one slot source")
        values = {
            "policy_version": policy_version,
            "slot_id": slot.slot_id if slot is not None else inherited.slot_id,
            "slot_contract_digest": (
                slot.contract_digest if slot is not None else inherited.slot_contract_digest
            ),
            "parent_artifact_id": parent_artifact_id,
            "parent_spec_digest": parent_spec_digest,
            "spec": spec,
            "edits": edits,
            "compiled_body": compile_extraction_policy_spec(spec),
            "body_digest": text_digest(compile_extraction_policy_spec(spec)),
            "compiler_id": EXTRACTION_POLICY_COMPILER_ID,
            "compiler_digest": EXTRACTION_POLICY_COMPILER_DIGEST,
            "frozen_wrapper_digest": (
                slot.frozen_wrapper_digest
                if slot is not None
                else inherited.frozen_wrapper_digest
            ),
            "input_schema_digest": (
                slot.input_schema_digest if slot is not None else inherited.input_schema_digest
            ),
            "output_schema_digest": (
                slot.output_schema_digest if slot is not None else inherited.output_schema_digest
            ),
            "required_placeholders": (
                slot.required_placeholders
                if slot is not None
                else inherited.required_placeholders
            ),
            "model_profile": (
                slot.model_profile if slot is not None else inherited.model_profile
            ),
            "max_body_chars": max_body_chars,
            "source_provenance": source_provenance,
            "generation_provenance": generation_provenance,
            "artifact_schema": EXTRACTION_PROMPT_ARTIFACT_SCHEMA,
            "schema_version": EXTRACTION_POLICY_SCHEMA_VERSION,
        }
        identity = cls._identity_from_values(values)
        digest = content_digest(identity)
        return cls(
            artifact_id=f"extraction-prompt.{digest[:40]}",
            artifact_digest=digest,
            **values,
        )

    @staticmethod
    def _identity_from_values(values: Mapping[str, object]) -> dict[str, object]:
        result = {
            **{
                key: value
                for key, value in values.items()
                if key not in {"spec", "edits", "generation_provenance"}
            },
            "spec": values["spec"].payload(),
            "edits": [edit.payload() for edit in values["edits"]],
            "generation_provenance": (
                values["generation_provenance"].payload()
                if values["generation_provenance"] is not None
                else None
            ),
        }
        result["required_placeholders"] = list(values["required_placeholders"])
        return result

    def identity_payload(self) -> dict[str, object]:
        return self._identity_from_values({
            "policy_version": self.policy_version,
            "slot_id": self.slot_id,
            "slot_contract_digest": self.slot_contract_digest,
            "parent_artifact_id": self.parent_artifact_id,
            "parent_spec_digest": self.parent_spec_digest,
            "spec": self.spec,
            "edits": self.edits,
            "compiled_body": self.compiled_body,
            "body_digest": self.body_digest,
            "compiler_id": self.compiler_id,
            "compiler_digest": self.compiler_digest,
            "frozen_wrapper_digest": self.frozen_wrapper_digest,
            "input_schema_digest": self.input_schema_digest,
            "output_schema_digest": self.output_schema_digest,
            "required_placeholders": self.required_placeholders,
            "model_profile": self.model_profile,
            "max_body_chars": self.max_body_chars,
            "source_provenance": self.source_provenance,
            "generation_provenance": self.generation_provenance,
            "artifact_schema": self.artifact_schema,
            "schema_version": self.schema_version,
        })

    def payload(self) -> dict[str, object]:
        return {
            **self.identity_payload(),
            "artifact_id": self.artifact_id,
            "artifact_digest": self.artifact_digest,
        }

    def to_prompt_component(
        self,
        slot: PromptSlotDescriptor,
    ) -> PromptComponentArtifact:
        if (
            slot.slot_id != self.slot_id
            or slot.contract_digest != self.slot_contract_digest
            or slot.frozen_wrapper_digest != self.frozen_wrapper_digest
            or slot.input_schema_digest != self.input_schema_digest
            or slot.output_schema_digest != self.output_schema_digest
            or slot.required_placeholders != self.required_placeholders
            or slot.model_profile != self.model_profile
        ):
            raise ValueError("extraction prompt artifact differs from its runtime slot")
        return PromptComponentArtifact.create(
            slot=slot,
            version=self.policy_version,
            policy_body=self.compiled_body,
            source_provenance=f"extraction-policy:{self.artifact_id}",
        )

    @classmethod
    def from_payload(cls, value: object) -> "ExtractionPromptPolicyArtifact":
        fields = {
            "schema_version",
            "artifact_schema",
            "artifact_id",
            "artifact_digest",
            "policy_version",
            "slot_id",
            "slot_contract_digest",
            "parent_artifact_id",
            "parent_spec_digest",
            "spec",
            "edits",
            "compiled_body",
            "body_digest",
            "compiler_id",
            "compiler_digest",
            "frozen_wrapper_digest",
            "input_schema_digest",
            "output_schema_digest",
            "required_placeholders",
            "model_profile",
            "max_body_chars",
            "source_provenance",
            "generation_provenance",
        }
        payload = _strict_mapping(value, fields, "extraction prompt policy artifact")
        edits = payload["edits"]
        placeholders = payload["required_placeholders"]
        if not isinstance(edits, list) or not isinstance(placeholders, list):
            raise ValueError("extraction prompt artifact collections must be lists")
        try:
            return cls(
                artifact_id=payload["artifact_id"],
                artifact_digest=payload["artifact_digest"],
                policy_version=payload["policy_version"],
                slot_id=payload["slot_id"],
                slot_contract_digest=payload["slot_contract_digest"],
                parent_artifact_id=payload["parent_artifact_id"],
                parent_spec_digest=payload["parent_spec_digest"],
                spec=ExtractionPolicySpec.from_payload(payload["spec"]),
                edits=tuple(ExtractionRuleEdit.from_payload(edit) for edit in edits),
                compiled_body=payload["compiled_body"],
                body_digest=payload["body_digest"],
                compiler_id=payload["compiler_id"],
                compiler_digest=payload["compiler_digest"],
                frozen_wrapper_digest=payload["frozen_wrapper_digest"],
                input_schema_digest=payload["input_schema_digest"],
                output_schema_digest=payload["output_schema_digest"],
                required_placeholders=tuple(placeholders),
                model_profile=payload["model_profile"],
                max_body_chars=payload["max_body_chars"],
                source_provenance=payload["source_provenance"],
                generation_provenance=(
                    ExtractionGenerationProvenance.from_payload(
                        payload["generation_provenance"]
                    )
                    if payload["generation_provenance"] is not None
                    else None
                ),
                artifact_schema=payload["artifact_schema"],
                schema_version=payload["schema_version"],
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("malformed extraction prompt policy artifact") from exc


def serialize_extraction_prompt_artifact(
    artifact: ExtractionPromptPolicyArtifact,
) -> str:
    return canonical_json(artifact.payload())
