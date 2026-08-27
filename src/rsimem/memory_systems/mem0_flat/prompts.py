"""Versioned prompt artifacts derived from the pinned MemBase Mem0 baseline."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from string import Template
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

from ...lifecycle import RawResourceUsage


PROMPT_CONTRACT_SCHEMA_VERSION = 1
MEMBASE_REPOSITORY = "https://github.com/zjunlp/MemBase"
MEMBASE_COMMIT = "d2aca6c7abcb1d67b331586cb834495d037fa3a6"
MEMBASE_PROMPT_PATH = "membase/baselines/mem0/configs/prompts.py"
MEMBASE_SOURCE_DIGEST = "bf92192da5033a6793531d55d87945d8bf8728517e0c0c0690e83cb0e0042849"
MEMBASE_LICENSE = "MIT"
MEMBASE_ATTRIBUTION = "Copyright (c) 2026 ZJUNLP"

_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_MODEL_PROFILE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}$")


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _digest_value(value: object) -> str:
    return _digest_text(_canonical_json(value))


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError("prompt schemas must contain JSON-compatible values")


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain_json(item) for item in value]
    return value


def _validate_schema(schema: Mapping[str, Any], name: str) -> None:
    if schema.get("type") != "object":
        raise ValueError(f"{name} schema must describe an object")
    properties = schema.get("properties")
    required = schema.get("required")
    if not isinstance(properties, Mapping) or not properties:
        raise ValueError(f"{name} schema requires properties")
    if not isinstance(required, (list, tuple)) or not required:
        raise ValueError(f"{name} schema requires required fields")
    if any(not isinstance(item, str) or not item for item in required):
        raise ValueError(f"{name} schema required fields must be names")
    if not set(required).issubset(properties):
        raise ValueError(f"{name} schema required fields must exist in properties")


@dataclass(frozen=True, slots=True)
class PromptSourceProvenance:
    repository: str
    commit: str
    path: str
    upstream_symbol: str
    source_digest: str
    license: str
    attribution: str
    local_modification_digest: str
    excluded_instruction_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        required = (
            self.repository,
            self.commit,
            self.path,
            self.upstream_symbol,
            self.license,
            self.attribution,
        )
        if any(not value.strip() for value in required):
            raise ValueError("prompt source provenance is incomplete")
        if len(self.commit) != 40 or not re.fullmatch(r"[0-9a-f]{40}", self.commit):
            raise ValueError("prompt source commit must be a full git SHA")
        if not _DIGEST.fullmatch(self.source_digest):
            raise ValueError("prompt source digest must be sha256")
        if not _DIGEST.fullmatch(self.local_modification_digest):
            raise ValueError("local prompt modification digest must be sha256")
        if any(not _IDENTIFIER.fullmatch(code) for code in self.excluded_instruction_codes):
            raise ValueError("excluded prompt instructions require stable codes")

    def manifest_record(self) -> dict[str, object]:
        return {
            "repository": self.repository,
            "commit": self.commit,
            "path": self.path,
            "upstream_symbol": self.upstream_symbol,
            "source_digest": self.source_digest,
            "license": self.license,
            "attribution": self.attribution,
            "local_modification_digest": self.local_modification_digest,
            "excluded_instruction_codes": list(self.excluded_instruction_codes),
        }


@dataclass(frozen=True, slots=True)
class PromptArtifact:
    prompt_id: str
    version: str
    template_digest: str
    input_schema: Mapping[str, Any]
    output_schema: Mapping[str, Any]
    model_profile: str
    policy_version: str
    source: PromptSourceProvenance
    schema_version: int = PROMPT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PROMPT_CONTRACT_SCHEMA_VERSION:
            raise ValueError("unsupported prompt artifact schema version")
        if not _IDENTIFIER.fullmatch(self.prompt_id):
            raise ValueError("prompt_id must be a stable machine identifier")
        if not _IDENTIFIER.fullmatch(self.version):
            raise ValueError("prompt version must be a stable machine identifier")
        if not _DIGEST.fullmatch(self.template_digest):
            raise ValueError("prompt template_digest must be sha256")
        if not _MODEL_PROFILE.fullmatch(self.model_profile):
            raise ValueError("prompt model_profile must be a stable identifier")
        if not _IDENTIFIER.fullmatch(self.policy_version):
            raise ValueError("prompt policy_version must be a stable identifier")
        _validate_schema(self.input_schema, "prompt input")
        _validate_schema(self.output_schema, "prompt output")
        object.__setattr__(self, "input_schema", _freeze_json(self.input_schema))
        object.__setattr__(self, "output_schema", _freeze_json(self.output_schema))

    def manifest_record(self) -> dict[str, object]:
        """Return the prompt identity safe for manifests and observer evidence."""

        return {
            "schema_version": self.schema_version,
            "prompt_id": self.prompt_id,
            "version": self.version,
            "template_digest": self.template_digest,
            "input_schema": _plain_json(self.input_schema),
            "output_schema": _plain_json(self.output_schema),
            "model_profile": self.model_profile,
            "policy_version": self.policy_version,
            "source": self.source.manifest_record(),
        }


@dataclass(frozen=True, slots=True)
class RenderedPrompt:
    artifact: PromptArtifact
    render_id: str
    input_digest: str
    text: str = field(repr=False)

    def __post_init__(self) -> None:
        if not _IDENTIFIER.fullmatch(self.render_id):
            raise ValueError("render_id must be a stable machine identifier")
        if not _DIGEST.fullmatch(self.input_digest):
            raise ValueError("rendered prompt input_digest must be sha256")
        if not self.text.strip():
            raise ValueError("rendered prompt text must not be empty")

    def observer_evidence(self) -> dict[str, object]:
        return {
            "prompt_id": self.artifact.prompt_id,
            "prompt_version": self.artifact.version,
            "template_digest": self.artifact.template_digest,
            "model_profile": self.artifact.model_profile,
            "policy_version": self.artifact.policy_version,
            "render_id": self.render_id,
            "input_digest": self.input_digest,
        }


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    artifact: PromptArtifact
    template: str = field(repr=False)

    def __post_init__(self) -> None:
        if not self.template.strip():
            raise ValueError("prompt template must not be empty")
        if _digest_text(self.template) != self.artifact.template_digest:
            raise ValueError("prompt artifact digest does not match template")
        placeholders = {
            match.group("named") or match.group("braced")
            for match in Template.pattern.finditer(self.template)
            if match.group("named") or match.group("braced")
        }
        required = set(self.artifact.input_schema["required"])
        if placeholders != required:
            raise ValueError("prompt placeholders must equal required input fields")

    def render(self, values: Mapping[str, Any]) -> RenderedPrompt:
        required = set(self.artifact.input_schema["required"])
        if set(values) != required:
            raise ValueError("render values must exactly match required input fields")
        encoded = {key: _canonical_json(values[key]) for key in sorted(values)}
        input_digest = _digest_value(encoded)
        text = Template(self.template).substitute(encoded)
        render_identity = {
            'prompt_id': self.artifact.prompt_id,
            'version': self.artifact.version,
            'template_digest': self.artifact.template_digest,
            'input_digest': input_digest,
        }
        render_id = f"render.{_digest_value(render_identity)[:40]}"
        return RenderedPrompt(self.artifact, render_id, input_digest, text)


@dataclass(frozen=True, slots=True)
class CompletionResult:
    completion_id: str
    render_id: str
    output_text: str = field(repr=False)
    usage: RawResourceUsage = RawResourceUsage()

    def __post_init__(self) -> None:
        if not _IDENTIFIER.fullmatch(self.completion_id):
            raise ValueError("completion_id must be a stable machine identifier")
        if not _IDENTIFIER.fullmatch(self.render_id):
            raise ValueError("completion render_id must be a stable machine identifier")

    def observer_evidence(self) -> dict[str, object]:
        return {
            "completion_id": self.completion_id,
            "render_id": self.render_id,
            "output_digest": _digest_text(self.output_text),
            "usage": self.usage.to_dict(),
        }


@runtime_checkable
class CompletionClient(Protocol):
    def complete(self, prompt: RenderedPrompt) -> CompletionResult: ...


class FakeCompletionClient:
    """Deterministic completion client for prompt contract tests only."""

    def __init__(
        self,
        responses: Mapping[str, str | Exception | Callable[[RenderedPrompt], str]],
        *,
        usage: RawResourceUsage = RawResourceUsage(model_requests=1),
    ) -> None:
        self._responses = dict(responses)
        self._usage = usage
        self._calls: list[dict[str, object]] = []

    @property
    def calls(self) -> tuple[dict[str, object], ...]:
        return tuple(self._calls)

    def complete(self, prompt: RenderedPrompt) -> CompletionResult:
        self._calls.append(prompt.observer_evidence())
        try:
            configured = self._responses[prompt.artifact.prompt_id]
        except KeyError as exc:
            raise KeyError("fake completion has no response for prompt artifact") from exc
        if isinstance(configured, Exception):
            raise configured
        output = configured(prompt) if callable(configured) else configured
        completion_identity = {
            'render_id': prompt.render_id,
            'output_digest': _digest_text(output),
        }
        completion_id = f"completion.{_digest_value(completion_identity)[:40]}"
        return CompletionResult(completion_id, prompt.render_id, output, self._usage)


_OBJECT = {"type": "object", "additionalProperties": False}

FACT_EXTRACTION_INPUT_SCHEMA = {
    **_OBJECT,
    "properties": {
        "source_messages": {"type": "array", "items": {"type": "object"}},
        "exit_evidence": {"type": "object"},
    },
    "required": ["source_messages", "exit_evidence"],
}
FACT_EXTRACTION_OUTPUT_SCHEMA = {
    **_OBJECT,
    "properties": {
        "facts": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["facts"],
}
INTERNAL_OPERATION_INPUT_SCHEMA = {
    **_OBJECT,
    "properties": {
        "new_facts": {"type": "array", "items": {"type": "object"}},
        "related_memories": {"type": "array", "items": {"type": "object"}},
    },
    "required": ["new_facts", "related_memories"],
}
INTERNAL_OPERATION_OUTPUT_SCHEMA = {
    **_OBJECT,
    "properties": {
        "operations": {"type": "array", "items": {"type": "object"}},
    },
    "required": ["operations"],
}
RETRIEVAL_SCORER_INPUT_SCHEMA = {
    **_OBJECT,
    "properties": {
        "query_features": {"type": "object"},
        "candidate_features": {"type": "array", "items": {"type": "object"}},
    },
    "required": ["query_features", "candidate_features"],
}
RETRIEVAL_SCORER_OUTPUT_SCHEMA = {
    **_OBJECT,
    "properties": {
        "scores": {"type": "array", "items": {"type": "object"}},
    },
    "required": ["scores"],
}

_FACT_EXTRACTION_TEMPLATE = """Extract minimal, durable semantic memory candidates from the completed experience.
Keep only stable facts, preferences, rules, or constraints that can help a future task.
Do not copy transcripts, tool payloads, failed attempts, credentials, machine paths, or unresolved claims.
Use only the supplied source and deterministic exit evidence. Return JSON matching the output schema.

Source messages JSON:
$source_messages

Exit evidence JSON:
$exit_evidence
"""

_INTERNAL_OPERATION_TEMPLATE = """Compare new semantic facts with trusted related-memory candidates.
For each fact choose ADD, UPDATE, DELETE, or NONE. UPDATE and DELETE may reference only candidate IDs supplied below.
Use NONE for duplicates. Do not invent backend names, artifact IDs, revisions, or candidate IDs.
Return JSON matching the output schema.

New facts JSON:
$new_facts

Related-memory candidates JSON:
$related_memories
"""

_RETRIEVAL_SCORER_TEMPLATE = """Score host-neutral semantic candidates for relevance to the supplied query features.
Use only candidate IDs supplied below. Return bounded scores in JSON matching the output schema.

Query features JSON:
$query_features

Candidate features JSON:
$candidate_features
"""


def _source(
    upstream_symbol: str,
    template: str,
    *,
    excluded_instruction_codes: tuple[str, ...] = (),
) -> PromptSourceProvenance:
    template_digest = _digest_text(template)
    local_modification_digest = _digest_value({
        "source_digest": MEMBASE_SOURCE_DIGEST,
        "upstream_symbol": upstream_symbol,
        "template_digest": template_digest,
        "excluded_instruction_codes": excluded_instruction_codes,
    })
    return PromptSourceProvenance(
        repository=MEMBASE_REPOSITORY,
        commit=MEMBASE_COMMIT,
        path=MEMBASE_PROMPT_PATH,
        upstream_symbol=upstream_symbol,
        source_digest=MEMBASE_SOURCE_DIGEST,
        license=MEMBASE_LICENSE,
        attribution=MEMBASE_ATTRIBUTION,
        local_modification_digest=local_modification_digest,
        excluded_instruction_codes=excluded_instruction_codes,
    )


def _prompt(
    prompt_id: str,
    template: str,
    input_schema: Mapping[str, Any],
    output_schema: Mapping[str, Any],
    upstream_symbol: str,
    *,
    model_profile: str,
    policy_version: str,
    version: str = "v1",
    excluded_instruction_codes: tuple[str, ...] = (),
) -> PromptTemplate:
    artifact = PromptArtifact(
        prompt_id=prompt_id,
        version=version,
        template_digest=_digest_text(template),
        input_schema=input_schema,
        output_schema=output_schema,
        model_profile=model_profile,
        policy_version=policy_version,
        source=_source(
            upstream_symbol,
            template,
            excluded_instruction_codes=excluded_instruction_codes,
        ),
    )
    return PromptTemplate(artifact, template)


def build_prompt_catalog(
    *,
    model_profile: str = "semantic-ingestion-default-v1",
    policy_version: str = "mem0-flat-v1",
) -> tuple[PromptTemplate, ...]:
    return (
        _prompt(
            "mem0-flat.fact-extraction",
            _FACT_EXTRACTION_TEMPLATE,
            FACT_EXTRACTION_INPUT_SCHEMA,
            FACT_EXTRACTION_OUTPUT_SCHEMA,
            "FACT_RETRIEVAL_PROMPT",
            model_profile=model_profile,
            policy_version=policy_version,
            excluded_instruction_codes=(
                "answer-source-fabrication",
                "dynamic-wall-clock-date",
                "prompt-secrecy-answer-rule",
            ),
        ),
        _prompt(
            "mem0-flat.internal-operation",
            _INTERNAL_OPERATION_TEMPLATE,
            INTERNAL_OPERATION_INPUT_SCHEMA,
            INTERNAL_OPERATION_OUTPUT_SCHEMA,
            "DEFAULT_UPDATE_MEMORY_PROMPT",
            model_profile=model_profile,
            policy_version=policy_version,
        ),
        _prompt(
            "mem0-flat.semantic-retrieval-scorer",
            _RETRIEVAL_SCORER_TEMPLATE,
            RETRIEVAL_SCORER_INPUT_SCHEMA,
            RETRIEVAL_SCORER_OUTPUT_SCHEMA,
            "DEFAULT_UPDATE_MEMORY_PROMPT",
            model_profile=model_profile,
            policy_version=policy_version,
        ),
    )


(
    FACT_EXTRACTION_PROMPT,
    INTERNAL_OPERATION_PROMPT,
    SEMANTIC_RETRIEVAL_SCORER_PROMPT,
) = build_prompt_catalog()


_POLICY_FACT_EXTRACTION_TEMPLATE = """Extract minimal durable semantic memory candidates from a completed agent experience.
Keep only user-supplied facts, preferences, rules, or constraints that can help a future task.
Assistant acknowledgements, tool output, transcripts, failures, credentials, machine paths, temporary requests, unresolved claims, and example text are not memories.
Each candidate must be independently understandable and contain no conversation wrapper.
Use only the supplied source messages and deterministic exit evidence. Return exactly one JSON object with a facts string list and no other fields.

Source messages JSON:
$source_messages

Exit evidence JSON:
$exit_evidence
"""

_POLICY_INTERNAL_OPERATION_TEMPLATE = """Compare each new durable semantic fact with the trusted related-memory candidates.
Return exactly one operation for every fact. Choose ADD for new information, UPDATE for a more current replacement, DELETE only when the new evidence explicitly withdraws an existing memory, and NONE for duplicates.
UPDATE and DELETE may reference only a supplied candidate_id with mutable=true. ADD and NONE must use null candidate_id.
Do not invent backend names, artifact IDs, revisions, candidate IDs, facts, or text. Return exactly one JSON object with an operations list. Every operation object must contain only fact_index, action, and candidate_id.

New facts JSON:
$new_facts

Related-memory candidates JSON:
$related_memories
"""


def build_policy_prompt_catalog(
    *,
    model_profile: str = "semantic-ingestion-default-v1",
    policy_version: str = "mem0-flat-v2",
) -> tuple[PromptTemplate, PromptTemplate]:
    return (
        _prompt(
            "mem0-flat.fact-extraction",
            _POLICY_FACT_EXTRACTION_TEMPLATE,
            FACT_EXTRACTION_INPUT_SCHEMA,
            FACT_EXTRACTION_OUTPUT_SCHEMA,
            "FACT_RETRIEVAL_PROMPT",
            model_profile=model_profile,
            policy_version=policy_version,
            version="v2",
            excluded_instruction_codes=(
                "answer-source-fabrication",
                "assistant-claim-extraction",
                "dynamic-wall-clock-date",
                "prompt-secrecy-answer-rule",
            ),
        ),
        _prompt(
            "mem0-flat.internal-operation",
            _POLICY_INTERNAL_OPERATION_TEMPLATE,
            INTERNAL_OPERATION_INPUT_SCHEMA,
            INTERNAL_OPERATION_OUTPUT_SCHEMA,
            "DEFAULT_UPDATE_MEMORY_PROMPT",
            model_profile=model_profile,
            policy_version=policy_version,
            version="v2",
        ),
    )


(
    POLICY_FACT_EXTRACTION_PROMPT,
    POLICY_INTERNAL_OPERATION_PROMPT,
) = build_policy_prompt_catalog()
