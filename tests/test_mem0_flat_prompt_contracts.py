from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from rsimem.lifecycle import RawResourceUsage
from rsimem.memory_systems.mem0_flat import (
    FACT_EXTRACTION_PROMPT,
    INTERNAL_OPERATION_PROMPT,
    MEMBASE_COMMIT,
    MEMBASE_LICENSE,
    MEMBASE_PROMPT_PATH,
    MEMBASE_SOURCE_DIGEST,
    PROMPT_CONTRACT_SCHEMA_VERSION,
    SEMANTIC_RETRIEVAL_SCORER_PROMPT,
    POLICY_FACT_EXTRACTION_PROMPT,
    POLICY_INTERNAL_OPERATION_PROMPT,
    FakeCompletionClient,
    PromptTemplate,
    build_prompt_catalog,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def test_catalog_uses_one_prompt_contract_and_pinned_membase_provenance() -> None:
    catalog = build_prompt_catalog(
        model_profile="fixture-model-v1",
        policy_version="fixture-policy-v1",
    )
    assert len(catalog) == 3
    assert len({item.artifact.prompt_id for item in catalog}) == 3
    assert {item.artifact.prompt_id for item in catalog} == {
        "mem0-flat.fact-extraction",
        "mem0-flat.internal-operation",
        "mem0-flat.semantic-retrieval-scorer",
    }
    for prompt in catalog:
        artifact = prompt.artifact
        assert artifact.version == "v1"
        assert artifact.model_profile == "fixture-model-v1"
        assert artifact.policy_version == "fixture-policy-v1"
        assert artifact.template_digest == _sha(prompt.template)
        assert artifact.input_schema["type"] == "object"
        assert artifact.output_schema["type"] == "object"
        assert artifact.source.commit == MEMBASE_COMMIT
        assert artifact.source.path == MEMBASE_PROMPT_PATH
        assert artifact.source.source_digest == MEMBASE_SOURCE_DIGEST
        assert artifact.source.license == MEMBASE_LICENSE
        assert len(artifact.source.local_modification_digest) == 64


def test_local_fact_prompt_excludes_non_construction_answer_instructions() -> None:
    source = FACT_EXTRACTION_PROMPT.artifact.source
    assert source.excluded_instruction_codes == (
        "answer-source-fabrication",
        "dynamic-wall-clock-date",
        "prompt-secrecy-answer-rule",
    )
    lowered = FACT_EXTRACTION_PROMPT.template.lower()
    assert "publicly available sources" not in lowered
    assert "today's date" not in lowered
    assert "don't reveal your prompt" not in lowered


def test_template_digest_and_required_artifact_fields_fail_closed() -> None:
    artifact = FACT_EXTRACTION_PROMPT.artifact
    assert PROMPT_CONTRACT_SCHEMA_VERSION == 1
    with pytest.raises(ValueError, match="unsupported prompt artifact schema"):
        replace(artifact, schema_version=2)
    with pytest.raises(ValueError, match="digest does not match"):
        PromptTemplate(artifact, FACT_EXTRACTION_PROMPT.template + "changed")
    with pytest.raises(ValueError, match="input schema"):
        replace(artifact, input_schema={})
    with pytest.raises(ValueError, match="model_profile"):
        replace(artifact, model_profile="")
    with pytest.raises(ValueError, match="prompt version"):
        replace(artifact, version="")
    with pytest.raises(ValueError, match="policy_version"):
        replace(artifact, policy_version="")


def test_rendered_prompt_and_completion_text_never_enter_observer_evidence() -> None:
    sentinel = "SENTINEL_RAW_PROMPT_CONTENT_7f3a"
    rendered = FACT_EXTRACTION_PROMPT.render({
        "source_messages": [{"role": "user", "content": sentinel}],
        "exit_evidence": {"safe_to_evict": True},
    })
    assert sentinel in rendered.text

    client = FakeCompletionClient(
        {FACT_EXTRACTION_PROMPT.artifact.prompt_id: '{"facts": ["private result"]}'},
        usage=RawResourceUsage(input_tokens=9, output_tokens=4, model_requests=1),
    )
    result = client.complete(rendered)
    observer_payload = json.dumps({
        "artifact": FACT_EXTRACTION_PROMPT.artifact.manifest_record(),
        "render": rendered.observer_evidence(),
        "completion": result.observer_evidence(),
        "calls": client.calls,
    }, sort_keys=True)
    assert sentinel not in observer_payload
    assert "private result" not in observer_payload
    assert "rendered_prompt" not in observer_payload
    assert result.usage.model_requests == 1


def test_render_contract_requires_exact_schema_inputs_and_fake_errors_propagate() -> None:
    with pytest.raises(ValueError, match="exactly match"):
        FACT_EXTRACTION_PROMPT.render({"source_messages": []})
    with pytest.raises(ValueError, match="exactly match"):
        FACT_EXTRACTION_PROMPT.render({
            "source_messages": [],
            "exit_evidence": {},
            "backend": "forged",
        })

    rendered = INTERNAL_OPERATION_PROMPT.render({
        "new_facts": [],
        "related_memories": [],
    })
    client = FakeCompletionClient({rendered.artifact.prompt_id: TimeoutError("fixture")})
    with pytest.raises(TimeoutError, match="fixture"):
        client.complete(rendered)


def test_default_prompt_constants_are_stable_catalog_members() -> None:
    assert (
        FACT_EXTRACTION_PROMPT,
        INTERNAL_OPERATION_PROMPT,
        SEMANTIC_RETRIEVAL_SCORER_PROMPT,
    ) == build_prompt_catalog()


def test_policy_v2_prompts_preserve_v1_contract_and_narrow_memory_ownership() -> None:
    fact = POLICY_FACT_EXTRACTION_PROMPT
    operation = POLICY_INTERNAL_OPERATION_PROMPT
    assert fact.artifact.version == operation.artifact.version == "v2"
    assert fact.artifact.policy_version == operation.artifact.policy_version == "mem0-flat-v2"
    assert fact.artifact.input_schema == FACT_EXTRACTION_PROMPT.artifact.input_schema
    assert operation.artifact.output_schema == INTERNAL_OPERATION_PROMPT.artifact.output_schema
    lowered = fact.template.lower()
    assert "user-supplied facts" in lowered
    assert "assistant acknowledgements" in lowered
    assert "temporary requests" in lowered
    assert "publicly available sources" not in lowered
    assert "answer-source-fabrication" in fact.artifact.source.excluded_instruction_codes
    assert "assistant-claim-extraction" in fact.artifact.source.excluded_instruction_codes
