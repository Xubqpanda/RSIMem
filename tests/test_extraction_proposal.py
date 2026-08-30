from __future__ import annotations

import json
from pathlib import Path

import pytest

from rsimem.memory.extraction_optimizer_store import JsonExtractionOptimizerCorpusStore
from rsimem.memory.extraction_prompt_optimizer import (
    CapturedExtractionOptimizerClient,
    ExtractionOptimizerDecision,
)
from rsimem.memory.extraction_optimizer_contracts import (
    EXTRACTION_OPTIMIZER_OUTPUT_SCHEMA_DIGEST,
    FROZEN_EXTRACTION_OPTIMIZER_CONFIG,
    build_extraction_optimizer_gate_request,
    build_extraction_optimizer_request,
)
from rsimem.memory.extraction_optimizer_corpus import (
    EXTRACTION_OPTIMIZER_CORPUS_SCHEMA_VERSION,
    PROCESS_SIGNAL_GATE_NOT_BOUND,
    OptimizerCorpusRetention,
    OptimizerCorpusSplit,
)
from rsimem.extraction_proposal import (
    _DeferredExtractionOptimizerClient,
    _read_api_key_file,
    main,
    prepare_extraction_proposal,
    prepare_pure_extraction_proposal,
)
from rsimem.memory.pure_extraction import JsonPureExtractionOptimizerCorpusStore
from rsimem.memory.pure_extraction_optimizer import (
    JsonPureExtractionOptimizerContentCaptureStore,
)
from rsimem.memory.prompt_components import canonical_json
from rsimem.memory.evidence_planes import EvidencePlane, EvidenceSourceKind
from rsimem.memory.revocation import JsonRevocationRegistry, RevocationEntry
from test_extraction_optimizer_contracts import _multi_corpus, _parent, _proposal_output
from test_pure_extraction_optimizer import _corpus as _pure_corpus
from test_pure_extraction_optimizer import _fixture as _pure_fixture


def _store(tmp_path: Path, corpus):
    owner = tmp_path / "owner"
    attempt = owner / "attempt.multi-v1"
    store = JsonExtractionOptimizerCorpusStore(
        attempt,
        owner_controlled_root=owner,
        attempt_id="attempt.multi-v1",
        split=OptimizerCorpusSplit.TRAIN,
    )
    store.write(corpus)
    return store, owner


def test_optimizer_key_reader_accepts_operator_config_without_endpoint_metadata(
    tmp_path: Path,
) -> None:
    path = tmp_path / "api_key.md"
    path.write_text('key: "secret-token"\nurl: https://provider.invalid/v1\n')
    assert _read_api_key_file(path) == "secret-token"

    plain = tmp_path / "plain.key"
    plain.write_text("plain-token\n")
    assert _read_api_key_file(plain) == "plain-token"


def test_proposal_persists_candidate_and_request_without_recalling_provider(
    tmp_path: Path,
) -> None:
    corpus = _multi_corpus(("useful", "useful"))
    store, owner = _store(tmp_path, corpus)
    client = CapturedExtractionOptimizerClient(_proposal_output)
    result = prepare_extraction_proposal(
        corpus_store=store,
        output_root=owner / "proposal",
        client=client,
    )

    assert result.decision == ExtractionOptimizerDecision.PROPOSE
    assert result.candidate is not None
    assert len(client.requests) == 1
    candidate_payload = json.loads(
        (owner / "proposal" / "candidate-artifact.json").read_text()
    )
    assert candidate_payload["artifact_id"] == result.candidate.artifact_id
    request_payload = json.loads(
        (owner / "proposal" / "optimizer-request.json").read_text()
    )
    assert request_payload["requestDigest"] == result.request.request_digest
    feasibility_payload = json.loads(
        (owner / "proposal" / "feasibility-hypothesis.json").read_text()
    )
    assert feasibility_payload["decision"] == "PROPOSE"
    assert feasibility_payload["candidate_artifact_id"] == result.candidate.artifact_id
    assert all(
        path.stat().st_mode & 0o777 == 0o600
        for path in (owner / "proposal").glob("*.json")
    )


def test_no_signal_proposal_does_not_call_provider_or_write_candidate(
    tmp_path: Path,
) -> None:
    corpus = _multi_corpus(("unresolved", "censored"), ownerships=(
        "unresolved",
        "unresolved",
    ))
    store, owner = _store(tmp_path, corpus)
    client = CapturedExtractionOptimizerClient(
        json.dumps({"decision": "PROPOSE"})
    )
    result = prepare_extraction_proposal(
        corpus_store=store,
        output_root=owner / "proposal",
        client=client,
    )

    assert result.decision == ExtractionOptimizerDecision.NO_PROPOSAL
    assert result.candidate is None
    assert client.requests == []
    assert not (owner / "proposal" / "candidate-artifact.json").exists()
    assert result.request.provider_eligible is False
    assert '"request_mode":"deterministic_signal_gate"' in (
        result.request.input_json
    )
    assert "source_messages" not in result.request.input_json
    payload = json.loads((owner / "proposal" / "optimizer-result.json").read_text())
    assert payload["reasonCodes"] == ["no_actionable_extraction_signal"]
    feasibility = json.loads(
        (owner / "proposal" / "feasibility-hypothesis.json").read_text()
    )
    assert feasibility["decision"] == "NO_PROPOSAL"
    assert feasibility["candidate_artifact_id"] is None


def test_pure_proposal_preparation_uses_pure_stores_without_legacy_projection(
    tmp_path: Path,
) -> None:
    parent, example, capture = _pure_fixture()
    owner = tmp_path / "owner"
    corpus_path = owner / "pure-corpus.json"
    capture_path = owner / "pure-captures.jsonl"
    corpus_store = JsonPureExtractionOptimizerCorpusStore(corpus_path)
    corpus_store.write(_pure_corpus(example))
    capture_store = JsonPureExtractionOptimizerContentCaptureStore(capture_path)
    capture_store.append(capture)
    client = CapturedExtractionOptimizerClient("{")
    result = prepare_pure_extraction_proposal(
        corpus_store=corpus_store,
        capture_store=capture_store,
        output_root=owner / "proposal",
        client=client,
    )
    # The frozen config requires two independent actionable logical cases, so
    # this one-case fixture must produce a deterministic gate without a model
    # call while still persisting the pure hypothesis projection.
    assert result.decision == ExtractionOptimizerDecision.NO_PROPOSAL
    assert result.reason_codes == ("insufficient_actionable_extraction_signal",)
    assert client.requests == []
    hypothesis = json.loads(
        (owner / "proposal" / "feasibility-hypothesis.json").read_text()
    )
    assert hypothesis["schema"] == "rsimem-pure-extraction-optimizer-hypothesis-v1"
    assert hypothesis["corpus_id"] == corpus_store.read().corpus_id


def test_rejected_candidate_is_persisted_without_deployable_artifact(
    tmp_path: Path,
) -> None:
    corpus = _multi_corpus(("missed", "missed"))
    store, owner = _store(tmp_path, corpus)
    client = CapturedExtractionOptimizerClient(
        lambda request: _proposal_output(
            request,
            rule_text="Remember Project Apollo as a durable preference.",
        )
    )

    result = prepare_extraction_proposal(
        corpus_store=store,
        output_root=owner / "proposal-rejected",
        client=client,
    )

    assert result.decision == ExtractionOptimizerDecision.NO_PROPOSAL
    assert result.reason_codes == ("candidate_corpus_value",)
    assert result.completion_id is not None
    assert result.usage == client.usage
    output = owner / "proposal-rejected"
    persisted = json.loads((output / "optimizer-result.json").read_text())
    assert persisted["decision"] == "NO_PROPOSAL"
    assert persisted["reasonCodes"] == ["candidate_corpus_value"]
    assert persisted["completionId"] == result.completion_id
    assert not (output / "candidate-artifact.json").exists()
    hypothesis = json.loads(
        (output / "feasibility-hypothesis.json").read_text()
    )
    assert hypothesis["decision"] == "NO_PROPOSAL"
    assert hypothesis["candidate_artifact_id"] is None


def test_proposal_revocation_registry_rejects_parent_before_provider_call(
    tmp_path: Path,
) -> None:
    corpus = _multi_corpus(("useful", "useful"))
    store, owner = _store(tmp_path, corpus)
    parent = _parent()
    registry = JsonRevocationRegistry(tmp_path / "revocations.jsonl")
    registry.initialize()
    registry.append(RevocationEntry.create(
        artifact_id=parent.artifact_id,
        artifact_schema_version=parent.schema_version,
        artifact_digest=parent.artifact_digest,
        evidence_plane=EvidencePlane.PURE_PROCESS,
        evidence_source=EvidenceSourceKind.RUNTIME_OBSERVATION,
        revoked_at="2026-08-30T01:02:03Z",
        reason_code="stale_schema",
    ))
    client = CapturedExtractionOptimizerClient(_proposal_output)
    with pytest.raises(ValueError, match="artifact is revoked"):
        prepare_extraction_proposal(
            corpus_store=store,
            output_root=owner / "proposal-revoked",
            client=client,
            revocation_registry=registry,
        )
    assert client.requests == []


def test_proposal_revocation_registry_rejects_corpus_before_provider_call(
    tmp_path: Path,
) -> None:
    corpus = _multi_corpus(("useful", "useful"))
    store, owner = _store(tmp_path, corpus)
    registry = JsonRevocationRegistry(tmp_path / "revocations-corpus.jsonl")
    registry.initialize()
    registry.append(RevocationEntry.create(
        artifact_id=corpus.corpus_id,
        artifact_schema_version=EXTRACTION_OPTIMIZER_CORPUS_SCHEMA_VERSION,
        artifact_digest=corpus.corpus_digest,
        evidence_plane=EvidencePlane.PURE_PROCESS,
        evidence_source=EvidenceSourceKind.RUNTIME_OBSERVATION,
        revoked_at="2026-08-30T01:02:03Z",
        reason_code="stale_corpus",
    ))
    client = CapturedExtractionOptimizerClient(_proposal_output)
    with pytest.raises(ValueError, match="artifact is revoked"):
        prepare_extraction_proposal(
            corpus_store=store,
            output_root=owner / "proposal-revoked-corpus",
            client=client,
            revocation_registry=registry,
        )
    assert client.requests == []


def test_formal_provider_entry_requires_revocation_registry_before_credentials(
    tmp_path: Path,
) -> None:
    request = build_extraction_optimizer_request(
        _parent(),
        _multi_corpus(("useful", "useful")),
    )
    key_file = tmp_path / "api-key"
    key_file.write_text("should-not-be-read\n", encoding="utf-8")
    client = _DeferredExtractionOptimizerClient(
        key_file,
        "https://coding.tu-zi.com/v1",
    )
    with pytest.raises(ValueError, match="revocation registry"):
        client.complete(request, FROZEN_EXTRACTION_OPTIMIZER_CONFIG)


def test_formal_provider_entry_requires_bound_process_signal_gate(tmp_path: Path) -> None:
    from rsimem.memory.extraction_prompt_optimizer import ExtractionPromptOptimizer

    corpus = _multi_corpus(("useful", "useful"))
    assert corpus.process_signal_gate == PROCESS_SIGNAL_GATE_NOT_BOUND
    registry = JsonRevocationRegistry(tmp_path / "revocations.jsonl")
    registry.initialize()
    client = _DeferredExtractionOptimizerClient(None, "https://coding.tu-zi.com/v1")
    result = ExtractionPromptOptimizer(
        client,
        revocation_registry=registry,
    ).propose(_parent(), corpus)
    assert result.decision == ExtractionOptimizerDecision.NO_PROPOSAL
    assert result.reason_codes == ("process_signal_gate_missing",)


def test_deferred_provider_rejects_gate_request_before_reading_credentials(
    tmp_path: Path,
) -> None:
    corpus = _multi_corpus(("useful", "useful"))
    request = build_extraction_optimizer_gate_request(
        _parent(),
        corpus,
        reason_codes=("process_signal_gate_missing",),
    )
    client = _DeferredExtractionOptimizerClient(
        tmp_path / "missing-api-key",
        "https://coding.tu-zi.com/v1",
    )

    with pytest.raises(ValueError, match="gate request cannot reach the provider"):
        client.complete(request, FROZEN_EXTRACTION_OPTIMIZER_CONFIG)


def test_malformed_completion_is_persisted_without_deployable_artifact(
    tmp_path: Path,
) -> None:
    corpus = _multi_corpus(("missed", "missed"))
    store, owner = _store(tmp_path, corpus)

    def malformed(request) -> str:
        return json.dumps({
            "decision": "PROPOSE",
            "reason_codes": ["actionable_extraction_signal"],
            "edits": [{
                "edit_id": "edit.malformed",
                "action": "REPLACE_RULE",
                "target_rule_id": "future-useful-scope",
                "rule_id": "future-useful-scope",
                "rule_text": None,
                "after_rule_id": None,
                "evidence_example_ids": [request.primary_example_ids[0]],
                "reason_codes": ["malformed_rule"],
            }],
        })

    client = CapturedExtractionOptimizerClient(malformed)
    result = prepare_extraction_proposal(
        corpus_store=store,
        output_root=owner / "proposal-malformed",
        client=client,
    )

    assert result.decision == ExtractionOptimizerDecision.NO_PROPOSAL
    assert result.reason_codes == ("completion_contract_invalid",)
    assert result.completion_id is not None
    output = owner / "proposal-malformed"
    persisted = json.loads((output / "optimizer-result.json").read_text())
    assert persisted["reasonCodes"] == ["completion_contract_invalid"]
    assert persisted["completionId"] == result.completion_id
    assert not (output / "candidate-artifact.json").exists()


def test_no_signal_cli_does_not_read_credentials_or_validate_provider(
    tmp_path: Path,
) -> None:
    corpus = _multi_corpus(("unresolved", "censored"), ownerships=(
        "unresolved",
        "unresolved",
    ))
    store, owner = _store(tmp_path, corpus)
    output = owner / "proposal-cli"

    result = main([
        str(store.attempt_root),
        "--owner-controlled-root",
        str(owner),
        "--attempt-id",
        store.attempt_id,
        "--output",
        str(output),
        "--api-key-file",
        str(tmp_path / "missing-key"),
        "--base-url",
        "http://provider.invalid/v1",
    ])

    assert result == 0
    payload = json.loads((output / "optimizer-result.json").read_text())
    assert payload["decision"] == "NO_PROPOSAL"
    assert payload["usage"]["model_requests"] == 0
    assert not (output / "candidate-artifact.json").exists()
