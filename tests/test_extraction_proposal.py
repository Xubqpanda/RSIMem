from __future__ import annotations

import json
from pathlib import Path

from rsimem.memory.extraction_optimizer_store import JsonExtractionOptimizerCorpusStore
from rsimem.memory.extraction_prompt_optimizer import (
    CapturedExtractionOptimizerClient,
    ExtractionOptimizerDecision,
)
from rsimem.memory.extraction_optimizer_contracts import (
    EXTRACTION_OPTIMIZER_OUTPUT_SCHEMA_DIGEST,
)
from rsimem.memory.extraction_optimizer_corpus import (
    OptimizerCorpusRetention,
    OptimizerCorpusSplit,
)
from rsimem.extraction_proposal import (
    _read_api_key_file,
    main,
    prepare_extraction_proposal,
)
from rsimem.memory.prompt_components import canonical_json
from test_extraction_optimizer_contracts import _multi_corpus, _parent, _proposal_output


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
