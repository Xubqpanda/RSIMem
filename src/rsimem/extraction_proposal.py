"""Generate and persist one extraction prompt proposal from a train corpus."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .memory.extraction_optimizer_contracts import FROZEN_EXTRACTION_OPTIMIZER_CONFIG
from .memory.extraction_optimizer_contracts import (
    ExtractionOptimizerConfig,
    ExtractionOptimizerRequest,
)
from .memory.extraction_optimizer_provider import OpenAICompatibleExtractionOptimizerClient
from .memory.extraction_optimizer_store import JsonExtractionOptimizerCorpusStore
from .memory.extraction_policy_artifact import ExtractionPromptPolicyArtifact
from .memory.extraction_policy_store import JsonExtractionPolicyStore
from .memory.extraction_prompt_optimizer import (
    ExtractionOptimizerDecision,
    ExtractionOptimizerResult,
    ExtractionPromptOptimizer,
    OptimizerCompletionValidationError,
)
from .memory.policy_feasibility import project_optimizer_result
from .memory.prompt_components import canonical_json, content_digest
from .memory.revocation import JsonRevocationRegistry
from .memory_systems.mem0_flat import (
    MEM0_FLAT_EXTRACTION_SLOT,
    MEM0_FLAT_EXTRACTION_SLOT_ID,
    Mem0FlatPromptAdapter,
)


class _DeferredExtractionOptimizerClient:
    """Keep credentials and provider transport unreachable on NO_PROPOSAL."""

    def __init__(
        self,
        api_key_file: Path | None,
        base_url: str,
        revocation_registry: JsonRevocationRegistry | None = None,
    ) -> None:
        self.api_key_file = api_key_file
        self.base_url = base_url
        self.revocation_registry = revocation_registry

    def complete(
        self,
        request: ExtractionOptimizerRequest,
        config: ExtractionOptimizerConfig,
    ):
        # This deferred client is the formal provider entry point.  A local
        # deterministic gate may return NO_PROPOSAL without credentials, but
        # any actual model call must bind the parent artifact to an explicit,
        # owner-controlled revocation registry first.
        if self.revocation_registry is None:
            raise ValueError(
                "optimizer provider proposal requires a revocation registry"
            )
        self.revocation_registry.assert_active(
            artifact_id=request.parent_artifact_id,
            artifact_schema_version=1,
            artifact_digest=request.parent_artifact_digest,
            evidence_plane="pure_process",
            evidence_source="runtime_observation",
        )
        if self.api_key_file is None:
            raise ValueError("optimizer provider API key file is required")
        api_key = _read_api_key_file(self.api_key_file)
        return OpenAICompatibleExtractionOptimizerClient(
            api_key=api_key,
            base_url=self.base_url,
        ).complete(request, config)


def _read_api_key_file(path: Path) -> str:
    """Read a credential from either a plain file or ``key: ...`` config.

    The repository's operator-managed ``api_key.md`` also contains endpoint
    metadata.  Passing that whole document to an HTTP client is both invalid
    and an avoidable credential-boundary mistake, so only the explicit key
    field is accepted when present.
    """

    raw = path.read_text(encoding="utf-8")
    candidates = []
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("key:"):
            candidates.append(stripped.split(":", 1)[1].strip())
    value = candidates[0] if candidates else raw.strip()
    value = value.strip('"').strip("'").strip()
    if not value or any(char.isspace() for char in value):
        raise ValueError("optimizer provider API key file is malformed")
    return value


def _write_immutable(path: Path, value: object) -> None:
    serialized = canonical_json(value) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != serialized:
            raise ValueError("proposal output conflicts with existing content")
        return
    path.write_text(serialized, encoding="utf-8")
    path.chmod(0o600)


def result_payload(result: ExtractionOptimizerResult) -> dict[str, object]:
    return {
        "schemaVersion": 1,
        "resultSchema": "extraction-optimizer-result-v1",
        "resultId": result.result_id,
        "decision": result.decision.value,
        "reasonCodes": list(result.reason_codes),
        "request": {
            "requestId": result.request.request_id,
            "requestDigest": result.request.request_digest,
            "parentArtifactId": result.request.parent_artifact_id,
            "parentArtifactDigest": result.request.parent_artifact_digest,
            "corpusId": result.request.corpus_id,
            "corpusDigest": result.request.corpus_digest,
            "optimizerConfigDigest": result.request.optimizer_config_digest,
            "providerEligible": result.request.provider_eligible,
        },
        "completionId": result.completion_id,
        "edits": [value.payload() for value in result.edits],
        "candidate": result.candidate.payload() if result.candidate else None,
        "usage": result.usage.to_dict(),
    }


def prepare_extraction_proposal(
    *,
    corpus_store: JsonExtractionOptimizerCorpusStore,
    output_root: Path,
    client,
    revocation_registry: JsonRevocationRegistry | None = None,
) -> ExtractionOptimizerResult:
    corpus = corpus_store.read_for_optimizer()
    parent = Mem0FlatPromptAdapter().export_root_policy_artifact(
        MEM0_FLAT_EXTRACTION_SLOT_ID
    )
    try:
        result = ExtractionPromptOptimizer(
            client,
            config=FROZEN_EXTRACTION_OPTIMIZER_CONFIG,
            revocation_registry=revocation_registry,
        ).propose(parent, corpus)
    except OptimizerCompletionValidationError as exc:
        # Persist the rejected completion metadata and usage before building
        # the content-free hypothesis.  Candidate text remains unreachable
        # from the durable result and can never be deployed.
        result = ExtractionPromptOptimizer._result(
            ExtractionOptimizerDecision.NO_PROPOSAL,
            (exc.reason_code,),
            exc.request,
            exc.completion_id,
            (),
            None,
            exc.usage,
        )
    output = output_root.expanduser().resolve()
    _write_immutable(output / "optimizer-result.json", result_payload(result))
    _write_immutable(
        output / "optimizer-request.json",
        {
            "requestId": result.request.request_id,
            "requestDigest": result.request.request_digest,
            "parentArtifactId": result.request.parent_artifact_id,
            "parentArtifactDigest": result.request.parent_artifact_digest,
            "corpusId": result.request.corpus_id,
            "corpusDigest": result.request.corpus_digest,
            "optimizerConfigDigest": result.request.optimizer_config_digest,
            "providerEligible": result.request.provider_eligible,
            "primaryExampleIds": list(result.request.primary_example_ids),
        },
    )
    projection = project_optimizer_result(
        result,
        corpus,
        parent_artifact_id=parent.artifact_id,
    )
    _write_immutable(output / "feasibility-hypothesis.json", projection.payload())
    if result.decision == ExtractionOptimizerDecision.PROPOSE:
        assert result.candidate is not None
        _write_immutable(
            output / "candidate-artifact.json",
            result.candidate.payload(),
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus_attempt_root", type=Path)
    parser.add_argument("--owner-controlled-root", type=Path, required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--base-url", default="https://coding.tu-zi.com/v1")
    parser.add_argument(
        "--revocation-registry",
        type=Path,
        help="append-only artifact revocation registry required for production proposals",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    revocation_registry = None
    if args.revocation_registry is not None:
        revocation_registry = JsonRevocationRegistry(args.revocation_registry)
    client = _DeferredExtractionOptimizerClient(
        args.api_key_file,
        args.base_url,
        revocation_registry,
    )
    store = JsonExtractionOptimizerCorpusStore(
        args.corpus_attempt_root,
        owner_controlled_root=args.owner_controlled_root,
        attempt_id=args.attempt_id,
        split="train",
    )
    result = prepare_extraction_proposal(
        corpus_store=store,
        output_root=args.output,
        client=client,
        revocation_registry=revocation_registry,
    )
    print(canonical_json({
        "resultId": result.result_id,
        "decision": result.decision.value,
        "candidateArtifactId": (
            result.candidate.artifact_id if result.candidate is not None else None
        ),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
