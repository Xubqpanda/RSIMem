"""Privacy isolation audit for content-bearing optimizer corpora."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .extraction_optimizer_corpus import ExtractionOptimizerCorpus


def audit_optimizer_corpus_isolation(
    corpus: ExtractionOptimizerCorpus,
    public_payloads: Mapping[str, object],
) -> tuple[str, ...]:
    """Report corpus body text leaked into content-free public evidence."""

    protected = {
        value
        for example in corpus.examples
        for value in (
            *(message.content.text for message in example.source_messages),
            *(fact.content.text for fact in example.extracted_facts),
            example.delayed_evidence.opportunity.text,
            example.delayed_evidence.use.text,
            example.delayed_evidence.outcome.text,
        )
        if value
    }
    issues = set()

    def inspect(value: object, owner: str) -> None:
        if isinstance(value, Mapping):
            for child in value.values():
                inspect(child, owner)
        elif isinstance(value, Sequence) and not isinstance(
            value,
            (str, bytes, bytearray),
        ):
            for child in value:
                inspect(child, owner)
        elif isinstance(value, str) and any(
            secret in value for secret in protected
        ):
            issues.add(f"corpus_content_leak:{owner}")

    for owner, payload in public_payloads.items():
        inspect(payload, owner)
    return tuple(sorted(issues))
