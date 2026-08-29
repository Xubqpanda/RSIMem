"""Evidence-gated extraction prompt optimizer and candidate safety checks."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Callable

from ..lifecycle import RawResourceUsage
from .extraction_feedback import AttributionConfidence, ExtractionFeedbackLabel
from .extraction_optimizer_contracts import (
    ExtractionOptimizerClient,
    ExtractionOptimizerCompletion,
    ExtractionOptimizerConfig,
    ExtractionOptimizerRequest,
    EXTRACTION_OPTIMIZER_SYSTEM_INSTRUCTION,
    FROZEN_EXTRACTION_OPTIMIZER_CONFIG,
    build_extraction_optimizer_gate_request,
    build_extraction_optimizer_request,
    logical_primary_examples,
)
from .extraction_optimizer_corpus import (
    ExtractionOptimizerCorpus,
    ExtractionOptimizerCorpusExample,
    OptimizerComponentOwnership,
)
from .extraction_policy_artifact import (
    ExtractionGenerationProvenance,
    ExtractionPolicyRule,
    ExtractionPromptPolicyArtifact,
    ExtractionRuleEdit,
    ExtractionRuleEditAction,
)
from .prompt_components import content_digest, text_digest
from .revocation import JsonRevocationRegistry
from .evidence_planes import EvidencePlane, EvidenceSourceKind


_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$")
_TOKEN = re.compile(r"[A-Za-z0-9_]+")
_URL_OR_EMAIL = re.compile(
    r"https?://[^\s\"'<>]+|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
)
_QUOTED_VALUE = re.compile(r"[\"']([^\"']{4,80})[\"']")
_GENERIC_ALLOWED_TOKENS = {
    "agent",
    "completed",
    "context",
    "durable",
    "extract",
    "fact",
    "future",
    "information",
    "memory",
    "preference",
    "project",
    "rule",
    "source",
    "task",
    "user",
}
_FORBIDDEN_RULE = re.compile(
    r"(?:\bSM\d{2}\b|\bTSV\b|due_date|owner\s*[|,/ ]+\s*priority|"
    r"ignore (?:all |any )?(?:previous|prior)|system prompt|developer message|"
    r"prompt injection|answer[ _-]?key|official[ _-]?(?:grader|score)|"
    r"hidden[ _-]?expectation|judge[ _-]?feedback|benchmark|family_id|task_id|"
    r"artifact_id|authorization|api[ _-]?key|credential|secret|redacted_|"
    r"output schema|return (?:exactly )?(?:one )?json|facts (?:field|key))",
    re.IGNORECASE,
)


class ExtractionOptimizerDecision(StrEnum):
    NO_PROPOSAL = "NO_PROPOSAL"
    PROPOSE = "PROPOSE"


class OptimizerCompletionValidationError(ValueError):
    """A provider completion rejected after transport succeeded.

    The exception keeps the completion metadata available to the outer
    preparation layer.  That layer can persist a rejected result and usage
    without retaining untrusted completion content or weakening the direct
    optimizer API's fail-closed exception semantics.
    """

    def __init__(
        self,
        message: str,
        *,
        reason_code: str,
        request: ExtractionOptimizerRequest,
        completion_id: str,
        usage: RawResourceUsage,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.request = request
        self.completion_id = completion_id
        self.usage = usage


class CandidateValidationError(OptimizerCompletionValidationError):
    """A parsed proposal rejected by the candidate-content safety boundary."""


@dataclass(frozen=True, slots=True)
class EvidenceBoundRuleEdit:
    edit: ExtractionRuleEdit
    evidence_example_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.evidence_example_ids or len(self.evidence_example_ids) != len(
            set(self.evidence_example_ids)
        ):
            raise ValueError("optimizer edit evidence IDs must be nonempty and unique")
        if not self.reason_codes or len(self.reason_codes) != len(set(self.reason_codes)):
            raise ValueError("optimizer edit reason codes must be nonempty and unique")
        for value in (*self.evidence_example_ids, *self.reason_codes):
            if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
                raise ValueError("optimizer edit evidence must use stable identifiers")

    def payload(self) -> dict[str, object]:
        return {
            "edit": self.edit.payload(),
            "evidence_example_ids": list(self.evidence_example_ids),
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True, slots=True)
class ExtractionOptimizerResult:
    result_id: str
    decision: ExtractionOptimizerDecision
    reason_codes: tuple[str, ...]
    request: ExtractionOptimizerRequest
    completion_id: str | None
    edits: tuple[EvidenceBoundRuleEdit, ...]
    candidate: ExtractionPromptPolicyArtifact | None
    usage: RawResourceUsage

    def __post_init__(self) -> None:
        object.__setattr__(self, "decision", ExtractionOptimizerDecision(self.decision))
        if not self.reason_codes:
            raise ValueError("optimizer result requires reason codes")
        if self.decision == ExtractionOptimizerDecision.NO_PROPOSAL:
            if self.edits or self.candidate is not None:
                raise ValueError("NO_PROPOSAL result cannot carry a candidate")
        elif not self.edits or self.candidate is None or self.completion_id is None:
            raise ValueError("PROPOSE result requires completion, edits, and candidate")
        expected = f"optimizer-result.{content_digest(self.identity_payload())[:40]}"
        if self.result_id != expected:
            raise ValueError("optimizer result ID mismatch")

    def identity_payload(self) -> dict[str, object]:
        return {
            "decision": self.decision.value,
            "reason_codes": list(self.reason_codes),
            "request_id": self.request.request_id,
            "completion_id": self.completion_id,
            "edits": [value.payload() for value in self.edits],
            "candidate_artifact_id": (
                self.candidate.artifact_id if self.candidate is not None else None
            ),
            "usage": self.usage.to_dict(),
        }


class CapturedExtractionOptimizerClient:
    """Deterministic captured completion used by fixtures and replay."""

    def __init__(
        self,
        output: str | Callable[[ExtractionOptimizerRequest], str],
        *,
        usage: RawResourceUsage = RawResourceUsage(),
    ) -> None:
        self.output = output
        self.usage = usage
        self.requests: list[ExtractionOptimizerRequest] = []

    def complete(
        self,
        request: ExtractionOptimizerRequest,
        config: ExtractionOptimizerConfig,
    ) -> ExtractionOptimizerCompletion:
        if config.config_digest != request.optimizer_config_digest:
            raise ValueError("captured optimizer config differs from request")
        self.requests.append(request)
        output = self.output(request) if callable(self.output) else self.output
        digest = content_digest({
            "request_id": request.request_id,
            "output_digest": text_digest(output),
        })
        return ExtractionOptimizerCompletion(
            f"optimizer-completion.{digest[:40]}",
            request.request_id,
            output,
            self.usage,
        )


class ExtractionPromptOptimizer:
    def __init__(
        self,
        client: ExtractionOptimizerClient,
        *,
        config: ExtractionOptimizerConfig = FROZEN_EXTRACTION_OPTIMIZER_CONFIG,
        revocation_registry: JsonRevocationRegistry | None = None,
    ) -> None:
        self.client = client
        self.config = config
        self.revocation_registry = revocation_registry

    def propose(
        self,
        parent: ExtractionPromptPolicyArtifact,
        corpus: ExtractionOptimizerCorpus,
    ) -> ExtractionOptimizerResult:
        if self.revocation_registry is not None:
            self.revocation_registry.assert_active(
                artifact_id=parent.artifact_id,
                artifact_schema_version=parent.schema_version,
                artifact_digest=parent.artifact_digest,
                evidence_plane=EvidencePlane.PURE_PROCESS,
                evidence_source=EvidenceSourceKind.RUNTIME_OBSERVATION,
            )
        try:
            primary = logical_primary_examples(corpus)
        except ValueError as exc:
            reason = "conflicting_logical_case_signal"
            request = build_extraction_optimizer_gate_request(
                parent,
                corpus,
                reason_codes=(reason,),
                config=self.config,
            )
            return self._result(
                ExtractionOptimizerDecision.NO_PROPOSAL,
                (reason,),
                request,
                None,
                (),
                None,
                RawResourceUsage(),
            )
        actionable = tuple(value for value in primary if _is_actionable(value))
        if len(actionable) < self.config.minimum_actionable_primary_examples:
            reason = (
                "no_actionable_extraction_signal"
                if not actionable
                else "insufficient_actionable_extraction_signal"
            )
            request = build_extraction_optimizer_gate_request(
                parent,
                corpus,
                reason_codes=(reason,),
                config=self.config,
            )
            return self._result(
                ExtractionOptimizerDecision.NO_PROPOSAL,
                (reason,),
                request,
                None,
                (),
                None,
                RawResourceUsage(),
            )
        by_source: dict[str, set[ExtractionFeedbackLabel]] = {}
        for value in actionable:
            by_source.setdefault(
                value.audit_join.source_record_id,
                set(),
            ).add(value.label)
        if any(len(labels) > 1 for labels in by_source.values()):
            request = build_extraction_optimizer_gate_request(
                parent,
                corpus,
                reason_codes=("conflicting_extraction_signal",),
                config=self.config,
            )
            return self._result(
                ExtractionOptimizerDecision.NO_PROPOSAL,
                ("conflicting_extraction_signal",),
                request,
                None,
                (),
                None,
                RawResourceUsage(),
            )
        request = build_extraction_optimizer_request(
            parent,
            corpus,
            config=self.config,
        )
        completion = self.client.complete(request, self.config)
        if completion.request_id != request.request_id:
            raise ValueError("optimizer completion belongs to another request")
        try:
            decision, reasons, edits = self._parse_completion(
                completion.output_text,
                corpus,
            )
        except ValueError as exc:
            raise OptimizerCompletionValidationError(
                str(exc),
                reason_code="completion_contract_invalid",
                request=request,
                completion_id=completion.completion_id,
                usage=completion.usage,
            ) from exc
        if decision == ExtractionOptimizerDecision.NO_PROPOSAL:
            return self._result(
                decision,
                reasons,
                request,
                completion.completion_id,
                (),
                None,
                completion.usage,
            )
        try:
            self._validate_candidate_content(parent, corpus, edits)
        except ValueError as exc:
            raise CandidateValidationError(
                str(exc),
                reason_code=_candidate_validation_reason(str(exc)),
                request=request,
                completion_id=completion.completion_id,
                usage=completion.usage,
            ) from exc
        provenance = ExtractionGenerationProvenance(
            optimizer_model=self.config.model_id,
            optimizer_config_digest=self.config.config_digest,
            training_corpus_id=corpus.corpus_id,
            training_cutoff=corpus.observation_cutoff,
            proposal_request_digest=request.request_digest,
            completion_digest=text_digest(completion.output_text),
            usage=completion.usage,
        )
        candidate = ExtractionPromptPolicyArtifact.create_child(
            parent=parent,
            policy_version=(
                f"candidate.{text_digest(completion.output_text)[:16]}"
            ),
            edits=tuple(value.edit for value in edits),
            generation_provenance=provenance,
        )
        return self._result(
            decision,
            reasons,
            request,
            completion.completion_id,
            edits,
            candidate,
            completion.usage,
        )

    def _parse_completion(
        self,
        raw: str,
        corpus: ExtractionOptimizerCorpus,
    ) -> tuple[
        ExtractionOptimizerDecision,
        tuple[str, ...],
        tuple[EvidenceBoundRuleEdit, ...],
    ]:
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("optimizer completion is not valid JSON") from exc
        if not isinstance(value, dict) or set(value) != {
            "decision",
            "reason_codes",
            "edits",
        }:
            raise ValueError("optimizer completion fields are invalid")
        reasons = _identifier_list(value["reason_codes"], "optimizer result reasons")
        raw_edits = value["edits"]
        if not isinstance(raw_edits, list):
            raise ValueError("optimizer edits must be a list")
        try:
            decision = ExtractionOptimizerDecision(value["decision"])
        except (TypeError, ValueError) as exc:
            raise ValueError("optimizer decision is invalid") from exc
        if decision == ExtractionOptimizerDecision.NO_PROPOSAL:
            if raw_edits or not reasons:
                raise ValueError("NO_PROPOSAL output shape is invalid")
            return decision, reasons, ()
        if not raw_edits or len(raw_edits) > self.config.maximum_rule_edits:
            raise ValueError("optimizer proposal edit count is invalid")
        eligible = {
            value.example_id: value
            for value in corpus.examples
            if value.primary and _is_actionable(value)
        }
        parsed = tuple(self._parse_edit(item, eligible) for item in raw_edits)
        if len({value.edit.edit_id for value in parsed}) != len(parsed):
            raise ValueError("optimizer proposal has duplicate edit IDs")
        return decision, reasons, parsed

    @staticmethod
    def _parse_edit(
        value: object,
        eligible: dict[str, ExtractionOptimizerCorpusExample],
    ) -> EvidenceBoundRuleEdit:
        fields = {
            "edit_id",
            "action",
            "target_rule_id",
            "rule_id",
            "rule_text",
            "after_rule_id",
            "evidence_example_ids",
            "reason_codes",
        }
        if not isinstance(value, dict) or set(value) != fields:
            raise ValueError("optimizer edit fields are invalid")
        evidence_ids = _identifier_list(
            value["evidence_example_ids"],
            "optimizer edit evidence IDs",
        )
        reasons = _identifier_list(value["reason_codes"], "optimizer edit reasons")
        if not evidence_ids or not set(evidence_ids).issubset(eligible):
            raise ValueError("optimizer edit cites ineligible evidence")
        action_names = {
            "ADD_RULE": ExtractionRuleEditAction.ADD,
            "REPLACE_RULE": ExtractionRuleEditAction.REPLACE,
            "DELETE_RULE": ExtractionRuleEditAction.DELETE,
        }
        try:
            action = action_names[value["action"]]
        except (KeyError, TypeError) as exc:
            raise ValueError("optimizer edit action is invalid") from exc
        rule = None
        if value["rule_id"] is not None or value["rule_text"] is not None:
            if not isinstance(value["rule_id"], str) or not isinstance(
                value["rule_text"],
                str,
            ):
                raise ValueError("optimizer edit rule is incomplete")
            rule = ExtractionPolicyRule(value["rule_id"], value["rule_text"])
        try:
            edit = ExtractionRuleEdit(
                edit_id=value["edit_id"],
                action=action,
                target_rule_id=value["target_rule_id"],
                rule=rule,
                after_rule_id=value["after_rule_id"],
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("optimizer edit contract is invalid") from exc
        return EvidenceBoundRuleEdit(edit, evidence_ids, reasons)

    def _validate_candidate_content(
        self,
        parent: ExtractionPromptPolicyArtifact,
        corpus: ExtractionOptimizerCorpus,
        edits: tuple[EvidenceBoundRuleEdit, ...],
    ) -> None:
        candidate_texts = tuple(
            value.edit.rule.text
            for value in edits
            if value.edit.rule is not None
        )
        if any(_FORBIDDEN_RULE.search(value) or "$" in value for value in candidate_texts):
            raise ValueError("optimizer candidate contains forbidden instructions")
        identifiers = {
            identifier.casefold()
            for example in corpus.examples
            for identifier in (
                example.audit_join.family_id,
                example.audit_join.source_record_id,
                example.audit_join.source_run_id,
                example.audit_join.source_episode_id,
                example.audit_join.source_session_id,
                example.audit_join.source_task_id,
                example.audit_join.feedback_record_id,
                example.audit_join.feedback_run_id,
                example.audit_join.feedback_trace_id,
                example.audit_join.feedback_episode_id,
                example.audit_join.feedback_session_id,
                example.audit_join.feedback_task_id,
                example.audit_join.feedback_example_id,
                example.primary_unit_id,
                *(fact.fact_id for fact in example.extracted_facts),
                *(fact.persisted_artifact_id or "" for fact in example.extracted_facts),
            )
            if identifier
        }
        for text in candidate_texts:
            normalized = text.casefold()
            if any(identifier in normalized for identifier in identifiers):
                raise ValueError("optimizer candidate copies corpus identity")
        source_text = tuple({
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
        })
        generic_tokens = {
            token.casefold()
            for token in _TOKEN.findall(
                parent.compiled_body + " " + EXTRACTION_OPTIMIZER_SYSTEM_INSTRUCTION
            )
        } | _GENERIC_ALLOWED_TOKENS
        sensitive_values = set()
        for value in source_text:
            sensitive_values.update(
                match.group(0).casefold() for match in _URL_OR_EMAIL.finditer(value)
            )
            sensitive_values.update(
                match.group(1).strip().casefold()
                for match in _QUOTED_VALUE.finditer(value)
                if match.group(1).strip()
            )
            for token in _TOKEN.findall(value):
                normalized = token.casefold()
                if (
                    len(token) >= 4
                    and normalized not in generic_tokens
                    and (
                        token[0].isupper()
                        or any(character.isdigit() for character in token)
                        or "_" in token
                    )
                ):
                    sensitive_values.add(normalized)
        for text in candidate_texts:
            normalized = text.casefold()
            if any(value in normalized for value in sensitive_values):
                raise ValueError("optimizer candidate copies a corpus-specific value")
        parent_ngrams = _ngrams(parent.compiled_body, self.config.long_ngram_tokens)
        forbidden_ngrams = set().union(*(
            _ngrams(value, self.config.long_ngram_tokens) for value in source_text
        )) - parent_ngrams
        for text in candidate_texts:
            if _ngrams(text, self.config.long_ngram_tokens) & forbidden_ngrams:
                raise ValueError("optimizer candidate copies corpus content")

    @staticmethod
    def _result(
        decision: ExtractionOptimizerDecision,
        reasons: tuple[str, ...],
        request: ExtractionOptimizerRequest,
        completion_id: str | None,
        edits: tuple[EvidenceBoundRuleEdit, ...],
        candidate: ExtractionPromptPolicyArtifact | None,
        usage: RawResourceUsage,
    ) -> ExtractionOptimizerResult:
        values = {
            "decision": decision,
            "reason_codes": reasons,
            "request": request,
            "completion_id": completion_id,
            "edits": edits,
            "candidate": candidate,
            "usage": usage,
        }
        identity = {
            "decision": decision.value,
            "reason_codes": list(reasons),
            "request_id": request.request_id,
            "completion_id": completion_id,
            "edits": [value.payload() for value in edits],
            "candidate_artifact_id": (
                candidate.artifact_id if candidate is not None else None
            ),
            "usage": usage.to_dict(),
        }
        return ExtractionOptimizerResult(
            result_id=f"optimizer-result.{content_digest(identity)[:40]}",
            **values,
        )


def _identifier_list(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or _IDENTIFIER.fullmatch(item) is None
        for item in value
    ):
        raise ValueError(f"{name} must be a stable identifier list")
    result = tuple(value)
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must be unique")
    return result


def _candidate_validation_reason(message: str) -> str:
    """Map internal safety diagnostics to non-sensitive stable reason codes."""

    if message == "optimizer candidate contains forbidden instructions":
        return "candidate_forbidden_instruction"
    if message == "optimizer candidate copies corpus identity":
        return "candidate_corpus_identity"
    if message == "optimizer candidate copies a corpus-specific value":
        return "candidate_corpus_value"
    if message == "optimizer candidate copies corpus content":
        return "candidate_corpus_content"
    return "candidate_validation_failed"


def _is_actionable(value: ExtractionOptimizerCorpusExample) -> bool:
    return (
        value.primary
        and value.label in {
            ExtractionFeedbackLabel.USEFUL,
            ExtractionFeedbackLabel.HARMFUL,
            ExtractionFeedbackLabel.MISSED,
        }
        and value.component_ownership == OptimizerComponentOwnership.EXTRACTION
        and value.attribution_confidence in {
            AttributionConfidence.HIGH,
            AttributionConfidence.MEDIUM,
        }
    )


def _ngrams(value: str, size: int) -> set[tuple[str, ...]]:
    tokens = tuple(token.casefold() for token in _TOKEN.findall(value))
    return {
        tokens[index:index + size]
        for index in range(max(0, len(tokens) - size + 1))
    }
