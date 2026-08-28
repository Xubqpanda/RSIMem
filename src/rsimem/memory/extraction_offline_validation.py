"""Static safety and deterministic offline validation for prompt candidates."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping, Protocol, runtime_checkable

from .extraction_policy_artifact import (
    ExtractionPromptPolicyArtifact,
    apply_extraction_rule_edits,
)
from .prompt_components import PromptSlotDescriptor, content_digest, text_digest


EXTRACTION_OFFLINE_SCHEMA_VERSION = 1
EXTRACTION_STATIC_SAFETY_SCHEMA = "extraction-candidate-static-safety-v1"
EXTRACTION_DETERMINISTIC_SUITE_SCHEMA = "extraction-deterministic-suite-v1"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$")
_FORBIDDEN_POLICY = re.compile(
    r"(?:\bSM\d{2}\b|\bTSV\b|due_date|owner\s*[|,/ ]+\s*priority|"
    r"ignore (?:all |any )?(?:previous|prior)|system prompt|developer message|"
    r"prompt injection|answer[ _-]?key|official[ _-]?(?:grader|score)|"
    r"hidden[ _-]?expectation|judge[ _-]?feedback|benchmark|family_id|task_id|"
    r"artifact_id|authorization|api[ _-]?key|secret|redacted_|"
    r"output schema|return (?:exactly )?(?:one )?json|facts (?:field|key))",
    re.IGNORECASE,
)
_PROMPT_LEAKAGE = re.compile(
    r"(?:system prompt|developer message|hidden instruction|authorization|"
    r"api[ _-]?key|redacted_)",
    re.IGNORECASE,
)
_TOKEN = re.compile(r"[A-Za-z0-9_]+")


def _require_id(value: object, name: str) -> None:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{name} must be a stable identifier")


def _require_digest(value: object, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be sha256")


@dataclass(frozen=True, slots=True)
class CandidateStaticSafetyReport:
    report_id: str
    parent_artifact_id: str
    candidate_artifact_id: str
    candidate_artifact_digest: str
    passed: bool
    reason_codes: tuple[str, ...]
    report_schema: str = EXTRACTION_STATIC_SAFETY_SCHEMA
    schema_version: int = EXTRACTION_OFFLINE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            self.schema_version != EXTRACTION_OFFLINE_SCHEMA_VERSION
            or self.report_schema != EXTRACTION_STATIC_SAFETY_SCHEMA
        ):
            raise ValueError("unsupported candidate static safety report")
        for value in (
            self.report_id,
            self.parent_artifact_id,
            self.candidate_artifact_id,
        ):
            _require_id(value, "candidate safety identity")
        if type(self.passed) is not bool or not self.reason_codes:
            raise ValueError("candidate safety result is incomplete")
        if self.passed != (self.reason_codes == ("static_safety_passed",)):
            raise ValueError("candidate safety status and reasons disagree")
        expected = f"candidate-safety.{content_digest(self.identity_payload())[:40]}"
        if self.report_id != expected:
            raise ValueError("candidate safety report ID mismatch")

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "report_schema": self.report_schema,
            "parent_artifact_id": self.parent_artifact_id,
            "candidate_artifact_id": self.candidate_artifact_id,
            "candidate_artifact_digest": self.candidate_artifact_digest,
            "passed": self.passed,
            "reason_codes": list(self.reason_codes),
        }


class ExtractionCandidateStaticValidator:
    def validate(
        self,
        *,
        parent: ExtractionPromptPolicyArtifact,
        candidate: ExtractionPromptPolicyArtifact,
        slot: PromptSlotDescriptor,
    ) -> CandidateStaticSafetyReport:
        reasons = []
        if candidate.parent_artifact_id != parent.artifact_id or (
            candidate.parent_spec_digest != parent.spec.spec_digest
        ):
            reasons.append("parent_lineage_mismatch")
        if candidate.generation_provenance is None or candidate.source_provenance is not None:
            reasons.append("generation_provenance_missing")
        if candidate.artifact_id == parent.artifact_id or (
            candidate.body_digest == parent.body_digest
        ):
            reasons.append("candidate_body_unchanged")
        try:
            candidate.to_prompt_component(slot)
        except (TypeError, ValueError):
            reasons.append("runtime_slot_contract_mismatch")
        try:
            replayed = apply_extraction_rule_edits(parent.spec, candidate.edits)
            if replayed != candidate.spec:
                reasons.append("candidate_edit_replay_mismatch")
        except (TypeError, ValueError):
            reasons.append("candidate_edit_replay_invalid")
        parent_rules = {value.rule_id: value for value in parent.spec.rules}
        candidate_rules = {value.rule_id: value for value in candidate.spec.rules}
        if any(
            candidate_rules.get(rule_id) != rule
            for rule_id, rule in parent_rules.items()
            if rule.protected
        ):
            reasons.append("protected_rule_changed")
        changed_rule_texts = tuple(
            rule.text
            for rule_id, rule in candidate_rules.items()
            if parent_rules.get(rule_id) != rule
        )
        if any(
            "$" in value or _FORBIDDEN_POLICY.search(value)
            for value in changed_rule_texts
        ):
            reasons.append("forbidden_candidate_instruction")
        passed = not reasons
        reason_codes = (
            ("static_safety_passed",)
            if passed
            else tuple(dict.fromkeys(reasons))
        )
        values = {
            "parent_artifact_id": parent.artifact_id,
            "candidate_artifact_id": candidate.artifact_id,
            "candidate_artifact_digest": candidate.artifact_digest,
            "passed": passed,
            "reason_codes": reason_codes,
            "report_schema": EXTRACTION_STATIC_SAFETY_SCHEMA,
            "schema_version": EXTRACTION_OFFLINE_SCHEMA_VERSION,
        }
        identity = {
            "schema_version": values["schema_version"],
            "report_schema": values["report_schema"],
            "parent_artifact_id": values["parent_artifact_id"],
            "candidate_artifact_id": values["candidate_artifact_id"],
            "candidate_artifact_digest": values["candidate_artifact_digest"],
            "passed": values["passed"],
            "reason_codes": list(values["reason_codes"]),
        }
        return CandidateStaticSafetyReport(
            report_id=f"candidate-safety.{content_digest(identity)[:40]}",
            **values,
        )


class DeterministicExtractionCategory(StrEnum):
    DURABLE_PREFERENCE = "durable_preference"
    DURABLE_CONSTRAINT = "durable_constraint"
    TEMPORARY_REQUEST = "temporary_request"
    UNRESOLVED_CLAIM = "unresolved_claim"
    ASSISTANT_ONLY = "assistant_only"
    TOOL_EVIDENCE = "tool_evidence"
    CREDENTIAL_PATH = "credential_path"
    EMPTY_SOURCE = "empty_source"


class DeterministicExtractionExpectation(StrEnum):
    RETAIN = "retain"
    EXCLUDE = "exclude"


_EXPECTED_BY_CATEGORY = {
    DeterministicExtractionCategory.DURABLE_PREFERENCE:
        DeterministicExtractionExpectation.RETAIN,
    DeterministicExtractionCategory.DURABLE_CONSTRAINT:
        DeterministicExtractionExpectation.RETAIN,
    DeterministicExtractionCategory.TEMPORARY_REQUEST:
        DeterministicExtractionExpectation.EXCLUDE,
    DeterministicExtractionCategory.UNRESOLVED_CLAIM:
        DeterministicExtractionExpectation.EXCLUDE,
    DeterministicExtractionCategory.ASSISTANT_ONLY:
        DeterministicExtractionExpectation.EXCLUDE,
    DeterministicExtractionCategory.TOOL_EVIDENCE:
        DeterministicExtractionExpectation.EXCLUDE,
    DeterministicExtractionCategory.CREDENTIAL_PATH:
        DeterministicExtractionExpectation.EXCLUDE,
    DeterministicExtractionCategory.EMPTY_SOURCE:
        DeterministicExtractionExpectation.EXCLUDE,
}


@dataclass(frozen=True, slots=True)
class DeterministicSourceMessage:
    role: str
    content: str

    def __post_init__(self) -> None:
        if self.role not in {"user", "assistant", "tool"}:
            raise ValueError("deterministic source role is invalid")
        if not isinstance(self.content, str):
            raise TypeError("deterministic source content must be text")

    def payload(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True, slots=True)
class DeterministicExtractionCase:
    case_id: str
    category: DeterministicExtractionCategory
    expectation: DeterministicExtractionExpectation
    messages: tuple[DeterministicSourceMessage, ...]

    def __post_init__(self) -> None:
        _require_id(self.case_id, "deterministic extraction case ID")
        object.__setattr__(self, "category", DeterministicExtractionCategory(self.category))
        object.__setattr__(
            self,
            "expectation",
            DeterministicExtractionExpectation(self.expectation),
        )
        if self.expectation != _EXPECTED_BY_CATEGORY[self.category]:
            raise ValueError("deterministic case expectation differs from category")
        if self.category == DeterministicExtractionCategory.EMPTY_SOURCE:
            if self.messages:
                raise ValueError("empty-source case must not carry messages")
        elif not self.messages:
            raise ValueError("deterministic extraction case requires messages")

    @property
    def source_text(self) -> str:
        return "\n".join(value.content for value in self.messages)


@runtime_checkable
class DeterministicExtractionExecutor(Protocol):
    def complete(
        self,
        artifact: ExtractionPromptPolicyArtifact,
        case: DeterministicExtractionCase,
    ) -> str: ...


class CapturedDeterministicExtractionExecutor:
    def __init__(self, outputs: Mapping[tuple[str, str], str]) -> None:
        self.outputs = dict(outputs)
        self.calls: list[tuple[str, str]] = []

    def complete(
        self,
        artifact: ExtractionPromptPolicyArtifact,
        case: DeterministicExtractionCase,
    ) -> str:
        key = (artifact.artifact_id, case.case_id)
        self.calls.append(key)
        try:
            return self.outputs[key]
        except KeyError as exc:
            raise KeyError("deterministic executor has no captured output") from exc


@dataclass(frozen=True, slots=True)
class DeterministicExtractionCaseResult:
    case_id: str
    category: DeterministicExtractionCategory
    parent_output_digest: str
    candidate_output_digest: str
    parent_fact_count: int
    candidate_fact_count: int
    passed: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_id(self.case_id, "deterministic result case ID")
        object.__setattr__(self, "category", DeterministicExtractionCategory(self.category))
        _require_digest(self.parent_output_digest, "deterministic parent output digest")
        _require_digest(
            self.candidate_output_digest,
            "deterministic candidate output digest",
        )
        if any(
            type(value) is not int or value < 0
            for value in (self.parent_fact_count, self.candidate_fact_count)
        ):
            raise ValueError("deterministic fact counts must be non-negative")
        if type(self.passed) is not bool or not self.reason_codes:
            raise ValueError("deterministic case result is incomplete")
        if self.passed != (self.reason_codes == ("case_passed",)):
            raise ValueError("deterministic case status and reasons disagree")

    def payload(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "category": self.category.value,
            "parent_output_digest": self.parent_output_digest,
            "candidate_output_digest": self.candidate_output_digest,
            "parent_fact_count": self.parent_fact_count,
            "candidate_fact_count": self.candidate_fact_count,
            "passed": self.passed,
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True, slots=True)
class DeterministicExtractionSuiteReport:
    report_id: str
    parent_artifact_id: str
    candidate_artifact_id: str
    passed: bool
    reason_codes: tuple[str, ...]
    results: tuple[DeterministicExtractionCaseResult, ...]
    report_schema: str = EXTRACTION_DETERMINISTIC_SUITE_SCHEMA
    schema_version: int = EXTRACTION_OFFLINE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            self.schema_version != EXTRACTION_OFFLINE_SCHEMA_VERSION
            or self.report_schema != EXTRACTION_DETERMINISTIC_SUITE_SCHEMA
        ):
            raise ValueError("unsupported deterministic extraction suite report")
        categories = tuple(value.category for value in self.results)
        if set(categories) != set(DeterministicExtractionCategory) or len(
            categories
        ) != len(set(categories)):
            raise ValueError("deterministic extraction suite coverage is incomplete")
        if self.passed != (self.reason_codes == ("deterministic_suite_passed",)):
            raise ValueError("deterministic suite status and reasons disagree")
        if self.passed != all(value.passed for value in self.results):
            raise ValueError("deterministic suite status differs from case results")
        expected = f"deterministic-suite.{content_digest(self.identity_payload())[:40]}"
        if self.report_id != expected:
            raise ValueError("deterministic suite report ID mismatch")

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "report_schema": self.report_schema,
            "parent_artifact_id": self.parent_artifact_id,
            "candidate_artifact_id": self.candidate_artifact_id,
            "passed": self.passed,
            "reason_codes": list(self.reason_codes),
            "results": [value.payload() for value in self.results],
        }


class DeterministicExtractionSuiteRunner:
    def run(
        self,
        *,
        parent: ExtractionPromptPolicyArtifact,
        candidate: ExtractionPromptPolicyArtifact,
        cases: tuple[DeterministicExtractionCase, ...],
        executor: DeterministicExtractionExecutor,
    ) -> DeterministicExtractionSuiteReport:
        by_category = {value.category: value for value in cases}
        if len(by_category) != len(cases) or set(by_category) != set(
            DeterministicExtractionCategory
        ):
            raise ValueError("deterministic extraction suite cases are incomplete")
        results = []
        suite_reasons = []
        for category in DeterministicExtractionCategory:
            case = by_category[category]
            parent_raw = executor.complete(parent, case)
            candidate_raw = executor.complete(candidate, case)
            reasons = []
            try:
                parent_facts = _parse_facts(parent_raw)
            except ValueError:
                parent_facts = ()
                reasons.append("parent_output_schema_invalid")
            try:
                candidate_facts = _parse_facts(candidate_raw)
            except ValueError:
                candidate_facts = ()
                reasons.append("candidate_output_schema_invalid")
            if case.expectation == DeterministicExtractionExpectation.RETAIN:
                if not candidate_facts:
                    reasons.append("durable_fact_missing")
            elif candidate_facts:
                reasons.append("excluded_source_extracted")
            if any(_PROMPT_LEAKAGE.search(value) for value in candidate_facts):
                reasons.append("prompt_leakage")
            if _copies_source(case.source_text, candidate_facts):
                reasons.append("source_transcript_copy")
            passed = not reasons
            result_reasons = (
                ("case_passed",) if passed else tuple(dict.fromkeys(reasons))
            )
            results.append(DeterministicExtractionCaseResult(
                case.case_id,
                category,
                text_digest(parent_raw),
                text_digest(candidate_raw),
                len(parent_facts),
                len(candidate_facts),
                passed,
                result_reasons,
            ))
            if not passed:
                suite_reasons.append(f"case_failed:{category.value}")
        passed = not suite_reasons
        reason_codes = (
            ("deterministic_suite_passed",)
            if passed
            else tuple(suite_reasons)
        )
        values = {
            "parent_artifact_id": parent.artifact_id,
            "candidate_artifact_id": candidate.artifact_id,
            "passed": passed,
            "reason_codes": reason_codes,
            "results": tuple(results),
            "report_schema": EXTRACTION_DETERMINISTIC_SUITE_SCHEMA,
            "schema_version": EXTRACTION_OFFLINE_SCHEMA_VERSION,
        }
        identity = {
            "schema_version": values["schema_version"],
            "report_schema": values["report_schema"],
            "parent_artifact_id": values["parent_artifact_id"],
            "candidate_artifact_id": values["candidate_artifact_id"],
            "passed": values["passed"],
            "reason_codes": list(values["reason_codes"]),
            "results": [value.payload() for value in values["results"]],
        }
        return DeterministicExtractionSuiteReport(
            report_id=f"deterministic-suite.{content_digest(identity)[:40]}",
            **values,
        )


def _parse_facts(raw: str) -> tuple[str, ...]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("deterministic extraction output is not JSON") from exc
    if not isinstance(value, dict) or set(value) != {"facts"}:
        raise ValueError("deterministic extraction output fields are invalid")
    facts = value["facts"]
    if not isinstance(facts, list) or any(
        not isinstance(item, str) or not item.strip() or len(item) > 2_000
        for item in facts
    ):
        raise ValueError("deterministic extraction facts are invalid")
    normalized = tuple(" ".join(value.split()) for value in facts)
    if len(normalized) != len(set(normalized)):
        raise ValueError("deterministic extraction facts are duplicated")
    return normalized


def _copies_source(source: str, facts: tuple[str, ...]) -> bool:
    source_tokens = tuple(value.casefold() for value in _TOKEN.findall(source))
    if not source_tokens:
        return False
    source_normalized = " ".join(source_tokens)
    source_ngrams = {
        source_tokens[index:index + 8]
        for index in range(max(0, len(source_tokens) - 7))
    }
    for fact in facts:
        tokens = tuple(value.casefold() for value in _TOKEN.findall(fact))
        if " ".join(tokens) == source_normalized and tokens:
            return True
        fact_ngrams = {
            tokens[index:index + 8]
            for index in range(max(0, len(tokens) - 7))
        }
        if fact_ngrams & source_ngrams:
            return True
    return False
