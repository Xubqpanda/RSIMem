"""Versioned content-free delayed feedback labels over operation evidence."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Mapping, Sequence

from ..lifecycle import RawResourceUsage
from .attribution import (
    AttributionMethod,
    AttributionReport,
    FailureCategory,
)
from .ingestion import InternalMemoryAction
from .operation_graph import (
    ArtifactKind,
    OperationGraph,
    OperationKind,
    OperationRecord,
    OperationStatus,
)


DELAYED_FEEDBACK_SCHEMA_VERSION = 1
DELAYED_FEEDBACK_DATASET_VERSION = "semantic-delayed-feedback-v1"
DELAYED_FEEDBACK_LABEL_SCHEMA = "semantic-delayed-utility-label-v1"
DELAYED_FEEDBACK_WINDOW_VERSION = "semantic-observation-window-v1"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$")
_REASON_CODE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_FORBIDDEN_KEYS = {
    "answer",
    "content",
    "expectation",
    "grader",
    "hidden",
    "memory",
    "prompt",
    "query",
    "response",
    "score",
}
_DELAYED_LABEL_OPERATION_KINDS = {
    OperationKind.RETRIEVAL,
    OperationKind.INJECTION,
    OperationKind.USE,
    OperationKind.TOOL_BEHAVIOR,
    OperationKind.DOWNSTREAM_OUTCOME,
    OperationKind.SUPERSESSION,
    OperationKind.RECOVERY,
}


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _stable_id(prefix: str, value: object) -> str:
    return f"{prefix}.{_digest(value)[:40]}"


def _require_identifier(value: str, name: str) -> None:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{name} must be a stable identifier")


def _require_ids(values: tuple[str, ...], name: str) -> None:
    if len(values) != len(set(values)) or any(
        not _IDENTIFIER.fullmatch(value) for value in values
    ):
        raise ValueError(f"{name} must be unique stable identifiers")


def _is_delayed_deterministic_failure(
    *,
    method: AttributionMethod,
    category: FailureCategory,
    candidate_operation_ids: Sequence[str],
    operations: Mapping[str, OperationRecord],
) -> bool:
    return (
        category != FailureCategory.UNRESOLVED_TASK_FAILURE
        and method not in {AttributionMethod.MODEL, AttributionMethod.UNRESOLVED}
        and any(
            operations[operation_id].kind in _DELAYED_LABEL_OPERATION_KINDS
            for operation_id in candidate_operation_ids
            if operation_id in operations
        )
    )


class FeedbackLabel(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    UNRESOLVED = "unresolved"
    CENSORED = "censored"


class ExposureState(StrEnum):
    NOT_RETRIEVED = "not_retrieved"
    RETRIEVED_NOT_INJECTED = "retrieved_not_injected"
    INJECTED_NOT_USED = "injected_not_used"
    USED = "used"
    SUPERSEDED = "superseded"
    CENSORED = "censored"


class CandidateDisposition(StrEnum):
    INCLUDED = "included"
    FILTERED = "filtered"
    NOT_ELIGIBLE = "not_eligible"
    UNKNOWN = "unknown"


class PropensitySource(StrEnum):
    DETERMINISTIC = "deterministic"
    LOGGED = "logged"
    MISSING = "missing"


class FeedbackEstimator(StrEnum):
    DIRECT = "direct"
    INVERSE_PROPENSITY_WEIGHTED = "inverse_propensity_weighted"
    DOUBLY_ROBUST = "doubly_robust"


@dataclass(frozen=True, slots=True)
class FeedbackObservationWindow:
    window_id: str
    version: str
    cutoff_operation_id: str
    visible_operation_ids: tuple[str, ...]
    complete: bool
    censor_reason: str | None = None
    schema_version: int = DELAYED_FEEDBACK_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != DELAYED_FEEDBACK_SCHEMA_VERSION:
            raise ValueError("unsupported feedback observation window schema")
        for value in (self.window_id, self.version, self.cutoff_operation_id):
            _require_identifier(value, "feedback observation window identity")
        _require_ids(self.visible_operation_ids, "feedback visible operation IDs")
        if (
            not self.visible_operation_ids
            or self.visible_operation_ids[-1] != self.cutoff_operation_id
        ):
            raise ValueError("feedback cutoff must terminate the visible operation prefix")
        if type(self.complete) is not bool:
            raise TypeError("feedback observation completeness must be bool")
        if self.complete and self.censor_reason is not None:
            raise ValueError("complete feedback window cannot carry a censor reason")
        if not self.complete and (
            not isinstance(self.censor_reason, str)
            or not _REASON_CODE.fullmatch(self.censor_reason)
        ):
            raise ValueError("incomplete feedback window requires a censor reason")

    @classmethod
    def create(
        cls,
        graph: OperationGraph,
        *,
        cutoff_operation_id: str | None = None,
        complete: bool,
        censor_reason: str | None = None,
        version: str = DELAYED_FEEDBACK_WINDOW_VERSION,
    ) -> "FeedbackObservationWindow":
        if not graph.operations:
            raise ValueError("feedback observation requires operations")
        cutoff = cutoff_operation_id or graph.operations[-1].operation_id
        indexes = [
            index
            for index, operation in enumerate(graph.operations)
            if operation.operation_id == cutoff
        ]
        if len(indexes) != 1:
            raise ValueError("feedback cutoff must identify one operation")
        visible = tuple(
            operation.operation_id for operation in graph.operations[: indexes[0] + 1]
        )
        identity = {
            "schema_version": DELAYED_FEEDBACK_SCHEMA_VERSION,
            "version": version,
            "cutoff_operation_id": cutoff,
            "visible_operation_ids": visible,
            "complete": complete,
            "censor_reason": censor_reason,
        }
        return cls(
            _stable_id("feedback-window", identity),
            version,
            cutoff,
            visible,
            complete,
            censor_reason,
        )

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "window_id": self.window_id,
            "version": self.version,
            "cutoff_operation_id": self.cutoff_operation_id,
            "visible_operation_ids": list(self.visible_operation_ids),
            "complete": self.complete,
            "censor_reason": self.censor_reason,
        }


@dataclass(frozen=True, slots=True)
class DelayedFeedbackConfig:
    policy_version: str
    feature_schema: str
    dataset_version: str = DELAYED_FEEDBACK_DATASET_VERSION
    label_schema: str = DELAYED_FEEDBACK_LABEL_SCHEMA
    window_version: str = DELAYED_FEEDBACK_WINDOW_VERSION
    schema_version: int = DELAYED_FEEDBACK_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != DELAYED_FEEDBACK_SCHEMA_VERSION:
            raise ValueError("unsupported delayed feedback config schema")
        for value in (
            self.policy_version,
            self.feature_schema,
            self.dataset_version,
            self.label_schema,
            self.window_version,
        ):
            _require_identifier(value, "delayed feedback config identity")

    @property
    def digest(self) -> str:
        return _digest(self.payload())

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "policy_version": self.policy_version,
            "feature_schema": self.feature_schema,
            "dataset_version": self.dataset_version,
            "label_schema": self.label_schema,
            "window_version": self.window_version,
        }


@dataclass(frozen=True, slots=True)
class DelayedFeedbackExample:
    example_id: str
    run_id: str
    source_episode_id: str
    observation_episode_ids: tuple[str, ...]
    session_ids: tuple[str, ...]
    task_ids: tuple[str, ...]
    memory_artifact_id: str
    memory_revision: str | None
    mutation_id: str
    mutation_operation_id: str
    mutation_action: InternalMemoryAction
    proposal_operation_ids: tuple[str, ...]
    source_operation_ids: tuple[str, ...]
    extraction_operation_ids: tuple[str, ...]
    related_retrieval_operation_ids: tuple[str, ...]
    decision_operation_ids: tuple[str, ...]
    target_resolution_operation_ids: tuple[str, ...]
    validation_operation_ids: tuple[str, ...]
    verification_operation_ids: tuple[str, ...]
    query_operation_ids: tuple[str, ...]
    retrieval_operation_ids: tuple[str, ...]
    injection_operation_ids: tuple[str, ...]
    use_operation_ids: tuple[str, ...]
    outcome_operation_ids: tuple[str, ...]
    tool_operation_ids: tuple[str, ...]
    supersession_operation_ids: tuple[str, ...]
    recovery_operation_ids: tuple[str, ...]
    attribution_record_ids: tuple[str, ...]
    attribution_methods: tuple[AttributionMethod, ...]
    failure_categories: tuple[FailureCategory, ...]
    attributed_operation_ids: tuple[str, ...]
    failure_subgraph_operation_ids: tuple[str, ...]
    policy_parameter_ids: tuple[str, ...]
    exposure_opportunity: bool
    entered_candidate_set: bool
    candidate_disposition: CandidateDisposition
    selection_propensity: float | None
    propensity_source: PropensitySource
    exposure_state: ExposureState
    label: FeedbackLabel
    label_reason_codes: tuple[str, ...]
    observation_cutoff_operation_id: str
    policy_version: str
    resources: RawResourceUsage
    schema_version: int = DELAYED_FEEDBACK_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != DELAYED_FEEDBACK_SCHEMA_VERSION:
            raise ValueError("unsupported delayed feedback example schema")
        object.__setattr__(self, "mutation_action", InternalMemoryAction(self.mutation_action))
        object.__setattr__(self, "exposure_state", ExposureState(self.exposure_state))
        object.__setattr__(self, "label", FeedbackLabel(self.label))
        object.__setattr__(
            self,
            "candidate_disposition",
            CandidateDisposition(self.candidate_disposition),
        )
        object.__setattr__(
            self,
            "propensity_source",
            PropensitySource(self.propensity_source),
        )
        object.__setattr__(
            self,
            "attribution_methods",
            tuple(AttributionMethod(value) for value in self.attribution_methods),
        )
        object.__setattr__(
            self,
            "failure_categories",
            tuple(FailureCategory(value) for value in self.failure_categories),
        )
        for value in (
            self.example_id,
            self.run_id,
            self.source_episode_id,
            self.memory_artifact_id,
            self.mutation_id,
            self.mutation_operation_id,
            self.observation_cutoff_operation_id,
            self.policy_version,
        ):
            _require_identifier(value, "delayed feedback example identity")
        if self.memory_revision is not None:
            _require_identifier(self.memory_revision, "feedback memory revision")
        id_groups = (
            self.observation_episode_ids,
            self.session_ids,
            self.task_ids,
            self.proposal_operation_ids,
            self.source_operation_ids,
            self.extraction_operation_ids,
            self.related_retrieval_operation_ids,
            self.decision_operation_ids,
            self.target_resolution_operation_ids,
            self.validation_operation_ids,
            self.verification_operation_ids,
            self.query_operation_ids,
            self.retrieval_operation_ids,
            self.injection_operation_ids,
            self.use_operation_ids,
            self.outcome_operation_ids,
            self.tool_operation_ids,
            self.supersession_operation_ids,
            self.recovery_operation_ids,
            self.attribution_record_ids,
            self.attributed_operation_ids,
            self.failure_subgraph_operation_ids,
            self.policy_parameter_ids,
        )
        for values in id_groups:
            _require_ids(values, "delayed feedback references")
        if not (
            len(self.attribution_record_ids)
            == len(self.attribution_methods)
            == len(self.failure_categories)
        ):
            raise ValueError("feedback attribution references must align")
        if type(self.exposure_opportunity) is not bool or type(
            self.entered_candidate_set
        ) is not bool:
            raise TypeError("feedback exposure opportunity fields must be bool")
        if self.entered_candidate_set and not self.exposure_opportunity:
            raise ValueError("candidate entry requires an exposure opportunity")
        if (
            self.candidate_disposition == CandidateDisposition.INCLUDED
        ) != self.entered_candidate_set:
            raise ValueError("candidate inclusion disposition is inconsistent")
        if (
            self.candidate_disposition == CandidateDisposition.NOT_ELIGIBLE
        ) != (not self.exposure_opportunity):
            raise ValueError("candidate eligibility disposition is inconsistent")
        if self.selection_propensity is not None and (
            not isinstance(self.selection_propensity, (int, float))
            or isinstance(self.selection_propensity, bool)
            or not math.isfinite(float(self.selection_propensity))
            or not 0.0 <= float(self.selection_propensity) <= 1.0
        ):
            raise ValueError("feedback propensity must be finite in [0,1]")
        if (
            self.propensity_source == PropensitySource.MISSING
        ) != (self.selection_propensity is None):
            raise ValueError("feedback propensity source is inconsistent")
        if self.selection_propensity is not None:
            object.__setattr__(
                self,
                "selection_propensity",
                float(self.selection_propensity),
            )
        if self.propensity_source == PropensitySource.DETERMINISTIC:
            expected_propensity = (
                1.0
                if self.candidate_disposition == CandidateDisposition.INCLUDED
                else 0.0
                if self.candidate_disposition == CandidateDisposition.FILTERED
                else None
            )
            if self.selection_propensity != expected_propensity:
                raise ValueError("deterministic propensity is inconsistent")
        if not self.proposal_operation_ids or not self.label_reason_codes:
            raise ValueError("feedback example requires proposal and label evidence")
        if any(not _REASON_CODE.fullmatch(value) for value in self.label_reason_codes):
            raise ValueError("feedback label reasons must be machine-readable")

    @property
    def operation_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((
            *self.proposal_operation_ids,
            *self.source_operation_ids,
            *self.extraction_operation_ids,
            *self.related_retrieval_operation_ids,
            *self.decision_operation_ids,
            *self.target_resolution_operation_ids,
            *self.validation_operation_ids,
            *self.verification_operation_ids,
            self.mutation_operation_id,
            *self.query_operation_ids,
            *self.retrieval_operation_ids,
            *self.injection_operation_ids,
            *self.use_operation_ids,
            *self.outcome_operation_ids,
            *self.tool_operation_ids,
            *self.supersession_operation_ids,
            *self.recovery_operation_ids,
            *self.attributed_operation_ids,
            *self.failure_subgraph_operation_ids,
        )))

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "example_id": self.example_id,
            "run_id": self.run_id,
            "source_episode_id": self.source_episode_id,
            "observation_episode_ids": list(self.observation_episode_ids),
            "session_ids": list(self.session_ids),
            "task_ids": list(self.task_ids),
            "memory_artifact_id": self.memory_artifact_id,
            "memory_revision": self.memory_revision,
            "mutation_id": self.mutation_id,
            "mutation_operation_id": self.mutation_operation_id,
            "mutation_action": self.mutation_action.value,
            "proposal_operation_ids": list(self.proposal_operation_ids),
            "source_operation_ids": list(self.source_operation_ids),
            "extraction_operation_ids": list(self.extraction_operation_ids),
            "related_retrieval_operation_ids": list(
                self.related_retrieval_operation_ids
            ),
            "decision_operation_ids": list(self.decision_operation_ids),
            "target_resolution_operation_ids": list(
                self.target_resolution_operation_ids
            ),
            "validation_operation_ids": list(self.validation_operation_ids),
            "verification_operation_ids": list(self.verification_operation_ids),
            "query_operation_ids": list(self.query_operation_ids),
            "retrieval_operation_ids": list(self.retrieval_operation_ids),
            "injection_operation_ids": list(self.injection_operation_ids),
            "use_operation_ids": list(self.use_operation_ids),
            "outcome_operation_ids": list(self.outcome_operation_ids),
            "tool_operation_ids": list(self.tool_operation_ids),
            "supersession_operation_ids": list(self.supersession_operation_ids),
            "recovery_operation_ids": list(self.recovery_operation_ids),
            "attribution_record_ids": list(self.attribution_record_ids),
            "attribution_methods": [value.value for value in self.attribution_methods],
            "failure_categories": [value.value for value in self.failure_categories],
            "attributed_operation_ids": list(self.attributed_operation_ids),
            "failure_subgraph_operation_ids": list(
                self.failure_subgraph_operation_ids
            ),
            "policy_parameter_ids": list(self.policy_parameter_ids),
            "exposure_opportunity": self.exposure_opportunity,
            "entered_candidate_set": self.entered_candidate_set,
            "candidate_disposition": self.candidate_disposition.value,
            "selection_propensity": self.selection_propensity,
            "propensity_source": self.propensity_source.value,
            "exposure_state": self.exposure_state.value,
            "label": self.label.value,
            "label_reason_codes": list(self.label_reason_codes),
            "observation_cutoff_operation_id": self.observation_cutoff_operation_id,
            "policy_version": self.policy_version,
            "resources": self.resources.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class DelayedFeedbackDataset:
    dataset_id: str
    config: DelayedFeedbackConfig
    window: FeedbackObservationWindow
    examples: tuple[DelayedFeedbackExample, ...]
    source_operation_count: int
    schema_version: int = DELAYED_FEEDBACK_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != DELAYED_FEEDBACK_SCHEMA_VERSION:
            raise ValueError("unsupported delayed feedback dataset schema")
        _require_identifier(self.dataset_id, "delayed feedback dataset identity")
        if self.window.version != self.config.window_version:
            raise ValueError("feedback window version differs from dataset config")
        if self.source_operation_count < 1:
            raise ValueError("feedback dataset requires source operations")

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "config": self.config.payload(),
            "window": self.window.payload(),
            "examples": [example.payload() for example in self.examples],
            "source_operation_count": self.source_operation_count,
        }


class JsonDelayedFeedbackDatasetStore:
    """Persist immutable content-addressed datasets without replacement."""

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, dataset: DelayedFeedbackDataset) -> tuple[Path, bool]:
        path = self.root / f"{dataset.dataset_id}.json"
        canonical = _canonical(dataset.payload()) + "\n"
        if path.exists():
            try:
                existing = path.read_text(encoding="utf-8")
                parsed = json.loads(existing)
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError("stored delayed feedback dataset is malformed") from exc
            if not isinstance(parsed, dict) or existing != canonical:
                raise ValueError("stored delayed feedback dataset conflicts with its identity")
            return path, False
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(canonical, encoding="utf-8")
        temporary.replace(path)
        return path, True


@dataclass(frozen=True, slots=True)
class FeedbackDatasetAudit:
    ok: bool
    issues: tuple[str, ...]
    example_count: int
    label_counts: tuple[tuple[FeedbackLabel, int], ...]


@dataclass(frozen=True, slots=True)
class FeedbackDatasetReport:
    observation_count: int
    opportunity_count: int
    candidate_count: int
    filtered_count: int
    missing_propensity_count: int
    censored_count: int
    censoring_rate: float
    label_counts: tuple[tuple[FeedbackLabel, int], ...]
    exposure_counts: tuple[tuple[ExposureState, int], ...]


def build_feedback_dataset_report(
    dataset: DelayedFeedbackDataset,
) -> FeedbackDatasetReport:
    observations = len(dataset.examples)
    label_counts = {label: 0 for label in FeedbackLabel}
    exposure_counts = {exposure: 0 for exposure in ExposureState}
    for example in dataset.examples:
        label_counts[example.label] += 1
        exposure_counts[example.exposure_state] += 1
    censored = label_counts[FeedbackLabel.CENSORED]
    return FeedbackDatasetReport(
        observation_count=observations,
        opportunity_count=sum(
            example.exposure_opportunity for example in dataset.examples
        ),
        candidate_count=sum(
            example.entered_candidate_set for example in dataset.examples
        ),
        filtered_count=sum(
            example.candidate_disposition == CandidateDisposition.FILTERED
            for example in dataset.examples
        ),
        missing_propensity_count=sum(
            example.selection_propensity is None for example in dataset.examples
        ),
        censored_count=censored,
        censoring_rate=censored / observations if observations else 0.0,
        label_counts=tuple((label, label_counts[label]) for label in FeedbackLabel),
        exposure_counts=tuple(
            (exposure, exposure_counts[exposure]) for exposure in ExposureState
        ),
    )


def validate_feedback_estimator(
    dataset: DelayedFeedbackDataset,
    estimator: FeedbackEstimator,
) -> None:
    estimator = FeedbackEstimator(estimator)
    if estimator == FeedbackEstimator.DIRECT:
        return
    if any(example.selection_propensity is None for example in dataset.examples):
        raise ValueError(f"{estimator.value} requires propensity for every observation")
    if any(
        example.selection_propensity == 0.0 for example in dataset.examples
    ):
        raise ValueError(f"{estimator.value} requires strictly positive propensity")


def _resolve_exposure_bias(
    *,
    target: str,
    queries: Sequence[OperationRecord],
    retrievals: Sequence[OperationRecord],
) -> tuple[bool, bool, CandidateDisposition, float | None, PropensitySource]:
    retrieved = any(
        operation.status == OperationStatus.SUCCESS
        and target in operation.input_artifact_ids
        for operation in retrievals
    )
    explicitly_filtered = any(
        operation.status == OperationStatus.NONE
        and operation.reason_code == "policy_filtered"
        and target in operation.input_artifact_ids
        for operation in retrievals
    )
    opportunity = bool(queries)
    if retrieved:
        return (
            opportunity,
            True,
            CandidateDisposition.INCLUDED,
            1.0,
            PropensitySource.DETERMINISTIC,
        )
    if explicitly_filtered:
        return (
            opportunity,
            False,
            CandidateDisposition.FILTERED,
            0.0,
            PropensitySource.DETERMINISTIC,
        )
    if not opportunity:
        return (
            False,
            False,
            CandidateDisposition.NOT_ELIGIBLE,
            None,
            PropensitySource.MISSING,
        )
    return (
        True,
        False,
        CandidateDisposition.UNKNOWN,
        None,
        PropensitySource.MISSING,
    )


def _ordered_ids(
    operations: Sequence[OperationRecord],
    selected: set[str],
    kind: OperationKind,
) -> tuple[str, ...]:
    return tuple(
        operation.operation_id
        for operation in operations
        if operation.operation_id in selected and operation.kind == kind
    )


def _combine_resources(operations: Sequence[OperationRecord]) -> RawResourceUsage:
    def total(name: str) -> int | None:
        values = [getattr(operation.usage, name) for operation in operations]
        return None if any(value is None for value in values) else sum(values)

    return RawResourceUsage(
        input_tokens=total("input_tokens"),
        output_tokens=total("output_tokens"),
        cache_read_tokens=total("cache_read_tokens"),
        cache_write_tokens=total("cache_write_tokens"),
        reasoning_tokens=total("reasoning_tokens"),
        model_requests=sum(operation.usage.model_requests for operation in operations),
        retry_count=sum(operation.usage.retry_count for operation in operations),
        duration_ms=sum(operation.latency_ms for operation in operations),
        storage_bytes=sum(operation.usage.storage_bytes for operation in operations),
    )


class DelayedFeedbackDatasetBuilder:
    def __init__(self, config: DelayedFeedbackConfig) -> None:
        self.config = config

    def build(
        self,
        graph: OperationGraph,
        window: FeedbackObservationWindow,
        *,
        attribution_reports: Sequence[AttributionReport] = (),
    ) -> DelayedFeedbackDataset:
        if window.version != self.config.window_version:
            raise ValueError("feedback builder window version mismatch")
        visible_ids = set(window.visible_operation_ids)
        operations = tuple(
            operation
            for operation in graph.operations
            if operation.operation_id in visible_ids
        )
        if tuple(operation.operation_id for operation in operations) != (
            window.visible_operation_ids
        ):
            raise ValueError("feedback window is not the graph operation prefix")
        by_id = {operation.operation_id: operation for operation in operations}
        artifacts = {artifact.artifact_id: artifact for artifact in graph.artifacts}
        mutation_by_operation = {
            mutation.operation_id: mutation for mutation in graph.mutations
        }

        def ancestors(operation_id: str) -> set[str]:
            selected = set()
            pending = [operation_id]
            while pending:
                current = pending.pop()
                if current in selected or current not in by_id:
                    continue
                selected.add(current)
                pending.extend(by_id[current].parent_operation_ids)
            return selected

        ancestor_cache = {
            operation.operation_id: ancestors(operation.operation_id)
            for operation in operations
        }
        examples = []
        for operation in operations:
            mutation = mutation_by_operation.get(operation.operation_id)
            if (
                mutation is None
                or mutation.action == InternalMemoryAction.NONE
                or mutation.target_artifact_id is None
            ):
                continue
            if operation.context.policy_version != self.config.policy_version:
                raise ValueError("mutation policy version differs from dataset config")
            target = mutation.target_artifact_id
            origin = ancestor_cache[operation.operation_id]
            direct = {
                item.operation_id
                for item in operations
                if target in (*item.input_artifact_ids, *item.output_artifact_ids)
            }
            descendants = {operation.operation_id} | direct
            changed = True
            while changed:
                changed = False
                for item in operations:
                    if item.operation_id in descendants:
                        continue
                    if set(item.parent_operation_ids) & descendants:
                        descendants.add(item.operation_id)
                        changed = True
            related = set(origin) | descendants
            for direct_id in tuple(direct):
                related.update(ancestor_cache[direct_id])
            related_operations = tuple(
                item for item in operations if item.operation_id in related
            )
            attributed = []
            failure_subgraph = []
            policy_parameters = []
            attribution_record_ids = []
            attribution_methods = []
            failure_categories = []
            deterministic_failure = False
            for report in attribution_reports:
                if not set(report.window.visible_operation_ids).issubset(by_id):
                    raise ValueError("attribution report contains future operation evidence")
                for record in report.records:
                    attributed_memory_ids = {
                        artifact_id
                        for artifact_id in record.artifact_ids
                        if artifacts.get(artifact_id) is not None
                        and artifacts[artifact_id].kind == ArtifactKind.MEMORY_ARTIFACT
                    }
                    relevant = (
                        target in attributed_memory_ids
                        if attributed_memory_ids
                        else bool(
                            set(record.candidate_operation_ids) & related
                            or any(
                                operation.operation_id
                                in ancestor_cache.get(candidate, set())
                                for candidate in record.candidate_operation_ids
                            )
                        )
                    )
                    if not relevant:
                        continue
                    attribution_record_ids.append(record.attribution_id)
                    attribution_methods.append(record.method)
                    failure_categories.append(record.category)
                    attributed.extend(record.candidate_operation_ids)
                    failure_subgraph.extend(record.candidate_operation_ids)
                    policy_parameters.extend(record.policy_parameter_ids)
                    deterministic_failure = deterministic_failure or (
                        _is_delayed_deterministic_failure(
                            method=record.method,
                            category=record.category,
                            candidate_operation_ids=record.candidate_operation_ids,
                            operations=by_id,
                        )
                    )

            def matching(kind: OperationKind) -> tuple[OperationRecord, ...]:
                return tuple(item for item in related_operations if item.kind == kind)

            retrievals = matching(OperationKind.RETRIEVAL)
            queries = matching(OperationKind.FUTURE_QUERY)
            injections = matching(OperationKind.INJECTION)
            uses = matching(OperationKind.USE)
            outcomes = matching(OperationKind.DOWNSTREAM_OUTCOME)
            supersessions = matching(OperationKind.SUPERSESSION)
            censored = any(
                item.status == OperationStatus.NONE
                and item.reason_code == "observation_censored"
                for item in outcomes
            )
            used = any(
                item.status == OperationStatus.SUCCESS
                and target in item.input_artifact_ids
                for item in uses
            )
            injected = any(
                item.status == OperationStatus.SUCCESS
                and target in item.input_artifact_ids
                for item in injections
            )
            retrieved = any(
                item.status == OperationStatus.SUCCESS
                and target in item.input_artifact_ids
                for item in retrievals
            )
            superseded = any(
                item.status == OperationStatus.SUCCESS
                and target in item.input_artifact_ids
                for item in supersessions
            )
            (
                exposure_opportunity,
                entered_candidate_set,
                candidate_disposition,
                selection_propensity,
                propensity_source,
            ) = _resolve_exposure_bias(
                target=target,
                queries=queries,
                retrievals=retrievals,
            )
            successful_outcome = any(
                item.status == OperationStatus.SUCCESS for item in outcomes
            )
            failed_outcome = any(
                item.status in {OperationStatus.FAILED, OperationStatus.REJECTED}
                for item in outcomes
            )
            if not window.complete or censored:
                exposure = ExposureState.CENSORED
                label = FeedbackLabel.CENSORED
                reasons = (window.censor_reason or "observation_censored",)
            elif used and successful_outcome and (
                deterministic_failure or failed_outcome
            ):
                exposure = ExposureState.USED
                label = FeedbackLabel.UNRESOLVED
                reasons = ("conflicting_future_evidence",)
            elif used and successful_outcome:
                exposure = ExposureState.USED
                label = FeedbackLabel.POSITIVE
                reasons = ("used_with_successful_outcome",)
            elif deterministic_failure:
                exposure = (
                    ExposureState.USED
                    if used
                    else ExposureState.INJECTED_NOT_USED
                    if injected
                    else ExposureState.RETRIEVED_NOT_INJECTED
                    if retrieved
                    else ExposureState.NOT_RETRIEVED
                )
                label = FeedbackLabel.NEGATIVE
                reasons = ("deterministically_attributed_failure",)
            elif superseded:
                exposure = ExposureState.SUPERSEDED
                label = FeedbackLabel.NEGATIVE
                reasons = ("superseded_without_observed_use",)
            elif injected and not used:
                exposure = ExposureState.INJECTED_NOT_USED
                label = FeedbackLabel.NEGATIVE
                reasons = ("injected_not_used",)
            elif used:
                exposure = ExposureState.USED
                label = FeedbackLabel.UNRESOLVED
                reasons = ("used_without_attributed_success",)
            else:
                exposure = (
                    ExposureState.RETRIEVED_NOT_INJECTED
                    if retrieved
                    else ExposureState.NOT_RETRIEVED
                )
                label = FeedbackLabel.UNRESOLVED
                reasons = (
                    "retrieved_not_injected" if retrieved else "not_retrieved",
                )

            memory_node = artifacts.get(target)
            memory_revision = (
                memory_node.revision
                if memory_node is not None
                and memory_node.kind == ArtifactKind.MEMORY_ARTIFACT
                else mutation.expected_revision
            )
            contexts = [item.context for item in related_operations]
            operation_set = {item.operation_id for item in related_operations}
            value = DelayedFeedbackExample(
                "feedback-example.placeholder",
                operation.context.run_id,
                operation.context.episode_id,
                tuple(dict.fromkeys(item.episode_id for item in contexts)),
                tuple(dict.fromkeys(item.session_id for item in contexts)),
                tuple(dict.fromkeys(item.task_id for item in contexts)),
                target,
                memory_revision,
                mutation.mutation_id,
                mutation.operation_id,
                mutation.action,
                mutation.proposal_operation_ids,
                _ordered_ids(operations, operation_set, OperationKind.SOURCE_OBSERVATION),
                _ordered_ids(operations, operation_set, OperationKind.FACT_EXTRACTION),
                _ordered_ids(
                    operations,
                    operation_set,
                    OperationKind.RELATED_MEMORY_RETRIEVAL,
                ),
                _ordered_ids(
                    operations,
                    operation_set,
                    OperationKind.INTERNAL_OPERATION_DECISION,
                ),
                _ordered_ids(
                    operations,
                    operation_set,
                    OperationKind.TARGET_RESOLUTION,
                ),
                _ordered_ids(operations, operation_set, OperationKind.VALIDATION),
                _ordered_ids(
                    operations,
                    operation_set,
                    OperationKind.REREAD_VERIFICATION,
                ),
                _ordered_ids(operations, operation_set, OperationKind.FUTURE_QUERY),
                _ordered_ids(operations, operation_set, OperationKind.RETRIEVAL),
                _ordered_ids(operations, operation_set, OperationKind.INJECTION),
                _ordered_ids(operations, operation_set, OperationKind.USE),
                _ordered_ids(
                    operations,
                    operation_set,
                    OperationKind.DOWNSTREAM_OUTCOME,
                ),
                _ordered_ids(operations, operation_set, OperationKind.TOOL_BEHAVIOR),
                _ordered_ids(operations, operation_set, OperationKind.SUPERSESSION),
                _ordered_ids(operations, operation_set, OperationKind.RECOVERY),
                tuple(attribution_record_ids),
                tuple(attribution_methods),
                tuple(failure_categories),
                tuple(dict.fromkeys(attributed)),
                tuple(dict.fromkeys(failure_subgraph)),
                tuple(dict.fromkeys(policy_parameters)),
                exposure_opportunity,
                entered_candidate_set,
                candidate_disposition,
                selection_propensity,
                propensity_source,
                exposure,
                label,
                reasons,
                window.cutoff_operation_id,
                operation.context.policy_version,
                _combine_resources(related_operations),
            )
            payload = value.payload()
            payload.pop("example_id")
            examples.append(replace(
                value,
                example_id=_stable_id("feedback-example", payload),
            ))

        core = {
            "schema_version": DELAYED_FEEDBACK_SCHEMA_VERSION,
            "config": self.config.payload(),
            "window": window.payload(),
            "examples": [example.payload() for example in examples],
            "source_operation_count": len(operations),
        }
        dataset = DelayedFeedbackDataset(
            _stable_id("feedback-dataset", core),
            self.config,
            window,
            tuple(examples),
            len(operations),
        )
        audit = audit_feedback_dataset(dataset, graph)
        if not audit.ok:
            raise ValueError("feedback dataset integrity failed: " + ",".join(audit.issues))
        return dataset


def audit_feedback_dataset(
    dataset: DelayedFeedbackDataset,
    graph: OperationGraph,
) -> FeedbackDatasetAudit:
    issues = set()
    operations = {item.operation_id: item for item in graph.operations}
    artifacts = {item.artifact_id: item for item in graph.artifacts}
    mutations = {item.mutation_id: item for item in graph.mutations}
    if len(operations) != len(graph.operations):
        issues.add("duplicate_operation")
    if len(artifacts) != len(graph.artifacts):
        issues.add("duplicate_artifact")
    if len(mutations) != len(graph.mutations):
        issues.add("duplicate_mutation")
    for operation in graph.operations:
        if not set(operation.parent_operation_ids).issubset(operations):
            issues.add("orphan_operation")
        if not set((
            *operation.input_artifact_ids,
            *operation.output_artifact_ids,
        )).issubset(artifacts):
            issues.add("orphan_artifact")
    for mutation in graph.mutations:
        if (
            mutation.operation_id not in operations
            or not set(mutation.proposal_operation_ids).issubset(operations)
        ):
            issues.add("orphan_operation")
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(operation_id: str) -> None:
        if operation_id in visiting:
            issues.add("operation_cycle")
            return
        if operation_id in visited or operation_id not in operations:
            return
        visiting.add(operation_id)
        for parent in operations[operation_id].parent_operation_ids:
            visit(parent)
        visiting.remove(operation_id)
        visited.add(operation_id)

    for operation_id in operations:
        visit(operation_id)
    visiting = set()
    visited = set()

    def visit(operation_id: str) -> None:
        if operation_id in visiting:
            issues.add("operation_cycle")
            return
        if operation_id in visited or operation_id not in operations:
            return
        visiting.add(operation_id)
        for parent in operations[operation_id].parent_operation_ids:
            visit(parent)
        visiting.remove(operation_id)
        visited.add(operation_id)

    for operation_id in operations:
        visit(operation_id)
    visible = set(dataset.window.visible_operation_ids)
    cutoff_index = {
        item.operation_id: index for index, item in enumerate(graph.operations)
    }.get(dataset.window.cutoff_operation_id)
    if cutoff_index is None:
        issues.add("missing_cutoff")
    elif tuple(
        item.operation_id for item in graph.operations[: cutoff_index + 1]
    ) != dataset.window.visible_operation_ids:
        issues.add("window_prefix_mismatch")
    example_ids = [example.example_id for example in dataset.examples]
    if len(example_ids) != len(set(example_ids)):
        issues.add("duplicate_example")
    for example in dataset.examples:
        example_payload = example.payload()
        example_payload.pop("example_id")
        if example.example_id != _stable_id("feedback-example", example_payload):
            issues.add("example_identity_mismatch")
        if not set(example.operation_ids).issubset(operations):
            issues.add("orphan_operation")
        if not set(example.operation_ids).issubset(visible):
            issues.add("future_leakage")
        kind_buckets = (
            (example.source_operation_ids, OperationKind.SOURCE_OBSERVATION),
            (example.extraction_operation_ids, OperationKind.FACT_EXTRACTION),
            (
                example.related_retrieval_operation_ids,
                OperationKind.RELATED_MEMORY_RETRIEVAL,
            ),
            (
                example.decision_operation_ids,
                OperationKind.INTERNAL_OPERATION_DECISION,
            ),
            (
                example.target_resolution_operation_ids,
                OperationKind.TARGET_RESOLUTION,
            ),
            (example.validation_operation_ids, OperationKind.VALIDATION),
            (
                example.verification_operation_ids,
                OperationKind.REREAD_VERIFICATION,
            ),
            (example.query_operation_ids, OperationKind.FUTURE_QUERY),
            (example.retrieval_operation_ids, OperationKind.RETRIEVAL),
            (example.injection_operation_ids, OperationKind.INJECTION),
            (example.use_operation_ids, OperationKind.USE),
            (example.outcome_operation_ids, OperationKind.DOWNSTREAM_OUTCOME),
            (example.tool_operation_ids, OperationKind.TOOL_BEHAVIOR),
            (example.supersession_operation_ids, OperationKind.SUPERSESSION),
            (example.recovery_operation_ids, OperationKind.RECOVERY),
        )
        if any(
            operation_id in operations
            and operations[operation_id].kind != expected_kind
            for operation_ids, expected_kind in kind_buckets
            for operation_id in operation_ids
        ) or (
            example.mutation_operation_id in operations
            and operations[example.mutation_operation_id].kind
            != OperationKind.MUTATION
        ):
            issues.add("operation_kind_mismatch")
        if example.memory_artifact_id not in artifacts:
            issues.add("orphan_artifact")
        if not set(example.policy_parameter_ids).issubset(artifacts):
            issues.add("orphan_artifact")
        elif any(
            artifacts[artifact_id].kind != ArtifactKind.POLICY_PARAMETER
            for artifact_id in example.policy_parameter_ids
        ):
            issues.add("policy_parameter_mismatch")
        mutation = mutations.get(example.mutation_id)
        if (
            mutation is None
            or mutation.operation_id != example.mutation_operation_id
            or mutation.target_artifact_id != example.memory_artifact_id
            or mutation.action != example.mutation_action
            or mutation.proposal_operation_ids != example.proposal_operation_ids
        ):
            issues.add("mutation_mismatch")
        if example.policy_version != dataset.config.policy_version:
            issues.add("policy_version_mismatch")
        if example.observation_cutoff_operation_id != dataset.window.cutoff_operation_id:
            issues.add("cutoff_mismatch")
        mutation_operation = operations.get(example.mutation_operation_id)
        if mutation_operation is not None and (
            example.run_id != mutation_operation.context.run_id
            or example.source_episode_id != mutation_operation.context.episode_id
        ):
            issues.add("provenance_mismatch")
        referenced = [
            operation
            for operation in graph.operations
            if operation.operation_id in set(example.operation_ids)
        ]
        if referenced and (
            example.observation_episode_ids
            != tuple(dict.fromkeys(item.context.episode_id for item in referenced))
            or example.session_ids
            != tuple(dict.fromkeys(item.context.session_id for item in referenced))
            or example.task_ids
            != tuple(dict.fromkeys(item.context.task_id for item in referenced))
        ):
            issues.add("provenance_mismatch")
        artifact = artifacts.get(example.memory_artifact_id)
        if (
            artifact is not None
            and artifact.revision is not None
            and example.memory_revision != artifact.revision
        ):
            issues.add("revision_mismatch")
        retrievals = tuple(
            operations[operation_id]
            for operation_id in example.retrieval_operation_ids
            if operation_id in operations
        )
        queries = tuple(
            operations[operation_id]
            for operation_id in example.query_operation_ids
            if operation_id in operations
        )
        injections = tuple(
            operations[operation_id]
            for operation_id in example.injection_operation_ids
            if operation_id in operations
        )
        uses = tuple(
            operations[operation_id]
            for operation_id in example.use_operation_ids
            if operation_id in operations
        )
        outcomes = tuple(
            operations[operation_id]
            for operation_id in example.outcome_operation_ids
            if operation_id in operations
        )
        supersessions = tuple(
            operations[operation_id]
            for operation_id in example.supersession_operation_ids
            if operation_id in operations
        )
        target = example.memory_artifact_id
        used = any(
            item.status == OperationStatus.SUCCESS
            and target in item.input_artifact_ids
            for item in uses
        )
        injected = any(
            item.status == OperationStatus.SUCCESS
            and target in item.input_artifact_ids
            for item in injections
        )
        retrieved = any(
            item.status == OperationStatus.SUCCESS
            and target in item.input_artifact_ids
            for item in retrievals
        )
        expected_exposure_bias = _resolve_exposure_bias(
            target=target,
            queries=queries,
            retrievals=retrievals,
        )
        actual_exposure_bias = (
            example.exposure_opportunity,
            example.entered_candidate_set,
            example.candidate_disposition,
            example.selection_propensity,
            example.propensity_source,
        )
        if actual_exposure_bias != expected_exposure_bias:
            issues.add("exposure_bias_evidence_mismatch")
        superseded = any(
            item.status == OperationStatus.SUCCESS
            and target in item.input_artifact_ids
            for item in supersessions
        )
        successful_outcome = any(
            item.status == OperationStatus.SUCCESS for item in outcomes
        )
        failed_outcome = any(
            item.status in {OperationStatus.FAILED, OperationStatus.REJECTED}
            for item in outcomes
        )
        censored = any(
            item.status == OperationStatus.NONE
            and item.reason_code == "observation_censored"
            for item in outcomes
        )
        deterministic_failure = any(
            _is_delayed_deterministic_failure(
                method=method,
                category=category,
                candidate_operation_ids=example.attributed_operation_ids,
                operations=operations,
            )
            for method, category in zip(
                example.attribution_methods,
                example.failure_categories,
            )
        )
        if not dataset.window.complete or censored:
            expected_exposure = ExposureState.CENSORED
            expected_label = FeedbackLabel.CENSORED
        elif used and successful_outcome and (
            deterministic_failure or failed_outcome
        ):
            expected_exposure = ExposureState.USED
            expected_label = FeedbackLabel.UNRESOLVED
        elif used and successful_outcome:
            expected_exposure = ExposureState.USED
            expected_label = FeedbackLabel.POSITIVE
        elif deterministic_failure:
            expected_exposure = (
                ExposureState.USED
                if used
                else ExposureState.INJECTED_NOT_USED
                if injected
                else ExposureState.RETRIEVED_NOT_INJECTED
                if retrieved
                else ExposureState.NOT_RETRIEVED
            )
            expected_label = FeedbackLabel.NEGATIVE
        elif superseded:
            expected_exposure = ExposureState.SUPERSEDED
            expected_label = FeedbackLabel.NEGATIVE
        elif injected and not used:
            expected_exposure = ExposureState.INJECTED_NOT_USED
            expected_label = FeedbackLabel.NEGATIVE
        elif used:
            expected_exposure = ExposureState.USED
            expected_label = FeedbackLabel.UNRESOLVED
        else:
            expected_exposure = (
                ExposureState.RETRIEVED_NOT_INJECTED
                if retrieved
                else ExposureState.NOT_RETRIEVED
            )
            expected_label = FeedbackLabel.UNRESOLVED
        if (
            example.exposure_state != expected_exposure
            or example.label != expected_label
        ):
            issues.add("label_evidence_mismatch")
    core = {
        "schema_version": dataset.schema_version,
        "config": dataset.config.payload(),
        "window": dataset.window.payload(),
        "examples": [example.payload() for example in dataset.examples],
        "source_operation_count": dataset.source_operation_count,
    }
    if dataset.dataset_id != _stable_id("feedback-dataset", core):
        issues.add("dataset_identity_mismatch")
    if dataset.source_operation_count != len(dataset.window.visible_operation_ids):
        issues.add("source_operation_count_mismatch")
    serialized = json.loads(_canonical(dataset.payload()))

    def inspect(value: object, key: str | None = None) -> None:
        if key is not None and key.casefold() in _FORBIDDEN_KEYS:
            issues.add("raw_or_hidden_field")
        if isinstance(value, Mapping):
            for child_key, child in value.items():
                inspect(child, str(child_key))
        elif isinstance(value, list):
            for child in value:
                inspect(child)

    inspect(serialized)
    counts = {label: 0 for label in FeedbackLabel}
    for example in dataset.examples:
        counts[example.label] += 1
    return FeedbackDatasetAudit(
        not issues,
        tuple(sorted(issues)),
        len(dataset.examples),
        tuple((label, counts[label]) for label in FeedbackLabel),
    )
