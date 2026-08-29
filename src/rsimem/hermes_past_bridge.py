"""Opt-in RSIMem read bridge for the in-process PAST-Bench Hermes adapter."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Mapping, Sequence

from .hermes_integration import (
    HermesAdapterExecutionError,
    HermesAdapterFailurePolicy,
    HermesExecutionMode,
    HermesExperimentConfig,
    _bound_hermes_skills_dir,
    _materialize_procedural_hits,
    build_configured_hermes_runtime,
)
from .ledger import LifecycleLedgerObserver, MemoryLedgerObserver
from .lifecycle import (
    EvaluationTrigger,
    ContextSnapshot,
    HermesLifecycleConfig,
    HermesLifecycleDryRunResult,
    HermesLifecycleDryRunRuntime,
    HermesStateSnapshotCollector,
    TaskLifecycleState,
)
from .memory import MemoryEvent, MemoryEventKind, MemoryKind, MemoryQuery
from .memory.trigger_policy import (
    DeterministicTriggerPolicy,
    HermesTriggerEventAdapter,
    TriggerObservation,
)
from .memory.source_selection_policy import DeterministicSourceSelectionPolicy
from .memory.policy_contracts import (
    AdmissionDecision,
    DecisionAction,
    ExecutionStatus,
    ExtractionDecision,
    MutationKind,
    SourceSelectionDecision,
    TriggerEvent,
)
from .memory.admission_policy import DeterministicAdmissionPolicy
from .memory.exposure_policy import DeterministicExposurePolicy
from .memory.policy_evidence import JsonPolicyDecisionLedger
from .memory.process_feedback import (
    JsonProcessFeedbackLedger,
    ProcessEvent,
    ProcessEventKind,
    ProcessEventStatus,
)
from .memory.live_writeback import (
    ExtractionPromptRuntimeScope,
    StaticSemanticBoundaryResult,
    StaticSemanticWritebackConfig,
    StaticSemanticWritebackRuntime,
)
from .extraction_validation_runtime import load_extraction_matched_trial_profile
from .memory.adaptive_policy_store import JsonAdaptivePolicyStore
from .memory.future_trace import (
    SemanticFeedbackContract,
    SemanticFeedbackResolver,
    SemanticFutureEvidence,
    SemanticFutureTraceRecorder,
    SemanticOutcomeEvidence,
)
from .memory.extraction_feedback import (
    ArtifactSemanticBinding,
    DeploymentObservation,
    ExposureMode,
    ExtractionFeedbackBuilder,
    FeedbackOperationJoin,
    FutureMemoryEvidence,
    ObservableToolEvent,
    detect_current_input_semantic_keys,
    detect_source_semantic_keys,
)
from .memory.extraction_optimizer_builder import ExtractionFactContent
from .memory.extraction_optimizer_capture import (
    ExtractionOptimizerFeedbackCapture,
    ExtractionOptimizerSourceCapture,
    JsonExtractionOptimizerCaptureLog,
)
from .memory.extraction_projection import (
    JsonLiveExtractionFeedbackRecordLog,
    JsonExtractionSourceRecordStore,
    LiveExtractionFeedbackRecord,
    Mem0FlatExtractionSourceProjector,
)
from .memory.operation_graph import OperationContext
from .memory_systems.mem0_flat import CompletionClient, FrozenMem0UtilityGate


class _PromptMemoryStore:
    def __init__(self, bridge: "HermesPastBenchBridge", native_store: object) -> None:
        self._bridge = bridge
        self._native = native_store
        self._snapshots: dict[str, tuple[Any, ...]] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self._native, name)

    def format_for_system_prompt(self, target: str) -> str | None:
        if not self._bridge.uses_adapter:
            result = self._native.format_for_system_prompt(target)
            self._bridge.observe_query(
                MemoryKind.SEMANTIC,
                "",
                namespace=target,
                limit=100,
                surface="system_prompt" if result else None,
            )
            self._bridge.record_semantic_prompt(result, target)
            return result

        def adapter_read() -> str | None:
            hits = self._snapshots.get(target)
            if hits is None:
                hits = self._bridge.runtime.query(MemoryQuery(
                    MemoryKind.SEMANTIC,
                    "",
                    namespace=target,
                    limit=100,
                ))
                self._snapshots[target] = hits
            query_digest = hashlib.sha256(
                json.dumps(
                    {"kind": "semantic", "namespace": target, "query": "", "limit": 100},
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            self._bridge._record_process_observation(
                kind=ProcessEventKind.RETRIEVAL,
                status=ProcessEventStatus.SUCCESS if hits else ProcessEventStatus.FAILED,
                host_event_id=self._bridge._last_host_event_id or f"event.semantic-query.{query_digest[:40]}",
                source_revision=self._bridge._last_host_source_revision or self._bridge._exposure_context_revision(),
                input_payload={"query_digest": query_digest, "namespace": target, "limit": 100},
                output_payload={"hit_count": len(hits), "exposure": "eager_system_prompt"},
                reason_codes=("decision_observed",) if hits else ("retrieval_miss",),
                execution_receipt_ids=(f"receipt.semantic-query.{query_digest[:24]}",),
            )
            if not hits:
                return None
            self._bridge.runtime.mark_injected(hits, surface="system_prompt")
            return self._native._render_block(
                target,
                [hit.artifact.content for hit in hits],
            )

        adapter_result = self._bridge.adapter_call(
            "system_prompt",
            adapter_read,
            lambda: self._native.format_for_system_prompt(target),
        )
        self._bridge.verify_projection(
            MemoryKind.SEMANTIC,
            "hermes-native-semantic",
            "system_prompt",
            adapter_result,
            lambda: self._native.format_for_system_prompt(target),
        )
        self._bridge.record_semantic_prompt(
            adapter_result,
            target,
            artifact_ids=tuple(hit.artifact.artifact_id for hit in self._snapshots.get(target, ())),
        )
        return adapter_result


class _SessionDb:
    def __init__(
        self,
        bridge: "HermesPastBenchBridge",
        native_db: object,
        current_session_id: str | None,
    ) -> None:
        self._bridge = bridge
        self._native = native_db
        self._current_session_id = current_session_id
        self._hits_by_session: dict[str, dict[str, Any]] = {}
        self._sessions: dict[str, dict[str, Any]] = {}
        self._conversations: dict[str, tuple[dict[str, Any], ...]] = {}
        self._injected_sessions: set[str] = set()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._native, name)

    def search_messages(
        self,
        *,
        query: str,
        source_filter: list[str] | None = None,
        role_filter: list[str] | None = None,
        exclude_sources: list[str] | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        native_call = lambda: self._native.search_messages(
            query=query,
            source_filter=source_filter,
            role_filter=role_filter,
            exclude_sources=exclude_sources,
            limit=limit,
            offset=offset,
        )
        if not self._bridge.uses_adapter:
            results = native_call()
            self._bridge.record_native_search(query, limit, results)
            return results

        def adapter_read() -> list[dict[str, Any]]:
            hits = self._bridge.runtime.query(MemoryQuery(
                MemoryKind.EPISODIC,
                query,
                limit=limit,
                filters={
                    "source_filter": source_filter,
                    "role_filter": role_filter,
                    "exclude_sources": exclude_sources,
                    "offset": offset,
                },
            ))
            query_digest = hashlib.sha256(
                json.dumps(
                    {"kind": "episodic", "query": query, "limit": limit, "offset": offset},
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            self._bridge._record_process_observation(
                kind=ProcessEventKind.RETRIEVAL,
                status=ProcessEventStatus.SUCCESS if hits else ProcessEventStatus.FAILED,
                host_event_id=self._bridge._last_host_event_id or f"event.episodic-query.{query_digest[:40]}",
                source_revision=self._bridge._last_host_source_revision or self._bridge._exposure_context_revision(),
                input_payload={"query_digest": query_digest, "limit": limit, "offset": offset},
                output_payload={"hit_count": len(hits)},
                reason_codes=("decision_observed",) if hits else ("retrieval_miss",),
                execution_receipt_ids=(f"receipt.episodic-query.{query_digest[:24]}",),
            )
            results = []
            for hit in hits:
                metadata = hit.artifact.metadata
                role = str(metadata.get("role") or "")
                source = str(metadata.get("source") or "")
                self._cache_projection(hit)
                results.append({
                    "id": metadata.get("message_id"),
                    "session_id": hit.artifact.namespace,
                    "role": role,
                    "snippet": metadata.get("snippet"),
                    "timestamp": metadata.get("timestamp"),
                    "tool_name": metadata.get("tool_name"),
                    "source": source,
                    "model": metadata.get("model"),
                    "session_started": metadata.get("session_started"),
                    "context": [
                        {"role": str(role), "content": str(content)}
                        for role, content in metadata.get("context") or ()
                    ],
                })
            return results

        adapter_result = self._bridge.adapter_call("session_search", adapter_read, native_call)
        self._bridge.verify_projection(
            MemoryKind.EPISODIC,
            "hermes-native-episodic",
            "session_search",
            adapter_result,
            native_call,
        )
        return adapter_result

    def _cache_projection(self, hit: Any) -> None:
        lineage = hit.artifact.metadata.get("session_lineage")
        if not isinstance(lineage, (list, tuple)) or not lineage:
            raise ValueError("episodic hit requires a session_lineage projection")
        projected_ids = []
        for item in lineage:
            if not isinstance(item, (list, tuple)) or len(item) != 3:
                raise ValueError("invalid episodic session_lineage entry")
            session_id, session, conversation = item
            session_id = str(session_id or "")
            if (
                not session_id
                or not isinstance(session, dict)
                or not isinstance(conversation, (list, tuple))
                or any(not isinstance(message, dict) for message in conversation)
            ):
                raise ValueError("invalid episodic session projection")
            session_value = dict(session)
            conversation_value = tuple(dict(message) for message in conversation)
            if session_id in self._sessions and self._sessions[session_id] != session_value:
                raise ValueError("conflicting episodic session projection")
            if (
                session_id in self._conversations
                and self._conversations[session_id] != conversation_value
            ):
                raise ValueError("conflicting episodic conversation projection")
            self._sessions[session_id] = session_value
            self._conversations[session_id] = conversation_value
            projected_ids.append(session_id)
        if hit.artifact.namespace != projected_ids[0]:
            raise ValueError("episodic hit namespace must start its session lineage")
        for session_id in projected_ids:
            self._hits_by_session.setdefault(session_id, {})[
                hit.artifact.artifact_id
            ] = hit

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        if not self._bridge.uses_adapter:
            return self._native.get_session(session_id)
        session = self._sessions.get(session_id)
        if session is not None:
            result = dict(session)
            self._bridge.verify_projection(
                MemoryKind.EPISODIC,
                "hermes-native-episodic",
                "session_get",
                result,
                lambda: self._native.get_session(session_id),
            )
            return result
        if session_id == self._current_session_id:
            return self._native.get_session(session_id)
        raise KeyError(f"session is absent from episodic projection: {session_id}")

    def get_messages_as_conversation(self, session_id: str) -> list[dict[str, Any]]:
        if not self._bridge.uses_adapter:
            return self._native.get_messages_as_conversation(session_id)
        messages = self._conversations.get(session_id)
        if messages is None:
            raise KeyError(f"conversation is absent from episodic projection: {session_id}")
        hits = tuple(self._hits_by_session.get(session_id, {}).values())
        if hits and session_id not in self._injected_sessions:
            self._bridge.runtime.mark_injected(hits, surface="session_search")
            self._injected_sessions.add(session_id)
        result = [dict(message) for message in messages]
        self._bridge.verify_projection(
            MemoryKind.EPISODIC,
            "hermes-native-episodic",
            "session_conversation",
            result,
            lambda: self._native.get_messages_as_conversation(session_id),
        )
        return result


class HermesPastBenchBridge:
    """Attach typed memory reads to one live PAST-Bench Hermes agent."""

    def __init__(
        self,
        hermes_home: Path,
        config: HermesExperimentConfig,
        *,
        evidence_path: Path,
        run_id: str,
        trace_id: str,
        episode_id: str,
        session_id: str,
        task_id: str,
        experiment_variant: str,
        family_id: str | None = None,
        stage: str | None = None,
        lifecycle_config: HermesLifecycleConfig | None = None,
        lifecycle_evidence_path: Path | None = None,
        lifecycle_receipt_path: Path | None = None,
        process_feedback_path: Path | None = None,
        lifecycle_complete: Callable[[str], str] | None = None,
        static_writeback_config: StaticSemanticWritebackConfig | None = None,
        static_completion_client: CompletionClient | None = None,
        static_operation_evidence_path: Path | None = None,
        static_mutation_receipt_path: Path | None = None,
    ) -> None:
        if config.mode == HermesExecutionMode.NATIVE:
            raise ValueError("native mode must not construct an RSIMem bridge")
        self.config = config
        self.evidence_path = evidence_path.expanduser().resolve()
        self._run_id = run_id
        self._trace_id = trace_id
        self._episode_id = episode_id
        self._session_id = session_id
        self._task_id = task_id
        self._family_id = family_id
        self._stage = stage
        self._snapshot_collector = HermesStateSnapshotCollector()
        self.ledger = MemoryLedgerObserver(
            run_id=run_id,
            variant=experiment_variant,
            trace_id=trace_id,
            episode_id=episode_id,
            session_id=session_id,
            task_id=task_id,
            family_id=family_id,
            stage=stage,
            execution_mode=config.mode.value,
            output_path=self.evidence_path,
        )
        self.runtime = build_configured_hermes_runtime(
            hermes_home,
            config,
            observers=(self.ledger,),
        )
        self._tool_handlers: dict[str, Callable[..., str]] = {}
        self._agent: object | None = None
        self._lifecycle_results: list[HermesLifecycleDryRunResult] = []
        self._lifecycle_failures: list[tuple[str, str]] = []
        self._trigger_adapter = HermesTriggerEventAdapter()
        self._trigger_policy = DeterministicTriggerPolicy()
        self._trigger_observations: list[TriggerObservation] = []
        self._source_selection_policy = DeterministicSourceSelectionPolicy()
        self._source_selection_decisions: list[SourceSelectionDecision] = []
        self._last_host_event_id: str | None = None
        self._last_host_source_revision: str | None = None
        self._policy_evidence = JsonPolicyDecisionLedger(
            self.evidence_path.with_name("rsimem_policy_decisions.jsonl"),
            variant=experiment_variant,
            trace_id=trace_id,
            family_id=family_id,
            stage=stage,
        )
        self._process_feedback = JsonProcessFeedbackLedger(
            process_feedback_path
            or self.evidence_path.with_name("rsimem_process_feedback.jsonl")
        )
        self._admission_policy = DeterministicAdmissionPolicy()
        self._exposure_policy = DeterministicExposurePolicy()
        self._policy_decision_ids: set[str] = {
            str(event.get("decisionId"))
            for event in self._policy_evidence.events
            if isinstance(event.get("decisionId"), str)
        }
        self._static_results: list[StaticSemanticBoundaryResult] = []
        self._static_failures: list[tuple[str, str]] = []
        self._semantic_futures: list[tuple[SemanticFutureEvidence, str]] = []
        self._semantic_outcomes_recorded = False
        lifecycle_config = lifecycle_config or HermesLifecycleConfig()
        resolved_lifecycle_evidence_path = (
            lifecycle_evidence_path
            or self.evidence_path.with_name("rsimem_lifecycle_events.jsonl")
        )
        self.lifecycle = (
            HermesLifecycleDryRunRuntime(
                lifecycle_config,
                run_id=run_id,
                episode_id=episode_id,
                session_id=session_id,
                task_id=task_id,
                variant=experiment_variant,
                trace_id=trace_id,
                receipt_path=(
                    lifecycle_receipt_path
                    or self.evidence_path.with_name("rsimem_lifecycle_receipts.json")
                ),
                evidence_path=resolved_lifecycle_evidence_path,
                family_id=family_id,
                stage=stage,
                injected_complete=lifecycle_complete,
            )
            if lifecycle_config.enabled
            else None
        )
        static_writeback_config = (
            static_writeback_config or StaticSemanticWritebackConfig()
        )
        if static_writeback_config.enabled:
            if config.mode != HermesExecutionMode.NATIVE_LEDGER:
                raise ValueError("static semantic writeback requires native+ledger mode")
            if static_completion_client is None:
                raise ValueError("static semantic writeback requires a completion client")
            static_ingestion_observer = (
                self.lifecycle.observer
                if self.lifecycle is not None
                else LifecycleLedgerObserver(
                    variant=experiment_variant,
                    trace_id=trace_id,
                    family_id=family_id,
                    stage=stage,
                    output_path=resolved_lifecycle_evidence_path,
                )
            )
            adaptive_store = None
            extraction_profile = None
            if static_writeback_config.adaptive_enabled:
                relative_store = Path(
                    static_writeback_config.adaptive_policy_store_path or ""
                )
                if relative_store.is_absolute() or ".." in relative_store.parts:
                    raise ValueError(
                        "adaptive policy store must be relative to Hermes home"
                    )
                store_path = (hermes_home / relative_store).resolve()
                if not store_path.is_relative_to(hermes_home.resolve()):
                    raise ValueError("adaptive policy store escapes Hermes home")
                adaptive_store = JsonAdaptivePolicyStore(
                    store_path,
                    trusted_root_policy_versions=(
                        static_writeback_config.adaptive_trusted_roots
                    ),
                )
            if static_writeback_config.matched_extraction_enabled:
                extraction_config_path = Path(
                    static_writeback_config.extraction_runtime_config_path or ""
                ).expanduser().resolve()
                capture_root = self.evidence_path.parent.resolve()
                if not extraction_config_path.is_relative_to(capture_root):
                    raise ValueError(
                        "extraction runtime config must stay inside capture artifacts"
                    )
                extraction_profile = load_extraction_matched_trial_profile(
                    extraction_config_path
                )
            self.static_writeback = StaticSemanticWritebackRuntime(
                hermes_home,
                static_completion_client,
                operation_evidence_path=(
                    static_operation_evidence_path
                    or self.evidence_path.with_name(
                        "rsimem_semantic_operations.jsonl"
                    )
                ),
                mutation_receipt_path=(
                    static_mutation_receipt_path
                    or hermes_home / ".rsimem" / "semantic_mutation_receipts.json"
                ),
                observer=self.ledger,
                ingestion_observer=static_ingestion_observer,
                utility_gate=(
                    FrozenMem0UtilityGate()
                    if static_writeback_config.utility_enabled
                    else None
                ),
                adaptive_policy_store=adaptive_store,
                adaptive_parameters=static_writeback_config.adaptive_parameters,
                require_adaptive_policy=static_writeback_config.adaptive_enabled,
                extraction_policy_artifact=(
                    extraction_profile.candidate
                    if extraction_profile is not None
                    else None
                ),
                expected_extraction_policy_artifact_id=(
                    extraction_profile.candidate.artifact_id
                    if extraction_profile is not None
                    else None
                ),
                expected_extraction_policy_artifact_digest=(
                    extraction_profile.candidate.artifact_digest
                    if extraction_profile is not None
                    else None
                ),
                extraction_runtime_scope=(
                    ExtractionPromptRuntimeScope.MATCHED_VALIDATION
                    if extraction_profile is not None
                    else ExtractionPromptRuntimeScope.ROOT_STATIC
                ),
                extraction_trial_id=(
                    extraction_profile.trial_id
                    if extraction_profile is not None
                    else None
                ),
            )
            if (
                static_writeback_config.feedback_contract
                != SemanticFeedbackContract.DISABLED
            ):
                if not family_id or not stage:
                    raise ValueError(
                        "semantic feedback contract requires family and stage"
                    )
                descriptor = self.static_writeback.policy.descriptor
                self.semantic_future_recorder = SemanticFutureTraceRecorder(
                    self.static_writeback.operation_recorder,
                    OperationContext(
                        run_id,
                        episode_id,
                        session_id,
                        task_id,
                        descriptor.policy_version,
                        descriptor.prompt_version,
                        descriptor.framework_version,
                    ),
                )
                self.semantic_feedback_resolver = SemanticFeedbackResolver(
                    static_writeback_config.feedback_contract,
                    family_id=family_id,
                    stage=stage,
                )
                feedback_registry = self.semantic_feedback_resolver.registry
                self.extraction_source_projector = (
                    Mem0FlatExtractionSourceProjector(feedback_registry)
                )
                self.extraction_source_store = JsonExtractionSourceRecordStore(
                    Path(hermes_home) / ".rsimem" / "extraction_sources.jsonl"
                )
                self.extraction_feedback_builder = ExtractionFeedbackBuilder(
                    feedback_registry
                )
                self.extraction_feedback_log = JsonLiveExtractionFeedbackRecordLog(
                    self.evidence_path.with_name(
                        "rsimem_extraction_feedback.jsonl"
                    )
                )
                self.extraction_optimizer_capture_log = (
                    JsonExtractionOptimizerCaptureLog(
                        self.evidence_path.with_name(
                            "extraction_optimizer_capture.jsonl"
                        )
                    )
                )
            else:
                self.semantic_future_recorder = None
                self.semantic_feedback_resolver = None
                self.extraction_source_projector = None
                self.extraction_source_store = None
                self.extraction_feedback_builder = None
                self.extraction_feedback_log = None
                self.extraction_optimizer_capture_log = None
        else:
            self.static_writeback = None
            self.semantic_future_recorder = None
            self.semantic_feedback_resolver = None
            self.extraction_source_projector = None
            self.extraction_source_store = None
            self.extraction_feedback_builder = None
            self.extraction_feedback_log = None
            self.extraction_optimizer_capture_log = None
        self._closed = False

    @property
    def uses_adapter(self) -> bool:
        return self.config.mode == HermesExecutionMode.ADAPTER_LEDGER

    def attach(self, agent: object) -> None:
        self._agent = agent
        memory_store = getattr(agent, "_memory_store", None)
        if memory_store is not None:
            agent._memory_store = _PromptMemoryStore(self, memory_store)
        session_db = getattr(agent, "_session_db", None)
        if session_db is not None:
            agent._session_db = _SessionDb(
                self,
                session_db,
                str(getattr(agent, "session_id", "") or "") or None,
            )
        self._wrap_skill_handlers()

    @property
    def lifecycle_results(self) -> tuple[HermesLifecycleDryRunResult, ...]:
        return tuple(self._lifecycle_results)

    @property
    def lifecycle_failures(self) -> tuple[tuple[str, str], ...]:
        return tuple(self._lifecycle_failures)

    @property
    def trigger_observations(self) -> tuple[TriggerObservation, ...]:
        """All observed trigger candidates, including shadow SKIP decisions."""

        return tuple(self._trigger_observations)

    @property
    def source_selection_decisions(self) -> tuple[SourceSelectionDecision, ...]:
        """Deterministic source choices observed at successful boundaries."""

        return tuple(self._source_selection_decisions)

    @property
    def policy_evidence(self) -> tuple[dict[str, object], ...]:
        return self._policy_evidence.events

    @property
    def process_feedback(self) -> tuple[ProcessEvent, ...]:
        """Content-free process observations captured for this bridge run."""

        return self._process_feedback.events

    @property
    def static_results(self) -> tuple[StaticSemanticBoundaryResult, ...]:
        return tuple(self._static_results)

    @property
    def static_failures(self) -> tuple[tuple[str, str], ...]:
        return tuple(self._static_failures)

    def on_task_completed(self, result: Mapping[str, Any]) -> None:
        """Receive the explicit post-conversation task boundary from PAST."""

        self._record_semantic_outcomes(result)
        if result.get("completed") is not True:
            return
        if self.lifecycle is not None:
            self._process_lifecycle_boundary(
                EvaluationTrigger.TASK_COMPLETED,
                TaskLifecycleState.COMPLETED,
            )
        if self.static_writeback is None:
            return
        try:
            snapshot = self._collect_completed_snapshot()
            trigger_event = self._trigger_adapter.from_snapshot(
                snapshot,
                EvaluationTrigger.TASK_COMPLETED.value,
                turn_index=sum(
                    1 for segment in snapshot.segments if segment.role == "user"
                ),
            )
            source_decision = self._observe_policy_boundary(snapshot, trigger_event)
            results = self.static_writeback.process_completed_snapshot(
                snapshot,
                selected_segment_ids=(
                    source_decision.selected_segment_ids
                    if source_decision is not None
                    and source_decision.action == DecisionAction.RUN
                    else None
                ),
            )
            for compiled in results:
                if not any(
                    item.compilation_id == compiled.compilation_id
                    for item in self._static_results
                ):
                    self._static_results.append(compiled)
                self._record_extraction_source(compiled)
                self._record_static_policy_evidence(compiled, snapshot, trigger_event)
        except Exception as exc:
            self._static_failures.append((
                EvaluationTrigger.TASK_COMPLETED.value,
                type(exc).__name__,
            ))
            if self.extraction_source_store is not None:
                raise

    def on_session_end(self, *, task_state: TaskLifecycleState = TaskLifecycleState.COMPLETED) -> None:
        """Observe a real session-end boundary without opening writeback."""

        self._observe_shadow_host_boundary(
            EvaluationTrigger.SESSION_END,
            task_state=TaskLifecycleState(task_state),
        )

    def on_turn_interval(self, *, turn_index: int) -> None:
        """Observe a turn interval; this boundary is shadow-only in 2B."""

        self._observe_shadow_host_boundary(
            EvaluationTrigger.TURN_INTERVAL,
            task_state=TaskLifecycleState.ACTIVE,
            turn_index=turn_index,
        )

    def on_tool_boundary(self, *, turn_index: int | None = None) -> None:
        """Observe a tool boundary, preserving any open tool closure."""

        self._observe_shadow_host_boundary(
            EvaluationTrigger.TOOL_BOUNDARY,
            task_state=TaskLifecycleState.ACTIVE,
            turn_index=turn_index,
            tool_boundary_observed=True,
            allow_open_tool_closure=True,
        )

    def on_context_pressure(self, *, context_tokens: int) -> None:
        """Observe context pressure from a host-provided token count."""

        self._observe_shadow_host_boundary(
            EvaluationTrigger.CONTEXT_PRESSURE,
            task_state=TaskLifecycleState.ACTIVE,
            context_tokens=context_tokens,
        )

    def on_manual_trigger(self) -> None:
        """Observe an explicitly authorized manual boundary."""

        self._observe_shadow_host_boundary(
            EvaluationTrigger.MANUAL,
            task_state=TaskLifecycleState.ACTIVE,
            manual_authorized=True,
        )

    def _observe_shadow_host_boundary(
        self,
        trigger: EvaluationTrigger,
        *,
        task_state: TaskLifecycleState,
        turn_index: int | None = None,
        context_tokens: int | None = None,
        tool_boundary_observed: bool = False,
        manual_authorized: bool = False,
        allow_open_tool_closure: bool = False,
    ) -> None:
        agent = self._agent
        if agent is None:
            raise ValueError("Hermes shadow boundary requires an attached agent")
        session_db = getattr(agent, "_session_db", None)
        native_session_id = str(getattr(agent, "session_id", "") or "")
        if session_db is None or not native_session_id:
            raise ValueError("Hermes shadow boundary requires a persisted native session")
        rows = session_db.get_messages(native_session_id)
        snapshot = self._snapshot_collector.collect(
            rows,
            run_id=self._run_id,
            episode_id=self._episode_id,
            session_id=self._session_id,
            task_id=self._task_id,
            task_state=task_state,
            lifecycle_state=trigger.value,
            source_ref=f"hermes_state:session:{native_session_id}",
            allow_open_tool_closure=allow_open_tool_closure,
        )
        event = self._trigger_adapter.from_snapshot(
            snapshot,
            trigger.value,
            context_tokens=context_tokens,
            turn_index=turn_index,
            tool_boundary_observed=tool_boundary_observed,
            manual_authorized=manual_authorized,
        )
        self._observe_policy_boundary(snapshot, event)

    def _collect_completed_snapshot(self) -> ContextSnapshot:
        agent = self._agent
        if agent is None:
            raise ValueError("Hermes semantic bridge is not attached")
        session_db = getattr(agent, "_session_db", None)
        native_session_id = str(getattr(agent, "session_id", "") or "")
        if session_db is None or not native_session_id:
            raise ValueError("semantic compilation requires a persisted native session")
        rows = session_db.get_messages(native_session_id)
        return self._snapshot_collector.collect(
            rows,
            run_id=self._run_id,
            episode_id=self._episode_id,
            session_id=self._session_id,
            task_id=self._task_id,
            task_state=TaskLifecycleState.COMPLETED,
            lifecycle_state=EvaluationTrigger.TASK_COMPLETED.value,
            source_ref=f"hermes_state:session:{native_session_id}",
        )

    def record_semantic_prompt(
        self,
        prompt: str | None,
        namespace: str,
        *,
        artifact_ids: tuple[str, ...] = (),
    ) -> None:
        if not artifact_ids and prompt:
            try:
                artifact_ids = tuple(
                    hit.artifact.artifact_id
                    for hit in self.runtime.query(MemoryQuery(
                        MemoryKind.SEMANTIC,
                        "",
                        namespace=namespace,
                        limit=100,
                    ))
                    if hit.artifact.content in prompt
                )
            except Exception:
                artifact_ids = ()
        exposure_event = TriggerEvent.create(
            event_type="memory_exposure",
            source_revision=self._exposure_context_revision(),
            input_payload={
                "namespace": namespace,
                "artifact_ids": list(artifact_ids),
                "prompt_digest": hashlib.sha256((prompt or "").encode("utf-8")).hexdigest(),
            },
            session_id=self._session_id,
            task_id=self._task_id,
            supported=True,
        )
        exposure = self._exposure_policy.decide(exposure_event, artifact_ids)
        injection_receipt_ids: tuple[str, ...] = ()
        if exposure.action == DecisionAction.RUN:
            from .memory.exposure_policy import DeterministicExposurePolicy

            receipt = DeterministicExposurePolicy.bind_injection(
                exposure,
                context_revision=exposure_event.source_revision,
                render_fingerprint=hashlib.sha256((prompt or "").encode("utf-8")).hexdigest(),
            )
            injection_receipt_ids = (receipt.receipt_id,)
        self._record_policy_decision(
            exposure,
            snapshot=self._synthetic_policy_snapshot(exposure_event.source_revision),
            injection_receipt_ids=injection_receipt_ids,
        )
        if self.semantic_future_recorder is None:
            return
        step_id = f"future-semantic.{namespace}.{len(self._semantic_futures) + 1}"
        future = self.semantic_future_recorder.record_prompt_injection(
            self.runtime.registry,
            prompt or "",
            namespace=namespace,
            parent_operation_ids=(),
            step_id=step_id,
        )
        self._semantic_futures.append((future, step_id))

    def _exposure_context_revision(self) -> str:
        agent = self._agent
        session_db = getattr(agent, "_session_db", None) if agent is not None else None
        native_session_id = str(getattr(agent, "session_id", "") or "") if agent is not None else ""
        rows = ()
        if session_db is not None and native_session_id:
            try:
                rows = tuple(session_db.get_messages(native_session_id))
            except Exception:
                rows = ()
        payload = json.dumps(rows, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str)
        return "exposure-rev." + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:40]

    def _synthetic_policy_snapshot(self, revision: str) -> ContextSnapshot:
        """Create content-free identity for exposure evidence before task snapshot."""

        from .lifecycle.snapshot import ProvenanceRef, SnapshotSegment

        segment = SnapshotSegment(
            "exposure.segment",
            "exposure.message",
            "system",
            "memory exposure boundary",
            "exposure.turn",
            0,
            completed=True,
        )
        return ContextSnapshot(
            self._run_id,
            self._episode_id,
            self._session_id,
            self._task_id,
            "exposure.snapshot." + hashlib.sha256(revision.encode("utf-8")).hexdigest()[:40],
            revision,
            (segment,),
            (),
            None,
            TaskLifecycleState.ACTIVE,
            "memory_exposure",
            (),
            0,
            ProvenanceRef(
                self._run_id,
                self._episode_id,
                self._session_id,
                self._task_id,
                "exposure.snapshot." + hashlib.sha256(revision.encode("utf-8")).hexdigest()[:40],
                "hermes:memory_exposure",
            ),
        )

    def _record_semantic_outcomes(self, result: Mapping[str, Any]) -> None:
        if self._semantic_outcomes_recorded:
            return
        if self.semantic_future_recorder is not None and self.semantic_feedback_resolver is not None:
            for future, step_id in self._semantic_futures:
                current_input = self._current_input(result)
                observation = self._semantic_deployment_observation(result)
                resolution = self.semantic_feedback_resolver.resolve(future, observation)
                outcome = self.semantic_future_recorder.record_use_and_outcome(
                    future,
                    used_artifact_ids=resolution.used_artifact_ids,
                    outcome_status=resolution.outcome_status,
                    outcome_reason_code=resolution.outcome_reason_code,
                    step_id=step_id,
                )
                self._record_extraction_feedback(
                    future,
                    observation,
                    outcome,
                    current_input,
                )
        completed = result.get("completed") is True
        outcome_digest = hashlib.sha256(
            json.dumps(
                {
                    "completed": completed,
                    "partial": result.get("partial") is True,
                    "interrupted": result.get("interrupted") is True,
                },
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        self._record_process_observation(
            kind=ProcessEventKind.TASK_OUTCOME,
            status=(ProcessEventStatus.SUCCESS if completed else ProcessEventStatus.FAILED),
            host_event_id=f"event.task-outcome.{outcome_digest[:40]}",
            source_revision=self._exposure_context_revision(),
            input_payload={"task_id": self._task_id},
            output_payload={"outcome_digest": outcome_digest},
            reason_codes=("task_completed",) if completed else ("task_failure",),
            execution_receipt_ids=(f"receipt.task-outcome.{outcome_digest[:24]}",),
        )
        self._semantic_outcomes_recorded = True

    def _record_extraction_source(
        self,
        boundary: StaticSemanticBoundaryResult,
    ) -> None:
        if (
            self.static_writeback is None
            or self.extraction_source_projector is None
            or self.extraction_source_store is None
            or self._family_id is None
            or self._stage is None
        ):
            return
        if boundary.duplicate:
            if not any(
                record.compilation_id == boundary.compilation_id
                for record in self.extraction_source_store.records()
            ):
                raise ValueError(
                    "duplicate semantic compilation has no source evidence"
                )
            return
        projection = self.static_writeback.source_projection_for(
            boundary.compilation_id
        )
        if projection is None:
            raise ValueError("semantic compilation source projection is unavailable")
        available_keys = detect_source_semantic_keys(
            self._family_id,
            tuple(message.content for message in projection.messages),
        )
        record = self.extraction_source_projector.project_record(
            boundary,
            self.static_writeback.policy,
            self.static_writeback.extraction_runtime_binding,
            family_id=self._family_id,
            stage=self._stage,
            available_semantic_keys=available_keys,
        )
        self.extraction_source_store.append(record)
        capture_log = self.extraction_optimizer_capture_log
        if capture_log is None:
            raise ValueError("optimizer capture log is unavailable")
        assert boundary.writeback is not None
        ingestion = boundary.writeback.ingestion
        if ingestion is None:
            raise ValueError("optimizer capture requires ingestion evidence")
        trace = self.static_writeback.policy.operation_trace(
            ingestion.idempotency_key
        )
        if trace is None:
            raise ValueError("optimizer capture requires extraction trace")
        fact_contents = []
        for extracted in trace.fact_extractions:
            fact = self.static_writeback.policy.fact_for_digest(
                extracted.content_digest
            )
            if fact is None or fact.fact_id != extracted.fact_id:
                raise ValueError("optimizer fact capture owner mismatch")
            fact_contents.append(ExtractionFactContent(
                extracted.fact_id,
                fact.content,
                extracted.accepted,
                extracted.reason_code,
            ))
        capture_log.append(ExtractionOptimizerSourceCapture.create(
            captured_at=self._utc_now(),
            source_record_id=record.record_id,
            source_record_digest=record.content_digest,
            projection=projection,
            fact_contents=tuple(fact_contents),
        ))

    def _record_static_policy_evidence(
        self,
        boundary: StaticSemanticBoundaryResult,
        snapshot: ContextSnapshot,
        trigger_event: object,
    ) -> None:
        """Join extraction/admission/commit outcomes to policy evidence.

        This is an observer-only projection.  It never changes the existing
        Mem0-flat execution or mutation safety gates.
        """

        from .memory.policy_contracts import TriggerEvent

        if not isinstance(trigger_event, TriggerEvent):
            raise ValueError("policy evidence requires a typed trigger event")
        writeback = boundary.writeback
        ingestion = writeback.ingestion if writeback is not None else None
        projection = self.static_writeback.source_projection_for(boundary.compilation_id) if self.static_writeback is not None else None
        if ingestion is None or projection is None:
            return
        trace = self.static_writeback.policy.operation_trace(ingestion.idempotency_key) if self.static_writeback is not None else None
        candidate_ids = tuple(trace.fact_artifact_ids) if trace is not None else ()
        extraction_action = DecisionAction.RUN if ingestion.status.value == "success" else DecisionAction.SKIP
        extraction_status = ExecutionStatus.PENDING if extraction_action == DecisionAction.RUN else ExecutionStatus.SKIPPED
        extraction = ExtractionDecision.create(
            policy_version=ingestion.policy_version,
            source_revision=snapshot.context_revision,
            input_payload={"compilation_id": boundary.compilation_id, "source_digest": projection.projection_digest},
            output_payload={"candidate_fact_ids": list(candidate_ids), "status": ingestion.status.value},
            action=extraction_action,
            execution_status=extraction_status,
            reason_codes=("extraction_completed" if extraction_action == DecisionAction.RUN else "extraction_failed",),
            lineage_id=f"lineage.{trigger_event.event_id}",
            trigger_event_id=trigger_event.event_id,
            execution_receipt_id=ingestion.execution_id if extraction_action == DecisionAction.RUN else None,
            candidate_fact_ids=candidate_ids,
            source_digest=projection.projection_digest,
            request_id=boundary.compilation_id,
        )
        self._record_policy_decision(extraction, snapshot)

        operations = tuple(ingestion.operations)
        if operations:
            operation_actions = tuple(item.action.value for item in operations)
            mutation_kind = (
                MutationKind(operation_actions[0].upper())
                if len(set(operation_actions)) == 1
                else MutationKind.NONE
            )
            accepted = candidate_ids if mutation_kind != MutationKind.NONE else ()
            targets = tuple(item.target_artifact_id for item in operations if item.target_artifact_id)
            expected_revision = next((item.expected_revision for item in operations if item.expected_revision), "backend.revision.unobserved")
            admission = AdmissionDecision.create(
                policy_version=ingestion.policy_version,
                source_revision=snapshot.context_revision,
                input_payload={"extraction_decision_id": extraction.decision_id, "operation_ids": [item.operation_id for item in operations]},
                output_payload={"actions": list(operation_actions), "targets": list(targets)},
                action=DecisionAction.RUN,
                execution_status=ExecutionStatus.EXECUTED,
                reason_codes=("admission_resolved",),
                lineage_id=extraction.lineage_id,
                trigger_event_id=trigger_event.event_id,
                execution_receipt_id=ingestion.execution_id,
                candidate_fact_ids=candidate_ids,
                accepted_fact_ids=accepted,
                mutation_kind=mutation_kind,
                backend_revision=expected_revision,
                target_artifact_ids=targets,
                update_supported=True,
            )
            self._record_policy_decision(admission, snapshot)
            for execution in writeback.executions if writeback is not None else ():
                commit = self._commit_decision_for_execution(execution, admission, snapshot, trigger_event)
                self._record_policy_decision(commit, snapshot)

    def _commit_decision_for_execution(self, execution: object, admission: AdmissionDecision, snapshot: ContextSnapshot, trigger_event: object):
        from .memory.policy_contracts import CommitDecision

        mutation_id = str(getattr(execution, "mutation_id", ""))
        receipt_id = getattr(execution, "receipt_id", None)
        status = getattr(getattr(execution, "status", None), "value", "failed")
        action = DecisionAction.RUN if mutation_id else DecisionAction.SKIP
        return CommitDecision.create(
            policy_version=admission.policy_version,
            source_revision=snapshot.context_revision,
            input_payload={"admission_decision_id": admission.decision_id, "mutation_id": mutation_id},
            output_payload={"status": status, "receipt_id": receipt_id},
            action=action,
            execution_status=ExecutionStatus.EXECUTED if action == DecisionAction.RUN else ExecutionStatus.SKIPPED,
            reason_codes=("mutation_committed" if status == "committed" else "mutation_not_committed",),
            lineage_id=admission.lineage_id,
            trigger_event_id=trigger_event.event_id,
            execution_receipt_id=receipt_id,
            mutation_ids=(mutation_id,) if mutation_id else (),
            expected_revision=admission.backend_revision or "backend.revision.unobserved",
            final_receipt_id=receipt_id,
        )

    def _record_policy_decision(
        self,
        decision: object,
        snapshot: ContextSnapshot,
        *,
        injection_receipt_ids: tuple[str, ...] = (),
    ) -> None:
        decision_id = getattr(decision, "decision_id")
        if decision_id in self._policy_decision_ids:
            return
        self._policy_evidence.record_decision(
            decision,
            run_id=snapshot.run_id,
            episode_id=snapshot.episode_id,
            session_id=snapshot.session_id,
            task_id=snapshot.task_id,
            snapshot_id=snapshot.snapshot_id,
            injection_receipt_ids=injection_receipt_ids,
        )
        self._policy_decision_ids.add(decision_id)
        # Every policy decision is also projected into the process corpus.  The
        # projection carries the exact decision fingerprints and receipt join,
        # but never copies source, prompt, response or memory content.
        from .memory.policy_contracts import PolicyDecision

        if isinstance(decision, PolicyDecision):
            host_event_id = decision.trigger_event_id or (
                "event.snapshot."
                + hashlib.sha256(snapshot.snapshot_id.encode("utf-8")).hexdigest()[:40]
            )
            self._process_feedback.record(ProcessEvent.from_policy_decision(
                decision,
                run_id=snapshot.run_id,
                variant=self.ledger.variant,
                trace_id=self._trace_id,
                episode_id=snapshot.episode_id,
                session_id=snapshot.session_id,
                task_id=snapshot.task_id,
                host_event_id=host_event_id,
                family_id=self._family_id,
                stage=self._stage,
                execution_receipt_ids=injection_receipt_ids,
            ))

    def _record_process_observation(
        self,
        *,
        kind: ProcessEventKind,
        status: ProcessEventStatus,
        host_event_id: str,
        source_revision: str,
        input_payload: object,
        output_payload: object,
        reason_codes: tuple[str, ...],
        execution_receipt_ids: tuple[str, ...] = (),
        policy_decision_id: str | None = None,
        policy_layer: object | None = None,
        lineage_id: str | None = None,
    ) -> None:
        """Append one non-policy host/process observation.

        This helper is used for retrieval, tool and downstream outcome events
        where no learned policy decision exists at the observation boundary.
        The host event and source revision remain mandatory, and all payloads
        are reduced to digests by :class:`ProcessEvent`.
        """

        # Prefer the most recent real host boundary.  Synthetic IDs are only
        # used for an observation that happens before the first boundary has
        # been delivered by the host adapter.
        if self._last_host_event_id is not None:
            host_event_id = self._last_host_event_id
        if self._last_host_source_revision is not None:
            source_revision = self._last_host_source_revision
        self._process_feedback.record(ProcessEvent.create(
            kind=kind,
            status=status,
            run_id=self._run_id,
            variant=self.ledger.variant,
            trace_id=self._trace_id,
            episode_id=self._episode_id,
            session_id=self._session_id,
            task_id=self._task_id,
            host_event_id=host_event_id,
            source_revision=source_revision,
            input_payload=input_payload,
            output_payload=output_payload,
            reason_codes=reason_codes,
            execution_receipt_ids=execution_receipt_ids,
            policy_decision_id=policy_decision_id,
            policy_layer=policy_layer,
            lineage_id=lineage_id,
            family_id=self._family_id,
            stage=self._stage,
        ))

    def _record_extraction_feedback(
        self,
        future: SemanticFutureEvidence,
        observation: DeploymentObservation,
        outcome: SemanticOutcomeEvidence,
        current_input: str,
    ) -> None:
        if (
            self.semantic_feedback_resolver is None
            or self.extraction_source_store is None
            or self.extraction_feedback_builder is None
            or self.extraction_feedback_log is None
            or self._family_id is None
        ):
            return
        contract = self.semantic_feedback_resolver.registry.resolver(
            self._family_id
        ).contract
        if observation.stage not in contract.opportunity.eligible_stages:
            return
        records = tuple(
            record
            for record in self.extraction_source_store.records()
            if record.family_id == self._family_id
        )
        for record in records:
            keys_by_artifact: dict[str, list[str]] = {}
            for fact in record.source.facts:
                if (
                    fact.artifact_id is None
                    or fact.artifact_id not in future.memory_artifact_ids
                ):
                    continue
                values = keys_by_artifact.setdefault(fact.artifact_id, [])
                for key in fact.semantic_keys:
                    if key not in values:
                        values.append(key)
            bindings = tuple(
                ArtifactSemanticBinding(artifact_id, tuple(keys))
                for artifact_id, keys in keys_by_artifact.items()
                if keys
            )
            exposed = bool(bindings) and future.injection_artifact_id is not None
            source_future = FutureMemoryEvidence(
                f"opportunity.{future.query_operation_id}",
                (
                    ExposureMode.EAGER_SYSTEM_PROMPT
                    if exposed
                    else ExposureMode.NOT_EXPOSED
                ),
                bindings,
                future.query_operation_id,
                future.injection_operation_id if exposed else None,
            )
            operation_join = FeedbackOperationJoin(
                future.query_operation_id,
                outcome.use_operation_id,
                outcome.outcome_operation_id,
            )
            missed = self.extraction_feedback_builder.derive_missed(
                record.source,
                observation,
                source_future,
                operation_join=operation_join,
            )
            dataset = self.extraction_feedback_builder.build(
                record.source,
                observation,
                source_future,
                missed=missed,
                operation_join=operation_join,
            )
            feedback_record = LiveExtractionFeedbackRecord.create(
                family_id=self._family_id,
                stage=observation.stage,
                run_id=self._run_id,
                trace_id=self._trace_id,
                episode_id=self._episode_id,
                session_id=self._session_id,
                task_id=self._task_id,
                deployment_observation_id=observation.observation_id,
                source_record_id=record.record_id,
                opportunity_operation_id=future.query_operation_id,
                use_operation_id=outcome.use_operation_id,
                outcome_operation_id=outcome.outcome_operation_id,
                dataset=dataset,
            )
            self.extraction_feedback_log.append(feedback_record)
            # Preserve stage diagnosis as process feedback.  A strict label is
            # not itself a process event: these observations only say whether
            # an exposure/use/outcome stage was seen, and therefore remain safe
            # for deployments without an output evaluator.
            for example in dataset.examples:
                if not example.primary:
                    continue
                if example.label.value == "useful":
                    process_kind = ProcessEventKind.TASK_OUTCOME
                    process_status = ProcessEventStatus.SUCCESS
                    process_reason = ("decision_observed",)
                elif example.label.value == "harmful":
                    process_kind = ProcessEventKind.TASK_OUTCOME
                    process_status = ProcessEventStatus.FAILED
                    process_reason = ("task_failure",)
                elif "injected_not_used" in example.reason_codes:
                    process_kind = ProcessEventKind.EXPOSURE
                    process_status = ProcessEventStatus.SUCCESS
                    process_reason = ("non_use",)
                elif "use_not_bound_to_memory" in example.reason_codes or "not_exposed" in example.reason_codes:
                    process_kind = ProcessEventKind.EXPOSURE
                    process_status = ProcessEventStatus.SKIPPED
                    process_reason = ("absence",)
                elif "observation_censored" in example.reason_codes:
                    process_kind = ProcessEventKind.TASK_OUTCOME
                    process_status = ProcessEventStatus.UNKNOWN
                    process_reason = ("observation_censored",)
                else:
                    process_kind = ProcessEventKind.TASK_OUTCOME
                    process_status = ProcessEventStatus.UNKNOWN
                    process_reason = ("decision_observed",)
                self._record_process_observation(
                    kind=process_kind,
                    status=process_status,
                    host_event_id=self._last_host_event_id or f"event.feedback.{record.record_id}",
                    source_revision=self._last_host_source_revision or self._exposure_context_revision(),
                    input_payload={
                        "feedback_record_id": feedback_record.record_id,
                        "example_id": example.example_id,
                        "future_opportunity_id": example.future_opportunity_id,
                    },
                    output_payload={
                        "label": example.label.value,
                        "exposure_mode": example.exposure_mode.value,
                    },
                    reason_codes=process_reason,
                    execution_receipt_ids=(
                        f"receipt.feedback.{hashlib.sha256(example.example_id.encode('utf-8')).hexdigest()[:24]}",
                    ),
                    lineage_id=(future.query_operation_id or None),
                )
            capture_log = self.extraction_optimizer_capture_log
            if capture_log is None:
                raise ValueError("optimizer capture log is unavailable")
            capture_log.append(ExtractionOptimizerFeedbackCapture.create(
                captured_at=self._utc_now(),
                feedback_record_id=feedback_record.record_id,
                source_record_id=record.record_id,
                observation=observation,
                current_input=current_input,
            ))

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(UTC).isoformat(timespec="microseconds").replace(
            "+00:00",
            "Z",
        )

    @staticmethod
    def _current_input(result: Mapping[str, Any]) -> str:
        raw_messages = result.get("messages")
        messages = (
            tuple(value for value in raw_messages if isinstance(value, Mapping))
            if isinstance(raw_messages, (list, tuple))
            else ()
        )
        user_inputs = tuple(
            str(message.get("content") or "")
            for message in messages
            if message.get("role") == "user"
        )
        return user_inputs[-1] if user_inputs else ""

    def _semantic_deployment_observation(
        self,
        result: Mapping[str, Any],
    ) -> DeploymentObservation:
        resolver = self.semantic_feedback_resolver
        if resolver is None or self._family_id is None or self._stage is None:
            raise ValueError("semantic feedback resolver identity is unavailable")
        contract = resolver.registry.resolver(self._family_id).contract
        messages = ()
        raw_messages = result.get("messages")
        if isinstance(raw_messages, (list, tuple)):
            messages = tuple(
                value for value in raw_messages if isinstance(value, Mapping)
            )
        current_input = self._current_input(result)
        current_keys = detect_current_input_semantic_keys(
            self._family_id,
            current_input,
        )
        task_keys = (
            contract.opportunity.memory_scope_keys
            if self._stage in contract.opportunity.eligible_stages
            else ()
        )
        observation_complete = not (
            result.get("partial") is True or result.get("interrupted") is True
        )
        identity = json.dumps({
            "family_id": self._family_id,
            "stage": self._stage,
            "task_id": self._task_id,
            "current_input_digest": hashlib.sha256(
                current_input.encode("utf-8")
            ).hexdigest(),
        }, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        return DeploymentObservation(
            observation_id=(
                "observation."
                + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:40]
            ),
            family_id=self._family_id,
            stage=self._stage,
            task_id=self._task_id,
            current_input_projection_digest=hashlib.sha256(
                current_input.encode("utf-8")
            ).hexdigest(),
            current_input_semantic_keys=current_keys,
            task_semantic_keys=task_keys,
            final_response=str(result.get("final_response") or ""),
            tool_events=self._observable_tool_events(messages),
            completed=result.get("completed") is True,
            observation_complete=observation_complete,
            censor_reason=(None if observation_complete else "execution_incomplete"),
        )

    @staticmethod
    def _observable_tool_events(
        messages: tuple[Mapping[str, Any], ...],
    ) -> tuple[ObservableToolEvent, ...]:
        calls: dict[str, tuple[str, Mapping[str, Any]]] = {}
        events = []
        for message in messages:
            if message.get("role") == "assistant":
                raw_calls = message.get("tool_calls")
                if not isinstance(raw_calls, (list, tuple)):
                    continue
                for ordinal, call in enumerate(raw_calls, start=1):
                    if not isinstance(call, Mapping):
                        continue
                    function = call.get("function")
                    if not isinstance(function, Mapping):
                        continue
                    name = str(function.get("name") or "")
                    if name != "notes_share":
                        continue
                    call_id = str(call.get("id") or f"notes-share-{ordinal}")
                    arguments = function.get("arguments")
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except json.JSONDecodeError:
                            arguments = {}
                    calls[call_id] = (
                        name,
                        arguments if isinstance(arguments, Mapping) else {},
                    )
                continue
            if message.get("role") != "tool":
                continue
            call_id = str(message.get("tool_call_id") or "")
            call = calls.get(call_id)
            if call is None:
                continue
            name, arguments = call
            content = message.get("content")
            if isinstance(content, str):
                try:
                    content = json.loads(content)
                except json.JSONDecodeError:
                    content = {}
            success = isinstance(content, Mapping) and content.get("success") is True

            def normalized(value: object) -> str:
                text = re.sub(r"[^a-z0-9]+", "_", str(value).casefold()).strip("_")
                return text or "unknown"

            recipients = arguments.get("recipients")
            recipient_ids = tuple(dict.fromkeys(
                normalized(value)
                for value in recipients
            )) if isinstance(recipients, (list, tuple)) else ()
            note_id = arguments.get("note_id")
            subject_ids = (normalized(note_id),) if note_id is not None else ()
            event_identity = json.dumps({
                "call_id": call_id,
                "tool_name": name,
                "subject_ids": subject_ids,
                "recipient_ids": recipient_ids,
                "success": success,
            }, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
            events.append(ObservableToolEvent(
                "tool-event."
                + hashlib.sha256(event_identity.encode("utf-8")).hexdigest()[:40],
                name,
                success,
                subject_ids,
                recipient_ids,
            ))
        return tuple(events)

    def _process_lifecycle_boundary(
        self,
        trigger: EvaluationTrigger,
        task_state: TaskLifecycleState,
    ) -> HermesLifecycleDryRunResult | None:
        assert self.lifecycle is not None
        evidence_count = len(self.lifecycle.observer.events)
        try:
            agent = self._agent
            if agent is None:
                raise ValueError("Hermes lifecycle bridge is not attached")
            session_db = getattr(agent, "_session_db", None)
            native_session_id = str(getattr(agent, "session_id", "") or "")
            if session_db is None or not native_session_id:
                raise ValueError("Hermes lifecycle requires a persisted native session")
            rows = session_db.get_messages(native_session_id)
            result = self.lifecycle.process(
                rows,
                trigger=trigger,
                task_state=task_state,
                source_ref=f"hermes_state:session:{native_session_id}",
            )
        except Exception as exc:
            if len(self.lifecycle.observer.events) == evidence_count:
                self.lifecycle.record_boundary_rejection(trigger, exc)
            self._lifecycle_failures.append((trigger.value, type(exc).__name__))
            return None
        if not any(
            item.evaluation.evaluation_id == result.evaluation.evaluation_id
            for item in self._lifecycle_results
        ):
            self._lifecycle_results.append(result)
        trigger_event = self._trigger_adapter.from_snapshot(
            result.snapshot,
            trigger.value,
            turn_index=sum(1 for row in rows if row.get("role") == "user"),
        )
        self._observe_policy_boundary(result.snapshot, trigger_event)
        return result

    def _observe_policy_boundary(
        self,
        snapshot: ContextSnapshot,
        trigger_event: TriggerEvent,
    ) -> SourceSelectionDecision:
        """Record trigger/source decisions for every trusted host boundary."""

        self._last_host_event_id = trigger_event.event_id
        self._last_host_source_revision = snapshot.context_revision

        existing_source = next(
            (
                item
                for item in self._source_selection_decisions
                if item.trigger_event_id == trigger_event.event_id
                and item.source_revision == snapshot.context_revision
            ),
            None,
        )
        if existing_source is not None:
            return existing_source

        observation = self._trigger_policy.decide(trigger_event)
        if not any(item.event.event_id == observation.event.event_id for item in self._trigger_observations):
            self._trigger_observations.append(observation)
        self._record_policy_decision(observation.decision, snapshot)
        source_decision = (
            self._source_selection_policy.select(snapshot, trigger_event)
            if observation.decision.action == DecisionAction.RUN
            else self._source_selection_policy.skip(
                snapshot,
                trigger_event,
                reason="trigger_not_run",
            )
        )
        if not any(item.decision_id == source_decision.decision_id for item in self._source_selection_decisions):
            self._source_selection_decisions.append(source_decision)
        self._record_policy_decision(source_decision, snapshot)
        return source_decision

    def adapter_call(
        self,
        operation: str,
        adapter_call: Callable[[], Any],
        native_call: Callable[[], Any],
    ) -> Any:
        try:
            return adapter_call()
        except Exception as exc:
            failure_type = type(exc).__name__
            if (
                self.config.adapter_failure_policy
                == HermesAdapterFailurePolicy.BYPASS_NATIVE
            ):
                if operation == "session_search":
                    kind = MemoryKind.EPISODIC
                    backend = "hermes-native-episodic"
                elif operation.startswith("skill"):
                    kind = MemoryKind.PROCEDURAL
                    backend = "hermes-native-procedural"
                else:
                    kind = MemoryKind.SEMANTIC
                    backend = "hermes-native-semantic"
                self.ledger.record(MemoryEvent(
                    MemoryEventKind.QUERY,
                    kind,
                    backend,
                    reason_code="adapter_failure_native_bypass",
                    attributes={
                        "surface": operation,
                        "failure_type": failure_type,
                    },
                ))
                process_kind = (
                    ProcessEventKind.RETRIEVAL
                    if operation == "session_search"
                    else ProcessEventKind.TOOL_RESULT
                    if operation.startswith("skill")
                    else ProcessEventKind.EXPOSURE
                )
                self._record_process_observation(
                    kind=process_kind,
                    status=ProcessEventStatus.SUCCESS,
                    host_event_id=self._last_host_event_id or f"event.adapter.{operation}",
                    source_revision=self._last_host_source_revision or self._exposure_context_revision(),
                    input_payload={"operation": operation},
                    output_payload={"route": "native_bypass", "failure_type": failure_type},
                    reason_codes=("adapter_failure",),
                    execution_receipt_ids=(
                        "receipt.adapter-failure."
                        + hashlib.sha256(operation.encode("utf-8")).hexdigest()[:24],
                    ),
                )
                return native_call()
            process_kind = (
                ProcessEventKind.RETRIEVAL
                if operation == "session_search"
                else ProcessEventKind.TOOL_RESULT
                if operation.startswith("skill")
                else ProcessEventKind.EXPOSURE
            )
            self._record_process_observation(
                kind=process_kind,
                status=ProcessEventStatus.FAILED,
                host_event_id=self._last_host_event_id or f"event.adapter.{operation}",
                source_revision=self._last_host_source_revision or self._exposure_context_revision(),
                input_payload={"operation": operation},
                output_payload={"failure_type": failure_type},
                reason_codes=("adapter_failure",),
                execution_receipt_ids=(
                    "receipt.adapter-failure."
                    + hashlib.sha256(operation.encode("utf-8")).hexdigest()[:24],
                ),
            )
            raise HermesAdapterExecutionError(
                f"Hermes adapter operation failed closed: {operation} ({failure_type})"
            ) from exc

    def verify_projection(
        self,
        kind: MemoryKind,
        backend: str,
        surface: str,
        adapter_value: Any,
        native_call: Callable[[], Any],
    ) -> None:
        if not self.config.verify_native_projection:
            return
        native_value = native_call()
        equivalent = adapter_value == native_value
        if isinstance(adapter_value, str):
            content_chars = len(adapter_value)
        else:
            content_chars = len(json.dumps(
                adapter_value,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ))
        self.ledger.record(MemoryEvent(
            MemoryEventKind.PROJECTION_CHECK,
            kind,
            backend,
            content_chars=content_chars,
            reason_code=None if equivalent else "projection_mismatch",
            attributes={"surface": surface, "equivalent": equivalent},
        ))
        if not equivalent:
            raise HermesAdapterExecutionError(
                f"Hermes adapter projection mismatch: {surface}"
            )

    def observe_query(
        self,
        kind: MemoryKind,
        text: str,
        *,
        namespace: str = "default",
        limit: int,
        surface: str | None,
    ) -> None:
        try:
            hits = self.runtime.query(MemoryQuery(
                kind,
                text,
                namespace=namespace,
                limit=limit,
            ))
            if surface and hits:
                self.runtime.mark_injected(hits, surface=surface)
            self._record_process_observation(
                kind=ProcessEventKind.RETRIEVAL,
                status=(ProcessEventStatus.SUCCESS if hits else ProcessEventStatus.FAILED),
                host_event_id=(
                    "event.query."
                    + hashlib.sha256(
                        json.dumps(
                            {"kind": kind.value, "namespace": namespace, "query_digest": hashlib.sha256(text.encode("utf-8")).hexdigest(), "limit": limit},
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    ).hexdigest()[:40]
                ),
                source_revision=self._exposure_context_revision(),
                input_payload={"query_digest": hashlib.sha256(text.encode("utf-8")).hexdigest(), "namespace": namespace, "limit": limit},
                output_payload={"hit_count": len(hits), "surface": surface},
                reason_codes=("retrieval_miss",) if not hits else ("decision_observed",),
                execution_receipt_ids=(
                    "receipt.query."
                    + hashlib.sha256(
                        f"{kind.value}:{namespace}:{text}:{limit}".encode("utf-8")
                    ).hexdigest()[:24],
                ),
            )
        except Exception as exc:
            query_digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            self._record_process_observation(
                kind=ProcessEventKind.RETRIEVAL,
                status=ProcessEventStatus.FAILED,
                host_event_id=f"event.query-failure.{query_digest[:40]}",
                source_revision=self._exposure_context_revision(),
                input_payload={"query_digest": query_digest, "namespace": namespace, "limit": limit},
                output_payload={"failure_type": type(exc).__name__},
                reason_codes=("retrieval_failure",),
                execution_receipt_ids=(f"receipt.query-failure.{query_digest[:24]}",),
            )
            return

    def record_native_search(
        self,
        query: str,
        limit: int,
        results: list[dict[str, Any]],
    ) -> None:
        self.ledger.record(MemoryEvent(
            MemoryEventKind.QUERY,
            MemoryKind.EPISODIC,
            "hermes-native-episodic",
            query_chars=len(query),
            attributes={"limit": limit, "namespace": "default"},
        ))
        self.ledger.record(MemoryEvent(
            MemoryEventKind.RETRIEVED,
            MemoryKind.EPISODIC,
            "hermes-native-episodic",
            artifact_ids=tuple(
                f"native-episodic:message:{item.get('id')}" for item in results
            ),
            content_chars=sum(len(str(item.get("content") or "")) for item in results),
            attributes={"count": len(results)},
        ))
        query_digest = hashlib.sha256(query.encode("utf-8")).hexdigest()
        self._record_process_observation(
            kind=ProcessEventKind.RETRIEVAL,
            status=ProcessEventStatus.SUCCESS if results else ProcessEventStatus.FAILED,
            host_event_id=f"event.native-search.{query_digest[:40]}",
            source_revision=self._exposure_context_revision(),
            input_payload={"query_digest": query_digest, "limit": limit},
            output_payload={"result_count": len(results)},
            reason_codes=("decision_observed",) if results else ("retrieval_miss",),
            execution_receipt_ids=(f"receipt.native-search.{query_digest[:24]}",),
        )

    def _wrap_skill_handlers(self) -> None:
        from tools.registry import registry

        for tool_name in ("skills_list", "skill_view"):
            entry = registry._tools.get(tool_name)
            if entry is None or tool_name in self._tool_handlers:
                continue
            original = entry.handler
            self._tool_handlers[tool_name] = original

            def handler(
                args: dict[str, Any],
                _tool_name: str = tool_name,
                _original: Callable[..., str] = original,
                **kwargs: Any,
            ) -> str:
                query = "" if _tool_name == "skills_list" else str(args.get("name") or "")
                native_call = lambda: _original(args, **kwargs)
                if not self.uses_adapter:
                    result = native_call()
                    self.observe_query(
                        MemoryKind.PROCEDURAL,
                        query,
                        limit=100 if _tool_name == "skills_list" else 5,
                        surface=_tool_name,
                    )
                    self._record_skill_process(_tool_name, query, result)
                    return result

                def adapter_read() -> str:
                    hits = self.runtime.query(MemoryQuery(
                        MemoryKind.PROCEDURAL,
                        query,
                        limit=100 if _tool_name == "skills_list" else 5,
                    ))
                    query_digest = hashlib.sha256(
                        json.dumps(
                            {"kind": "procedural", "query": query, "limit": 100 if _tool_name == "skills_list" else 5},
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    ).hexdigest()
                    self._record_process_observation(
                        kind=ProcessEventKind.RETRIEVAL,
                        status=ProcessEventStatus.SUCCESS if hits else ProcessEventStatus.FAILED,
                        host_event_id=self._last_host_event_id or f"event.procedural-query.{query_digest[:40]}",
                        source_revision=self._last_host_source_revision or self._exposure_context_revision(),
                        input_payload={"query_digest": query_digest, "tool": _tool_name},
                        output_payload={"hit_count": len(hits)},
                        reason_codes=("decision_observed",) if hits else ("retrieval_miss",),
                        execution_receipt_ids=(f"receipt.procedural-query.{query_digest[:24]}",),
                    )
                    projected_hits = hits
                    if _tool_name == "skill_view" and not projected_hits:
                        projected_hits = self.runtime.query(MemoryQuery(
                            MemoryKind.PROCEDURAL,
                            "",
                            limit=100,
                        ))
                    with TemporaryDirectory(
                        prefix="rsimem-hermes-live-skills-"
                    ) as directory:
                        skills_dir = Path(directory) / "skills"
                        _materialize_procedural_hits(skills_dir, projected_hits)
                        with _bound_hermes_skills_dir(skills_dir):
                            result = _original(args, **kwargs)
                    payload = json.loads(result)
                    if hits and payload.get("success") is True:
                        self.runtime.mark_injected(hits, surface=_tool_name)
                    return result

                adapter_result = self.adapter_call(_tool_name, adapter_read, native_call)
                self.verify_projection(
                    MemoryKind.PROCEDURAL,
                    "hermes-native-procedural",
                    _tool_name,
                    adapter_result,
                    native_call,
                )
                self._record_skill_process(_tool_name, query, adapter_result)
                return adapter_result

            entry.handler = handler

    def _record_skill_process(self, tool_name: str, query: str, result: str) -> None:
        """Record a tool result without persisting the returned skill text."""

        try:
            payload = json.loads(result)
        except (TypeError, json.JSONDecodeError):
            payload = {}
        success = isinstance(payload, Mapping) and payload.get("success") is True
        digest = hashlib.sha256(
            f"{tool_name}:{query}".encode("utf-8")
        ).hexdigest()
        self._record_process_observation(
            kind=ProcessEventKind.TOOL_RESULT,
            status=ProcessEventStatus.SUCCESS if success else ProcessEventStatus.FAILED,
            host_event_id=f"event.skill.{digest[:40]}",
            source_revision=self._exposure_context_revision(),
            input_payload={"tool": tool_name, "query_digest": hashlib.sha256(query.encode("utf-8")).hexdigest()},
            output_payload={"success": success},
            reason_codes=("decision_observed",) if success else ("tool_failure",),
            execution_receipt_ids=(f"receipt.skill.{digest[:24]}",),
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            from tools.registry import registry

            for tool_name, handler in self._tool_handlers.items():
                entry = registry._tools.get(tool_name)
                if entry is not None:
                    entry.handler = handler
            self._tool_handlers.clear()
        finally:
            self._agent = None
            try:
                if self.static_writeback is not None:
                    self.static_writeback.close()
            finally:
                self.runtime.close()
