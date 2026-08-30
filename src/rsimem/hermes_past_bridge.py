"""Opt-in RSIMem read bridge for the in-process PAST-Bench Hermes adapter."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Iterable, Mapping, Sequence

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
from .extraction_validation_runtime import (
    load_extraction_matched_trial_profile,
    load_extraction_offline_validation_profile,
)
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
    ExtractionSourceEvidence,
    FeedbackOperationJoin,
    FutureMemoryEvidence,
    ObservableToolEvent,
    detect_current_input_semantic_keys,
    detect_user_source_semantic_keys,
)
from .memory.tool_exact_join import ToolCallResultJoin
from .memory.extraction_optimizer_builder import ExtractionFactContent
from .memory.extraction_optimizer_capture import (
    ExtractionOptimizerFeedbackCapture,
    ExtractionOptimizerSourceCapture,
    JsonExtractionOptimizerCaptureLog,
)
from .memory.opportunity import (
    JsonOpportunityEvidenceLog,
    OpportunityEvidence,
    OpportunitySurface,
)
from .memory.use_attribution import (
    JsonMemoryUseEvidenceLog,
    MemoryUseEvidence,
    OutcomeEvidenceKind,
)
from .memory.extraction_projection import (
    JsonLiveExtractionFeedbackRecordLog,
    JsonExtractionSourceRecordStore,
    LiveExtractionFeedbackRecord,
    Mem0FlatExtractionSourceProjector,
)
from .memory.operation_graph import OperationContext
from .memory.operation_graph import materialize_operation_graph
from .memory.use_attribution import resolve_memory_use
from .memory.artifact_set import (
    ArtifactSetSemanticBinding,
    JsonArtifactSetBindingLog,
)
from .memory.pure_extraction import (
    PureExtractionSourceProjector,
    PureExtractionSourceRecord,
)
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
        opportunity_evidence_path: Path | None = None,
        opportunity_evidence_provider: Callable[
            [Mapping[str, Any]], Iterable[OpportunityEvidence]
        ] | None = None,
        memory_use_evidence_path: Path | None = None,
        artifact_set_binding_path: Path | None = None,
        artifact_set_binding_provider: Callable[
            [object], Iterable[ArtifactSetSemanticBinding]
        ] | None = None,
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
        self._opportunity_evidence_provider = opportunity_evidence_provider
        self._opportunity_evidence_log = JsonOpportunityEvidenceLog(
            opportunity_evidence_path
            or self.evidence_path.with_name("rsimem_opportunities.jsonl")
        )
        self._memory_use_evidence_log = JsonMemoryUseEvidenceLog(
            memory_use_evidence_path
            or self.evidence_path.with_name("rsimem_memory_use_evidence.jsonl")
        )
        self._artifact_set_binding_provider = artifact_set_binding_provider
        self._artifact_set_binding_log = JsonArtifactSetBindingLog(
            artifact_set_binding_path
            or self.evidence_path.with_name("rsimem_artifact_set_bindings.jsonl")
        )
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
        self._tool_call_result_joins: dict[str, ToolCallResultJoin] = {}
        self._tool_call_ids_seen: set[str] = set()
        self._tool_result_ids_seen: set[str] = set()
        self._skill_invocation_counts: dict[tuple[str, str, str], int] = {}
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
        existing_process_events = self._process_feedback.events
        self._tool_call_ids_seen.update(
            event.tool_call_id
            for event in existing_process_events
            if event.kind is ProcessEventKind.TOOL_CALL
            and event.tool_call_id is not None
        )
        self._tool_result_ids_seen.update(
            event.tool_result_id
            for event in existing_process_events
            if event.kind is ProcessEventKind.TOOL_RESULT
            and event.tool_result_id is not None
        )
        # Recover synthetic skill invocation ordinals from the persisted
        # content-free call IDs.  This prevents a restarted bridge from
        # treating a second identical skill invocation as the first one and
        # silently collapsing its process evidence into a prior retry.
        for event in existing_process_events:
            call_id = event.tool_call_id
            if event.kind is not ProcessEventKind.TOOL_CALL or call_id is None:
                continue
            parts = call_id.split(".")
            if len(parts) != 6 or parts[:2] != ["call", "skill"]:
                continue
            try:
                ordinal = int(parts[5])
            except ValueError:
                continue
            if ordinal < 0:
                continue
            key = (parts[3], parts[2], parts[4])
            self._skill_invocation_counts[key] = max(
                self._skill_invocation_counts.get(key, 0), ordinal + 1
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
            if static_writeback_config.matched_extraction_enabled or (
                static_writeback_config.extraction_runtime_scope
                == ExtractionPromptRuntimeScope.OFFLINE_VALIDATION
            ):
                extraction_config_path = Path(
                    static_writeback_config.extraction_runtime_config_path or ""
                ).expanduser().resolve()
                capture_root = self.evidence_path.parent.resolve()
                if not extraction_config_path.is_relative_to(capture_root):
                    raise ValueError(
                        "extraction runtime config must stay inside capture artifacts"
                    )
                if static_writeback_config.matched_extraction_enabled:
                    extraction_profile = load_extraction_matched_trial_profile(
                        extraction_config_path
                    )
                else:
                    extraction_profile = load_extraction_offline_validation_profile(
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
                    static_writeback_config.extraction_runtime_scope
                    if extraction_profile is not None
                    else ExtractionPromptRuntimeScope.ROOT_STATIC
                ),
                extraction_trial_id=(
                    (
                        extraction_profile.trial_id
                        if hasattr(extraction_profile, "trial_id")
                        else extraction_profile.validation_id
                    )
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
    def tool_call_result_joins(self) -> tuple[ToolCallResultJoin, ...]:
        """Exact tool call/result closures observed by this bridge."""

        return tuple(self._tool_call_result_joins.values())

    @property
    def opportunity_evidence(self) -> tuple[OpportunityEvidence, ...]:
        """Deployment-visible opportunities emitted by the host adapter.

        Family/stage benchmark contracts are intentionally absent from this
        collection.  They remain available only through the explicit audit
        projection used by :class:`SemanticFeedbackResolver`.
        """

        return self._opportunity_evidence_log.records()

    @property
    def memory_use_evidence(self) -> tuple[MemoryUseEvidence, ...]:
        """Generic retrieval/injection/use/outcome joins observed by Hermes."""

        return self._memory_use_evidence_log.records()

    @property
    def artifact_set_bindings(self) -> tuple[ArtifactSetSemanticBinding, ...]:
        return self._artifact_set_binding_log.records()

    def project_pure_extraction_source(
        self,
        boundary: StaticSemanticBoundaryResult,
        *,
        source_projection_id: str,
        context_revision: str,
        provenance_id: str,
        visible_semantic_keys: tuple[str, ...] = (),
        fact_semantic_keys: Mapping[str, tuple[str, ...]] | None = None,
    ) -> PureExtractionSourceRecord:
        """Project one live extraction boundary into the pure-process plane.

        This is deliberately opt-in.  It does not consult ``family_id`` or
        ``stage`` and does not persist a benchmark feedback label.  Callers
        that have a trusted runtime observation can persist the returned
        record with :class:`JsonPureExtractionSourceRecordStore` and join it
        later with :class:`PureExtractionFeedbackRecord`.
        """

        if self.static_writeback is None:
            raise ValueError("pure extraction projection requires static writeback")
        return PureExtractionSourceProjector().project_record(
            boundary,
            self.static_writeback.policy,
            self.static_writeback.extraction_runtime_binding,
            source_projection_id=source_projection_id,
            context_revision=context_revision,
            provenance_id=provenance_id,
            visible_semantic_keys=visible_semantic_keys,
            fact_semantic_keys=fact_semantic_keys,
        )

    def _record_runtime_opportunities(
        self,
        result: Mapping[str, Any],
    ) -> tuple[OpportunityEvidence, ...]:
        provider = self._opportunity_evidence_provider
        if provider is None:
            return ()
        # Keep benchmark scope out of the runtime opportunity surface.  The
        # bridge may itself be running a PAST-Bench case, but family/stage are
        # audit metadata and must not influence a deployment-visible provider.
        visible = self._strip_benchmark_scope(result)
        return self.record_opportunity_evidence(provider(visible))

    @staticmethod
    def _strip_benchmark_scope(value: object) -> object:
        """Return a recursively scope-free view for runtime providers.

        ``family_id``/``stage`` are removed at every nesting level so a host
        adapter cannot accidentally branch on benchmark identity.  Other
        deployment-visible fields (messages, tool schemas, resource state)
        are preserved byte-for-byte as far as the mapping representation
        permits.
        """

        forbidden = frozenset({
            "familyid", "stage", "benchmarkfamily", "benchmarkstage",
        })
        if isinstance(value, Mapping):
            return {
                key: HermesPastBenchBridge._strip_benchmark_scope(child)
                for key, child in value.items()
                if "".join(char for char in str(key).casefold() if char.isalnum())
                not in forbidden
            }
        if isinstance(value, list):
            return [HermesPastBenchBridge._strip_benchmark_scope(child) for child in value]
        if isinstance(value, tuple):
            return tuple(
                HermesPastBenchBridge._strip_benchmark_scope(child)
                for child in value
            )
        return value

    def record_opportunity_evidence(
        self,
        values: OpportunityEvidence | Iterable[OpportunityEvidence],
    ) -> tuple[OpportunityEvidence, ...]:
        """Record deployment-visible opportunity evidence from a host adapter.

        This is the public bridge boundary for application-owned opportunity
        adapters.  It never accepts family/stage labels as a substitute for a
        visible evidence payload.
        """

        if isinstance(values, OpportunityEvidence):
            values = (values,)
        if isinstance(values, (str, bytes, Mapping)):
            raise TypeError("opportunity provider must return an iterable")
        recorded: list[OpportunityEvidence] = []
        for evidence in values:
            if not isinstance(evidence, OpportunityEvidence):
                raise TypeError("opportunity provider returned a non-evidence value")
            # The contract itself enforces pure-process plane/source identity;
            # this check makes the bridge boundary explicit and protects
            # against future contract widening.
            if evidence.evidence_plane.value != "pure_process":
                raise ValueError("runtime opportunity must be pure_process evidence")
            self._opportunity_evidence_log.append(evidence)
            recorded.append(evidence)
        return tuple(recorded)

    @property
    def process_feedback_event_ids(self) -> tuple[str, ...]:
        # ProcessCorpus canonicalizes events by logical identity rather than
        # append order.  Mirror that ordering here so concurrent/restarted
        # writers cannot change the response digest while representing the
        # same process evidence set.
        return tuple(sorted(event.event_id for event in self.process_feedback))

    @property
    def process_feedback_digest(self) -> str:
        """Stable identity for the content-free process corpus in this run."""

        return hashlib.sha256(
            json.dumps(
                list(self.process_feedback_event_ids),
                ensure_ascii=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    @property
    def static_results(self) -> tuple[StaticSemanticBoundaryResult, ...]:
        return tuple(self._static_results)

    @property
    def static_failures(self) -> tuple[tuple[str, str], ...]:
        return tuple(self._static_failures)

    def on_task_completed(self, result: Mapping[str, Any]) -> None:
        """Receive the explicit post-conversation task boundary from PAST."""

        self._record_semantic_outcomes(result)
        completed = result.get("completed") is True
        if not completed:
            return
        # PAST-Bench reflection is a separate review episode.  It has no
        # completed-task source projection or extraction invocation of its own;
        # running semantic writeback here would either duplicate the parent
        # compilation or make extraction evidence unverifiable.  Keep process
        # observation, but reserve semantic compilation for the primary
        # task-completed boundary.
        if self._stage != "reflection":
            if self.lifecycle is not None:
                self._process_lifecycle_boundary(
                    EvaluationTrigger.TASK_COMPLETED,
                    TaskLifecycleState.COMPLETED,
                )
            if self.static_writeback is not None:
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
        memory_use_operation_ids: list[str] = []
        if self.semantic_future_recorder is not None and self.semantic_feedback_resolver is not None:
            for future, step_id in self._semantic_futures:
                current_input = self._current_input(result)
                observation = self._semantic_deployment_observation(result)
                # Runtime observations must not be seeded with benchmark
                # contract scope.  The feedback resolver may use a
                # contract-scoped projection for audit attribution, but that
                # projection remains local to this benchmark-audit path and
                # is never emitted as pure-process evidence.
                audit_observation = self._semantic_audit_observation(
                    observation,
                    current_input=current_input,
                )
                resolution = self.semantic_feedback_resolver.resolve(
                    future,
                    audit_observation,
                )
                outcome = self.semantic_future_recorder.record_use_and_outcome(
                    future,
                    used_artifact_ids=resolution.used_artifact_ids,
                    outcome_status=resolution.outcome_status,
                    outcome_reason_code=resolution.outcome_reason_code,
                    step_id=step_id,
                )
                memory_use_operation_ids.append(outcome.use_operation_id)
                self._record_memory_use_evidence(future, outcome, result)
                self._record_extraction_feedback(
                    future,
                    audit_observation,
                    outcome,
                    current_input,
                )
        self._record_tool_call_results(
            result,
            memory_use_operation_id=(
                memory_use_operation_ids[0]
                if len(memory_use_operation_ids) == 1
                else None
            ),
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

    def _record_memory_use_evidence(
        self,
        future: SemanticFutureEvidence,
        outcome: SemanticOutcomeEvidence,
        result: Mapping[str, Any],
    ) -> None:
        """Persist a generic operation-bound use/outcome join.

        This evidence deliberately carries no benchmark family or parser
        labels.  It is emitted even when no artifact was used (exposure-only
        and non-use remain resolvable states), provided retrieval returned a
        concrete artifact set.
        """

        if not future.memory_artifact_ids:
            return
        completed = result.get("completed") is True
        observation_complete = not (
            result.get("partial") is True or result.get("interrupted") is True
        )
        outcome_kind = self._outcome_kind(result)
        matching_binding = next(
            (
                binding
                for binding in self.artifact_set_bindings
                if binding.complete
                and set(binding.member_artifact_ids)
                == set(future.memory_artifact_ids)
            ),
            None,
        )
        evidence = MemoryUseEvidence.create(
            artifact_ids=(
                () if matching_binding is not None
                else tuple(future.memory_artifact_ids)
            ),
            artifact_set_id=(
                matching_binding.binding_id if matching_binding is not None else None
            ),
            retrieval_operation_id=future.retrieval_operation_id,
            retrieved_artifact_ids=tuple(future.memory_artifact_ids),
            injection_operation_id=(
                future.injection_operation_id
                if future.injected_artifact_ids
                else None
            ),
            injected_artifact_ids=tuple(future.injected_artifact_ids),
            downstream_operation_id=outcome.use_operation_id,
            used_artifact_ids=tuple(outcome.used_artifact_ids),
            outcome_operation_id=outcome.outcome_operation_id,
            outcome_kind=outcome_kind,
            outcome_success=(
                None
                if (
                    not observation_complete
                    or outcome_kind is None
                    or outcome_kind is OutcomeEvidenceKind.TOOL_FAILURE
                )
                else completed
            ),
            observation_cutoff=self._observation_cutoff(result),
            provenance_id=future.query_operation_id,
            observation_complete=observation_complete,
            behavioral_consistency=False,
        )
        if self.static_writeback is not None:
            graph = materialize_operation_graph(
                self.static_writeback.operation_log.events
            )
            joined = resolve_memory_use(
                evidence,
                artifact_set_binding=matching_binding,
                operation_graph=graph,
            )
            if joined.reason_code == "operation_join_invalid":
                raise ValueError("semantic memory-use operation join is invalid")
        self._memory_use_evidence_log.append(evidence)

    @staticmethod
    def _outcome_kind(
        result: Mapping[str, Any],
    ) -> OutcomeEvidenceKind | None:
        raw_messages = result.get("messages")
        if isinstance(raw_messages, (list, tuple)):
            for message in raw_messages:
                if not isinstance(message, Mapping) or message.get("role") != "tool":
                    continue
                content = message.get("content")
                if not isinstance(content, str):
                    return None
                try:
                    payload = json.loads(content)
                except json.JSONDecodeError:
                    return None
                if not isinstance(payload, Mapping) or type(payload.get("success")) is not bool:
                    return None
                if payload.get("success") is False:
                    return OutcomeEvidenceKind.TOOL_FAILURE
        return OutcomeEvidenceKind.TASK_COMPLETION

    @staticmethod
    def _observation_cutoff(result: Mapping[str, Any]) -> str:
        """Return a stable host timestamp, with an explicit deterministic fallback."""

        for key in ("observation_cutoff", "observed_at", "completed_at"):
            value = result.get(key)
            if isinstance(value, str):
                try:
                    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
                except ValueError:
                    continue
                if parsed.tzinfo is not None:
                    return value.replace("+00:00", "Z")
        raw_messages = result.get("messages")
        timestamps = (
            message.get("timestamp")
            for message in raw_messages
            if isinstance(message, Mapping)
            and isinstance(message.get("timestamp"), (int, float))
            and not isinstance(message.get("timestamp"), bool)
        ) if isinstance(raw_messages, (list, tuple)) else ()
        values = tuple(timestamps)
        if values:
            return datetime.fromtimestamp(max(values), tz=UTC).isoformat(
                timespec="microseconds"
            ).replace("+00:00", "Z")
        # A missing host clock is an observation-quality limitation, not a
        # reason to invent wall-clock identity.  Epoch is deterministic and
        # callers can still inspect ``observation_complete``/join status.
        return "1970-01-01T00:00:00Z"

    def _record_tool_call_results(
        self,
        result: Mapping[str, Any],
        *,
        memory_use_operation_id: str | None = None,
    ) -> None:
        """Project every observed tool call/result into exact process events.

        Arguments and return bodies stay in the host-owned trace.  The public
        process corpus receives only stable call/result identity, tool-name
        digest, retry identity, status, host boundary and receipt joins.
        """

        raw_messages = result.get("messages")
        if not isinstance(raw_messages, (list, tuple)):
            return
        messages = tuple(value for value in raw_messages if isinstance(value, Mapping))
        calls: dict[str, list[tuple[str, str, str, str, bool]]] = {}
        call_counts: dict[str, int] = {}
        duplicate_call_ids: set[str] = set(self._tool_call_ids_seen)
        emitted_result_ids: set[str] = set()
        duplicate_result_ids: set[str] = set(self._tool_result_ids_seen)
        joins: list[ToolCallResultJoin] = []
        host_event_id = self._last_host_event_id or "event.tool-observation"
        source_revision = self._last_host_source_revision or self._exposure_context_revision()
        for message in messages:
            if message.get("role") != "assistant":
                continue
            raw_calls = message.get("tool_calls")
            if not isinstance(raw_calls, (list, tuple)):
                continue
            for ordinal, call in enumerate(raw_calls, start=1):
                if not isinstance(call, Mapping):
                    continue
                function = call.get("function")
                if not isinstance(function, Mapping):
                    continue
                name = str(function.get("name") or "unknown_tool")
                raw_call_id = str(call.get("id") or f"tool-call-{ordinal}")
                count = call_counts.get(raw_call_id, 0)
                call_counts[raw_call_id] = count + 1
                retry_identity = (
                    "retry."
                    + hashlib.sha256(
                        f"{raw_call_id}:{name}:{count}".encode("utf-8")
                    ).hexdigest()[:24]
                )
                call_id = raw_call_id if re.fullmatch(
                    r"[A-Za-z0-9][A-Za-z0-9_.:+-]{0,255}", raw_call_id
                ) else (
                    "tool-call."
                    + hashlib.sha256(raw_call_id.encode("utf-8")).hexdigest()[:24]
                )
                if count or call_id in self._tool_call_ids_seen:
                    duplicate_call_ids.add(raw_call_id)
                declared_task_id = call.get("task_id", message.get("task_id"))
                cross_task = (
                    declared_task_id is not None
                    and str(declared_task_id) != self._task_id
                )
                calls.setdefault(raw_call_id, []).append(
                    (
                        call_id,
                        name,
                        retry_identity,
                        "receipt.tool-call."
                        + hashlib.sha256(
                            f"{call_id}:{retry_identity}".encode("utf-8")
                        ).hexdigest()[:24],
                        cross_task,
                    )
                )
        for message in messages:
            if message.get("role") != "tool":
                continue
            raw_call_id = str(message.get("tool_call_id") or "")
            call_occurrences = calls.get(raw_call_id, ())
            content = message.get("content")
            parsed = None
            if isinstance(content, str):
                try:
                    parsed = json.loads(content)
                except json.JSONDecodeError:
                    parsed = None
            # A malformed result (or a non-boolean success field) is not a
            # tool failure.  Preserve the call/result identity, but mark the
            # payload as a type mismatch so downstream attribution remains
            # fail-closed instead of manufacturing a negative signal.
            valid_result = (
                isinstance(parsed, Mapping)
                and type(parsed.get("success")) is bool
            )
            success = parsed.get("success") if valid_result else None
            result_seed = json.dumps(
                {"call_id": raw_call_id, "content_digest": hashlib.sha256(
                    str(content or "").encode("utf-8")
                ).hexdigest()},
                sort_keys=True,
                separators=(",", ":"),
            )
            result_id = str(message.get("id") or (
                "tool-result."
                + hashlib.sha256(result_seed.encode("utf-8")).hexdigest()[:24]
            ))
            duplicate_result = (
                result_id in emitted_result_ids
                or result_id in duplicate_result_ids
            )
            emitted_result_ids.add(result_id)
            declared_task_id = message.get("task_id")
            result_cross_task = (
                declared_task_id is not None
                and str(declared_task_id) != self._task_id
            )
            if not call_occurrences:
                orphan_call_id = (
                    "orphan-call."
                    + hashlib.sha256(raw_call_id.encode("utf-8")).hexdigest()[:24]
                )
                joins.append(ToolCallResultJoin.create(
                    call_id=orphan_call_id,
                    result_id=result_id,
                    tool_name_digest=hashlib.sha256(b"unknown_tool").hexdigest(),
                    success=success,
                    retry_identity="retry.orphan",
                    run_id=self._run_id,
                    variant=self.ledger.variant,
                    trace_id=self._trace_id,
                    episode_id=self._episode_id,
                    session_id=self._session_id,
                    task_id=self._task_id,
                    source_revision=source_revision,
                    host_event_id=host_event_id,
                    memory_use_operation_id=memory_use_operation_id,
                    result_receipt_id="receipt.tool-result."
                    + hashlib.sha256(result_id.encode("utf-8")).hexdigest()[:24],
                    call_present=False,
                    result_present=True,
                    orphan_result=True,
                    cross_task=result_cross_task,
                ))
                continue
            # A result referring to a duplicated call ID cannot identify
            # which retry occurrence produced it.  Bind it to the first
            # occurrence for deterministic projection and retain the
            # duplicate flag; the remaining occurrences are emitted as
            # duplicate/missing call-only joins below.
            call_id, name, retry_identity, call_receipt, call_cross_task = (
                call_occurrences[0]
            )
            joins.append(ToolCallResultJoin.create(
                call_id=call_id,
                result_id=result_id,
                tool_name_digest=hashlib.sha256(name.encode("utf-8")).hexdigest(),
                success=success,
                retry_identity=retry_identity,
                run_id=self._run_id,
                variant=self.ledger.variant,
                trace_id=self._trace_id,
                episode_id=self._episode_id,
                session_id=self._session_id,
                task_id=self._task_id,
                source_revision=source_revision,
                host_event_id=host_event_id,
                memory_use_operation_id=memory_use_operation_id,
                call_receipt_id=call_receipt,
                result_receipt_id="receipt.tool-result."
                + hashlib.sha256(result_id.encode("utf-8")).hexdigest()[:24],
                duplicate_call=raw_call_id in duplicate_call_ids,
                duplicate_result=duplicate_result,
                type_mismatch=not valid_result,
                cross_task=call_cross_task or result_cross_task,
            ))
        represented_calls = {
            (join.call_id, join.retry_identity)
            for join in joins
            if join.call_present
        }
        for raw_call_id, occurrences in calls.items():
            for call_id, name, retry_identity, call_receipt, cross_task in occurrences:
                if (call_id, retry_identity) in represented_calls:
                    continue
                joins.append(ToolCallResultJoin.create(
                    call_id=call_id,
                    result_id=None,
                    tool_name_digest=hashlib.sha256(name.encode("utf-8")).hexdigest(),
                    success=None,
                    retry_identity=retry_identity,
                    run_id=self._run_id,
                    variant=self.ledger.variant,
                    trace_id=self._trace_id,
                    episode_id=self._episode_id,
                    session_id=self._session_id,
                    task_id=self._task_id,
                    source_revision=source_revision,
                    host_event_id=host_event_id,
                    memory_use_operation_id=memory_use_operation_id,
                    call_receipt_id=call_receipt,
                    call_present=True,
                    result_present=False,
                    duplicate_call=raw_call_id in duplicate_call_ids,
                    cross_task=cross_task,
                ))
        for join in joins:
            previous = self._tool_call_result_joins.get(join.join_id)
            if previous is not None and previous != join:
                raise ValueError("conflicting tool call/result join")
            self._tool_call_result_joins[join.join_id] = join
            for event in join.process_events(
                family_id=self._family_id,
                stage=self._stage,
            ):
                self._process_feedback.record(event)
        self._tool_call_ids_seen.update(
            join.call_id for join in joins if join.call_present and join.call_id is not None
        )
        self._tool_result_ids_seen.update(
            join.result_id
            for join in joins
            if join.result_present and join.result_id is not None
        )

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
        available_keys = detect_user_source_semantic_keys(
            self._family_id,
            tuple(
                (message.role, message.content)
                for message in projection.messages
            ),
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
        self._record_artifact_set_bindings(record.source)
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

    def _record_artifact_set_bindings(self, source: object) -> None:
        provider = self._artifact_set_binding_provider
        if provider is None:
            return
        values = provider(source)
        if isinstance(values, ArtifactSetSemanticBinding):
            values = (values,)
        if isinstance(values, (str, bytes, Mapping)):
            raise TypeError("artifact-set provider must return an iterable")
        if not isinstance(source, ExtractionSourceEvidence):
            raise TypeError("artifact-set source has the wrong type")
        source_artifacts = {
            fact.artifact_id for fact in source.facts if fact.artifact_id is not None
        }
        source_facts = {fact.fact_id for fact in source.facts}
        for binding in values:
            if not isinstance(binding, ArtifactSetSemanticBinding):
                raise TypeError("artifact-set provider returned a non-binding value")
            if binding.source_digest != source.source_projection_digest:
                raise ValueError("artifact-set binding source digest mismatch")
            if not set(binding.member_artifact_ids).issubset(source_artifacts):
                raise ValueError("artifact-set binding references foreign artifact")
            if not set(binding.member_fact_ids).issubset(source_facts):
                raise ValueError("artifact-set binding references foreign fact")
            self._artifact_set_binding_log.append(binding)

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
            # Keep benchmark identity explicit at the bridge boundary.  The
            # ledger defaults are a convenience for direct callers, not a
            # substitute for the host's evidence-plane declaration.
            family_id=self._family_id,
            stage=self._stage,
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
        messages = ()
        raw_messages = result.get("messages")
        if isinstance(raw_messages, (list, tuple)):
            messages = tuple(
                value for value in raw_messages if isinstance(value, Mapping)
            )
        current_input = self._current_input(result)
        # Runtime opportunity evidence is supplied by a host/application
        # adapter from deployment-visible input, environment or tool schema.
        # Do this before constructing the benchmark-audit projection so the
        # two evidence planes cannot be conflated.
        runtime_opportunities = self._record_runtime_opportunities(result)
        runtime_requirements = tuple(dict.fromkeys(
            evidence.semantic_requirement
            for evidence in runtime_opportunities
        ))
        runtime_current_input_requirements = tuple(dict.fromkeys(
            evidence.semantic_requirement
            for evidence in runtime_opportunities
            if evidence.source_surface is OpportunitySurface.CURRENT_INPUT
        ))
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
            "runtime_requirements": list(runtime_requirements),
            "runtime_current_input_requirements": list(
                runtime_current_input_requirements
            ),
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
            # Pure runtime observations do not run benchmark-family semantic
            # parsers.  Current-input requirements may be supplied by an
            # application-owned opportunity provider and are kept in the
            # separate audit projection when a registered benchmark contract
            # is explicitly requested.
            current_input_semantic_keys=runtime_current_input_requirements,
            # Task semantic requirements are application-owned runtime
            # evidence.  A benchmark family contract is not allowed to
            # manufacture them at the bridge boundary.
            task_semantic_keys=runtime_requirements,
            final_response=str(result.get("final_response") or ""),
            tool_events=self._observable_tool_events(messages),
            completed=result.get("completed") is True,
            observation_complete=observation_complete,
            censor_reason=(None if observation_complete else "execution_incomplete"),
        )

    def _semantic_audit_observation(
        self,
        observation: DeploymentObservation,
        *,
        current_input: str = "",
    ) -> DeploymentObservation:
        """Attach frozen benchmark scope only for audit attribution.

        ``DeploymentObservation`` is also used as the source for process
        captures.  Keeping the runtime observation scope-free prevents a
        benchmark contract from leaking into optimizer-visible evidence.  The
        semantic feedback resolver still needs its registered contract to
        decide whether an opportunity was observed, so it receives this
        short-lived audit projection instead.
        """

        resolver = self.semantic_feedback_resolver
        if resolver is None:
            raise ValueError("semantic feedback resolver is unavailable")
        contract = resolver.registry.resolver(observation.family_id).contract
        if observation.stage not in contract.opportunity.eligible_stages:
            return observation
        runtime_current_input_requirements = tuple(
            observation.current_input_semantic_keys
        )
        runtime_task_requirements = tuple(observation.task_semantic_keys)
        audit_current_input_requirements = detect_current_input_semantic_keys(
            observation.family_id,
            current_input,
        )
        audit_task_requirements = contract.opportunity.memory_scope_keys
        return replace(
            observation,
            current_input_semantic_keys=tuple(dict.fromkeys(
                (*runtime_current_input_requirements, *audit_current_input_requirements)
            )),
            task_semantic_keys=tuple(dict.fromkeys(
                (*runtime_task_requirements, *audit_task_requirements)
            )),
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
                try:
                    return native_call()
                except Exception as native_exc:
                    stage_reason = (
                        "retrieval_failure"
                        if operation == "session_search"
                        else "tool_failure"
                        if operation.startswith("skill")
                        else "injection_failure"
                    )
                    self._record_process_observation(
                        kind=process_kind,
                        status=ProcessEventStatus.FAILED,
                        host_event_id=self._last_host_event_id or f"event.native-bypass.{operation}",
                        source_revision=self._last_host_source_revision or self._exposure_context_revision(),
                        input_payload={"operation": operation, "route": "native_bypass"},
                        output_payload={
                            "adapter_failure_type": failure_type,
                            "native_failure_type": type(native_exc).__name__,
                        },
                        reason_codes=("adapter_failure", stage_reason),
                        execution_receipt_ids=(
                            "receipt.native-bypass-failure."
                            + hashlib.sha256(
                                f"{operation}:{failure_type}:{type(native_exc).__name__}".encode("utf-8")
                            ).hexdigest()[:24],
                        ),
                    )
                    raise
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
        """Record a skill call/result closure without persisting skill text.

        Hermes' registry wrapper does not expose the host's tool-call ID, so
        this bridge derives a stable per-invocation identity from the tool,
        query digest, host boundary and retry ordinal.  The closure still
        carries only digests and status into the process ledger; arguments and
        returned skill content stay in the owner-controlled Hermes trace.
        """

        try:
            payload = json.loads(result)
        except (TypeError, json.JSONDecodeError):
            payload = {}
        success = (
            payload.get("success")
            if isinstance(payload, Mapping) and type(payload.get("success")) is bool
            else None
        )
        type_mismatch = success is None
        query_digest = hashlib.sha256(query.encode("utf-8")).hexdigest()
        tool_digest = hashlib.sha256(tool_name.encode("utf-8")).hexdigest()
        source_revision = (
            self._last_host_source_revision or self._exposure_context_revision()
        )
        host_event_id = self._last_host_event_id or (
            "event.skill." + hashlib.sha256(
                f"{tool_name}:{query_digest}:{source_revision}".encode("utf-8")
            ).hexdigest()[:40]
        )
        host_digest = hashlib.sha256(host_event_id.encode("utf-8")).hexdigest()[:8]
        invocation_key = (tool_digest[:8], query_digest[:16], host_digest)
        ordinal = self._skill_invocation_counts.get(invocation_key, 0)
        self._skill_invocation_counts[invocation_key] = ordinal + 1
        invocation_digest = hashlib.sha256(json.dumps(
            {
                "tool": tool_name,
                "query_digest": query_digest,
                "host_event_id": host_event_id,
                "source_revision": source_revision,
                "ordinal": ordinal,
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
        call_id = (
            f"call.skill.{query_digest[:16]}.{tool_digest[:8]}."
            f"{host_digest}.{ordinal}"
        )
        result_id = (
            f"result.skill.{query_digest[:16]}.{tool_digest[:8]}."
            f"{host_digest}.{ordinal}"
        )
        retry_identity = "retry.skill." + invocation_digest[:24]
        call_receipt_id = "receipt.skill-call." + invocation_digest[:24]
        result_receipt_id = "receipt.skill-result." + invocation_digest[:24]
        join = ToolCallResultJoin.create(
            call_id=call_id,
            result_id=result_id,
            tool_name_digest=tool_digest,
            success=success,
            retry_identity=retry_identity,
            run_id=self._run_id,
            variant=self.ledger.variant,
            trace_id=self._trace_id,
            episode_id=self._episode_id,
            session_id=self._session_id,
            task_id=self._task_id,
            source_revision=source_revision,
            host_event_id=host_event_id,
            call_receipt_id=call_receipt_id,
            result_receipt_id=result_receipt_id,
            type_mismatch=type_mismatch,
        )
        previous = self._tool_call_result_joins.get(join.join_id)
        if previous is not None:
            if previous != join:
                raise ValueError("conflicting skill tool call/result join")
            return
        self._tool_call_result_joins[join.join_id] = join
        self._tool_call_ids_seen.add(call_id)
        self._tool_result_ids_seen.add(result_id)
        for event in join.process_events(
            family_id=self._family_id,
            stage=self._stage,
        ):
            self._process_feedback.record(event)

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
