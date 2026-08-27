"""Opt-in RSIMem read bridge for the in-process PAST-Bench Hermes adapter."""

from __future__ import annotations

import json
import hashlib
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Mapping

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
from .memory.live_writeback import (
    StaticSemanticBoundaryResult,
    StaticSemanticWritebackConfig,
    StaticSemanticWritebackRuntime,
)
from .memory.adaptive_policy_store import JsonAdaptivePolicyStore
from .memory.future_trace import (
    SemanticFeedbackContract,
    SemanticFeedbackResolver,
    SemanticFutureEvidence,
    SemanticFutureTraceRecorder,
)
from .memory.extraction_feedback import (
    DeploymentObservation,
    ObservableToolEvent,
    detect_current_input_semantic_keys,
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
            else:
                self.semantic_future_recorder = None
                self.semantic_feedback_resolver = None
        else:
            self.static_writeback = None
            self.semantic_future_recorder = None
            self.semantic_feedback_resolver = None
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
            results = self.static_writeback.process_completed_snapshot(snapshot)
            for compiled in results:
                if not any(
                    item.compilation_id == compiled.compilation_id
                    for item in self._static_results
                ):
                    self._static_results.append(compiled)
        except Exception as exc:
            self._static_failures.append((
                EvaluationTrigger.TASK_COMPLETED.value,
                type(exc).__name__,
            ))

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

    def record_semantic_prompt(self, prompt: str | None, namespace: str) -> None:
        if self.semantic_future_recorder is None or not prompt:
            return
        step_id = f"future-semantic.{namespace}.{len(self._semantic_futures) + 1}"
        future = self.semantic_future_recorder.record_prompt_injection(
            self.runtime.registry,
            prompt,
            namespace=namespace,
            parent_operation_ids=(),
            step_id=step_id,
        )
        self._semantic_futures.append((future, step_id))

    def _record_semantic_outcomes(self, result: Mapping[str, Any]) -> None:
        if (
            self.semantic_future_recorder is None
            or self.semantic_feedback_resolver is None
            or self._semantic_outcomes_recorded
        ):
            return
        for future, step_id in self._semantic_futures:
            observation = self._semantic_deployment_observation(result)
            resolution = self.semantic_feedback_resolver.resolve(future, observation)
            self.semantic_future_recorder.record_use_and_outcome(
                future,
                used_artifact_ids=resolution.used_artifact_ids,
                outcome_status=resolution.outcome_status,
                outcome_reason_code=resolution.outcome_reason_code,
                step_id=step_id,
            )
        self._semantic_outcomes_recorded = True

    def _semantic_deployment_observation(
        self,
        result: Mapping[str, Any],
    ) -> DeploymentObservation:
        resolver = self.semantic_feedback_resolver
        if resolver is None or self._family_id is None or self._stage is None:
            raise ValueError("semantic feedback resolver identity is unavailable")
        contract = resolver.registry.resolver(self._family_id).contract
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
        current_input = user_inputs[-1] if user_inputs else ""
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
        return result

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
                return native_call()
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
        except Exception:
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
                    return result

                def adapter_read() -> str:
                    hits = self.runtime.query(MemoryQuery(
                        MemoryKind.PROCEDURAL,
                        query,
                        limit=100 if _tool_name == "skills_list" else 5,
                    ))
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
                return adapter_result

            entry.handler = handler

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
