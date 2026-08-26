"""Opt-in RSIMem read bridge for the in-process PAST-Bench Hermes adapter."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable

from .hermes_integration import (
    HermesAdapterExecutionError,
    HermesAdapterFailurePolicy,
    HermesExecutionMode,
    HermesExperimentConfig,
    _bound_hermes_skills_dir,
    _materialize_procedural_hits,
    build_configured_hermes_runtime,
)
from .ledger import MemoryLedgerObserver
from .memory import MemoryEvent, MemoryEventKind, MemoryKind, MemoryQuery


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

        return self._bridge.adapter_call(
            "system_prompt",
            adapter_read,
            lambda: self._native.format_for_system_prompt(target),
        )


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

        return self._bridge.adapter_call("session_search", adapter_read, native_call)

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
            return dict(session)
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
        return [dict(message) for message in messages]


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
    ) -> None:
        if config.mode == HermesExecutionMode.NATIVE:
            raise ValueError("native mode must not construct an RSIMem bridge")
        self.config = config
        self.evidence_path = evidence_path.expanduser().resolve()
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
        self._closed = False

    @property
    def uses_adapter(self) -> bool:
        return self.config.mode == HermesExecutionMode.ADAPTER_LEDGER

    def attach(self, agent: object) -> None:
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

                return self.adapter_call(_tool_name, adapter_read, native_call)

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
            self.runtime.close()
