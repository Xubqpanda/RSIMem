"""Hermes host-owned memory projections used by the RSIMem bridge.

The wrappers keep native Hermes objects behind a narrow projection boundary.
They depend on bridge callbacks for ledger/evidence recording but do not know
PAST task or grader data.
"""

from __future__ import annotations

import hashlib
import json
from tempfile import TemporaryDirectory
from typing import Any, TYPE_CHECKING

from .adapter_contracts import (
    AdapterResult,
    AdapterStatus,
    CanonicalHostEvent,
    HostCapabilities,
    HostAdapter,
    MethodRunIdentity,
    MethodStateSnapshot,
    content_digest,
)
from .memory.contracts import MemoryKind, MemoryQuery
from .memory.process_feedback import ProcessEventKind, ProcessEventStatus
from .hermes_integration import _bound_hermes_skills_dir, _materialize_procedural_hits

if TYPE_CHECKING:
    from .hermes_past_bridge import HermesPastBenchBridge


class HermesHostAdapter:
    """Typed host boundary around one live Hermes bridge.

    The bridge still owns lifecycle/evidence policy; this object owns the
    host-facing attachment and identity checks so a future host implementation
    can replace Hermes without changing method or benchmark contracts.
    """

    def __init__(self, bridge: "HermesPastBenchBridge") -> None:
        self._bridge = bridge
        self._run: MethodRunIdentity | None = None
        self._events: set[str] = set()

    @property
    def capabilities(self) -> HostCapabilities:
        return HostCapabilities(
            memory_kinds=tuple(MemoryKind),
            tool_call_result_closure=True,
            usage_accounting=True,
            restart=False,
            context_snapshot=True,
            native_bypass=(
                str(self._bridge.config.adapter_failure_policy.value)
                == "bypass_native"
            ),
        )

    def attach(self, agent: object) -> None:
        memory_store = getattr(agent, "_memory_store", None)
        if memory_store is not None:
            agent._memory_store = _PromptMemoryStore(self._bridge, memory_store)
        session_db = getattr(agent, "_session_db", None)
        if session_db is not None:
            agent._session_db = _SessionDb(
                self._bridge,
                session_db,
                str(getattr(agent, "session_id", "") or "") or None,
            )
        wrap_skill_handlers(self._bridge)

    def prepare_session(self, run: MethodRunIdentity) -> AdapterResult:
        expected = (
            self._bridge._run_id,
            self._bridge._session_id,
            self._bridge._task_id,
        )
        if (run.run_id, run.session_id, run.task_id) != expected:
            return AdapterResult(AdapterStatus.REJECTED, "operation.hermes.prepare", "identity_mismatch")
        self._run = run
        return AdapterResult(AdapterStatus.SUPPORTED, "operation.hermes.prepare")

    def observe_event(self, event: CanonicalHostEvent) -> AdapterResult:
        if self._run is None:
            return AdapterResult(AdapterStatus.REJECTED, "operation.hermes.observe", "run_not_prepared")
        if event.session_id != self._run.session_id or event.task_id != self._run.task_id:
            return AdapterResult(AdapterStatus.REJECTED, "operation.hermes.observe", "identity_mismatch")
        if event.event_id in self._events:
            return AdapterResult(AdapterStatus.REJECTED, "operation.hermes.observe", "duplicate_event")
        self._events.add(event.event_id)
        return AdapterResult(AdapterStatus.SUPPORTED, "operation.hermes.observe")

    def snapshot_state(self) -> MethodStateSnapshot:
        snapshot = self._bridge._collect_completed_snapshot()
        digest = content_digest({
            "snapshot_id": snapshot.snapshot_id,
            "context_revision": snapshot.context_revision,
            "segment_ids": [segment.segment_id for segment in snapshot.segments],
            "active_segment_ids": list(snapshot.active_segment_ids),
            "current_turn_id": snapshot.current_turn_id,
        })
        return MethodStateSnapshot(
            state_id=f"state.hermes.{snapshot.snapshot_id}",
            revision=snapshot.context_revision,
            state_schema="hermes.context.snapshot.v1",
            state_digest=digest,
            active=True,
        )

    def restart(self, run: MethodRunIdentity) -> AdapterResult:
        return AdapterResult(
            AdapterStatus.UNSUPPORTED,
            "operation.hermes.restart",
            "restart_requires_new_bridge",
        )


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
            hits = self._bridge.observe_query(
                MemoryKind.SEMANTIC,
                "",
                namespace=target,
                limit=100,
                surface="system_prompt" if result else None,
            )
            self._bridge.record_semantic_prompt(
                result,
                target,
                artifact_ids=tuple(hit.artifact.artifact_id for hit in hits),
                retrieved_hits=hits,
                retrieval_limit=100,
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
            rendered = self._native._render_block(
                target,
                [hit.artifact.content for hit in hits],
            )
            # Rendering is part of the host exposure boundary.  Do not mark
            # artifacts as injected until the render succeeds; a renderer
            # failure is an injection failure, not an exposure/use event.
            self._bridge.runtime.mark_injected(hits, surface="system_prompt")
            return rendered

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
        # A successful adapter read already has the authoritative hit set.
        # Reuse it when creating the future trace instead of querying the
        # backend a second time.  Native-bypass reads are deliberately not
        # attributed to RSIMem artifacts: the adapter did not establish an
        # exact retrieval/injection join, so recording a synthetic future
        # would overstate memory use after a failure.
        if self._bridge._last_adapter_route == "adapter":
            hits = tuple(self._snapshots.get(target, ()))
            self._bridge.record_semantic_prompt(
                adapter_result,
                target,
                artifact_ids=tuple(hit.artifact.artifact_id for hit in hits),
                retrieved_hits=hits,
                retrieval_limit=100,
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


def wrap_skill_handlers(bridge: "HermesPastBenchBridge") -> None:
    """Install the procedural projection at the Hermes tool boundary."""

    from tools.registry import registry

    for tool_name in ("skills_list", "skill_view"):
        entry = registry._tools.get(tool_name)
        if entry is None or tool_name in bridge._tool_handlers:
            continue
        original = entry.handler
        bridge._tool_handlers[tool_name] = original

        def handler(
            args: dict[str, Any],
            _tool_name: str = tool_name,
            _original: Any = original,
            **kwargs: Any,
        ) -> str:
            query = "" if _tool_name == "skills_list" else str(args.get("name") or "")
            native_call = lambda: _original(args, **kwargs)
            if not bridge.uses_adapter:
                result = native_call()
                bridge.observe_query(
                    MemoryKind.PROCEDURAL,
                    query,
                    limit=100 if _tool_name == "skills_list" else 5,
                    surface=_tool_name,
                )
                bridge._record_skill_process(_tool_name, query, result)
                return result

            def adapter_read() -> str:
                hits = bridge.runtime.query(MemoryQuery(
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
                bridge._record_process_observation(
                    kind=ProcessEventKind.RETRIEVAL,
                    status=ProcessEventStatus.SUCCESS if hits else ProcessEventStatus.FAILED,
                    host_event_id=bridge._last_host_event_id or f"event.procedural-query.{query_digest[:40]}",
                    source_revision=bridge._last_host_source_revision or bridge._exposure_context_revision(),
                    input_payload={"query_digest": query_digest, "tool": _tool_name},
                    output_payload={"hit_count": len(hits)},
                    reason_codes=("decision_observed",) if hits else ("retrieval_miss",),
                    execution_receipt_ids=(f"receipt.procedural-query.{query_digest[:24]}",),
                )
                projected_hits = hits
                if _tool_name == "skill_view" and not projected_hits:
                    projected_hits = bridge.runtime.query(MemoryQuery(
                        MemoryKind.PROCEDURAL,
                        "",
                        limit=100,
                    ))
                with TemporaryDirectory(prefix="rsimem-hermes-live-skills-") as directory:
                    skills_dir = Path(directory) / "skills"
                    _materialize_procedural_hits(skills_dir, projected_hits)
                    with _bound_hermes_skills_dir(skills_dir):
                        result = _original(args, **kwargs)
                payload = json.loads(result)
                if hits and payload.get("success") is True:
                    bridge.runtime.mark_injected(hits, surface=_tool_name)
                return result

            adapter_result = bridge.adapter_call(_tool_name, adapter_read, native_call)
            bridge.verify_projection(
                MemoryKind.PROCEDURAL,
                "hermes-native-procedural",
                _tool_name,
                adapter_result,
                native_call,
            )
            bridge._record_skill_process(_tool_name, query, adapter_result)
            return adapter_result

        entry.handler = handler


__all__ = ["HermesHostAdapter", "_PromptMemoryStore", "_SessionDb", "wrap_skill_handlers"]
