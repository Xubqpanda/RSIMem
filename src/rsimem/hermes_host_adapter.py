"""Hermes host-owned memory projections used by the RSIMem bridge.

The wrappers keep native Hermes objects behind a narrow projection boundary.
They depend on bridge callbacks for ledger/evidence recording but do not know
PAST task or grader data.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, TYPE_CHECKING

from .memory.contracts import MemoryKind, MemoryQuery
from .memory.process_feedback import ProcessEventKind, ProcessEventStatus

if TYPE_CHECKING:
    from .hermes_past_bridge import HermesPastBenchBridge


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


__all__ = ["_PromptMemoryStore", "_SessionDb"]
