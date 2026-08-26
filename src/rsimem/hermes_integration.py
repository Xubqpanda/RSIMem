"""Opt-in Hermes runtime construction and deterministic surface equivalence."""

from __future__ import annotations

import os
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from .ledger import MemoryLedgerObserver
from .memory import (
    MemoryEvent,
    MemoryEventKind,
    MemoryKind,
    MemoryObserver,
    MemoryQuery,
    MemoryRuntime,
    build_hermes_native_registry,
)


_ENTRY_DELIMITER = "\n\u00a7\n"
_SEMANTIC_LIMITS = {"memory": 2200, "user": 1375}
_NATIVE_ROUTES = {
    MemoryKind.SEMANTIC: "hermes-native-semantic",
    MemoryKind.EPISODIC: "hermes-native-episodic",
    MemoryKind.PROCEDURAL: "hermes-native-procedural",
}


class HermesExecutionMode(StrEnum):
    NATIVE = "native"
    NATIVE_LEDGER = "native+ledger"
    ADAPTER_LEDGER = "native+adapter+ledger"


@dataclass(frozen=True, slots=True)
class HermesExperimentConfig:
    """Explicit experiment path. Direct native behavior is the default."""

    mode: HermesExecutionMode = HermesExecutionMode.NATIVE
    routes: Mapping[MemoryKind, str] = field(default_factory=lambda: dict(_NATIVE_ROUTES))

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", HermesExecutionMode(self.mode))
        routes = {MemoryKind(kind): backend for kind, backend in self.routes.items()}
        if set(routes) != set(MemoryKind):
            raise ValueError("Hermes experiment config requires one route per memory kind")
        if any(not backend.strip() for backend in routes.values()):
            raise ValueError("Hermes backend route names must not be empty")
        object.__setattr__(self, "routes", MappingProxyType(routes))

    @property
    def uses_adapter(self) -> bool:
        return self.mode == HermesExecutionMode.ADAPTER_LEDGER

    @property
    def ledger_enabled(self) -> bool:
        return self.mode != HermesExecutionMode.NATIVE


def build_configured_hermes_runtime(
    hermes_home: Path,
    config: HermesExperimentConfig,
    *,
    observers: Iterable[MemoryObserver] = (),
) -> MemoryRuntime:
    """Build the selected typed runtime without changing the direct native path."""

    unsupported = {
        kind: backend
        for kind, backend in config.routes.items()
        if backend != _NATIVE_ROUTES[kind]
    }
    if unsupported:
        rendered = ", ".join(
            f"{kind.value}={backend}" for kind, backend in sorted(
                unsupported.items(), key=lambda item: item[0].value
            )
        )
        raise ValueError(f"unregistered Hermes backend routes: {rendered}")
    return MemoryRuntime(build_hermes_native_registry(hermes_home), observers=observers)


@dataclass(frozen=True, slots=True)
class HermesEquivalenceProbe:
    episodic_query: str
    episodic_limit: int = 5
    procedural_skill_name: str | None = None
    procedural_resource_path: str | None = None

    def __post_init__(self) -> None:
        if not self.episodic_query.strip():
            raise ValueError("equivalence probe requires an episodic query")
        if self.episodic_limit < 1:
            raise ValueError("episodic_limit must be positive")
        if (
            self.procedural_skill_name is not None
            and not self.procedural_skill_name.strip()
        ):
            raise ValueError("procedural_skill_name must be non-empty when present")
        if (
            self.procedural_resource_path is not None
            and not self.procedural_resource_path.strip()
        ):
            raise ValueError("procedural_resource_path must be non-empty when present")


@dataclass(frozen=True, slots=True)
class _SemanticView:
    memory_block: str
    user_block: str
    memory_entry_count: int
    user_entry_count: int
    memory_content_chars: int
    user_content_chars: int


@dataclass(frozen=True, slots=True)
class _EpisodicHitView:
    session_id: str
    role: str
    content: str
    snippet: str
    context: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class _ProceduralSkillView:
    name: str
    description: str
    category: str | None
    content: str
    resources: tuple[tuple[str, bytes], ...]


@dataclass(frozen=True, slots=True)
class _HermesSurfaceView:
    semantic: _SemanticView
    episodic: tuple[_EpisodicHitView, ...]
    procedural: tuple[_ProceduralSkillView, ...]


@dataclass(frozen=True, slots=True)
class HermesSurfaceCheck:
    memory_kind: MemoryKind
    equivalent: bool
    native_item_count: int
    candidate_item_count: int
    native_content_chars: int
    candidate_content_chars: int
    native_resource_bytes: int = 0
    candidate_resource_bytes: int = 0


@dataclass(frozen=True, slots=True)
class HermesVariantResult:
    mode: HermesExecutionMode
    equivalent_to_native: bool
    checks: tuple[HermesSurfaceCheck, ...]
    ledger_enabled: bool
    memory_event_count: int
    memory_event_kinds: tuple[MemoryEventKind, ...]
    ledger_event_count: int
    ledger_event_kinds: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class HermesEquivalenceReport:
    variants: tuple[HermesVariantResult, ...]

    @property
    def equivalent(self) -> bool:
        return all(variant.equivalent_to_native for variant in self.variants)


class HermesExecutionSurface(StrEnum):
    SYSTEM_PROMPT = "system_prompt"
    SESSION_SEARCH = "session_search"
    SKILLS_LIST = "skills_list"
    SKILL_VIEW = "skill_view"


@dataclass(frozen=True, slots=True)
class HermesExecutionSurfaceCheck:
    surface: HermesExecutionSurface
    equivalent: bool
    native_content_chars: int
    candidate_content_chars: int


@dataclass(frozen=True, slots=True)
class HermesExecutionVariantResult:
    mode: HermesExecutionMode
    equivalent_to_native: bool
    checks: tuple[HermesExecutionSurfaceCheck, ...]
    ledger_enabled: bool
    memory_event_count: int
    memory_event_kinds: tuple[MemoryEventKind, ...]
    ledger_event_count: int
    ledger_event_kinds: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class HermesExecutionEquivalenceReport:
    variants: tuple[HermesExecutionVariantResult, ...]

    @property
    def equivalent(self) -> bool:
        return all(variant.equivalent_to_native for variant in self.variants)


def _read_entries(path: Path) -> tuple[str, ...]:
    if not path.exists():
        return ()
    raw = path.read_text(encoding="utf-8")
    return tuple(dict.fromkeys(item.strip() for item in raw.split(_ENTRY_DELIMITER) if item.strip()))


def _render_semantic_block(target: str, entries: tuple[str, ...]) -> str:
    if not entries:
        return ""
    content = _ENTRY_DELIMITER.join(entries)
    limit = _SEMANTIC_LIMITS[target]
    percent = int((len(content) / limit) * 100)
    if target == "user":
        header = f"USER PROFILE (who the user is) [{percent}% \u2014 {len(content):,}/{limit:,} chars]"
    else:
        header = f"MEMORY (your personal notes) [{percent}% \u2014 {len(content):,}/{limit:,} chars]"
    separator = "\u2550" * 46
    return f"{separator}\n{header}\n{separator}\n{content}"


def _native_semantic(home: Path) -> _SemanticView:
    memories = home / "memories"
    memory_entries = _read_entries(memories / "MEMORY.md")
    user_entries = _read_entries(memories / "USER.md")
    return _SemanticView(
        _render_semantic_block("memory", memory_entries),
        _render_semantic_block("user", user_entries),
        len(memory_entries),
        len(user_entries),
        len(_ENTRY_DELIMITER.join(memory_entries)),
        len(_ENTRY_DELIMITER.join(user_entries)),
    )


def _native_episodic(home: Path, probe: HermesEquivalenceProbe) -> tuple[_EpisodicHitView, ...]:
    db_path = home / "state.db"
    if not db_path.exists():
        return ()
    connection = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            """SELECT m.id, m.session_id, m.role, m.content,
                      snippet(messages_fts, 0, '>>>', '<<<', '...', 40) AS snippet,
                      bm25(messages_fts) AS rank_score
               FROM messages_fts
               JOIN messages m ON m.id = messages_fts.rowid
               WHERE messages_fts MATCH ?
               ORDER BY rank_score
               LIMIT ?""",
            (probe.episodic_query, probe.episodic_limit),
        ).fetchall()
        result = []
        for row in rows:
            context_rows = connection.execute(
                """SELECT role, content FROM messages
                   WHERE session_id = ? AND id >= ? - 1 AND id <= ? + 1
                   ORDER BY id""",
                (row["session_id"], row["id"], row["id"]),
            ).fetchall()
            result.append(_EpisodicHitView(
                session_id=str(row["session_id"]),
                role=str(row["role"]),
                content=str(row["content"] or ""),
                snippet=str(row["snippet"] or ""),
                context=tuple(
                    (str(item["role"]), str(item["content"] or "")[:200])
                    for item in context_rows
                ),
            ))
        return tuple(result)
    except sqlite3.OperationalError:
        return ()
    finally:
        connection.close()


def _frontmatter(content: str, fallback: str) -> tuple[str, str]:
    name = fallback
    description = ""
    if not content.startswith("---"):
        return name, description
    parts = content.split("---", 2)
    if len(parts) < 3:
        return name, description
    for line in parts[1].splitlines():
        key, separator, value = line.partition(":")
        if not separator:
            continue
        if key.strip() == "name":
            name = value.strip().strip("\"'") or fallback
        elif key.strip() == "description":
            description = value.strip().strip("\"'")
    return name, description


def _native_procedural(home: Path) -> tuple[_ProceduralSkillView, ...]:
    root = home / "skills"
    views = []
    if not root.exists():
        return ()
    for skill_path in sorted(path for path in root.rglob("SKILL.md") if path.is_file()):
        content = skill_path.read_text(encoding="utf-8")
        name, description = _frontmatter(content, skill_path.parent.name)
        relative = skill_path.parent.relative_to(root)
        category = relative.parts[0] if len(relative.parts) > 1 else None
        resources = tuple(
            (path.relative_to(skill_path.parent).as_posix(), path.read_bytes())
            for path in sorted(skill_path.parent.rglob("*"))
            if path.is_file() and not path.is_symlink() and path.name != "SKILL.md"
        )
        views.append(_ProceduralSkillView(name, description, category, content, resources))
    views.sort(key=lambda item: (item.category or "", item.name))
    return tuple(views)


def _record_native_surface(
    view: _HermesSurfaceView,
    probe: HermesEquivalenceProbe,
    observer: MemoryObserver,
) -> None:
    semantic = view.semantic
    for namespace, count, content_chars in (
        ("memory", semantic.memory_entry_count, semantic.memory_content_chars),
        ("user", semantic.user_entry_count, semantic.user_content_chars),
    ):
        artifact_ids = tuple(
            f"native-semantic:{namespace}:{index}" for index in range(count)
        )
        observer.record(MemoryEvent(
            MemoryEventKind.QUERY,
            MemoryKind.SEMANTIC,
            "hermes-native-semantic",
            query_chars=0,
            attributes={"limit": 100, "namespace": namespace},
        ))
        observer.record(MemoryEvent(
            MemoryEventKind.RETRIEVED,
            MemoryKind.SEMANTIC,
            "hermes-native-semantic",
            artifact_ids=artifact_ids,
            content_chars=content_chars,
            attributes={"count": count},
        ))
        if artifact_ids:
            observer.record(MemoryEvent(
                MemoryEventKind.INJECTED,
                MemoryKind.SEMANTIC,
                "hermes-native-semantic",
                artifact_ids=artifact_ids,
                content_chars=content_chars,
                attributes={"count": count, "surface": "system_prompt"},
            ))

    episodic_ids = tuple(
        f"native-episodic:{index}" for index in range(len(view.episodic))
    )
    episodic_chars = sum(len(item.content) for item in view.episodic)
    observer.record(MemoryEvent(
        MemoryEventKind.QUERY,
        MemoryKind.EPISODIC,
        "hermes-native-episodic",
        query_chars=len(probe.episodic_query),
        attributes={"limit": probe.episodic_limit, "namespace": "default"},
    ))
    observer.record(MemoryEvent(
        MemoryEventKind.RETRIEVED,
        MemoryKind.EPISODIC,
        "hermes-native-episodic",
        artifact_ids=episodic_ids,
        content_chars=episodic_chars,
        attributes={"count": len(episodic_ids)},
    ))
    if episodic_ids:
        observer.record(MemoryEvent(
            MemoryEventKind.INJECTED,
            MemoryKind.EPISODIC,
            "hermes-native-episodic",
            artifact_ids=episodic_ids,
            content_chars=episodic_chars,
            attributes={"count": len(episodic_ids), "surface": "session_search"},
        ))

    procedural_ids = tuple(
        f"native-procedural:{index}" for index in range(len(view.procedural))
    )
    procedural_chars = sum(len(item.content) for item in view.procedural)
    observer.record(MemoryEvent(
        MemoryEventKind.QUERY,
        MemoryKind.PROCEDURAL,
        "hermes-native-procedural",
        query_chars=0,
        attributes={"limit": 100, "namespace": "default"},
    ))
    observer.record(MemoryEvent(
        MemoryEventKind.RETRIEVED,
        MemoryKind.PROCEDURAL,
        "hermes-native-procedural",
        artifact_ids=procedural_ids,
        content_chars=procedural_chars,
        attributes={"count": len(procedural_ids)},
    ))
    if procedural_ids:
        observer.record(MemoryEvent(
            MemoryEventKind.INJECTED,
            MemoryKind.PROCEDURAL,
            "hermes-native-procedural",
            artifact_ids=procedural_ids,
            content_chars=procedural_chars,
            attributes={"count": len(procedural_ids), "surface": "skill_view"},
        ))


def _capture_native(
    home: Path,
    probe: HermesEquivalenceProbe,
    *,
    observer: MemoryObserver | None = None,
) -> _HermesSurfaceView:
    view = _HermesSurfaceView(
        semantic=_native_semantic(home),
        episodic=_native_episodic(home, probe),
        procedural=_native_procedural(home),
    )
    if observer is not None:
        _record_native_surface(view, probe, observer)
    return view


class _MemoryEventCollector:
    def __init__(self, ledger: MemoryLedgerObserver | None = None) -> None:
        self.events: list[MemoryEvent] = []
        self.ledger = ledger

    def record(self, event: MemoryEvent) -> None:
        self.events.append(event)
        if self.ledger is not None:
            self.ledger.record(event)


_HERMES_NATIVE_BINDING_LOCK = threading.RLock()


@contextmanager
def _bound_hermes_memory_dir(home: Path):
    """Temporarily point Hermes' import-time memory path at an isolated home."""

    from tools import memory_tool

    with _HERMES_NATIVE_BINDING_LOCK:
        previous = memory_tool.MEMORY_DIR
        memory_tool.MEMORY_DIR = home / "memories"
        try:
            yield memory_tool
        finally:
            memory_tool.MEMORY_DIR = previous


@contextmanager
def _fixed_hermes_clock():
    """Keep the real Hermes prompt builder deterministic for matched fixtures."""

    import hermes_time

    with _HERMES_NATIVE_BINDING_LOCK:
        previous = hermes_time.now
        hermes_time.now = lambda: datetime(2026, 8, 26, 12, 0, 0)
        try:
            yield
        finally:
            hermes_time.now = previous


def _build_real_hermes_system_prompt(store: object) -> str:
    """Invoke Hermes' real prompt assembly without constructing a model client."""

    from run_agent import AIAgent

    agent = SimpleNamespace(
        skip_context_files=True,
        _honcho_config=None,
        valid_tool_names={"memory"},
        _honcho=None,
        _honcho_session_key=None,
        _memory_store=store,
        _memory_enabled=True,
        _user_profile_enabled=True,
        pass_session_id=False,
        session_id="execution-equivalence-session",
        model="fixture-model",
        provider="fixture-provider",
        platform="",
    )
    with _fixed_hermes_clock():
        return AIAgent._build_system_prompt(agent, "Matched fixture system message")


def _record_native_prompt_memory(
    store: object,
    observer: MemoryObserver,
) -> None:
    snapshots = (
        ("memory", tuple(store.memory_entries)),
        ("user", tuple(store.user_entries)),
    )
    for namespace, entries in snapshots:
        artifact_ids = tuple(
            f"native-semantic:{namespace}:{index}" for index in range(len(entries))
        )
        content_chars = sum(len(entry) for entry in entries)
        observer.record(MemoryEvent(
            MemoryEventKind.QUERY,
            MemoryKind.SEMANTIC,
            "hermes-native-semantic",
            query_chars=0,
            attributes={"limit": 100, "namespace": namespace},
        ))
        observer.record(MemoryEvent(
            MemoryEventKind.RETRIEVED,
            MemoryKind.SEMANTIC,
            "hermes-native-semantic",
            artifact_ids=artifact_ids,
            content_chars=content_chars,
            attributes={"count": len(entries)},
        ))
    for namespace, entries in snapshots:
        if entries:
            artifact_ids = tuple(
                f"native-semantic:{namespace}:{index}"
                for index in range(len(entries))
            )
            observer.record(MemoryEvent(
                MemoryEventKind.INJECTED,
                MemoryKind.SEMANTIC,
                "hermes-native-semantic",
                artifact_ids=artifact_ids,
                content_chars=sum(len(entry) for entry in entries),
                attributes={"count": len(entries), "surface": "system_prompt"},
            ))


def _capture_native_system_prompt(
    home: Path,
    *,
    observer: MemoryObserver | None = None,
) -> str:
    with _bound_hermes_memory_dir(home) as memory_tool:
        store = memory_tool.MemoryStore(
            memory_char_limit=_SEMANTIC_LIMITS["memory"],
            user_char_limit=_SEMANTIC_LIMITS["user"],
        )
        store.load_from_disk()
    prompt = _build_real_hermes_system_prompt(store)
    if observer is not None:
        _record_native_prompt_memory(store, observer)
    return prompt


class _AdapterPromptMemoryStore:
    """Hermes prompt-store surface backed by one frozen typed-runtime read."""

    def __init__(self, runtime: MemoryRuntime) -> None:
        self.runtime = runtime
        self._hits = {
            namespace: runtime.query(MemoryQuery(
                MemoryKind.SEMANTIC,
                "",
                namespace=namespace,
                limit=100,
            ))
            for namespace in ("memory", "user")
        }
        self._injected: set[str] = set()

    def format_for_system_prompt(self, target: str) -> str | None:
        hits = self._hits.get(target, ())
        if not hits:
            return None
        if target not in self._injected:
            self.runtime.mark_injected(hits, surface="system_prompt")
            self._injected.add(target)
        return _render_semantic_block(
            target,
            tuple(hit.artifact.content for hit in hits),
        )


def _capture_adapter_system_prompt(
    home: Path,
    collector: _MemoryEventCollector,
) -> str:
    runtime = build_configured_hermes_runtime(
        home,
        HermesExperimentConfig(HermesExecutionMode.ADAPTER_LEDGER),
        observers=(collector,),
    )
    try:
        return _build_real_hermes_system_prompt(_AdapterPromptMemoryStore(runtime))
    finally:
        runtime.close()


@contextmanager
def _deterministic_session_summarizer():
    """Replace only the external LLM call while preserving Hermes search logic."""

    from tools import session_search_tool

    async def summarize(conversation_text: str, query: str, session_meta: dict) -> str:
        return f"Deterministic summary for {query}:\n{conversation_text}"

    with _HERMES_NATIVE_BINDING_LOCK:
        previous = session_search_tool._summarize_session
        session_search_tool._summarize_session = summarize
        try:
            yield
        finally:
            session_search_tool._summarize_session = previous


class _ObservedNativeSessionDb:
    """Observer-only wrapper around Hermes' unmodified SessionDB results."""

    def __init__(self, db: object, observer: MemoryObserver) -> None:
        self.db = db
        self.observer = observer
        self._hits_by_session: dict[str, tuple[dict[str, Any], ...]] = {}
        self._injected_sessions: set[str] = set()

    def search_messages(self, *, query: str, **kwargs) -> list[dict[str, Any]]:
        self.observer.record(MemoryEvent(
            MemoryEventKind.QUERY,
            MemoryKind.EPISODIC,
            "hermes-native-episodic",
            query_chars=len(query),
            attributes={"limit": kwargs.get("limit", 50), "namespace": "default"},
        ))
        results = self.db.search_messages(query=query, **kwargs)
        for result in results:
            self._hits_by_session.setdefault(result["session_id"], tuple())
            self._hits_by_session[result["session_id"]] += (result,)
        artifact_ids = tuple(
            f"native-episodic:message:{result['id']}" for result in results
        )
        self.observer.record(MemoryEvent(
            MemoryEventKind.RETRIEVED,
            MemoryKind.EPISODIC,
            "hermes-native-episodic",
            artifact_ids=artifact_ids,
            content_chars=sum(len(str(result.get("content") or "")) for result in results),
            attributes={"count": len(results)},
        ))
        return results

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        return self.db.get_session(session_id)

    def get_messages_as_conversation(self, session_id: str) -> list[dict[str, Any]]:
        messages = self.db.get_messages_as_conversation(session_id)
        if session_id not in self._injected_sessions:
            hits = self._hits_by_session.get(session_id, ())
            self.observer.record(MemoryEvent(
                MemoryEventKind.INJECTED,
                MemoryKind.EPISODIC,
                "hermes-native-episodic",
                artifact_ids=tuple(
                    f"native-episodic:message:{item['id']}" for item in hits
                ),
                content_chars=sum(
                    len(str(message.get("content") or "")) for message in messages
                ),
                attributes={"count": len(hits), "surface": "session_search"},
            ))
            self._injected_sessions.add(session_id)
        return messages


class _AdapterSessionDb:
    """Subset of SessionDB consumed by Hermes session_search, backed by runtime hits."""

    def __init__(self, runtime: MemoryRuntime) -> None:
        self.runtime = runtime
        self._hits_by_session: dict[str, tuple[Any, ...]] = {}
        self._injected_sessions: set[str] = set()

    def search_messages(
        self,
        *,
        query: str,
        role_filter: list[str] | None = None,
        exclude_sources: list[str] | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        if offset:
            return []
        hits = self.runtime.query(MemoryQuery(
            MemoryKind.EPISODIC,
            query,
            limit=limit,
        ))
        results = []
        for hit in hits:
            metadata = hit.artifact.metadata
            role = str(metadata.get("role") or "")
            source = str(metadata.get("source") or "")
            if role_filter and role not in role_filter:
                continue
            if exclude_sources and source in exclude_sources:
                continue
            self._hits_by_session.setdefault(hit.artifact.namespace, tuple())
            self._hits_by_session[hit.artifact.namespace] += (hit,)
            results.append({
                "id": metadata.get("message_id"),
                "session_id": hit.artifact.namespace,
                "role": role,
                "snippet": metadata.get("snippet"),
                "content": hit.artifact.content,
                "timestamp": metadata.get("timestamp"),
                "tool_name": None,
                "source": source,
                "model": metadata.get("model"),
                "session_started": metadata.get("session_started"),
            })
        return results

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        hits = self._hits_by_session.get(session_id, ())
        if not hits:
            return None
        metadata = hits[0].artifact.metadata
        return {
            "id": session_id,
            "source": metadata.get("source"),
            "model": metadata.get("model"),
            "started_at": metadata.get("session_started"),
            "title": metadata.get("session_title"),
            "parent_session_id": metadata.get("parent_session_id"),
        }

    def get_messages_as_conversation(self, session_id: str) -> list[dict[str, Any]]:
        hits = self._hits_by_session.get(session_id, ())
        by_id: dict[int, tuple[str, str]] = {}
        for hit in hits:
            for message_id, role, content in (
                hit.artifact.metadata.get("context_messages") or ()
            ):
                by_id[int(message_id)] = (str(role), str(content))
        ordered = [
            {"role": role, "content": content}
            for _, (role, content) in sorted(by_id.items())
        ]
        if session_id not in self._injected_sessions:
            self.runtime.mark_injected(hits, surface="session_search")
            self._injected_sessions.add(session_id)
        return ordered


def _dispatch_real_session_search(
    db: object,
    probe: HermesEquivalenceProbe,
) -> str:
    from tools import session_search_tool  # noqa: F401 - registers the real tool
    from tools.registry import registry

    with _deterministic_session_summarizer():
        return registry.dispatch(
            "session_search",
            {"query": probe.episodic_query, "limit": probe.episodic_limit},
            db=db,
            current_session_id="execution-equivalence-current",
        )


def _capture_native_session_search(
    home: Path,
    probe: HermesEquivalenceProbe,
    *,
    observer: MemoryObserver | None = None,
) -> str:
    from hermes_state import SessionDB

    db = SessionDB(home / "state.db")
    try:
        execution_db = (
            _ObservedNativeSessionDb(db, observer) if observer is not None else db
        )
        return _dispatch_real_session_search(execution_db, probe)
    finally:
        db.close()


def _capture_adapter_session_search(
    home: Path,
    probe: HermesEquivalenceProbe,
    collector: _MemoryEventCollector,
) -> str:
    runtime = build_configured_hermes_runtime(
        home,
        HermesExperimentConfig(HermesExecutionMode.ADAPTER_LEDGER),
        observers=(collector,),
    )
    try:
        return _dispatch_real_session_search(_AdapterSessionDb(runtime), probe)
    finally:
        runtime.close()


@contextmanager
def _bound_hermes_skills_dir(skills_dir: Path):
    """Bind Hermes' import-time skill globals to an isolated directory."""

    from tools import skills_tool

    with _HERMES_NATIVE_BINDING_LOCK:
        previous_home = skills_tool.HERMES_HOME
        previous_skills = skills_tool.SKILLS_DIR
        previous_env = os.environ.get("HERMES_HOME")
        skills_tool.HERMES_HOME = skills_dir.parent
        skills_tool.SKILLS_DIR = skills_dir
        os.environ["HERMES_HOME"] = str(skills_dir.parent)
        try:
            yield
        finally:
            skills_tool.HERMES_HOME = previous_home
            skills_tool.SKILLS_DIR = previous_skills
            if previous_env is None:
                os.environ.pop("HERMES_HOME", None)
            else:
                os.environ["HERMES_HOME"] = previous_env


def _dispatch_real_skills(probe: HermesEquivalenceProbe) -> dict[HermesExecutionSurface, str]:
    if probe.procedural_skill_name is None:
        raise ValueError("execution equivalence requires procedural_skill_name")
    from tools import skills_tool  # noqa: F401 - registers the real tools
    from tools.registry import registry

    skills_list_result = registry.dispatch("skills_list", {})
    skill_view_result = registry.dispatch(
        "skill_view",
        {"name": probe.procedural_skill_name},
    )
    if probe.procedural_resource_path is not None:
        resource_result = registry.dispatch(
            "skill_view",
            {
                "name": probe.procedural_skill_name,
                "file_path": probe.procedural_resource_path,
            },
        )
        skill_view_result = f"{skill_view_result}\n{resource_result}"
    return {
        HermesExecutionSurface.SKILLS_LIST: skills_list_result,
        HermesExecutionSurface.SKILL_VIEW: skill_view_result,
    }


def _record_native_skills(
    home: Path,
    probe: HermesEquivalenceProbe,
    observer: MemoryObserver,
) -> None:
    skills = _native_procedural(home)
    all_ids = tuple(
        f"native-procedural:{index}" for index in range(len(skills))
    )
    observer.record(MemoryEvent(
        MemoryEventKind.QUERY,
        MemoryKind.PROCEDURAL,
        "hermes-native-procedural",
        query_chars=0,
        attributes={"limit": 100, "namespace": "default"},
    ))
    observer.record(MemoryEvent(
        MemoryEventKind.RETRIEVED,
        MemoryKind.PROCEDURAL,
        "hermes-native-procedural",
        artifact_ids=all_ids,
        content_chars=sum(len(skill.content) for skill in skills),
        attributes={"count": len(skills)},
    ))
    if skills:
        observer.record(MemoryEvent(
            MemoryEventKind.INJECTED,
            MemoryKind.PROCEDURAL,
            "hermes-native-procedural",
            artifact_ids=all_ids,
            content_chars=sum(len(skill.content) for skill in skills),
            attributes={"count": len(skills), "surface": "skills_list"},
        ))

    selected = tuple(
        (index, skill)
        for index, skill in enumerate(skills)
        if skill.name == probe.procedural_skill_name
    )
    observer.record(MemoryEvent(
        MemoryEventKind.QUERY,
        MemoryKind.PROCEDURAL,
        "hermes-native-procedural",
        query_chars=len(probe.procedural_skill_name or ""),
        attributes={"limit": 5, "namespace": "default"},
    ))
    observer.record(MemoryEvent(
        MemoryEventKind.RETRIEVED,
        MemoryKind.PROCEDURAL,
        "hermes-native-procedural",
        artifact_ids=tuple(f"native-procedural:{index}" for index, _ in selected),
        content_chars=sum(len(skill.content) for _, skill in selected),
        attributes={"count": len(selected)},
    ))
    if selected:
        observer.record(MemoryEvent(
            MemoryEventKind.INJECTED,
            MemoryKind.PROCEDURAL,
            "hermes-native-procedural",
            artifact_ids=tuple(
                f"native-procedural:{index}" for index, _ in selected
            ),
            content_chars=sum(len(skill.content) for _, skill in selected),
            attributes={"count": len(selected), "surface": "skill_view"},
        ))


def _capture_native_skills(
    home: Path,
    probe: HermesEquivalenceProbe,
    *,
    observer: MemoryObserver | None = None,
) -> dict[HermesExecutionSurface, str]:
    with _bound_hermes_skills_dir(home / "skills"):
        result = _dispatch_real_skills(probe)
    if observer is not None:
        _record_native_skills(home, probe, observer)
    return result


def _materialize_procedural_hits(skills_dir: Path, hits: Iterable[Any]) -> None:
    for hit in hits:
        relative = PurePosixPath(str(
            hit.artifact.metadata.get("relative_path") or hit.artifact.title or ""
        ))
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise ValueError("procedural artifact has an unsafe relative path")
        skill_dir = skills_dir.joinpath(*relative.parts)
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(
            hit.artifact.content,
            encoding="utf-8",
        )
        for resource in hit.artifact.resources:
            resource_path = skill_dir.joinpath(*PurePosixPath(resource.path).parts)
            resource_path.parent.mkdir(parents=True, exist_ok=True)
            resource_path.write_bytes(resource.content)


def _capture_adapter_skills(
    home: Path,
    probe: HermesEquivalenceProbe,
    collector: _MemoryEventCollector,
) -> dict[HermesExecutionSurface, str]:
    if probe.procedural_skill_name is None:
        raise ValueError("execution equivalence requires procedural_skill_name")
    runtime = build_configured_hermes_runtime(
        home,
        HermesExperimentConfig(HermesExecutionMode.ADAPTER_LEDGER),
        observers=(collector,),
    )
    try:
        list_hits = runtime.query(MemoryQuery(
            MemoryKind.PROCEDURAL,
            "",
            limit=100,
        ))
        with TemporaryDirectory(prefix="rsimem-hermes-skills-") as directory:
            skills_dir = Path(directory) / "skills"
            _materialize_procedural_hits(skills_dir, list_hits)
            with _bound_hermes_skills_dir(skills_dir):
                from tools import skills_tool  # noqa: F401 - registers real tools
                from tools.registry import registry

                skills_list_result = registry.dispatch("skills_list", {})
                runtime.mark_injected(list_hits, surface="skills_list")
                view_hits = runtime.query(MemoryQuery(
                    MemoryKind.PROCEDURAL,
                    probe.procedural_skill_name,
                    limit=5,
                ))
                skill_view_result = registry.dispatch(
                    "skill_view",
                    {"name": probe.procedural_skill_name},
                )
                if probe.procedural_resource_path is not None:
                    resource_result = registry.dispatch(
                        "skill_view",
                        {
                            "name": probe.procedural_skill_name,
                            "file_path": probe.procedural_resource_path,
                        },
                    )
                    skill_view_result = f"{skill_view_result}\n{resource_result}"
                runtime.mark_injected(view_hits, surface="skill_view")
        return {
            HermesExecutionSurface.SKILLS_LIST: skills_list_result,
            HermesExecutionSurface.SKILL_VIEW: skill_view_result,
        }
    finally:
        runtime.close()


def _capture_adapter(
    home: Path,
    probe: HermesEquivalenceProbe,
    collector: _MemoryEventCollector,
) -> _HermesSurfaceView:
    config = HermesExperimentConfig(HermesExecutionMode.ADAPTER_LEDGER)
    runtime = build_configured_hermes_runtime(home, config, observers=(collector,))
    try:
        semantic_hits: dict[str, tuple[Any, ...]] = {}
        for namespace in ("memory", "user"):
            hits = runtime.query(MemoryQuery(MemoryKind.SEMANTIC, "", namespace=namespace, limit=100))
            runtime.mark_injected(hits, surface="system_prompt")
            semantic_hits[namespace] = hits
        semantic = _SemanticView(
            _render_semantic_block(
                "memory", tuple(hit.artifact.content for hit in semantic_hits["memory"])
            ),
            _render_semantic_block(
                "user", tuple(hit.artifact.content for hit in semantic_hits["user"])
            ),
            len(semantic_hits["memory"]),
            len(semantic_hits["user"]),
            len(_ENTRY_DELIMITER.join(
                hit.artifact.content for hit in semantic_hits["memory"]
            )),
            len(_ENTRY_DELIMITER.join(
                hit.artifact.content for hit in semantic_hits["user"]
            )),
        )

        episodic_hits = runtime.query(MemoryQuery(
            MemoryKind.EPISODIC,
            probe.episodic_query,
            limit=probe.episodic_limit,
        ))
        runtime.mark_injected(episodic_hits, surface="session_search")
        episodic = tuple(
            _EpisodicHitView(
                session_id=hit.artifact.namespace,
                role=str(hit.artifact.metadata.get("role") or ""),
                content=hit.artifact.content,
                snippet=str(hit.artifact.metadata.get("snippet") or ""),
                context=tuple(hit.artifact.metadata.get("context") or ()),
            )
            for hit in episodic_hits
        )

        procedural_hits = runtime.query(MemoryQuery(
            MemoryKind.PROCEDURAL, "", limit=100
        ))
        runtime.mark_injected(procedural_hits, surface="skill_view")
        procedural = tuple(
            _ProceduralSkillView(
                name=hit.artifact.title or "",
                description=str(hit.artifact.metadata.get("description") or ""),
                category=(
                    None if hit.artifact.namespace == "default" else hit.artifact.namespace
                ),
                content=hit.artifact.content,
                resources=tuple(
                    (resource.path, resource.content) for resource in hit.artifact.resources
                ),
            )
            for hit in procedural_hits
        )
        return _HermesSurfaceView(semantic, episodic, procedural)
    finally:
        runtime.close()


def _surface_checks(
    native: _HermesSurfaceView,
    candidate: _HermesSurfaceView,
) -> tuple[HermesSurfaceCheck, ...]:
    native_semantic = native.semantic.memory_block + native.semantic.user_block
    candidate_semantic = candidate.semantic.memory_block + candidate.semantic.user_block
    native_episode_chars = sum(len(item.content) + len(item.snippet) for item in native.episodic)
    candidate_episode_chars = sum(
        len(item.content) + len(item.snippet) for item in candidate.episodic
    )
    native_skill_chars = sum(len(item.content) for item in native.procedural)
    candidate_skill_chars = sum(len(item.content) for item in candidate.procedural)
    native_resource_bytes = sum(
        len(content) for item in native.procedural for _, content in item.resources
    )
    candidate_resource_bytes = sum(
        len(content) for item in candidate.procedural for _, content in item.resources
    )
    return (
        HermesSurfaceCheck(
            MemoryKind.SEMANTIC,
            native.semantic == candidate.semantic,
            native.semantic.memory_entry_count + native.semantic.user_entry_count,
            candidate.semantic.memory_entry_count + candidate.semantic.user_entry_count,
            len(native_semantic),
            len(candidate_semantic),
        ),
        HermesSurfaceCheck(
            MemoryKind.EPISODIC,
            native.episodic == candidate.episodic,
            len(native.episodic),
            len(candidate.episodic),
            native_episode_chars,
            candidate_episode_chars,
        ),
        HermesSurfaceCheck(
            MemoryKind.PROCEDURAL,
            native.procedural == candidate.procedural,
            len(native.procedural),
            len(candidate.procedural),
            native_skill_chars,
            candidate_skill_chars,
            native_resource_bytes,
            candidate_resource_bytes,
        ),
    )


def run_hermes_equivalence_variants(
    hermes_home: Path,
    probe: HermesEquivalenceProbe,
) -> HermesEquivalenceReport:
    """Compare native, instrumented native, and adapter reads on one home."""

    home = hermes_home.expanduser().resolve()
    baseline = _capture_native(home, probe)
    variants = []
    for mode in HermesExecutionMode:
        config = HermesExperimentConfig(mode)
        ledger = (
            MemoryLedgerObserver(
                run_id="storage-boundary-fixture",
                variant=mode.value,
                trace_id="storage-boundary-trace",
                episode_id="storage-boundary-episode",
                session_id="storage-boundary-session",
                task_id="storage-boundary-task",
                family_id="storage-boundary",
                stage="fixture",
            )
            if config.ledger_enabled
            else None
        )
        collector = _MemoryEventCollector(ledger)
        if mode == HermesExecutionMode.ADAPTER_LEDGER:
            candidate = _capture_adapter(home, probe, collector)
        elif mode == HermesExecutionMode.NATIVE_LEDGER:
            candidate = _capture_native(home, probe, observer=collector)
        else:
            candidate = _capture_native(home, probe)
        checks = _surface_checks(baseline, candidate)
        variants.append(HermesVariantResult(
            mode=mode,
            equivalent_to_native=all(check.equivalent for check in checks),
            checks=checks,
            ledger_enabled=config.ledger_enabled,
            memory_event_count=len(collector.events),
            memory_event_kinds=tuple(event.kind for event in collector.events),
            ledger_event_count=len(ledger.events) if ledger is not None else 0,
            ledger_event_kinds=(
                tuple(event["kind"] for event in ledger.events)
                if ledger is not None
                else ()
            ),
        ))
    return HermesEquivalenceReport(tuple(variants))


def run_hermes_execution_equivalence_variants(
    hermes_home: Path,
    probe: HermesEquivalenceProbe,
) -> HermesExecutionEquivalenceReport:
    """Compare final model-visible output from real Hermes execution surfaces."""

    home = hermes_home.expanduser().resolve()
    baseline = {
        HermesExecutionSurface.SYSTEM_PROMPT: _capture_native_system_prompt(home),
        HermesExecutionSurface.SESSION_SEARCH: _capture_native_session_search(home, probe),
        **_capture_native_skills(home, probe),
    }
    variants = []
    for mode in HermesExecutionMode:
        config = HermesExperimentConfig(mode)
        ledger = (
            MemoryLedgerObserver(
                run_id="execution-surface-fixture",
                variant=mode.value,
                trace_id="execution-surface-trace",
                episode_id="execution-surface-episode",
                session_id="execution-surface-session",
                task_id="execution-surface-task",
                family_id="execution-surface",
                stage="fixture",
            )
            if config.ledger_enabled
            else None
        )
        collector = _MemoryEventCollector(ledger)
        if mode == HermesExecutionMode.ADAPTER_LEDGER:
            candidate = {
                HermesExecutionSurface.SYSTEM_PROMPT: _capture_adapter_system_prompt(
                    home, collector
                ),
                HermesExecutionSurface.SESSION_SEARCH: _capture_adapter_session_search(
                    home, probe, collector
                ),
                **_capture_adapter_skills(home, probe, collector),
            }
        elif mode == HermesExecutionMode.NATIVE_LEDGER:
            candidate = {
                HermesExecutionSurface.SYSTEM_PROMPT: _capture_native_system_prompt(
                    home, observer=collector
                ),
                HermesExecutionSurface.SESSION_SEARCH: _capture_native_session_search(
                    home, probe, observer=collector
                ),
                **_capture_native_skills(home, probe, observer=collector),
            }
        else:
            candidate = {
                HermesExecutionSurface.SYSTEM_PROMPT: _capture_native_system_prompt(home),
                HermesExecutionSurface.SESSION_SEARCH: _capture_native_session_search(
                    home, probe
                ),
                **_capture_native_skills(home, probe),
            }
        checks = tuple(
            HermesExecutionSurfaceCheck(
                surface=surface,
                equivalent=baseline[surface] == candidate[surface],
                native_content_chars=len(baseline[surface]),
                candidate_content_chars=len(candidate[surface]),
            )
            for surface in HermesExecutionSurface
        )
        variants.append(HermesExecutionVariantResult(
            mode=mode,
            equivalent_to_native=all(check.equivalent for check in checks),
            checks=checks,
            ledger_enabled=config.ledger_enabled,
            memory_event_count=len(collector.events),
            memory_event_kinds=tuple(event.kind for event in collector.events),
            ledger_event_count=len(ledger.events) if ledger is not None else 0,
            ledger_event_kinds=(
                tuple(event["kind"] for event in ledger.events)
                if ledger is not None
                else ()
            ),
        ))
    return HermesExecutionEquivalenceReport(tuple(variants))
