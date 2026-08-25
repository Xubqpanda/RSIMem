"""Opt-in Hermes runtime construction and deterministic surface equivalence."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
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

    def __post_init__(self) -> None:
        if not self.episodic_query.strip():
            raise ValueError("equivalence probe requires an episodic query")
        if self.episodic_limit < 1:
            raise ValueError("episodic_limit must be positive")


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
