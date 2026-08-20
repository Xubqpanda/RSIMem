from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from rsimem.memory import (
    HermesEpisodicBackend,
    HermesProceduralBackend,
    HermesSemanticBackend,
    MemoryArtifact,
    MemoryEvent,
    MemoryEventKind,
    MemoryKind,
    MemoryMutation,
    MemoryMutationAction,
    MemoryQuery,
    MemoryResource,
    MemoryRuntime,
    build_hermes_native_registry,
)


def _artifact(
    artifact_id: str,
    kind: MemoryKind,
    content: str,
    **kwargs,
) -> MemoryArtifact:
    return MemoryArtifact(artifact_id, kind, content, **kwargs)


def _state_db(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            model TEXT
        );
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT,
            timestamp REAL NOT NULL,
            tool_name TEXT
        );
        CREATE VIRTUAL TABLE messages_fts USING fts5(
            content,
            content=messages,
            content_rowid=id
        );
        CREATE TRIGGER messages_fts_insert AFTER INSERT ON messages BEGIN
            INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
        END;
        """
    )
    connection.execute(
        "INSERT INTO sessions(id, source, model) VALUES (?, ?, ?)",
        ("session-1", "cli", "gpt-test"),
    )
    connection.executemany(
        "INSERT INTO messages(session_id, role, content, timestamp, tool_name) VALUES (?, ?, ?, ?, ?)",
        [
            ("session-1", "user", "Deploy the billing service.", 1.0, None),
            ("session-1", "assistant", "Deployment completed successfully.", 2.0, None),
        ],
    )
    connection.commit()
    connection.close()


def test_semantic_backend_crud_and_native_limits(tmp_path: Path) -> None:
    memories = tmp_path / "memories"
    memories.mkdir()
    (memories / "MEMORY.md").write_text("Use Python 3.11.\n§\nDeploy on Friday.", encoding="utf-8")
    backend = HermesSemanticBackend(memories)

    hits = backend.query(MemoryQuery(MemoryKind.SEMANTIC, "Python", namespace="default"))
    assert len(hits) == 1
    original = hits[0].artifact
    assert backend.get(original.artifact_id) == original

    update = _artifact(
        original.artifact_id,
        MemoryKind.SEMANTIC,
        "Use Python 3.11 with uv.",
        namespace="memory",
    )
    result = backend.mutate(MemoryMutation(
        MemoryMutationAction.UPDATE,
        MemoryKind.SEMANTIC,
        artifact=update,
        expected_revision=original.revision,
    ))
    assert result.accepted is True
    assert "with uv" in (memories / "MEMORY.md").read_text(encoding="utf-8")

    stale = backend.mutate(MemoryMutation(
        MemoryMutationAction.DELETE,
        MemoryKind.SEMANTIC,
        artifact_id=result.artifact_id,
        expected_revision="stale",
    ))
    assert stale.reason_code == "revision_conflict"

    added = backend.mutate(MemoryMutation(
        MemoryMutationAction.ADD,
        MemoryKind.SEMANTIC,
        artifact=_artifact("new", MemoryKind.SEMANTIC, "Owner is Alice."),
    ))
    assert added.accepted is True
    deleted = backend.mutate(MemoryMutation(
        MemoryMutationAction.DELETE,
        MemoryKind.SEMANTIC,
        artifact_id=added.artifact_id,
    ))
    assert deleted.accepted is True
    assert "Owner is Alice" not in (memories / "MEMORY.md").read_text(encoding="utf-8")

    too_large = backend.mutate(MemoryMutation(
        MemoryMutationAction.ADD,
        MemoryKind.SEMANTIC,
        artifact=_artifact("large", MemoryKind.SEMANTIC, "x" * 2200),
    ))
    assert too_large.reason_code == "character_limit_exceeded"


def test_episodic_backend_searches_fts_and_is_read_only(tmp_path: Path) -> None:
    state_db = tmp_path / "state.db"
    _state_db(state_db)
    backend = HermesEpisodicBackend(state_db)

    hits = backend.query(MemoryQuery(MemoryKind.EPISODIC, "deployment"))
    assert len(hits) == 1
    assert hits[0].artifact.namespace == "session-1"
    stored = backend.get(hits[0].artifact.artifact_id)
    assert stored is not None
    assert stored.artifact_id == hits[0].artifact.artifact_id
    assert stored.content == hits[0].artifact.content
    assert backend.query(MemoryQuery(MemoryKind.EPISODIC, '"unterminated')) == ()

    rejected = backend.mutate(MemoryMutation(
        MemoryMutationAction.DELETE,
        MemoryKind.EPISODIC,
        artifact_id=hits[0].artifact.artifact_id,
    ))
    assert rejected.accepted is False
    assert rejected.reason_code == "read_only_backend"


def test_procedural_backend_crud_replaces_resource_set(tmp_path: Path) -> None:
    skills = tmp_path / "skills"
    backend = HermesProceduralBackend(skills)
    content = "---\nname: deploy-service\ndescription: Deploy a service\n---\nRun preflight checks."
    artifact = _artifact(
        "candidate",
        MemoryKind.PROCEDURAL,
        content,
        title="deploy-service",
        resources=(MemoryResource("scripts/check.sh", b"pytest -q\n"),),
    )

    created = backend.mutate(MemoryMutation(
        MemoryMutationAction.ADD,
        MemoryKind.PROCEDURAL,
        artifact=artifact,
    ))
    assert created.accepted is True
    hit = backend.query(MemoryQuery(MemoryKind.PROCEDURAL, "preflight"))[0]
    assert hit.artifact.resources[0].path == "scripts/check.sh"

    updated_artifact = _artifact(
        hit.artifact.artifact_id,
        MemoryKind.PROCEDURAL,
        content.replace("preflight", "release"),
        title="deploy-service",
        resources=(MemoryResource("references/runbook.md", b"Rollback safely.\n"),),
    )
    updated = backend.mutate(MemoryMutation(
        MemoryMutationAction.UPDATE,
        MemoryKind.PROCEDURAL,
        artifact=updated_artifact,
        expected_revision=hit.artifact.revision,
    ))
    assert updated.accepted is True
    current = backend.get(created.artifact_id or "")
    assert current is not None
    assert [resource.path for resource in current.resources] == ["references/runbook.md"]
    assert not (skills / "deploy-service" / "scripts").exists()

    deleted = backend.mutate(MemoryMutation(
        MemoryMutationAction.DELETE,
        MemoryKind.PROCEDURAL,
        artifact_id=current.artifact_id,
        expected_revision=current.revision,
    ))
    assert deleted.accepted is True
    assert backend.get(current.artifact_id) is None


class _Observer:
    def __init__(self) -> None:
        self.events: list[MemoryEvent] = []

    def record(self, event: MemoryEvent) -> None:
        self.events.append(event)


def test_registry_runtime_routes_all_kinds_without_content_in_events(tmp_path: Path) -> None:
    secret_memory = "PRIVATE_MEMORY_SENTINEL"
    memories = tmp_path / "memories"
    memories.mkdir()
    (memories / "MEMORY.md").write_text(secret_memory, encoding="utf-8")
    _state_db(tmp_path / "state.db")
    observer = _Observer()
    runtime = MemoryRuntime(build_hermes_native_registry(tmp_path), observers=(observer,))

    hits = runtime.query(MemoryQuery(MemoryKind.SEMANTIC, "PRIVATE_MEMORY"))
    runtime.mark_injected(hits, surface="system_prompt")
    rejected = runtime.mutate(MemoryMutation(
        MemoryMutationAction.DELETE,
        MemoryKind.EPISODIC,
        artifact_id="hermes-episodic:message:1",
    ))

    assert rejected.reason_code == "operation_not_supported"
    assert [event.kind for event in observer.events] == [
        MemoryEventKind.QUERY,
        MemoryEventKind.RETRIEVED,
        MemoryEventKind.INJECTED,
        MemoryEventKind.MUTATION_REQUESTED,
        MemoryEventKind.MUTATION_REJECTED,
    ]
    serialized = json.dumps([{
        "kind": event.kind,
        "memory_kind": event.memory_kind,
        "backend": event.backend,
        "artifact_ids": event.artifact_ids,
        "query_chars": event.query_chars,
        "content_chars": event.content_chars,
        "reason_code": event.reason_code,
        "attributes": dict(event.attributes),
    } for event in observer.events])
    assert secret_memory not in serialized
    assert "PRIVATE_MEMORY" not in serialized
