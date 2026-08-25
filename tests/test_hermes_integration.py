from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from pathlib import Path

import pytest

from rsimem.hermes_integration import (
    HermesEquivalenceProbe,
    HermesExecutionMode,
    HermesExperimentConfig,
    build_configured_hermes_runtime,
    run_hermes_equivalence_variants,
)
from rsimem.ledger import LifecycleLedgerObserver
from rsimem.lifecycle import run_sm01_preference_fixture
from rsimem.memory import MemoryKind


PRIVATE_PREFERENCE = "Use TSV with owner, priority, task, and due_date."


def _hermes_home(path: Path) -> Path:
    memories = path / "memories"
    memories.mkdir(parents=True)
    (memories / "MEMORY.md").write_text(
        f"{PRIVATE_PREFERENCE}\n§\nAlways include a header row.",
        encoding="utf-8",
    )
    (memories / "USER.md").write_text(
        "The user prefers concise status updates.",
        encoding="utf-8",
    )

    skill = path / "skills" / "operations" / "task-table"
    (skill / "references").mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: task-table\ndescription: Format a task table\n---\nUse the requested columns.",
        encoding="utf-8",
    )
    (skill / "references" / "columns.md").write_text(
        "owner, priority, task, due_date\n",
        encoding="utf-8",
    )

    connection = sqlite3.connect(path / "state.db")
    connection.executescript(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            model TEXT,
            started_at REAL
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
        "INSERT INTO sessions(id, source, model, started_at) VALUES (?, ?, ?, ?)",
        ("session-1", "cli", "fixture-model", 0.0),
    )
    connection.executemany(
        "INSERT INTO messages(session_id, role, content, timestamp, tool_name) VALUES (?, ?, ?, ?, ?)",
        [
            ("session-1", "user", "Please format the project tasks.", 1.0, None),
            ("session-1", "assistant", "I used the requested task table.", 2.0, None),
            ("session-1", "user", "The task table looks correct.", 3.0, None),
        ],
    )
    connection.commit()
    connection.close()
    return path


def test_config_defaults_to_direct_native_and_requires_three_routes(tmp_path: Path) -> None:
    config = HermesExperimentConfig()
    assert config.mode == HermesExecutionMode.NATIVE
    assert config.uses_adapter is False
    assert config.ledger_enabled is False
    assert set(config.routes) == set(MemoryKind)

    with pytest.raises(ValueError, match="one route per memory kind"):
        HermesExperimentConfig(routes={MemoryKind.SEMANTIC: "semantic"})

    unsupported = HermesExperimentConfig(routes={
        MemoryKind.SEMANTIC: "external-semantic",
        MemoryKind.EPISODIC: "hermes-native-episodic",
        MemoryKind.PROCEDURAL: "hermes-native-procedural",
    })
    with pytest.raises(ValueError, match="unregistered Hermes backend routes"):
        build_configured_hermes_runtime(tmp_path, unsupported)


def test_native_ledger_and_adapter_views_are_equivalent(tmp_path: Path) -> None:
    home = _hermes_home(tmp_path)
    report = run_hermes_equivalence_variants(
        home,
        HermesEquivalenceProbe(episodic_query="task table"),
    )

    assert report.equivalent is True
    assert [variant.mode for variant in report.variants] == list(HermesExecutionMode)
    assert all(variant.equivalent_to_native for variant in report.variants)
    assert [variant.ledger_enabled for variant in report.variants] == [False, True, True]
    adapter = next(
        item for item in report.variants
        if item.mode == HermesExecutionMode.ADAPTER_LEDGER
    )
    assert {check.memory_kind for check in adapter.checks} == set(MemoryKind)
    assert adapter.memory_event_count == 12
    semantic = next(
        check for check in adapter.checks if check.memory_kind == MemoryKind.SEMANTIC
    )
    assert semantic.native_item_count == semantic.candidate_item_count == 3
    serialized = json.dumps(asdict(report), default=str)
    assert PRIVATE_PREFERENCE not in serialized
    assert "requested task table" not in serialized


def test_lifecycle_events_join_ledger_without_context_content(tmp_path: Path) -> None:
    fixture = run_sm01_preference_fixture()

    def build() -> LifecycleLedgerObserver:
        observer = LifecycleLedgerObserver(
            variant="native+adapter+ledger",
            trace_id="trace-sm01-fixture",
            family_id="SM01",
            stage="learn",
        )
        observer.record_snapshot(fixture.snapshot)
        for event in fixture.events:
            observer.record(event)
        return observer

    first = build()
    second = build()
    assert first.events == second.events
    assert [event["kind"] for event in first.events] == [
        "context_snapshot",
        "plan_created",
        "plan_validated",
        "dry_run_mutation",
    ]
    assert all(event["episodeId"] == fixture.snapshot.episode_id for event in first.events)
    assert all(event["sessionId"] == fixture.snapshot.session_id for event in first.events)
    assert all(event["snapshotId"] == fixture.snapshot.snapshot_id for event in first.events)
    assert first.events[-1]["data"]["mutationId"] == fixture.receipts[0].mutation_id

    output = tmp_path / "lifecycle.jsonl"
    first.write(output)
    serialized = output.read_text(encoding="utf-8")
    assert PRIVATE_PREFERENCE not in serialized
    assert "current task is complete" not in serialized
    assert "/mnt/" not in serialized
