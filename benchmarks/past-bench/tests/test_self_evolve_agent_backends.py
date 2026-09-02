from __future__ import annotations

import argparse
import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from past_bench.models.tool import ToolEndpoint, ToolSpec
from past_bench.cli import _resolve_rsimem_sensitivity_paths
from past_bench.runner.self_evolve import (
    HermesPersistenceBackend,
    NanobotPersistenceBackend,
    ZeroClawPersistenceBackend,
    snapshot_nanobot_artifacts,
    snapshot_zeroclaw_artifacts,
)
from past_bench.runtime.adapters.nanobot import NanobotAdapter
from past_bench.runtime.adapters.zeroclaw import ZeroClawAdapter
from past_bench.runtime.adapters.http_task_tools import invoke_http_tool, matching_task_tools


class _EchoHandler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        self.send_response(200)
        self.end_headers()
        self.wfile.write(self.path.encode("utf-8"))

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8")
        self.send_response(200)
        self.end_headers()
        self.wfile.write(body.encode("utf-8"))

    def log_message(self, format, *args):  # noqa: A003
        return


def test_invoke_http_tool_supports_get_query_and_post_json():
    server = HTTPServer(("127.0.0.1", 0), _EchoHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}/tool"
        get_result = invoke_http_tool(
            url=base,
            method="GET",
            arguments={"query": "alpha", "tags": ["x", "y"]},
            timeout=5,
        )
        assert "/tool?" in get_result
        assert "query=alpha" in get_result
        assert "tags=x" in get_result and "tags=y" in get_result

        post_result = invoke_http_tool(
            url=base,
            method="POST",
            arguments={"query": "alpha", "count": 2},
            timeout=5,
        )
        assert json.loads(post_result) == {"query": "alpha", "count": 2}
    finally:
        server.shutdown()
        thread.join(timeout=2)


def test_matching_task_tools_only_returns_registered_endpoints():
    tools = [
        ToolSpec(name="a", description="A", input_schema={}),
        ToolSpec(name="b", description="B", input_schema={}),
    ]
    endpoints = [ToolEndpoint(tool_name="b", url="http://example.test/b", method="POST")]
    matches = matching_task_tools(tools, endpoints)
    assert [(tool.name, endpoint.tool_name) for tool, endpoint in matches] == [("b", "b")]


def test_full_oracle_home_seed_is_hermes_only_and_preserves_native_state(tmp_path: Path):
    oracle = tmp_path / "oracle-home"
    (oracle / "memories").mkdir(parents=True)
    (oracle / "memories" / "MEMORY.md").write_text("oracle fact", encoding="utf-8")
    (oracle / "skills" / "oracle-procedure").mkdir(parents=True)
    (oracle / "skills" / "oracle-procedure" / "SKILL.md").write_text(
        "---\nname: oracle-procedure\n---\noracle steps", encoding="utf-8"
    )
    (oracle / "sessions").mkdir()
    (oracle / "sessions" / "seed.json").write_text("{}", encoding="utf-8")
    (oracle / "state.db").write_bytes(b"sqlite-oracle")

    hermes = HermesPersistenceBackend()
    state_root, _ = hermes.family_paths(tmp_path / "variant", "family")
    hermes.reset_state(state_root)
    (state_root / "state.db").write_bytes(b"old-state")
    hermes.materialize_oracle_home(state_root=state_root, oracle_home_seed_dir=oracle)
    assert (state_root / "memories" / "MEMORY.md").read_text(encoding="utf-8") == "oracle fact"
    assert (state_root / "skills" / "oracle-procedure" / "SKILL.md").exists()
    assert (state_root / "sessions" / "seed.json").exists()
    assert (state_root / "state.db").read_bytes() == b"sqlite-oracle"

    with pytest.raises(ValueError, match="Hermes backend"):
        NanobotPersistenceBackend().materialize_oracle_home(
            state_root=tmp_path / "nanobot",
            oracle_home_seed_dir=oracle,
        )
    with pytest.raises(ValueError, match="Hermes backend"):
        ZeroClawPersistenceBackend().materialize_oracle_home(
            state_root=tmp_path / "zeroclaw",
            oracle_home_seed_dir=oracle,
        )


def test_sensitivity_paths_require_explicit_distinct_hermes_directories(tmp_path: Path):
    backend = HermesPersistenceBackend()
    assert _resolve_rsimem_sensitivity_paths(argparse.Namespace(), backend) is None
    with pytest.raises(SystemExit, match="both state and Hermes-home"):
        _resolve_rsimem_sensitivity_paths(
            argparse.Namespace(rsimem_sensitivity_state_dir=str(tmp_path / "state")),
            backend,
        )
    with pytest.raises(SystemExit, match="must differ"):
        _resolve_rsimem_sensitivity_paths(
            argparse.Namespace(
                rsimem_sensitivity_state_dir=str(tmp_path / "same"),
                rsimem_sensitivity_hermes_home_dir=str(tmp_path / "same"),
            ),
            backend,
        )
    paths = _resolve_rsimem_sensitivity_paths(
        argparse.Namespace(
            rsimem_sensitivity_state_dir=str(tmp_path / "state"),
            rsimem_sensitivity_hermes_home_dir=str(tmp_path / "home"),
        ),
        backend,
    )
    assert paths == ((tmp_path / "state").resolve(), (tmp_path / "home").resolve())


def test_nanobot_backend_history_and_artifact_snapshot(tmp_path: Path):
    backend = NanobotPersistenceBackend()
    variant_dir = tmp_path / "variant"
    state_root, anchors_dir = backend.family_paths(variant_dir, "family-a")
    backend.reset_state(state_root)

    preseed = tmp_path / "preseed"
    (preseed / "memories").mkdir(parents=True)
    (preseed / "memories" / "MEMORY.md").write_text("Rule one", encoding="utf-8")
    (preseed / "skills" / "triage").mkdir(parents=True)
    (preseed / "skills" / "triage" / "SKILL.md").write_text("Always triage", encoding="utf-8")
    (preseed / "sessions").mkdir(parents=True)
    (preseed / "sessions" / "cli_direct.jsonl").write_text("", encoding="utf-8")
    backend.materialize_inputs(
        state_root=state_root,
        initial_home_fixture_dir=None,
        preseed_artifacts_dir=preseed,
    )
    workspace = backend.workspace_dir(state_root)
    assert (workspace / "memory" / "MEMORY.md").exists()
    assert (workspace / "skills" / "triage" / "SKILL.md").exists()
    assert (workspace / "sessions").exists()

    (workspace / "skills" / "triage" / "extra.txt").write_text("v1", encoding="utf-8")
    backend.save_anchor(
        state_root=state_root,
        episode=SimpleNamespace(history_save_anchor="anchor-a"),
        anchors_dir=anchors_dir,
        history_anchors={},
    )
    history_anchors = {"anchor-a": anchors_dir / "anchor-a"}
    (workspace / "skills" / "triage" / "extra.txt").write_text("changed", encoding="utf-8")
    backend.prepare_history(
        state_root=state_root,
        episode=SimpleNamespace(history_mode="from_anchor", history_load_anchor="anchor-a", label="", task="t"),
        history_anchors=history_anchors,
    )
    assert (workspace / "skills" / "triage" / "extra.txt").read_text(encoding="utf-8") == "v1"

    backend.prepare_history(
        state_root=state_root,
        episode=SimpleNamespace(history_mode="fresh", history_load_anchor="", label="", task="t"),
        history_anchors=history_anchors,
    )
    assert backend.workspace_dir(state_root).exists()
    assert not (backend.workspace_dir(state_root) / "skills" / "triage").exists()

    artifacts_dir = tmp_path / "artifacts"
    (artifacts_dir / "memory").mkdir(parents=True)
    (artifacts_dir / "memory" / "MEMORY.md").write_text("Rule one", encoding="utf-8")
    (artifacts_dir / "skills" / "triage").mkdir(parents=True)
    (artifacts_dir / "skills" / "triage" / "SKILL.md").write_text("Always triage", encoding="utf-8")
    (artifacts_dir / "sessions").mkdir(parents=True)
    (artifacts_dir / "nanobot_metadata.json").write_text(
        json.dumps({"prior_history": True}),
        encoding="utf-8",
    )
    session_file = artifacts_dir / "sessions" / "cli_direct.jsonl"
    session_file.write_text(
        "\n".join(
            [
                json.dumps({"_type": "metadata", "key": "cli:direct"}),
                json.dumps(
                    {
                        "role": "assistant",
                        "timestamp": "2026-01-01T00:00:00",
                        "tool_calls": [
                            {
                                "name": "read_file",
                                "args": {"path": str(artifacts_dir / "skills" / "triage" / "SKILL.md")},
                            }
                        ],
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    snapshot = snapshot_nanobot_artifacts(artifacts_dir)
    assert snapshot["memory_file_exists"] is True
    assert snapshot["skill_count"] == 1
    assert snapshot["internal_tools"]["skill_read_count"] == 1
    assert snapshot["internal_tools"]["session_search_calls"] == 1


def test_nanobot_forced_consolidation_archives_pending_messages():
    session = SimpleNamespace(
        key="self-evolve:family-a",
        messages=[{"role": "user", "content": "Remember owner prefers terse summaries."}],
        last_consolidated=0,
    )
    saved = {"count": 0}

    class _Sessions:
        def get_or_create(self, key: str):
            assert key == session.key
            return session

        def save(self, current_session):
            assert current_session is session
            saved["count"] += 1

    class _Consolidator:
        def __init__(self) -> None:
            self.archived_batches: list[list[object]] = []
            self._lock = asyncio.Lock()

        def get_lock(self, key: str):
            assert key == session.key
            return self._lock

        async def archive_messages(self, messages):
            self.archived_batches.append(list(messages))
            return True

    agent = SimpleNamespace(
        sessions=_Sessions(),
        memory_consolidator=_Consolidator(),
    )

    archived = asyncio.run(
        NanobotAdapter._force_memory_consolidation(agent, session.key)
    )

    assert archived is True
    assert session.last_consolidated == 1
    assert saved["count"] == 1
    assert agent.memory_consolidator.archived_batches == [[session.messages[0]]]


def test_nanobot_forced_consolidation_is_noop_when_session_is_already_archived():
    session = SimpleNamespace(
        key="self-evolve:family-a",
        messages=[{"role": "assistant", "content": "Already consolidated."}],
        last_consolidated=1,
    )

    class _Sessions:
        def get_or_create(self, key: str):
            assert key == session.key
            return session

        def save(self, current_session):
            raise AssertionError("save should not be called when nothing is pending")

    class _Consolidator:
        def get_lock(self, key: str):
            raise AssertionError("lock should not be requested when nothing is pending")

    agent = SimpleNamespace(
        sessions=_Sessions(),
        memory_consolidator=_Consolidator(),
    )

    archived = asyncio.run(
        NanobotAdapter._force_memory_consolidation(agent, session.key)
    )

    assert archived is False
    assert session.last_consolidated == 1


def test_zeroclaw_backend_materializes_memory_and_parses_transcript(tmp_path: Path):
    backend = ZeroClawPersistenceBackend()
    state_root = tmp_path / "zeroclaw_state"
    backend.reset_state(state_root)

    preseed = tmp_path / "preseed"
    (preseed / "memories").mkdir(parents=True)
    (preseed / "memories" / "MEMORY.md").write_text(
        "Use the waiver template\n§\nUse the waiver template",
        encoding="utf-8",
    )
    (preseed / "skills" / "waiver").mkdir(parents=True)
    (preseed / "skills" / "waiver" / "SKILL.md").write_text("Check the waiver KB first", encoding="utf-8")
    (preseed / "session_seed.json").write_text(
        json.dumps(
            {
                "sessions": [
                    {
                        "id": "waiver-seed-001",
                        "source": "seed",
                        "title": "Waiver history",
                        "messages": [
                            {"role": "user", "content": "Review the temporary waiver."},
                            {"role": "assistant", "content": "Use waiver WAIV-118 for staging-eu only."},
                        ],
                    }
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    backend.materialize_inputs(
        state_root=state_root,
        initial_home_fixture_dir=None,
        preseed_artifacts_dir=preseed,
    )
    before = backend.snapshot_before(state_root, include_contents=True)
    assert before["memory_file_exists"] is True
    assert before["skill_count"] == 1
    assert sum("waiver template" in entry for entry in before["memory_entries"]) == 1
    session_history_files = list(backend.session_history_dir(state_root).glob("*.json"))
    assert len(session_history_files) == 1

    artifacts_dir = tmp_path / "artifacts"
    (artifacts_dir / ".zeroclaw").mkdir(parents=True)
    (artifacts_dir / ".zeroclaw" / "memory_store.json").write_text(
        json.dumps({"entry_001": "Use the waiver template"}, ensure_ascii=False),
        encoding="utf-8",
    )
    (artifacts_dir / "skills" / "waiver").mkdir(parents=True)
    skill_path = artifacts_dir / "skills" / "waiver" / "SKILL.md"
    skill_path.write_text("Check the waiver KB first", encoding="utf-8")
    (artifacts_dir / "zeroclaw_transcript.json").write_text(
        json.dumps(
            {
                "messages": [
                    {
                        "tool_calls": [
                            {"name": "memory_recall", "args": {"query": "waiver"}},
                            {"name": "file_read", "args": {"path": str(skill_path)}},
                            {"name": "session_search", "args": {"query": "waiver staging-eu"}},
                        ]
                    }
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    after = snapshot_zeroclaw_artifacts(artifacts_dir)
    assert after["memory_file_exists"] is True
    assert after["skill_count"] == 1
    assert after["internal_tools"]["memory_read_count"] == 1
    assert after["internal_tools"]["skill_read_count"] == 1
    assert after["internal_tools"]["session_search_calls"] == 1


def test_zeroclaw_memory_store_deduplicates_values(tmp_path: Path):
    home_dir = tmp_path / "home"
    first = ZeroClawAdapter._store_memory_value(
        home_dir=home_dir,
        key="entry_001",
        value="Use the waiver template",
    )
    second = ZeroClawAdapter._store_memory_value(
        home_dir=home_dir,
        key="entry_002",
        value="  Use   the waiver template  ",
    )
    payload = json.loads(
        (home_dir / ".zeroclaw" / "memory_store.json").read_text(encoding="utf-8")
    )
    assert first == "Stored: entry_001"
    assert second == "Already stored as entry_001"
    assert payload == {"entry_001": "Use the waiver template"}


def test_zeroclaw_session_search_returns_seed_and_transcript_matches(tmp_path: Path):
    session_history_dir = tmp_path / "session_history"
    session_history_dir.mkdir(parents=True)
    (session_history_dir / "001_seed.json").write_text(
        json.dumps(
            {
                "id": "pg04-waiv-118",
                "source": "seed",
                "title": "Ledger export temporary waiver staging-eu",
                "messages": [
                    {"role": "user", "content": "Check the temporary waiver for ledger-export staging-eu."},
                    {
                        "role": "assistant",
                        "content": "Waiver WAIV-118 applies only to owner data-platform, service ledger-export, scope staging-eu.",
                    },
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (session_history_dir / "002_runtime.json").write_text(
        json.dumps(
            {
                "messages": [
                    {"type": "HumanMessage", "role": "user", "content": "Continue the waiver review."},
                    {
                        "type": "AIMessage",
                        "role": "assistant",
                        "content": "Recovered WAIV-118 for staging-eu before updating the audit note.",
                        "tool_calls": [],
                    },
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    result = ZeroClawAdapter._search_session_history(
        session_history_dir,
        "waiver staging-eu",
        max_results=5,
    )

    assert result["success"] is True
    assert len(result["results"]) >= 2
    assert any(item["source"] == "seed" for item in result["results"])
    assert any("WAIV-118" in item["excerpt"] for item in result["results"])
