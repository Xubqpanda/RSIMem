from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
MODULE_PATH = REPO_ROOT / "agents" / "hermes-agent" / "tools" / "session_search_tool.py"


def _load_module():
    original_modules = {
        name: sys.modules.get(name)
        for name in ("agent", "agent.auxiliary_client", "tools", "tools.registry")
    }

    fake_agent_pkg = types.ModuleType("agent")
    fake_aux = types.ModuleType("agent.auxiliary_client")

    async def _unused_async_call_llm(*args, **kwargs):  # pragma: no cover
        raise RuntimeError("aux model unavailable")

    fake_aux.async_call_llm = _unused_async_call_llm
    fake_agent_pkg.auxiliary_client = fake_aux

    fake_tools_pkg = types.ModuleType("tools")
    fake_tools_pkg.__path__ = []  # mark as package
    fake_registry_mod = types.ModuleType("tools.registry")

    class _FakeRegistry:
        def register(self, **kwargs):
            return None

    fake_registry_mod.registry = _FakeRegistry()

    sys.modules["agent"] = fake_agent_pkg
    sys.modules["agent.auxiliary_client"] = fake_aux
    sys.modules["tools"] = fake_tools_pkg
    sys.modules["tools.registry"] = fake_registry_mod

    spec = importlib.util.spec_from_file_location("session_search_tool_under_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    try:
        spec.loader.exec_module(module)
    finally:
        for name, original in original_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original
    return module


class _FakeDB:
    def search_messages(self, **kwargs):
        return [
            {
                "session_id": "seed-session",
                "snippet": "Vendor X contract discussion staging SOC2",
                "source": "cli",
                "model": "test-model",
                "session_started": 1710000000,
            }
        ]

    def get_session(self, session_id: str):
        return {
            "id": session_id,
            "title": "Vendor X Contract Decision",
            "source": "cli",
            "started_at": 1710000000,
            "parent_session_id": None,
        }

    def get_messages_as_conversation(self, session_id: str):
        return [
            {"role": "user", "content": "Please record the Vendor X production decision for future reference."},
            {
                "role": "tool",
                "tool_name": "notes_get",
                "content": (
                    "Vendor X is NOT approved for production until SOC2. "
                    "Exception granted for staging-only pilot through May 31."
                ),
            },
            {
                "role": "assistant",
                "content": "Confirmed: not approved for production, staging-only pilot through May 31.",
            },
        ]


class _FallbackQueryDB:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def search_messages(self, **kwargs):
        query = kwargs["query"]
        self.queries.append(query)
        if query == "migration-wave decision cleared integrations suspended":
            return []
        if query == "migration-wave OR decision OR cleared OR integrations OR suspended":
            return [
                {
                    "session_id": "seed-session",
                    "snippet": "Approved exception set includes INTG-EU-002 and INTG-US-004",
                    "source": "cli",
                    "model": "test-model",
                    "session_started": 1710000000,
                }
            ]
        return []

    def get_session(self, session_id: str):
        return {
            "id": session_id,
            "title": "EP02 Seed Exception Set",
            "source": "cli",
            "started_at": 1710000000,
            "parent_session_id": None,
        }

    def get_messages_as_conversation(self, session_id: str):
        return [
            {"role": "user", "content": "Approve the migration-wave exception set."},
            {"role": "assistant", "content": "Approved INTG-EU-002, INTG-US-004, and INTG-AP-006."},
        ]


class _StructuredUpdateDB:
    def search_messages(self, **kwargs):
        return [
            {
                "session_id": "seed-session",
                "snippet": "Wave Cedar carry-forward subset",
                "source": "cli",
                "model": "test-model",
                "session_started": 1710000000,
            }
        ]

    def get_session(self, session_id: str):
        return {
            "id": session_id,
            "title": "EP02 Seed Exception Set",
            "source": "cli",
            "started_at": 1710000000,
            "parent_session_id": None,
        }

    def get_messages_as_conversation(self, session_id: str):
        return [
            {
                "role": "tool",
                "tool_name": "config_update_integration",
                "content": json.dumps(
                    {
                        "status": "updated",
                        "integration": {
                            "integration_id": "INTG-EU-002",
                            "status": "active",
                            "exception_approved": True,
                            "notes": "Wave Cedar carry-forward subset; session-bounded restart exception only.",
                        },
                    }
                ),
            },
            {
                "role": "tool",
                "tool_name": "config_update_integration",
                "content": json.dumps(
                    {
                        "status": "updated",
                        "integration": {
                            "integration_id": "INTG-US-004",
                            "status": "active",
                            "exception_approved": True,
                            "notes": "Wave Cedar carry-forward subset; session-bounded restart exception only.",
                        },
                    }
                ),
            },
            {
                "role": "tool",
                "tool_name": "config_update_integration",
                "content": json.dumps(
                    {
                        "status": "updated",
                        "integration": {
                            "integration_id": "INTG-AP-006",
                            "status": "active",
                            "exception_approved": True,
                            "notes": "Wave Cedar carry-forward subset; session-bounded restart exception only.",
                        },
                    }
                ),
            },
        ]


class _NestedSearchNoiseDB:
    def search_messages(self, **kwargs):
        return [
            {
                "session_id": "seed-session",
                "snippet": "Wave Cedar queue",
                "source": "cli",
                "model": "test-model",
                "session_started": 1710000000,
            }
        ]

    def get_session(self, session_id: str):
        return {
            "id": session_id,
            "title": "EP02 Context Session",
            "source": "cli",
            "started_at": 1710000000,
            "parent_session_id": None,
        }

    def get_messages_as_conversation(self, session_id: str):
        return [
            {
                "role": "tool",
                "tool_name": "session_search",
                "content": json.dumps(
                    {
                        "success": True,
                        "query": "Wave Cedar queue",
                        "results": [
                            {
                                "session_id": "old",
                                "summary": "Fallback recall for query 'Wave Cedar queue' from an older session.",
                            }
                        ],
                    }
                ),
            },
            {
                "role": "assistant",
                "content": "AUDIT|suspended_ids:INTG-AP-010,INTG-AP-011|notes:APAC backlog only.",
            },
        ]


async def _return_none(*args, **kwargs):
    return None


def test_session_search_returns_fallback_summary_when_llm_summary_missing(monkeypatch) -> None:
    session_search_tool = _load_module()
    monkeypatch.setattr(session_search_tool, "_summarize_session", _return_none)
    monkeypatch.setitem(
        sys.modules,
        "model_tools",
        types.SimpleNamespace(_run_async=lambda coro: asyncio.run(coro)),
    )

    result = json.loads(
        session_search_tool.session_search(
            query="Vendor X",
            db=_FakeDB(),
            current_session_id="current-session",
        )
    )

    assert result["success"] is True
    assert result["count"] == 1
    assert result["sessions_searched"] == 1
    assert result["results"][0]["summary_mode"] == "fallback_excerpt"
    assert "SOC2" in result["results"][0]["summary"]
    assert "staging" in result["results"][0]["summary"]


def test_session_search_retries_with_or_query_when_whitespace_query_misses(monkeypatch) -> None:
    session_search_tool = _load_module()
    monkeypatch.setattr(session_search_tool, "_summarize_session", _return_none)
    monkeypatch.setitem(
        sys.modules,
        "model_tools",
        types.SimpleNamespace(_run_async=lambda coro: asyncio.run(coro)),
    )

    db = _FallbackQueryDB()
    result = json.loads(
        session_search_tool.session_search(
            query="migration-wave decision cleared integrations suspended",
            db=db,
            current_session_id="current-session",
        )
    )

    assert result["success"] is True
    assert result["count"] == 1
    assert db.queries[:2] == [
        "migration-wave decision cleared integrations suspended",
        "migration-wave OR decision OR cleared OR integrations OR suspended",
    ]


def test_fallback_summary_surfaces_all_structured_tool_updates(monkeypatch) -> None:
    session_search_tool = _load_module()
    monkeypatch.setattr(session_search_tool, "_summarize_session", _return_none)
    monkeypatch.setitem(
        sys.modules,
        "model_tools",
        types.SimpleNamespace(_run_async=lambda coro: asyncio.run(coro)),
    )

    result = json.loads(
        session_search_tool.session_search(
            query="Wave Cedar carry-forward subset",
            db=_StructuredUpdateDB(),
            current_session_id="current-session",
        )
    )

    summary = result["results"][0]["summary"]
    assert "Recovered structured facts" in summary
    assert "INTG-EU-002" in summary
    assert "INTG-US-004" in summary
    assert "INTG-AP-006" in summary
    assert "Recovered approved/allowed IDs from tool-recorded state changes" in summary


def test_fallback_summary_skips_nested_session_search_noise(monkeypatch) -> None:
    session_search_tool = _load_module()
    monkeypatch.setattr(session_search_tool, "_summarize_session", _return_none)
    monkeypatch.setitem(
        sys.modules,
        "model_tools",
        types.SimpleNamespace(_run_async=lambda coro: asyncio.run(coro)),
    )

    result = json.loads(
        session_search_tool.session_search(
            query="Wave Cedar queue",
            db=_NestedSearchNoiseDB(),
            current_session_id="current-session",
        )
    )

    summary = result["results"][0]["summary"]
    assert "Assistant-stated subset references" not in summary
    assert "Fallback recall for query 'Wave Cedar queue' from an older session." not in summary
    assert "AUDIT|suspended_ids:INTG-AP-010,INTG-AP-011" in summary
