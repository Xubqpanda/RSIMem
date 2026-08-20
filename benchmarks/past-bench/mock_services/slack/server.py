"""Mock Slack API service for agent evaluation (FastAPI on port 3100).

Supports readMessages (conversations.history / conversations.replies) and
sendMessage (chat.postMessage) via a unified ``slack_action`` tool endpoint.
Fixture data is loaded from a JSON file specified by SLACK_FIXTURES env var.
"""

from __future__ import annotations

import json
import os
import copy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

app = FastAPI(title="Mock Slack API")

from mock_services._base import add_error_injection
add_error_injection(app)

FIXTURES_PATH = Path(os.environ.get(
    "SLACK_FIXTURES",
    str(Path(__file__).resolve().parent / "default_fixtures.json"),
))

# In-memory state
_members: dict[str, dict[str, str]] = {}
_channels: dict[str, list[dict[str, Any]]] = {}
_threads: dict[str, list[dict[str, Any]]] = {}
_audit_log: list[dict[str, Any]] = []
_posted_messages: list[dict[str, Any]] = []
_reactions: list[dict[str, Any]] = []


def _load_fixtures() -> None:
    global _members, _channels, _threads
    with open(FIXTURES_PATH) as f:
        data = json.load(f)
    _members = data.get("members", {})
    _channels = {}
    _threads = {}
    for ch_id, ch_data in data.get("channels", {}).items():
        _channels[ch_id] = ch_data.get("messages", [])
    # Support both flat threads {"ts": [...]} and channel-nested {"CH": {"ts": [...]}}
    for key, value in data.get("threads", {}).items():
        if isinstance(value, list):
            # Flat format: key is thread_ts, value is list of messages
            _threads[key] = value
        elif isinstance(value, dict):
            # Channel-nested format: key is channel_id, value is {thread_ts: [messages]}
            for thread_ts, thread_msgs in value.items():
                _threads[thread_ts] = thread_msgs


_load_fixtures()


def _log_call(method: str, params: dict[str, Any], response: Any) -> None:
    _audit_log.append({
        "method": method,
        "params": params,
        "response": response,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


# --- Slack API endpoints ---

@app.post("/api/conversations.history")
async def conversations_history(request: Request):
    body = await request.json()
    channel = body.get("channel", "")
    messages = _channels.get(channel, [])
    result = {"ok": True, "messages": copy.deepcopy(messages)}
    _log_call("conversations.history", {"channel": channel}, result)
    return result


@app.post("/api/conversations.replies")
async def conversations_replies(request: Request):
    body = await request.json()
    channel = body.get("channel", "")
    ts = body.get("ts", "")
    thread_msgs = _threads.get(ts, [])
    result = {"ok": True, "messages": copy.deepcopy(thread_msgs)}
    _log_call("conversations.replies", {"channel": channel, "ts": ts}, result)
    return result


@app.post("/api/chat.postMessage")
async def post_message(request: Request):
    body = await request.json()
    channel = body.get("channel", "")
    text = body.get("text", "")
    thread_ts = body.get("thread_ts")
    ts = f"{datetime.now(timezone.utc).timestamp():.6f}"
    msg = {"channel": channel, "text": text, "ts": ts, "user": "U_AGENT"}
    if thread_ts:
        msg["thread_ts"] = thread_ts
    _posted_messages.append(msg)
    if channel in _channels:
        _channels[channel].append(msg)
    result = {"ok": True, "channel": channel, "ts": ts}
    _log_call("chat.postMessage", {"channel": channel, "text": text, "thread_ts": thread_ts}, result)
    return result


@app.post("/api/reactions.add")
async def reactions_add(request: Request):
    body = await request.json()
    channel = body.get("channel", "")
    timestamp = body.get("timestamp", "")
    name = body.get("name", "")
    reaction = {"channel": channel, "timestamp": timestamp, "name": name}
    _reactions.append(reaction)
    result = {"ok": True}
    _log_call("reactions.add", {"channel": channel, "timestamp": timestamp, "name": name}, result)
    return result


@app.post("/api/pins.add")
async def pins_add(request: Request):
    body = await request.json()
    channel = body.get("channel", "")
    timestamp = body.get("timestamp", "")
    result = {"ok": True}
    _log_call("pins.add", {"channel": channel, "timestamp": timestamp}, result)
    return result


@app.post("/api/users.list")
async def users_list(request: Request):
    members_list = [
        {"id": uid, "name": info.get("name", ""), "real_name": info.get("realName", "")}
        for uid, info in _members.items()
    ]
    result = {"ok": True, "members": members_list}
    _log_call("users.list", {}, result)
    return result


# --- Unified slack_action tool endpoint ---

@app.post("/slack/action")
async def slack_action(request: Request):
    """Unified tool endpoint: maps action + channel to the correct Slack API."""
    body = await request.json()
    action = body.get("action", "")
    channel = body.get("channel", "")
    text = body.get("text", "")
    thread_id = body.get("threadId", "")

    if action == "readMessages":
        if thread_id:
            messages = _threads.get(thread_id, [])
            result = {"ok": True, "messages": copy.deepcopy(messages)}
            _log_call("conversations.replies", {"channel": channel, "ts": thread_id}, result)
        else:
            messages = _channels.get(channel, [])
            result = {"ok": True, "messages": copy.deepcopy(messages)}
            _log_call("conversations.history", {"channel": channel}, result)
        return result

    elif action == "sendMessage":
        ts = f"{datetime.now(timezone.utc).timestamp():.6f}"
        msg = {"channel": channel, "text": text, "ts": ts, "user": "U_AGENT"}
        if thread_id:
            msg["thread_ts"] = thread_id
        _posted_messages.append(msg)
        if channel in _channels:
            _channels[channel].append(msg)
        result = {"ok": True, "channel": channel, "ts": ts}
        _log_call("chat.postMessage", {"channel": channel, "text": text, "thread_ts": thread_id or None}, result)
        return result

    elif action == "react":
        emoji = body.get("emoji", body.get("name", ""))
        timestamp = body.get("timestamp", thread_id)
        reaction = {"channel": channel, "timestamp": timestamp, "name": emoji}
        _reactions.append(reaction)
        result = {"ok": True}
        _log_call("reactions.add", {"channel": channel, "timestamp": timestamp, "name": emoji}, result)
        return result

    elif action == "pinMessage":
        timestamp = body.get("timestamp", thread_id)
        result = {"ok": True}
        _log_call("pins.add", {"channel": channel, "timestamp": timestamp}, result)
        return result

    return {"ok": False, "error": f"Unknown action: {action}"}


# --- Management endpoints ---

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/slack/reset")
async def reset():
    global _audit_log, _posted_messages, _reactions
    _audit_log = []
    _posted_messages = []
    _reactions = []
    _load_fixtures()
    return {"status": "reset"}


@app.get("/slack/audit")
async def audit():
    return {"audit": _audit_log, "posted_messages": _posted_messages, "reactions": _reactions}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("SLACK_PORT", "3100"))
    uvicorn.run(app, host="0.0.0.0", port=port)
