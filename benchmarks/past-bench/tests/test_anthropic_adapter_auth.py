from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parent.parent
HERMES_ROOT = REPO_ROOT / "agents" / "hermes-agent"

if str(HERMES_ROOT) not in sys.path:
    sys.path.insert(0, str(HERMES_ROOT))

from agent import anthropic_adapter as anthropic_adapter_under_test  # noqa: E402


def test_anthropic_auth_kwargs_uses_api_key_for_minimax_compatible_gateway():
    assert anthropic_adapter_under_test._anthropic_auth_kwargs(
        "sk-cp-example",
        "https://api.minimax.io/anthropic",
    ) == {"api_key": "sk-cp-example"}


def test_anthropic_auth_kwargs_uses_auth_token_for_first_party_non_key_tokens():
    assert anthropic_adapter_under_test._anthropic_auth_kwargs(
        "sk-ant-oat-example",
        "https://api.anthropic.com",
    ) == {"auth_token": "sk-ant-oat-example"}


def test_build_anthropic_client_uses_api_key_for_third_party_anthropic_endpoint(monkeypatch):
    captured: dict[str, object] = {}

    class FakeAnthropic:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        anthropic_adapter_under_test,
        "_anthropic_sdk",
        SimpleNamespace(Anthropic=FakeAnthropic),
    )
    monkeypatch.setitem(sys.modules, "httpx", SimpleNamespace(Timeout=lambda **kwargs: kwargs))

    anthropic_adapter_under_test.build_anthropic_client(
        "sk-cp-example",
        "https://api.minimax.io/anthropic",
    )

    assert captured["api_key"] == "sk-cp-example"
    assert "auth_token" not in captured
    assert captured["base_url"] == "https://api.minimax.io/anthropic"


def test_normalize_anthropic_response_parses_minimax_invoke_markup_as_tool_call():
    response = SimpleNamespace(
        content=[
            SimpleNamespace(
                type="text",
                text='<invoke name="notes_list">\n</invoke>\n</minimax:tool_call>',
            )
        ],
        stop_reason="end_turn",
    )

    assistant_message, finish_reason = anthropic_adapter_under_test.normalize_anthropic_response(response)

    assert finish_reason == "tool_calls"
    assert assistant_message.content is None
    assert assistant_message.tool_calls is not None
    assert len(assistant_message.tool_calls) == 1
    tool_call = assistant_message.tool_calls[0]
    assert tool_call.function.name == "notes_list"
    assert tool_call.function.arguments == "{}"


def test_normalize_anthropic_response_keeps_non_tool_text_when_minimax_markup_is_mixed():
    response = SimpleNamespace(
        content=[
            SimpleNamespace(
                type="text",
                text='I will check first.\n<invoke name="notes_get">\n{"note_id":"DOC-431"}\n</invoke>\n</minimax:tool_call>',
            )
        ],
        stop_reason="end_turn",
    )

    assistant_message, finish_reason = anthropic_adapter_under_test.normalize_anthropic_response(response)

    assert finish_reason == "tool_calls"
    assert assistant_message.content == "I will check first."
    assert assistant_message.tool_calls is not None
    assert len(assistant_message.tool_calls) == 1
    tool_call = assistant_message.tool_calls[0]
    assert tool_call.function.name == "notes_get"
    assert tool_call.function.arguments == '{"note_id":"DOC-431"}'
