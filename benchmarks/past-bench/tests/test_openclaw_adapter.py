import json

from past_bench.models.content import ImageBlock, TextBlock, ToolResultBlock
from past_bench.models.message import Message
from past_bench.models.tool import ToolSpec
from past_bench.runtime.adapters.openclaw import (
    OpenClawResponsesAdapter,
    _response_from_openclaw_agent_output,
    _message_to_openclaw_items,
    _response_to_step_response,
    _tool_to_openclaw_definition,
    _tool_results_to_openclaw_items,
)
from past_bench.runtime.protocol import RuntimeConfigPayload, RuntimeModelConfig, StartSessionRequest
from past_bench.runtime.registry import AgentSpec


def test_message_to_openclaw_items_supports_text_and_image():
    message = Message(
        role="user",
        content=[
            TextBlock(text="这是什么吃的"),
            ImageBlock(data="ZmFrZQ==", mime_type="image/jpeg"),
        ],
    )

    items = _message_to_openclaw_items(message)

    assert items[0] == {
        "type": "message",
        "role": "user",
        "content": "这是什么吃的",
    }
    assert items[1]["type"] == "input_image"
    assert items[1]["source"]["type"] == "base64"
    assert items[1]["source"]["media_type"] == "image/jpeg"


def test_tool_results_to_openclaw_items_preserves_call_id():
    items = _tool_results_to_openclaw_items(
        [
            ToolResultBlock(
                tool_use_id="call_1",
                content=[TextBlock(text='{"ok": true}')],
                is_error=False,
            )
        ]
    )

    assert items == [
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": '{"ok": true}',
        }
    ]


def test_tool_to_openclaw_definition_uses_nested_function_shape():
    tool = ToolSpec(
        name="todo_create_task",
        description="Create a new todo task",
        input_schema={"type": "object", "properties": {"title": {"type": "string"}}},
    )

    definition = _tool_to_openclaw_definition(tool)

    assert definition == {
        "type": "function",
        "function": {
            "name": "todo_create_task",
            "description": "Create a new todo task",
            "parameters": {"type": "object", "properties": {"title": {"type": "string"}}},
        },
    }


def test_response_to_step_response_extracts_text_and_function_calls():
    response = _response_to_step_response(
        {
            "usage": {"input_tokens": 11, "output_tokens": 7},
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "我先查一下。"}],
                },
                {
                    "type": "function_call",
                    "call_id": "call_123",
                    "name": "lookup_order",
                    "arguments": '{"order_id": "A-42"}',
                },
            ],
        },
        model_time_s=0.25,
    )

    assert response.status == "acting"
    assert response.usage.input_tokens == 11
    assert response.usage.output_tokens == 7
    assert response.assistant_message is not None
    assert response.assistant_message.text == "我先查一下。"
    assert response.tool_calls[0].tool_use_id == "call_123"
    assert response.tool_calls[0].name == "lookup_order"
    assert response.tool_calls[0].arguments == {"order_id": "A-42"}


def test_response_to_step_response_finishes_on_text_only():
    response = _response_to_step_response(
        {
            "usage": {"input_tokens": 5, "output_tokens": 3},
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "这是陕北抿节。"}],
                }
            ],
        },
        model_time_s=0.1,
    )

    assert response.status == "finished"
    assert response.final_output == "这是陕北抿节。"


def test_openclaw_adapter_uses_loopback_gateway_without_auth(monkeypatch):
    adapter = OpenClawResponsesAdapter(
        AgentSpec(name="openclaw", adapter="openclaw_responses"),
        StartSessionRequest(
            session_id="sess-1",
            agent_name="openclaw",
            task_id="T_demo",
            task_name="Demo Task",
            max_turns=4,
            timeout_seconds=60,
            initial_messages=[
                Message(role="system", content=[TextBlock(text="You are helpful.")]),
                Message(role="user", content=[TextBlock(text="Say hi.")]),
            ],
            model=RuntimeModelConfig(
                model_id="MiniMax-M2.7",
                api_key="secret",
                base_url="https://api.minimaxi.com/anthropic",
                extra_body={
                    "openclaw": {
                        "provider_api": "anthropic-messages",
                        "provider_id": "minimax",
                    }
                },
            ),
            runtime_config=RuntimeConfigPayload(),
        ),
    )
    try:
        config = json.loads(adapter._config_path.read_text(encoding="utf-8"))
    finally:
        adapter.close("test")

    assert config["gateway"]["bind"] == "loopback"
    assert config["gateway"]["auth"] == {"mode": "none"}


def test_response_from_openclaw_agent_output_parses_json_after_log_noise():
    response = _response_from_openclaw_agent_output(
        '[agents/model-providers] warning\\n{\n'
        '  "payloads": [{"text": "done"}],\n'
        '  "meta": {"agentMeta": {"lastCallUsage": {"input": 12, "output": 3}}}\n'
        '}',
        "",
    )

    assert response.status == "finished"
    assert response.final_output == "done"
    assert response.usage.input_tokens == 12
    assert response.usage.output_tokens == 3


def test_response_from_openclaw_agent_output_parses_json_from_stderr():
    response = _response_from_openclaw_agent_output(
        "",
        '[agents/model-providers] warning\\n{\n'
        '  "payloads": [{"text": "stderr result"}],\n'
        '  "meta": {"agentMeta": {"lastCallUsage": {"input": 9, "output": 2}}}\n'
        '}',
    )

    assert response.status == "finished"
    assert response.final_output == "stderr result"
    assert response.usage.input_tokens == 9
    assert response.usage.output_tokens == 2
