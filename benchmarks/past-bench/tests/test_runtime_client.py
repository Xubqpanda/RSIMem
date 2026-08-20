from past_bench.config import RuntimeConfig
from past_bench.models.content import TextBlock
from past_bench.models.message import Message
from past_bench.runtime.client import ContainerRuntimeClient
from past_bench.runtime.protocol import (
    RuntimeConfigPayload,
    RuntimeModelConfig,
    StartSessionRequest,
    StepRequest,
)


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeHttpxClient:
    def __init__(self):
        self.calls = []

    def request(self, method, path, json=None, timeout=None):
        self.calls.append(
            {
                "method": method,
                "path": path,
                "json": json,
                "timeout": timeout,
            }
        )
        if path == "/start_session":
            return _FakeResponse(
                {
                    "session_id": "sess-1",
                    "agent_name": "hermes",
                    "adapter": "hermes",
                }
            )
        if path == "/step":
            return _FakeResponse(
                {
                    "status": "finished",
                    "assistant_message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "done"}],
                    },
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                    "tool_calls": [],
                    "final_output": "done",
                    "error": None,
                    "model_time_s": 0.0,
                }
            )
        raise AssertionError(f"Unexpected request path: {path}")


def _start_request(timeout_seconds: int) -> StartSessionRequest:
    return StartSessionRequest(
        session_id="sess-1",
        agent_name="hermes",
        task_id="T_demo",
        task_name="Demo",
        max_turns=8,
        timeout_seconds=timeout_seconds,
        initial_messages=[Message(role="user", content=[TextBlock(text="hi")])],
        model=RuntimeModelConfig(model_id="demo-model"),
        runtime_config=RuntimeConfigPayload(),
    )


def test_container_runtime_client_uses_task_timeout_for_step_requests():
    client = ContainerRuntimeClient(RuntimeConfig())
    client._client = _FakeHttpxClient()

    client.start_session(_start_request(timeout_seconds=600))
    client.step(StepRequest(session_id="sess-1", step_id=0))

    assert client._client.calls[-1]["path"] == "/step"
    assert client._client.calls[-1]["timeout"] == 630.0


def test_container_runtime_client_keeps_minimum_step_timeout():
    client = ContainerRuntimeClient(RuntimeConfig())
    client._client = _FakeHttpxClient()

    client.start_session(_start_request(timeout_seconds=30))
    client.step(StepRequest(session_id="sess-1", step_id=0))

    assert client._client.calls[-1]["timeout"] == 120.0
