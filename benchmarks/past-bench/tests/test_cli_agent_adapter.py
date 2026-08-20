import subprocess

from past_bench.models.content import TextBlock
from past_bench.models.message import Message
from past_bench.models.tool import ToolEndpoint, ToolSpec
from past_bench.runtime.adapters.cli_agent import ClaudeCodeCliAdapter, CodexCliAdapter
from past_bench.runtime.protocol import RuntimeConfigPayload, RuntimeModelConfig, StartSessionRequest, StepRequest
from past_bench.runtime.registry import AgentSpec


def _request(*, tools=None, endpoints=None):
    return StartSessionRequest(
        session_id="sess-1",
        agent_name="demo",
        task_id="T_demo",
        task_name="Demo Task",
        max_turns=4,
        timeout_seconds=60,
        initial_messages=[
            Message(role="system", content=[TextBlock(text="You are helpful.")]),
            Message(role="user", content=[TextBlock(text="Say hi.")]),
        ],
        tools=tools or [],
        tool_endpoints=endpoints or [],
        model=RuntimeModelConfig(model_id="demo-model", api_key="secret"),
        runtime_config=RuntimeConfigPayload(),
    )


def test_codex_cli_adapter_parses_jsonl(monkeypatch):
    monkeypatch.setattr(CodexCliAdapter, "_login_with_api_key", lambda self, env: None)

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout=(
                '{"type":"thread.started","thread_id":"abc"}\n'
                '{"type":"item.completed","item":{"type":"agent_message","text":"done"}}\n'
                '{"type":"turn.completed","usage":{"input_tokens":12,"output_tokens":3}}\n'
            ),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    adapter = CodexCliAdapter(
        AgentSpec(name="codex", adapter="codex_cli"),
        _request(),
    )
    response = adapter.step(StepRequest(session_id="sess-1", step_id=0))

    assert response.status == "finished"
    assert response.final_output == "done"
    assert response.usage.input_tokens == 12
    assert response.usage.output_tokens == 3


def test_claude_cli_adapter_parses_json(monkeypatch):
    captured = {}

    def fake_run(*args, **kwargs):
        captured["command"] = args[0]
        captured["env"] = kwargs["env"]
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout=(
                '{"type":"result","subtype":"success","is_error":false,"result":"finished",'
                '"usage":{"input_tokens":22163,"output_tokens":20}}'
            ),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    adapter = ClaudeCodeCliAdapter(
        AgentSpec(name="claude", adapter="claude_code_cli"),
        StartSessionRequest(
            session_id="sess-1",
            agent_name="demo",
            task_id="T_demo",
            task_name="Demo Task",
            max_turns=4,
            timeout_seconds=60,
            initial_messages=[
                Message(role="system", content=[TextBlock(text="You are helpful.")]),
                Message(role="user", content=[TextBlock(text="Say hi.")]),
            ],
            model=RuntimeModelConfig(
                model_id="MiniMax-M2.5",
                api_key="secret",
                base_url="https://api.minimaxi.com/anthropic",
                extra_body={
                    "claude_code": {
                        "auth_env": "ANTHROPIC_AUTH_TOKEN",
                        "env": {
                            "ANTHROPIC_BASE_URL": "https://api.minimaxi.com/anthropic",
                            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
                        },
                    }
                },
            ),
            runtime_config=RuntimeConfigPayload(),
        ),
    )
    response = adapter.step(StepRequest(session_id="sess-1", step_id=0))

    assert response.status == "finished"
    assert response.final_output == "finished"
    assert response.usage.input_tokens == 22163
    assert response.usage.output_tokens == 20
    assert captured["env"]["ANTHROPIC_AUTH_TOKEN"] == "secret"
    assert captured["env"]["ANTHROPIC_BASE_URL"] == "https://api.minimaxi.com/anthropic"
    assert captured["env"]["ANTHROPIC_MODEL"] == "MiniMax-M2.5"
    assert captured["command"][0] == "claude"
    assert captured["command"][1] == "-p"
    assert "Demo Task" in captured["command"][2]


def test_claude_cli_adapter_supports_custom_cli_model_and_optional_model_env(monkeypatch):
    captured = {}

    def fake_run(*args, **kwargs):
        captured["command"] = args[0]
        captured["env"] = kwargs["env"]
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout='{"type":"result","subtype":"success","is_error":false,"result":"ok"}',
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    adapter = ClaudeCodeCliAdapter(
        AgentSpec(name="claude", adapter="claude_code_cli"),
        StartSessionRequest(
            session_id="sess-1",
            agent_name="demo",
            task_id="T_demo",
            task_name="Demo Task",
            max_turns=4,
            timeout_seconds=60,
            initial_messages=[
                Message(role="system", content=[TextBlock(text="You are helpful.")]),
                Message(role="user", content=[TextBlock(text="Say hi.")]),
            ],
            model=RuntimeModelConfig(
                model_id="glm-4.7",
                api_key="secret",
                base_url="https://api.z.ai/api/anthropic",
                extra_body={
                    "claude_code": {
                        "auth_env": "ANTHROPIC_AUTH_TOKEN",
                        "cli_model": "sonnet",
                        "set_model_env": False,
                        "env": {
                            "API_TIMEOUT_MS": "3000000",
                        },
                    }
                },
            ),
            runtime_config=RuntimeConfigPayload(),
        ),
    )
    response = adapter.step(StepRequest(session_id="sess-1", step_id=0))

    assert response.status == "finished"
    assert captured["command"][6] == "sonnet"
    assert captured["env"]["ANTHROPIC_AUTH_TOKEN"] == "secret"
    assert captured["env"]["ANTHROPIC_BASE_URL"] == "https://api.z.ai/api/anthropic"
    assert "ANTHROPIC_MODEL" not in captured["env"]


def test_cli_adapter_requires_http_endpoints_for_declared_tools(monkeypatch):
    monkeypatch.setattr(CodexCliAdapter, "_login_with_api_key", lambda self, env: None)

    adapter = CodexCliAdapter(
        AgentSpec(name="codex", adapter="codex_cli"),
        _request(
            tools=[
                ToolSpec(
                    name="todo_create_task",
                    description="Create a task",
                    input_schema={"type": "object"},
                )
            ],
            endpoints=[],
        ),
    )
    response = adapter.step(StepRequest(session_id="sess-1", step_id=0))

    assert response.status == "error"
    assert "HTTP-backed tools" in (response.error or "")


def test_cli_adapter_accepts_http_backed_tools(monkeypatch):
    monkeypatch.setattr(CodexCliAdapter, "_login_with_api_key", lambda self, env: None)

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout='{"type":"item.completed","item":{"type":"agent_message","text":"ok"}}\n',
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    adapter = CodexCliAdapter(
        AgentSpec(name="codex", adapter="codex_cli"),
        _request(
            tools=[
                ToolSpec(
                    name="todo_create_task",
                    description="Create a task",
                    input_schema={"type": "object"},
                )
            ],
            endpoints=[
                ToolEndpoint(
                    tool_name="todo_create_task",
                    url="http://host.docker.internal:9102/todo/tasks/create",
                    method="POST",
                )
            ],
        ),
    )
    response = adapter.step(StepRequest(session_id="sess-1", step_id=0))

    assert response.status == "finished"
    assert response.final_output == "ok"
