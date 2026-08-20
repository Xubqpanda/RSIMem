import json
import os
import sys
import types
from pathlib import Path

from past_bench.models.content import TextBlock
from past_bench.models.message import Message
from past_bench.runner.providers.openai_compat import OpenAICompatProvider
from past_bench.runtime.adapters.hermes import HermesAdapter
from past_bench.runtime.adapters.openclaw import OpenClawResponsesAdapter
from past_bench.runtime.protocol import RuntimeConfigPayload, RuntimeModelConfig, StartSessionRequest, StepRequest
from past_bench.runtime.registry import AgentSpec


def test_openai_compat_provider_uses_configured_temperature():
    provider = OpenAICompatProvider(model_id="demo-model", temperature=0.25)
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        message = types.SimpleNamespace(content="ok", tool_calls=None, reasoning_content=None)
        choice = types.SimpleNamespace(message=message)
        usage = types.SimpleNamespace(prompt_tokens=3, completion_tokens=2)
        return types.SimpleNamespace(choices=[choice], usage=usage)

    provider.client = types.SimpleNamespace(
        chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(create=fake_create),
        )
    )

    assistant, usage = provider.chat([Message(role="user", content=[TextBlock(text="hi")])])

    assert captured["temperature"] == 0.25
    assert assistant.text == "ok"
    assert usage.input_tokens == 3
    assert usage.output_tokens == 2


def test_hermes_adapter_passes_runtime_temperature_to_ai_agent(monkeypatch, tmp_path):
    captured = {}

    fake_run_agent = types.ModuleType("run_agent")

    class FakeAIAgent:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs
            self.session_id = "hermes-session-1"

        def run_conversation(self, **kwargs):
            return {
                "final_response": "done",
                "input_tokens": 11,
                "output_tokens": 7,
                "prompt_tokens": 13,
                "completion_tokens": 9,
            }

    fake_run_agent.AIAgent = FakeAIAgent

    fake_hermes_state = types.ModuleType("hermes_state")
    fake_hermes_state.closed = False
    fake_hermes_state.instances = []

    class FakeSessionDB:
        def __init__(self):
            fake_hermes_state.instances.append(self)
            self.title = None

        def close(self):
            fake_hermes_state.closed = True

        def get_session_title(self, session_id):
            return self.title

        def set_session_title(self, session_id, title):
            self.title = title
            return True

    fake_hermes_state.SessionDB = FakeSessionDB

    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)
    monkeypatch.setitem(sys.modules, "hermes_state", fake_hermes_state)
    monkeypatch.setattr(HermesAdapter, "_register_past_bench_tools", lambda self: None)
    monkeypatch.setattr(HermesAdapter, "_reload_hermes_modules_if_needed", lambda *args, **kwargs: None)

    adapter = HermesAdapter(
        AgentSpec(name="hermes", adapter="hermes"),
        StartSessionRequest(
            session_id="sess-1",
            agent_name="hermes",
            task_id="T_demo",
            task_name="Demo Task",
            max_turns=4,
            timeout_seconds=60,
            initial_messages=[
                Message(role="system", content=[TextBlock(text="System text")]),
                Message(role="user", content=[TextBlock(text="User text")]),
            ],
            model=RuntimeModelConfig(
                model_id="MiniMax-M2.7",
                api_key="secret",
                base_url="https://api.minimaxi.com/anthropic",
                extra_body={
                    "hermes": {
                        "home_dir": str(tmp_path / "hermes_home"),
                        "session_search_enabled": True,
                    }
                },
            ),
            runtime_config=RuntimeConfigPayload(temperature=0.25),
        ),
    )
    try:
        response = adapter.step(StepRequest(session_id="sess-1", step_id=0))
    finally:
        adapter.close("test")

    assert response.status == "finished"
    assert response.final_output == "done"
    assert response.usage.input_tokens == 11
    assert response.usage.output_tokens == 7
    assert captured["kwargs"]["temperature"] == 0.25
    assert isinstance(captured["kwargs"]["session_db"], FakeSessionDB)
    assert len(fake_hermes_state.instances) == 1
    assert fake_hermes_state.instances[0].title == "T_demo - Demo Task"
    assert fake_hermes_state.closed is True


def test_hermes_adapter_sets_home_before_prepare_home(monkeypatch, tmp_path):
    captured = {}

    fake_run_agent = types.ModuleType("run_agent")

    class FakeAIAgent:
        def __init__(self, **kwargs):
            captured["session_db"] = kwargs.get("session_db")

        def run_conversation(self, **kwargs):
            return {
                "final_response": "done",
                "input_tokens": 1,
                "output_tokens": 1,
            }

    fake_run_agent.AIAgent = FakeAIAgent

    fake_hermes_state = types.ModuleType("hermes_state")

    class FakeSessionDB:
        def __init__(self):
            pass

        def close(self):
            pass

        def get_session_title(self, session_id):
            return None

        def set_session_title(self, session_id, title):
            return True

    fake_hermes_state.SessionDB = FakeSessionDB

    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)
    monkeypatch.setitem(sys.modules, "hermes_state", fake_hermes_state)
    monkeypatch.setattr(HermesAdapter, "_register_past_bench_tools", lambda self: None)
    monkeypatch.setattr(HermesAdapter, "_capture_hermes_artifacts", lambda *args, **kwargs: None)
    monkeypatch.setattr(HermesAdapter, "_reload_hermes_modules_if_needed", lambda *args, **kwargs: None)

    expected_home = (tmp_path / "hermes_home").resolve()
    seen_env = {}

    def fake_prepare(hermes_cfg):
        seen_env["value"] = os.environ.get("HERMES_HOME")
        expected_home.mkdir(parents=True, exist_ok=True)
        return expected_home

    monkeypatch.setattr(HermesAdapter, "_prepare_hermes_home", staticmethod(fake_prepare))

    adapter = HermesAdapter(
        AgentSpec(name="hermes", adapter="hermes"),
        StartSessionRequest(
            session_id="sess-1",
            agent_name="hermes",
            task_id="T_demo",
            task_name="Demo Task",
            max_turns=4,
            timeout_seconds=60,
            initial_messages=[],
            model=RuntimeModelConfig(
                model_id="MiniMax-M2.7",
                extra_body={
                    "hermes": {
                        "home_dir": str(expected_home),
                        "session_search_enabled": True,
                    }
                },
            ),
            runtime_config=RuntimeConfigPayload(),
        ),
    )
    try:
        response = adapter.step(StepRequest(session_id="sess-1", step_id=0))
    finally:
        adapter.close("test")

    assert response.status == "finished"
    assert seen_env["value"] == str(expected_home)
    assert isinstance(captured["session_db"], FakeSessionDB)


def test_reload_hermes_modules_reloads_stale_state_without_run_agent(monkeypatch, tmp_path):
    reloads = []

    fake_hermes_constants = types.ModuleType("hermes_constants")
    fake_hermes_state = types.ModuleType("hermes_state")
    fake_hermes_state.DEFAULT_DB_PATH = Path("/tmp/old-home/state.db")
    fake_skill_tool = types.ModuleType("tools.skills_tool")

    monkeypatch.delitem(sys.modules, "run_agent", raising=False)
    monkeypatch.setitem(sys.modules, "hermes_constants", fake_hermes_constants)
    monkeypatch.setitem(sys.modules, "hermes_state", fake_hermes_state)
    monkeypatch.setitem(sys.modules, "tools.skills_tool", fake_skill_tool)
    monkeypatch.setattr(
        "past_bench.runtime.adapters.hermes.importlib.reload",
        lambda module: reloads.append(module.__name__) or module,
    )

    HermesAdapter._reload_hermes_modules_if_needed(tmp_path / "new-home")

    assert "hermes_state" in reloads
    assert "hermes_constants" in reloads
    assert "tools.skills_tool" in reloads


def test_openclaw_adapter_writes_runtime_temperature_to_agent_params():
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
            runtime_config=RuntimeConfigPayload(temperature=0.25),
        ),
    )
    try:
        config = json.loads(adapter._config_path.read_text(encoding="utf-8"))
    finally:
        adapter.close("test")

    assert config["agents"]["defaults"]["params"]["temperature"] == 0.25
    assert config["agents"]["list"][0]["params"]["temperature"] == 0.25
