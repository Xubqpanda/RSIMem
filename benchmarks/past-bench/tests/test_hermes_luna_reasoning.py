from __future__ import annotations

import sys
from pathlib import Path


HERMES_ROOT = Path(__file__).resolve().parents[1] / "agents" / "hermes-agent"
if str(HERMES_ROOT) not in sys.path:
    sys.path.insert(0, str(HERMES_ROOT))

from run_agent import AIAgent


def test_luna_chat_request_explicitly_disables_reasoning() -> None:
    agent = AIAgent(
        model="gpt-5.6-luna", api_key="test", base_url="https://example.test/v1",
        reasoning_config={"enabled": False}, max_tokens=64, temperature=0.0,
    )
    kwargs = agent._build_api_kwargs([{"role": "user", "content": "hello"}])
    assert kwargs["extra_body"]["reasoning_effort"] == "none"


def test_other_models_do_not_receive_luna_override() -> None:
    agent = AIAgent(
        model="gpt-5.4", api_key="test", base_url="https://example.test/v1",
        reasoning_config={"enabled": False}, max_tokens=64, temperature=0.0,
    )
    kwargs = agent._build_api_kwargs([{"role": "user", "content": "hello"}])
    assert "extra_body" not in kwargs or "reasoning_effort" not in kwargs["extra_body"]
