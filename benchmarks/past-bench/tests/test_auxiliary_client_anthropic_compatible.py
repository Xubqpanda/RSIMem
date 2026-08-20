from __future__ import annotations

import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
HERMES_ROOT = REPO_ROOT / "agents" / "hermes-agent"

if str(HERMES_ROOT) not in sys.path:
    sys.path.insert(0, str(HERMES_ROOT))

from agent import auxiliary_client as auxiliary_client_under_test  # noqa: E402
from hermes_cli import auth as hermes_auth  # noqa: E402


def test_resolve_api_key_provider_uses_anthropic_adapter_for_minimax(monkeypatch) -> None:
    provider = hermes_auth.ProviderConfig(
        id="minimax",
        name="MiniMax",
        auth_type="api_key",
        inference_base_url="https://api.minimax.io/anthropic",
        api_key_env_vars=("MINIMAX_API_KEY",),
    )
    monkeypatch.setattr(hermes_auth, "PROVIDER_REGISTRY", {"minimax": provider})
    monkeypatch.setattr(
        hermes_auth,
        "resolve_api_key_provider_credentials",
        lambda provider_id: {"api_key": "test-minimax-key", "base_url": ""},
    )

    fake_anthropic = types.ModuleType("agent.anthropic_adapter")
    fake_anthropic.build_anthropic_client = lambda api_key, base_url: {
        "api_key": api_key,
        "base_url": base_url,
    }
    sys.modules["agent.anthropic_adapter"] = fake_anthropic

    client, model = auxiliary_client_under_test._resolve_api_key_provider()

    assert isinstance(client, auxiliary_client_under_test.AnthropicAuxiliaryClient)
    assert model == "MiniMax-M2.7-highspeed"
    assert client.api_key == "test-minimax-key"
    assert client.base_url == "https://api.minimax.io/anthropic"


def test_resolve_api_key_provider_uses_anthropic_adapter_for_any_anthropic_suffix(monkeypatch) -> None:
    provider = hermes_auth.ProviderConfig(
        id="alibaba",
        name="Alibaba Cloud (DashScope)",
        auth_type="api_key",
        inference_base_url="https://dashscope-intl.aliyuncs.com/apps/anthropic",
        api_key_env_vars=("DASHSCOPE_API_KEY",),
    )
    monkeypatch.setattr(hermes_auth, "PROVIDER_REGISTRY", {"alibaba": provider})
    monkeypatch.setattr(
        hermes_auth,
        "resolve_api_key_provider_credentials",
        lambda provider_id: {"api_key": "test-dashscope-key", "base_url": ""},
    )

    fake_anthropic = types.ModuleType("agent.anthropic_adapter")
    fake_anthropic.build_anthropic_client = lambda api_key, base_url: {
        "api_key": api_key,
        "base_url": base_url,
    }
    sys.modules["agent.anthropic_adapter"] = fake_anthropic

    client, model = auxiliary_client_under_test._resolve_api_key_provider()

    assert isinstance(client, auxiliary_client_under_test.AnthropicAuxiliaryClient)
    assert model == "default"
    assert client.api_key == "test-dashscope-key"
    assert client.base_url == "https://dashscope-intl.aliyuncs.com/apps/anthropic"
