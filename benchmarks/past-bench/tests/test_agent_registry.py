from pathlib import Path

from past_bench.runtime.registry import load_agent_registry, missing_required_env, resolve_model_config


def test_load_agent_registry_with_env_expansion(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("KIMI_CODE_API_KEY", "secret")
    registry_path = tmp_path / "agents.yaml"
    registry_path.write_text(
        """
agents:
  demo:
    adapter: openai_compat_chat
    required_env: [kimi-code-api-key]
    default_model:
      model_id: demo-model
      api_key_env: kimi-code-api-key
      base_url: https://example.test/v1
      extra_body:
        openclaw:
          provider_api: anthropic-messages
""".strip(),
        encoding="utf-8",
    )

    registry = load_agent_registry(registry_path)
    spec = registry["demo"]
    model = resolve_model_config(spec)

    assert spec.required_env == ["kimi-code-api-key"]
    assert model.model_id == "demo-model"
    assert model.api_key == "secret"
    assert model.base_url == "https://example.test/v1"
    assert model.extra_body == {"openclaw": {"provider_api": "anthropic-messages"}}


def test_resolve_model_config_uses_named_profile(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    registry_path = tmp_path / "agents.yaml"
    registry_path.write_text(
        """
agents:
  openclaw:
    adapter: openclaw_responses
    default_profile: kimi
    default_model:
      extra_body:
        openclaw:
          input_modalities: [text]
    model_profiles:
      kimi:
        model_id: kimi-k2.6
        api_key_env: KIMI_CODE_API_KEY
        base_url: https://api.kimi.com/coding
        extra_body:
          openclaw:
            provider_api: anthropic-messages
            provider_id: kimi
      openai:
        model_id: gpt-5
        api_key_env: OPENAI_API_KEY
        base_url: https://api.openai.com/v1
        extra_body:
          openclaw:
            provider_api: openai-completions
            provider_id: openai
            input_modalities: [text, image]
""".strip(),
        encoding="utf-8",
    )

    registry = load_agent_registry(registry_path)
    spec = registry["openclaw"]
    model = resolve_model_config(spec, profile="openai")

    assert model.model_id == "gpt-5"
    assert model.api_key == "openai-secret"
    assert model.base_url == "https://api.openai.com/v1"
    assert model.extra_body == {
        "openclaw": {
            "input_modalities": ["text", "image"],
            "provider_api": "openai-completions",
            "provider_id": "openai",
        }
    }
    assert missing_required_env(spec, profile="openai") == []


def test_resolve_model_config_supports_minimax_profile(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "minimax-secret")
    registry_path = tmp_path / "agents.yaml"
    registry_path.write_text(
        """
agents:
  openclaw:
    adapter: openclaw_responses
    default_profile: kimi
    default_model:
      extra_body:
        openclaw:
          input_modalities: [text]
          context_window: 32768
          max_tokens: 8192
    model_profiles:
      minimax:
        model_id: MiniMax-M2.5
        api_key_env: MINIMAX_API_KEY
        base_url: https://api.minimaxi.com/anthropic
        extra_body:
          openclaw:
            provider_api: anthropic-messages
            provider_id: minimax
""".strip(),
        encoding="utf-8",
    )

    registry = load_agent_registry(registry_path)
    spec = registry["openclaw"]
    model = resolve_model_config(spec, profile="minimax")

    assert model.model_id == "MiniMax-M2.5"
    assert model.api_key == "minimax-secret"
    assert model.base_url == "https://api.minimaxi.com/anthropic"
    assert model.extra_body == {
        "openclaw": {
            "input_modalities": ["text"],
            "context_window": 32768,
            "max_tokens": 8192,
            "provider_api": "anthropic-messages",
            "provider_id": "minimax",
        }
    }
    assert missing_required_env(spec, profile="minimax") == []


def test_resolve_model_config_supports_minimax_global_profile(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "minimax-secret")
    registry_path = tmp_path / "agents.yaml"
    registry_path.write_text(
        """
agents:
  openclaw:
    adapter: openclaw_responses
    default_model:
      extra_body:
        openclaw:
          input_modalities: [text]
    model_profiles:
      minimax_global:
        model_id: MiniMax-M2.5
        api_key_env: MINIMAX_API_KEY
        base_url: https://api.minimax.io/anthropic
        extra_body:
          openclaw:
            provider_api: anthropic-messages
            provider_id: minimax
""".strip(),
        encoding="utf-8",
    )

    registry = load_agent_registry(registry_path)
    spec = registry["openclaw"]
    model = resolve_model_config(spec, profile="minimax_global")

    assert model.model_id == "MiniMax-M2.5"
    assert model.api_key == "minimax-secret"
    assert model.base_url == "https://api.minimax.io/anthropic"
    assert model.extra_body == {
        "openclaw": {
            "input_modalities": ["text"],
            "provider_api": "anthropic-messages",
            "provider_id": "minimax",
        }
    }


def test_resolve_model_config_supports_claude_minimax_profile(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "minimax-secret")
    registry_path = tmp_path / "agents.yaml"
    registry_path.write_text(
        """
agents:
  claude:
    adapter: claude_code_cli
    default_profile: anthropic
    default_model:
      extra_body:
        claude_code:
          env:
            DISABLE_TELEMETRY: "1"
    model_profiles:
      anthropic:
        model_id: claude-sonnet-4-20250514
        api_key_env: ANTHROPIC_API_KEY
        base_url: https://api.anthropic.com/v1
        extra_body:
          claude_code:
            auth_env: ANTHROPIC_API_KEY
      minimax:
        model_id: MiniMax-M2.5
        api_key_env: MINIMAX_API_KEY
        base_url: https://api.minimaxi.com/anthropic
        extra_body:
          claude_code:
            auth_env: ANTHROPIC_AUTH_TOKEN
            env:
              ANTHROPIC_BASE_URL: https://api.minimaxi.com/anthropic
              ANTHROPIC_MODEL: MiniMax-M2.5
""".strip(),
        encoding="utf-8",
    )

    registry = load_agent_registry(registry_path)
    spec = registry["claude"]
    model = resolve_model_config(spec, profile="minimax")

    assert model.model_id == "MiniMax-M2.5"
    assert model.api_key == "minimax-secret"
    assert model.base_url == "https://api.minimaxi.com/anthropic"
    assert model.extra_body == {
        "claude_code": {
            "auth_env": "ANTHROPIC_AUTH_TOKEN",
            "env": {
                "DISABLE_TELEMETRY": "1",
                "ANTHROPIC_BASE_URL": "https://api.minimaxi.com/anthropic",
                "ANTHROPIC_MODEL": "MiniMax-M2.5",
            },
        }
    }
    assert missing_required_env(spec, profile="minimax") == []


def test_resolve_model_config_supports_openclaw_glm_profile(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ZAI_API_KEY", "zai-secret")
    registry_path = tmp_path / "agents.yaml"
    registry_path.write_text(
        """
agents:
  openclaw:
    adapter: openclaw_responses
    default_model:
      extra_body:
        openclaw:
          input_modalities: [text]
    model_profiles:
      glm:
        model_id: glm-4.7
        api_key_env: ZAI_API_KEY
        base_url: https://api.z.ai/api/anthropic
        extra_body:
          openclaw:
            provider_api: anthropic-messages
            provider_id: zai
            input_modalities: [text]
            context_window: 200000
            max_tokens: 128000
""".strip(),
        encoding="utf-8",
    )

    registry = load_agent_registry(registry_path)
    spec = registry["openclaw"]
    model = resolve_model_config(spec, profile="glm")

    assert model.model_id == "glm-4.7"
    assert model.api_key == "zai-secret"
    assert model.base_url == "https://api.z.ai/api/anthropic"
    assert model.extra_body == {
        "openclaw": {
            "input_modalities": ["text"],
            "provider_api": "anthropic-messages",
            "provider_id": "zai",
            "context_window": 200000,
            "max_tokens": 128000,
        }
    }
    assert missing_required_env(spec, profile="glm") == []


def test_resolve_model_config_supports_claude_kimi_and_glm_profiles(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("KIMI_CODE_API_KEY", "kimi-secret")
    monkeypatch.setenv("ZAI_API_KEY", "zai-secret")
    registry_path = tmp_path / "agents.yaml"
    registry_path.write_text(
        """
agents:
  claude:
    adapter: claude_code_cli
    default_profile: anthropic
    default_model:
      extra_body:
        claude_code:
          env:
            DISABLE_TELEMETRY: "1"
    model_profiles:
      kimi:
        model_id: kimi-k2.6
        api_key_env: KIMI_CODE_API_KEY
        base_url: https://api.kimi.com/coding/
        extra_body:
          claude_code:
            auth_env: ANTHROPIC_API_KEY
            cli_model: sonnet
            set_model_env: false
            env:
              ANTHROPIC_BASE_URL: https://api.kimi.com/coding/
              ENABLE_TOOL_SEARCH: "false"
      glm:
        model_id: glm-4.7
        api_key_env: ZAI_API_KEY
        base_url: https://api.z.ai/api/anthropic
        extra_body:
          claude_code:
            auth_env: ANTHROPIC_AUTH_TOKEN
            cli_model: sonnet
            set_model_env: false
            env:
              ANTHROPIC_BASE_URL: https://api.z.ai/api/anthropic
              API_TIMEOUT_MS: "3000000"
              ANTHROPIC_DEFAULT_HAIKU_MODEL: glm-4.5-air
              ANTHROPIC_DEFAULT_SONNET_MODEL: glm-4.7
              ANTHROPIC_DEFAULT_OPUS_MODEL: glm-4.7
""".strip(),
        encoding="utf-8",
    )

    registry = load_agent_registry(registry_path)
    spec = registry["claude"]

    kimi = resolve_model_config(spec, profile="kimi")
    assert kimi.model_id == "kimi-k2.6"
    assert kimi.api_key == "kimi-secret"
    assert kimi.base_url == "https://api.kimi.com/coding/"
    assert kimi.extra_body == {
        "claude_code": {
            "auth_env": "ANTHROPIC_API_KEY",
            "cli_model": "sonnet",
            "set_model_env": False,
            "env": {
                "DISABLE_TELEMETRY": "1",
                "ANTHROPIC_BASE_URL": "https://api.kimi.com/coding/",
                "ENABLE_TOOL_SEARCH": "false",
            },
        }
    }

    glm = resolve_model_config(spec, profile="glm")
    assert glm.model_id == "glm-4.7"
    assert glm.api_key == "zai-secret"
    assert glm.base_url == "https://api.z.ai/api/anthropic"
    assert glm.extra_body == {
        "claude_code": {
            "auth_env": "ANTHROPIC_AUTH_TOKEN",
            "cli_model": "sonnet",
            "set_model_env": False,
            "env": {
                "DISABLE_TELEMETRY": "1",
                "ANTHROPIC_BASE_URL": "https://api.z.ai/api/anthropic",
                "API_TIMEOUT_MS": "3000000",
                "ANTHROPIC_DEFAULT_HAIKU_MODEL": "glm-4.5-air",
                "ANTHROPIC_DEFAULT_SONNET_MODEL": "glm-4.7",
                "ANTHROPIC_DEFAULT_OPUS_MODEL": "glm-4.7",
            },
        }
    }
    assert missing_required_env(spec, profile="kimi") == []
    assert missing_required_env(spec, profile="glm") == []


def test_paper_model_profiles_are_registered_for_hermes_and_hermes_plus():
    registry = load_agent_registry(Path(__file__).resolve().parents[1] / "configs" / "agents.yaml")
    expected_models = {
        "glm": "glm-5.1",
        "kimi": "kimi-k2.6",
        "deepseek": "deepseek-v4-pro",
        "minimax": "MiniMax-M2.7",
        "openai_gpt54": "gpt-5.4",
    }

    for agent_name in ("hermes", "hermes-plus"):
        spec = registry[agent_name]
        for profile, model_id in expected_models.items():
            model = resolve_model_config(spec, profile=profile)
            assert model.model_id == model_id

        glm = resolve_model_config(spec, profile="glm")
        assert glm.base_url == "https://api.z.ai/api/anthropic"

        kimi = resolve_model_config(spec, profile="kimi")
        assert kimi.base_url == "https://api.kimi.com/coding/v1"


def test_hermes_plus_bootstrap_uses_runtime_python_pip_normalization():
    registry = load_agent_registry(Path(__file__).resolve().parents[1] / "configs" / "agents.yaml")

    assert registry["hermes-plus"].bootstrap_commands == ["pip install -e agents/hermes-plus"]
