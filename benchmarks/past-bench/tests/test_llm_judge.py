from __future__ import annotations

import sys
from types import SimpleNamespace

from past_bench.graders.llm_judge import (
    LLMJudge,
    _AnthropicCompletions,
    _anthropic_auth_kwargs,
    _uses_anthropic_sdk,
)


def test_uses_anthropic_sdk_for_anthropic_compatible_endpoints():
    assert _uses_anthropic_sdk("https://api.anthropic.com/v1")
    assert _uses_anthropic_sdk("https://api.minimaxi.com/anthropic")
    assert _uses_anthropic_sdk("https://api.z.ai/api/anthropic")
    assert not _uses_anthropic_sdk("https://api.openai.com/v1")


def test_llm_judge_passes_custom_anthropic_base_url(monkeypatch):
    captured: dict[str, str] = {}

    class FakeAnthropic:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=FakeAnthropic))

    LLMJudge(
        model_id="MiniMax-M2.7",
        api_key="test-key",
        base_url="https://api.minimaxi.com/anthropic",
    )

    assert captured == {
        "api_key": "test-key",
        "base_url": "https://api.minimaxi.com/anthropic",
    }


def test_anthropic_auth_kwargs_uses_api_key_for_first_party_keys():
    assert _anthropic_auth_kwargs("sk-ant-api-123") == {"api_key": "sk-ant-api-123"}
    assert _anthropic_auth_kwargs(
        "sk-cp-example",
        "https://api.minimaxi.com/anthropic",
    ) == {"api_key": "sk-cp-example"}


def test_anthropic_completions_uses_first_text_block():
    class FakeClient:
        class messages:
            @staticmethod
            def create(**kwargs):
                return SimpleNamespace(
                    content=[
                        SimpleNamespace(type="thinking", thinking="..."),
                        SimpleNamespace(type="text", text='{"score": 0.7, "reasoning": "ok"}'),
                    ]
                )

    resp = _AnthropicCompletions(FakeClient()).create(
        model="MiniMax-M2.7",
        messages=[{"role": "user", "content": "hi"}],
    )

    assert resp.choices[0].message.content == '{"score": 0.7, "reasoning": "ok"}'
