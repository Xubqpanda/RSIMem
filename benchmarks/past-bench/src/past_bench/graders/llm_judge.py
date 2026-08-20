"""LLM-as-judge for subjective communication quality scoring."""

from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from openai import OpenAI
from pydantic import BaseModel


class JudgeResult(BaseModel):
    score: float  # 0.0-1.0
    reasoning: str


_SYSTEM_PROMPT = """\
You are an evaluation judge for an AI assistant.
You will be given a task prompt, a conversation, a summary of actions taken, and a rubric.
Follow the rubric to score the assistant's response on a 0.0-1.0 scale.
Respond with JSON only: {"score": <float>, "reasoning": "<brief explanation>"}
"""


def _uses_anthropic_sdk(base_url: str | None) -> bool:
    """Whether the judge should call the Anthropic Messages API."""
    if not base_url:
        return False

    parsed = urlparse(base_url)
    host = parsed.netloc.lower()
    path = parsed.path.rstrip("/").lower()
    return host.endswith("anthropic.com") or "/anthropic" in path


def _anthropic_auth_kwargs(api_key: str, base_url: str | None = None) -> dict[str, str]:
    """Choose Anthropic auth style for first-party vs compatible gateways.

    First-party Anthropic keys (sk-ant-api*) always use ``api_key``.
    Third-party anthropic-compatible endpoints (GLM, MiniMax) also use
    ``api_key`` (sent as x-api-key header).  Only first-party non-key
    tokens (e.g. OAuth) use ``auth_token``.
    """
    if api_key.startswith("sk-ant-api"):
        return {"api_key": api_key}
    # Third-party endpoints (not api.anthropic.com) use api_key, not auth_token
    if base_url and "api.anthropic.com" not in base_url:
        return {"api_key": api_key}
    return {"auth_token": api_key}


# ---------------------------------------------------------------------------
# Thin wrapper so graders that call judge.client.chat.completions.create(...)
# work transparently with the Anthropic SDK.
# ---------------------------------------------------------------------------

@dataclass
class _AnthropicMessageChoice:
    message: Any = None

@dataclass
class _AnthropicChatResponse:
    choices: list = field(default_factory=list)


class _AnthropicCompletions:
    """Mimics ``openai.chat.completions`` using the Anthropic SDK."""

    def __init__(self, anthropic_client: Any) -> None:
        self._client = anthropic_client

    def create(self, *, model: str, messages: list[dict], **kwargs) -> _AnthropicChatResponse:
        # Split system message from user messages (Anthropic API convention)
        system_text = None
        user_messages = []
        for m in messages:
            if m["role"] == "system":
                system_text = m["content"]
            else:
                user_messages.append(m)

        create_kwargs: dict[str, Any] = {
            "model": model,
            "messages": user_messages,
            "max_tokens": kwargs.get("max_tokens", 4096),
        }
        if system_text:
            create_kwargs["system"] = system_text
        if "temperature" in kwargs:
            create_kwargs["temperature"] = kwargs["temperature"]

        resp = self._client.messages.create(**create_kwargs)

        # Build an OpenAI-shaped response object
        text = "{}"
        for block in resp.content or []:
            block_text = getattr(block, "text", None)
            if block_text:
                text = block_text
                break

        @dataclass
        class _Msg:
            content: str = ""
        choice = _AnthropicMessageChoice(message=_Msg(content=text))
        return _AnthropicChatResponse(choices=[choice])


class _AnthropicChatNamespace:
    """Mimics ``client.chat.completions``."""
    def __init__(self, anthropic_client: Any) -> None:
        self.completions = _AnthropicCompletions(anthropic_client)


class _AnthropicClientShim:
    """Drop-in for ``OpenAI(...)`` that routes to the Anthropic SDK."""
    def __init__(self, api_key: str, base_url: str | None = None) -> None:
        import anthropic

        kwargs: dict[str, Any] = _anthropic_auth_kwargs(api_key, base_url)
        if base_url:
            kwargs["base_url"] = base_url
        self._client = anthropic.Anthropic(**kwargs)
        self.chat = _AnthropicChatNamespace(self._client)


class LLMJudge:
    """Judge communication quality using an LLM via OpenAI-compatible API.

    Anthropic-style endpoints (official Anthropic or third-party ``/anthropic``
    gateways such as MiniMax) use the Anthropic SDK; all other endpoints use
    the OpenAI SDK.
    """

    def __init__(
        self,
        model_id: str = "google/gemini-2.5-flash",
        api_key: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
    ) -> None:
        if _uses_anthropic_sdk(base_url):
            self.client = _AnthropicClientShim(
                api_key=api_key or "",
                base_url=base_url,
            )
        else:
            self.client = OpenAI(api_key=api_key or "dummy", base_url=base_url)
        self.model_id = model_id

    def evaluate(
        self,
        task_prompt: str,
        conversation: str,
        actions_summary: str,
        rubric: str,
    ) -> JudgeResult:
        """Evaluate communication quality and return a JudgeResult."""
        user_msg = (
            f"## Task Prompt\n{task_prompt}\n\n"
            f"## Conversation\n{conversation}\n\n"
            f"## Actions Taken\n{actions_summary}\n\n"
            f"## Rubric\n{rubric}"
        )
        max_retries = 20
        last_exc: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=0.0,
                    max_tokens=8192,
                )
                raw = resp.choices[0].message.content or "{}"
                # Strip markdown code fences if present
                raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
                raw = re.sub(r"\s*```$", "", raw.strip())
                m = re.search(r'\{[^{}]*\}', raw)
                if m:
                    raw = m.group(0)
                try:
                    parsed = json.loads(raw)
                    score, reasoning = parsed["score"], parsed["reasoning"]
                except json.JSONDecodeError:
                    # Fallback: extract score and reasoning directly
                    score_m = re.search(r'"score"\s*:\s*([0-9.]+)', raw)
                    reason_m = re.search(r'"reasoning"\s*:\s*"((?:[^"\\]|\\.)*)"', raw)
                    if score_m:
                        parsed = {
                            "score": float(score_m.group(1)),
                            "reasoning": reason_m.group(1) if reason_m else "",
                        }
                    else:
                        raise json.JSONDecodeError("No score found in raw", raw, 0)

                return JudgeResult(
                    score=max(0.0, min(1.0, float(score))),
                    reasoning=str(reasoning),
                )
            except Exception as exc:
                last_exc = exc
                status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
                delay = min(2 ** (attempt + 1), 8) + random.uniform(0, 1)
                print(f"[judge-retry] ({status or type(exc).__name__}), "
                      f"attempt {attempt + 1}/{max_retries}, waiting {delay:.1f}s ...")
                time.sleep(delay)
