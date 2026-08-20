"""Stateful chat adapter backed by the existing OpenAI-compatible provider."""

from __future__ import annotations

import time

from ...models.message import Message
from ...runner.providers.openai_compat import OpenAICompatProvider
from ..protocol import StartSessionRequest, StepRequest, StepResponse, ToolCallAction
from ..registry import AgentSpec
from .base import RuntimeAdapter


class OpenAICompatChatAdapter(RuntimeAdapter):
    """Runtime adapter that reuses the existing chat provider and tool schema."""

    def __init__(self, spec: AgentSpec, request: StartSessionRequest) -> None:
        super().__init__(spec, request)
        self._messages = [msg.model_copy(deep=True) for msg in request.initial_messages]
        self._tools = [tool.model_copy(deep=True) for tool in request.tools]
        self._provider = OpenAICompatProvider(
            model_id=request.model.model_id,
            api_key=request.model.api_key,
            base_url=request.model.base_url,
            extra_body=request.model.extra_body,
            temperature=request.runtime_config.temperature,
        )

    def step(self, request: StepRequest) -> StepResponse:
        if request.tool_results:
            self._messages.append(
                Message(
                    role="user",
                    content=[block.model_copy(deep=True) for block in request.tool_results],
                )
            )

        started = time.monotonic()
        assistant_message, usage = self._provider.chat(self._messages, tools=self._tools)
        model_time_s = time.monotonic() - started
        self._messages.append(assistant_message)

        tool_calls = [
            ToolCallAction(
                tool_use_id=block.id,
                name=block.name,
                arguments=block.input,
            )
            for block in assistant_message.content
            if block.type == "tool_use"
        ]
        if tool_calls:
            return StepResponse(
                status="acting",
                assistant_message=assistant_message,
                usage=usage,
                tool_calls=tool_calls,
                model_time_s=model_time_s,
            )
        return StepResponse(
            status="finished",
            assistant_message=assistant_message,
            usage=usage,
            final_output=assistant_message.text,
            model_time_s=model_time_s,
        )
