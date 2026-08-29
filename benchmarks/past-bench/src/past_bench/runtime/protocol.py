"""Pydantic models for the decoupled agent runtime protocol.

Modified by RSIMem to transport request-level model usage evidence.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from ..models.content import ToolResultBlock
from ..models.message import Message
from ..models.tool import ToolEndpoint, ToolSpec
from ..models.trace import ModelCallRecord, TokenUsage


class RuntimeModelConfig(BaseModel):
    """Resolved model configuration passed from bench to runtime."""

    model_id: str
    api_key: str | None = None
    base_url: str | None = None
    extra_body: dict[str, Any] | None = None


class RuntimeConfigPayload(BaseModel):
    """Execution knobs owned by the bench."""

    temperature: float = 0.0
    max_output_tokens: int = 4096
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolCallAction(BaseModel):
    """A tool call emitted by the runtime."""

    tool_use_id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class StartSessionRequest(BaseModel):
    """Start a new agent session."""

    session_id: str
    agent_name: str
    task_id: str
    task_name: str
    max_turns: int
    timeout_seconds: int
    initial_messages: list[Message] = Field(default_factory=list)
    tools: list[ToolSpec] = Field(default_factory=list)
    tool_endpoints: list[ToolEndpoint] = Field(default_factory=list)
    model: RuntimeModelConfig
    runtime_config: RuntimeConfigPayload = Field(default_factory=RuntimeConfigPayload)


class StartSessionResponse(BaseModel):
    """Acknowledgement for a newly-created session."""

    session_id: str
    agent_name: str
    adapter: str


class StepRequest(BaseModel):
    """Send the latest tool results back to the runtime and request another step."""

    session_id: str
    step_id: int
    tool_results: list[ToolResultBlock] = Field(default_factory=list)


class StepResponse(BaseModel):
    """Runtime response for a single model step."""

    status: Literal["acting", "finished", "error"]
    assistant_message: Message | None = None
    usage: TokenUsage = Field(default_factory=TokenUsage)
    model_calls: list[ModelCallRecord] = Field(default_factory=list)
    tool_calls: list[ToolCallAction] = Field(default_factory=list)
    final_output: str | None = None
    error: str | None = None
    model_time_s: float = 0.0
    # Content-free RSIMem process evidence identity.  This is deliberately
    # separate from benchmark/evaluation score fields, which are owned by the
    # runner and reporter rather than the policy learner.
    process_feedback_event_ids: list[str] = Field(default_factory=list)
    process_feedback_digest: str | None = None


class InterruptRequest(BaseModel):
    """Interrupt an in-flight session."""

    session_id: str
    reason: str = ""


class CloseSessionRequest(BaseModel):
    """Terminate a session and release runtime state."""

    session_id: str
    reason: str = ""


class HealthResponse(BaseModel):
    """Health probe payload."""

    status: Literal["ok"] = "ok"
    agents: list[str] = Field(default_factory=list)


class BootstrapRequest(BaseModel):
    """Ensure an agent runtime is installed and ready."""

    agent_name: str
    force: bool = False


class BootstrapResponse(BaseModel):
    """Result of agent bootstrap inside the runtime."""

    agent_name: str
    installed: bool = True
    already_present: bool = False
    commands_run: list[str] = Field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
