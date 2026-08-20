"""JSONL trace event types for agent evaluation.

Modified by RSIMem to retain request-level model usage evidence.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field

from .message import Message


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    reasoning_tokens: int | None = None
    request_count: int | None = None
    retry_count: int | None = None
    usage_complete: bool = False


class ModelCallRecord(BaseModel):
    """Sanitized evidence for one physical model API request."""

    call_id: str
    sequence: int
    component: str = "agent"
    purpose: str = "task_execution"
    provider: str | None = None
    model: str | None = None
    api_mode: str | None = None
    attempt: int = 1
    status: Literal["success", "error", "invalid_response", "interrupted"]
    usage: TokenUsage = Field(default_factory=TokenUsage)
    usage_available: bool = False
    duration_ms: float = 0.0
    http_status: int | None = None
    error_category: str | None = None


class ModelCallUsage(ModelCallRecord):
    """Trace event emitted for one physical model API request."""

    type: Literal["model_call_usage"] = "model_call_usage"
    trace_id: str
    timestamp: str = Field(default_factory=_now)


class DimensionScores(BaseModel):
    completion: float = 0.0
    robustness: float = 0.0
    communication: float = 0.0
    safety: float = 1.0  # binary gate: 0 or 1
    efficiency_turns: int = 0
    efficiency_tokens: int = 0
    efficiency_wall_time_s: float = 0.0


# --- JSONL Event Types ---


class TraceStart(BaseModel):
    type: Literal["trace_start"] = "trace_start"
    trace_id: str
    task_id: str
    model: str
    persona: str = "default"
    runtime_temperature: float | None = None
    timestamp: str = Field(default_factory=_now)


class TraceMessage(BaseModel):
    type: Literal["message"] = "message"
    trace_id: str
    message: Message
    usage: TokenUsage = Field(default_factory=TokenUsage)
    timestamp: str = Field(default_factory=_now)


class ToolDispatch(BaseModel):
    type: Literal["tool_dispatch"] = "tool_dispatch"
    trace_id: str
    tool_use_id: str
    tool_name: str
    endpoint_url: str
    request_body: dict[str, Any] = Field(default_factory=dict)
    response_status: int = 200
    response_body: Any = None
    latency_ms: float = 0.0
    timestamp: str = Field(default_factory=_now)


class RuntimeRequest(BaseModel):
    type: Literal["runtime_request"] = "runtime_request"
    trace_id: str
    session_id: str
    step_id: int
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=_now)


class RuntimeResponse(BaseModel):
    type: Literal["runtime_response"] = "runtime_response"
    trace_id: str
    session_id: str
    step_id: int
    status: str
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=_now)


class AuditSnapshot(BaseModel):
    type: Literal["audit_snapshot"] = "audit_snapshot"
    trace_id: str
    service_name: str
    audit_url: str
    audit_data: dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=_now)


class MediaLoad(BaseModel):
    type: Literal["media_load"] = "media_load"
    trace_id: str
    modality: Literal["image", "audio", "video", "document"]
    source_path: str
    mime_type: str
    size_bytes: int
    sha256: str
    status: Literal["loaded", "skipped", "error"] = "loaded"
    note: str = ""
    timestamp: str = Field(default_factory=_now)


class TraceEnd(BaseModel):
    type: Literal["trace_end"] = "trace_end"
    trace_id: str
    total_turns: int = 0
    model_input_tokens: int = 0
    model_output_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    reasoning_tokens: int | None = None
    model_request_count: int | None = None
    model_retry_count: int | None = None
    model_usage_complete: bool = False
    model_time_s: float = 0.0
    tool_time_s: float = 0.0
    other_time_s: float = 0.0
    wall_time_s: float = 0.0
    scores: DimensionScores = Field(default_factory=DimensionScores)
    task_score: float = 0.0
    passed: bool = False
    failure_modes: list[str] = Field(default_factory=list)
    timestamp: str = Field(default_factory=_now)


class GradingResult(BaseModel):
    type: Literal["grading_result"] = "grading_result"
    trace_id: str
    task_id: str
    scores: DimensionScores = Field(default_factory=DimensionScores)
    task_score: float = 0.0
    passed: bool = False
    failure_modes: list[str] = Field(default_factory=list)
    timestamp: str = Field(default_factory=_now)


TraceEvent = Annotated[
    Union[
        TraceStart,
        TraceMessage,
        ModelCallUsage,
        ToolDispatch,
        RuntimeRequest,
        RuntimeResponse,
        AuditSnapshot,
        MediaLoad,
        TraceEnd,
        GradingResult,
    ],
    Field(discriminator="type"),
]
