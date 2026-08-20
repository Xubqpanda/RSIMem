"""RSIMem request-level usage telemetry tests."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from past_bench.models.trace import ModelCallRecord, ModelCallUsage, TokenUsage
from past_bench.runtime.adapters.hermes import HermesAdapter
from past_bench.trace.reader import read_events
from past_bench.trace.writer import TraceWriter


def test_model_call_usage_round_trips_without_payload_text(tmp_path) -> None:
    record = ModelCallRecord(
        call_id="model-call-0001",
        sequence=1,
        provider="test-provider",
        model="test-model",
        api_mode="codex_responses",
        attempt=2,
        status="error",
        usage=TokenUsage(request_count=1, retry_count=1),
        usage_available=False,
        duration_ms=12.5,
        http_status=503,
        error_category="ServiceUnavailableError",
    )
    path = tmp_path / "trace.jsonl"
    with TraceWriter(path) as writer:
        writer.write_event(ModelCallUsage(trace_id="trace-1", **record.model_dump()))

    events = list(read_events(path))
    assert len(events) == 1
    assert isinstance(events[0], ModelCallUsage)
    assert events[0].usage.cache_read_tokens is None
    serialized = path.read_text(encoding="utf-8")
    assert "request_body" not in serialized
    assert "response_body" not in serialized


def test_hermes_records_canonical_usage_and_redacts_error_text() -> None:
    from run_agent import AIAgent

    agent = object.__new__(AIAgent)
    agent.provider = "openai-codex"
    agent.model = "test-model"
    agent.api_mode = "codex_responses"
    agent.session_id = "session-1"
    agent.usage_component = "agent"
    agent.usage_purpose = "task_execution"
    agent.model_usage_callback = None
    agent.model_call_usage_records = []
    agent._model_request_sequence = 0

    context = agent._begin_model_call(attempt=1)
    response = SimpleNamespace(usage=SimpleNamespace(
        input_tokens=100,
        output_tokens=20,
        input_tokens_details=SimpleNamespace(
            cached_tokens=40,
            cache_creation_tokens=0,
        ),
        output_tokens_details=SimpleNamespace(reasoning_tokens=5),
    ))
    agent._finish_model_call(context, status="success", response=response)
    success = agent.model_call_usage_records[0]
    assert success["usage"] == {
        "input_tokens": 60,
        "output_tokens": 20,
        "cache_read_tokens": 40,
        "cache_write_tokens": 0,
        "reasoning_tokens": 5,
        "request_count": 1,
        "retry_count": 0,
        "usage_complete": True,
    }

    class SecretError(RuntimeError):
        status_code = 503

    context = agent._begin_model_call(attempt=2)
    agent._finish_model_call(
        context,
        status="error",
        error=SecretError("SENSITIVE_ERROR_VALUE_MUST_NOT_BE_RECORDED"),
    )
    failure = agent.model_call_usage_records[1]
    assert failure["http_status"] == 503
    assert failure["error_category"] == "SecretError"
    assert "SENSITIVE_ERROR_VALUE" not in json.dumps(failure)


def test_hermes_adapter_requires_complete_optional_buckets() -> None:
    complete = ModelCallRecord(
        call_id="one",
        sequence=1,
        status="success",
        usage=TokenUsage(cache_read_tokens=10),
        usage_available=True,
    )
    missing = ModelCallRecord(
        call_id="two",
        sequence=2,
        status="success",
        usage=TokenUsage(cache_read_tokens=None),
        usage_available=True,
    )
    assert HermesAdapter._complete_model_call_sum([complete], "cache_read_tokens") == 10
    assert HermesAdapter._complete_model_call_sum([complete, missing], "cache_read_tokens") is None


def test_auxiliary_calls_use_context_local_recorders(monkeypatch) -> None:
    from agent import auxiliary_client

    response = SimpleNamespace(choices=[], usage=None)

    class SyncCompletions:
        def create(self, **kwargs):
            return response

    class AsyncCompletions:
        async def create(self, **kwargs):
            return response

    sync_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SyncCompletions()),
    )
    async_client = SimpleNamespace(
        chat=SimpleNamespace(completions=AsyncCompletions()),
    )
    monkeypatch.setattr(
        auxiliary_client,
        "_resolve_task_provider_model",
        lambda *args: ("custom", "test-model", None, None),
    )
    monkeypatch.setattr(
        auxiliary_client,
        "_get_cached_client",
        lambda *args, **kwargs: (
            async_client if kwargs.get("async_mode") else sync_client,
            "test-model",
        ),
    )
    monkeypatch.setattr(auxiliary_client, "_build_call_kwargs", lambda *args, **kwargs: {})

    sync_evidence = []
    async_evidence = []

    def sync_executor(request, **metadata):
        sync_evidence.append(metadata)
        return request()

    async def async_executor(request, **metadata):
        async_evidence.append(metadata)
        return await request()

    with auxiliary_client.use_request_executors(sync_executor, async_executor):
        assert auxiliary_client.call_llm(task="compression", messages=[]) is response
        assert asyncio.run(
            auxiliary_client.async_call_llm(task="web_extract", messages=[])
        ) is response

    assert sync_evidence[0]["purpose"] == "compression"
    assert async_evidence[0]["purpose"] == "web_extract"
