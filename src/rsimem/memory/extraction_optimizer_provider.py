"""OpenAI-compatible provider adapter for the frozen extraction optimizer."""

from __future__ import annotations

import time
from typing import Any, Callable
from urllib.parse import urlparse

from ..lifecycle import RawResourceUsage
from .extraction_optimizer_contracts import (
    EXTRACTION_OPTIMIZER_TIMEOUT_SECONDS,
    EXTRACTION_OPTIMIZER_OUTPUT_SCHEMA,
    ExtractionOptimizerCompletion,
    ExtractionOptimizerConfig,
    ExtractionOptimizerRequest,
    FROZEN_EXTRACTION_OPTIMIZER_CONFIG,
)
from .prompt_components import content_digest, text_digest


class OpenAICompatibleExtractionOptimizerClient:
    # A real provider call is a formal proposal boundary and therefore must
    # be paired with an owner-controlled revocation registry by the optimizer.
    requires_revocation_registry = True
    requires_process_signal_gate = True

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        sdk_client: object | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("optimizer provider API key is missing")
        parsed = urlparse(base_url)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("optimizer provider base URL must be absolute HTTPS")
        self._clock = clock
        if sdk_client is None:
            from openai import OpenAI

            sdk_client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=EXTRACTION_OPTIMIZER_TIMEOUT_SECONDS,
                max_retries=0,
            )
        self._client = sdk_client

    def complete(
        self,
        request: ExtractionOptimizerRequest,
        config: ExtractionOptimizerConfig = FROZEN_EXTRACTION_OPTIMIZER_CONFIG,
    ) -> ExtractionOptimizerCompletion:
        if request.provider_eligible is not True:
            raise ValueError("optimizer gate request cannot reach the provider")
        if request.optimizer_config_digest != config.config_digest:
            raise ValueError("optimizer provider config differs from request")
        started = self._clock()
        response = self._client.chat.completions.create(
            model=config.model_id,
            messages=[
                {"role": "system", "content": request.system_instruction},
                {"role": "user", "content": request.input_json},
            ],
            temperature=float(config.temperature),
            max_tokens=config.max_output_tokens,
            timeout=config.timeout_seconds,
            # ``json_object`` only guarantees syntactic JSON and the Luna
            # endpoint has been observed to omit required contract fields in
            # that mode.  Send the frozen schema at the provider boundary so
            # the response is constrained before our strict parser sees it.
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "extraction_optimizer_result",
                    "strict": True,
                    "schema": EXTRACTION_OPTIMIZER_OUTPUT_SCHEMA,
                },
            },
        )
        duration_ms = max(0, round((self._clock() - started) * 1_000))
        output = self._output_text(response)
        usage = self._usage(getattr(response, "usage", None), duration_ms)
        identity = {
            "request_id": request.request_id,
            "provider_response_id_digest": text_digest(
                str(getattr(response, "id", ""))
            ),
            "output_digest": text_digest(output),
        }
        return ExtractionOptimizerCompletion(
            f"optimizer-completion.{content_digest(identity)[:40]}",
            request.request_id,
            output,
            usage,
        )

    @staticmethod
    def _output_text(response: object) -> str:
        choices = getattr(response, "choices", None)
        if not isinstance(choices, (list, tuple)) or len(choices) != 1:
            raise ValueError("optimizer provider returned an invalid choice count")
        content = getattr(getattr(choices[0], "message", None), "content", None)
        if not isinstance(content, str) or not content.strip():
            raise ValueError("optimizer provider returned empty output")
        return content

    @classmethod
    def _usage(cls, value: object, duration_ms: int) -> RawResourceUsage:
        input_tokens = cls._optional_int(value, "prompt_tokens", "input_tokens")
        output_tokens = cls._optional_int(
            value,
            "completion_tokens",
            "output_tokens",
        )
        input_details = cls._first_attr(
            value,
            "prompt_tokens_details",
            "input_tokens_details",
        )
        output_details = cls._first_attr(
            value,
            "completion_tokens_details",
            "output_tokens_details",
        )
        cache_read = cls._optional_int(input_details, "cached_tokens")
        cache_write = cls._optional_int(
            input_details,
            "cache_write_tokens",
            "cache_creation_input_tokens",
        )
        reasoning = cls._optional_int(output_details, "reasoning_tokens")
        return RawResourceUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
            reasoning_tokens=reasoning,
            model_requests=1,
            retry_count=0,
            duration_ms=duration_ms,
        )

    @staticmethod
    def _first_attr(value: object, *names: str) -> object | None:
        for name in names:
            candidate = getattr(value, name, None)
            if candidate is not None:
                return candidate
        return None

    @classmethod
    def _optional_int(cls, value: object, *names: str) -> int | None:
        candidate = cls._first_attr(value, *names)
        if candidate is None:
            return None
        if type(candidate) is not int or candidate < 0:
            raise ValueError("optimizer provider usage value is invalid")
        return candidate
