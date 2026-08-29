"""Small, secret-free completion probe for OpenAI-compatible providers.

The probe is deliberately outside benchmark execution and accounting.  It
checks that a configured endpoint can return non-empty assistant content, but
does not treat a successful HTTP response as evidence of complete usage.  The
raw response body and credential never cross the result boundary.
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, Sequence
from urllib.parse import urlparse


PROVIDER_PROBE_SCHEMA_VERSION = 1


Transport = Callable[[urllib.request.Request, float], tuple[int, bytes]]


def _urllib_transport(request: urllib.request.Request, timeout: float) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return int(response.status), response.read()
    except urllib.error.HTTPError as exc:
        # Keep the body private; callers only need the status classification.
        try:
            exc.read()
        except OSError:
            pass
        return int(exc.code), b""


def _endpoint(base_url: str) -> str:
    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError("provider base URL must not be empty")
    normalized = base_url.strip().rstrip("/")
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("provider base URL must be an HTTP(S) URL")
    return normalized + "/chat/completions"


@dataclass(frozen=True, slots=True)
class ProviderProbeResult:
    """Content-free outcome of one provider probe."""

    base_url: str
    model: str
    http_status: int | None
    content_available: bool
    usage_available: bool
    error_code: str | None = None
    schema_version: int = PROVIDER_PROBE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PROVIDER_PROBE_SCHEMA_VERSION:
            raise ValueError("unsupported provider probe schema version")
        if not isinstance(self.base_url, str) or not self.base_url.strip():
            raise ValueError("provider probe base URL must not be empty")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("provider probe model must not be empty")
        if self.http_status is not None and (
            type(self.http_status) is not int or not 100 <= self.http_status <= 599
        ):
            raise ValueError("provider probe HTTP status is invalid")
        if type(self.content_available) is not bool:
            raise ValueError("provider probe content flag must be bool")
        if type(self.usage_available) is not bool:
            raise ValueError("provider probe usage flag must be bool")
        if self.error_code is not None and (
            not isinstance(self.error_code, str) or not self.error_code.strip()
        ):
            raise ValueError("provider probe error code must be non-empty")
        if self.content_available and self.error_code is not None:
            raise ValueError("successful provider probe cannot carry an error")
        if self.content_available and (
            self.http_status is None or not 200 <= self.http_status < 300
        ):
            raise ValueError("successful provider probe requires a 2xx status")
        if not self.content_available and self.error_code is None:
            raise ValueError("failed provider probe requires an error code")

    @property
    def ok(self) -> bool:
        return self.content_available

    def payload(self) -> dict[str, object]:
        return {
            "schemaVersion": self.schema_version,
            "baseUrl": self.base_url,
            "model": self.model,
            "httpStatus": self.http_status,
            "contentAvailable": self.content_available,
            "usageAvailable": self.usage_available,
            "errorCode": self.error_code,
            "ok": self.ok,
        }

    def to_json(self) -> str:
        return json.dumps(self.payload(), ensure_ascii=True, sort_keys=True)


def probe_provider(
    base_url: str,
    api_key: str,
    model: str,
    *,
    timeout_seconds: float = 20.0,
    transport: Transport = _urllib_transport,
) -> ProviderProbeResult:
    """Probe one completion endpoint without exposing response content."""

    if not isinstance(api_key, str) or not api_key.strip():
        return ProviderProbeResult(
            base_url=base_url,
            model=model,
            http_status=None,
            content_available=False,
            usage_available=False,
            error_code="credential_missing",
        )
    if not isinstance(timeout_seconds, (int, float)) or isinstance(timeout_seconds, bool) or timeout_seconds <= 0:
        raise ValueError("provider probe timeout must be positive")
    try:
        endpoint = _endpoint(base_url)
    except ValueError:
        return ProviderProbeResult(
            base_url=base_url,
            model=model,
            http_status=None,
            content_available=False,
            usage_available=False,
            error_code="invalid_base_url",
        )
    if not isinstance(model, str) or not model.strip():
        raise ValueError("provider probe model must not be empty")
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": "Reply with exactly OK."}],
            "temperature": 0,
            "max_tokens": 16,
            "stream": False,
        },
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        status, response_body = transport(request, float(timeout_seconds))
    except (OSError, TimeoutError):
        return ProviderProbeResult(
            base_url=base_url,
            model=model,
            http_status=None,
            content_available=False,
            usage_available=False,
            error_code="transport_error",
        )
    if type(status) is not int or not 100 <= status <= 599:
        raise ValueError("provider probe transport returned an invalid HTTP status")
    if status < 200 or status >= 300:
        return ProviderProbeResult(
            base_url=base_url,
            model=model,
            http_status=status,
            content_available=False,
            usage_available=False,
            error_code="http_error",
        )
    try:
        decoded = json.loads(response_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return ProviderProbeResult(
            base_url=base_url,
            model=model,
            http_status=status,
            content_available=False,
            usage_available=False,
            error_code="invalid_json",
        )
    if not isinstance(decoded, dict):
        error_code = "malformed_response"
        usage_available = False
    else:
        choices = decoded.get("choices")
        usage = decoded.get("usage")
        usage_available = isinstance(usage, dict)
        message = choices[0].get("message") if isinstance(choices, list) and choices else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            error_code = "malformed_content"
        elif not content.strip():
            error_code = "empty_content"
        else:
            return ProviderProbeResult(
                base_url=base_url,
                model=model,
                http_status=status,
                content_available=True,
                usage_available=usage_available,
            )
    return ProviderProbeResult(
        base_url=base_url,
        model=model,
        http_status=status,
        content_available=False,
        usage_available=usage_available,
        error_code=error_code,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    args = parser.parse_args(argv)
    import os

    result = probe_provider(
        args.base_url,
        os.environ.get(args.api_key_env, ""),
        args.model,
        timeout_seconds=args.timeout_seconds,
    )
    print(result.to_json())
    return 0 if result.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["ProviderProbeResult", "probe_provider", "main"]
