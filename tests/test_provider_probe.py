from __future__ import annotations

import json

import pytest

from rsimem.provider_probe import ProviderProbeResult, probe_provider


def test_probe_accepts_content_and_keeps_secret_out_of_result() -> None:
    seen = {}

    def transport(request, timeout):
        seen["url"] = request.full_url
        seen["body"] = json.loads(request.data)
        seen["authorization"] = request.get_header("Authorization")
        return 200, json.dumps({
            "choices": [{"message": {"content": "OK"}}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 1},
        }).encode()

    result = probe_provider(
        "https://provider.example/v1/",
        "fixture-secret-token",
        "fixture-model",
        transport=transport,
    )
    assert result.ok
    assert result.usage_available
    assert seen["url"] == "https://provider.example/v1/chat/completions"
    assert seen["body"]["model"] == "fixture-model"
    assert "fixture-secret-token" not in result.to_json()


def test_probe_rejects_empty_content_and_html_response() -> None:
    empty = probe_provider(
        "https://provider.example/v1",
        "secret",
        "model",
        transport=lambda request, timeout: (
            200,
            b'{"choices":[{"message":{"content":"   "}}],"usage":{}}',
        ),
    )
    assert not empty.ok
    assert empty.error_code == "empty_content"
    assert empty.usage_available

    html = probe_provider(
        "https://provider.example/v1",
        "secret",
        "model",
        transport=lambda request, timeout: (200, b"<html>provider ui</html>"),
    )
    assert not html.ok
    assert html.error_code == "invalid_json"


def test_probe_preserves_http_and_transport_failure_without_body() -> None:
    unavailable = probe_provider(
        "https://provider.example/v1",
        "secret",
        "model",
        transport=lambda request, timeout: (503, b"capacity details"),
    )
    assert not unavailable.ok
    assert unavailable.http_status == 503
    assert unavailable.error_code == "http_error"

    failed = probe_provider(
        "https://provider.example/v1",
        "secret",
        "model",
        transport=lambda request, timeout: (_ for _ in ()).throw(TimeoutError()),
    )
    assert not failed.ok
    assert failed.http_status is None
    assert failed.error_code == "transport_error"


def test_probe_rejects_missing_credential_and_invalid_url() -> None:
    missing = probe_provider("https://provider.example/v1", "", "model")
    assert missing.error_code == "credential_missing"
    invalid = probe_provider("provider.example/v1", "secret", "model")
    assert invalid.error_code == "invalid_base_url"


def test_probe_result_rejects_inconsistent_manual_states() -> None:
    with pytest.raises(ValueError, match="requires a 2xx status"):
        ProviderProbeResult(
            "https://provider.example/v1",
            "model",
            503,
            True,
            False,
        )
    with pytest.raises(ValueError, match="requires an error code"):
        ProviderProbeResult(
            "https://provider.example/v1",
            "model",
            200,
            False,
            False,
        )
