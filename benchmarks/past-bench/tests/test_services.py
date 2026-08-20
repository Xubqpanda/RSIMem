from __future__ import annotations

from past_bench.models.task import ServiceDef
from past_bench.runner.services import ServiceManager


class _FakeResponse:
    def __init__(self, status_code: int, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class _FakeClient:
    def __init__(self, response: _FakeResponse):
        self._response = response

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get(self, *_args, **_kwargs):
        return self._response

    def post(self, *_args, **_kwargs):
        return self._response


def _config_service() -> ServiceDef:
    return ServiceDef(
        name="config",
        command="python mock_services/config/server.py",
        port=9210,
        health_check="http://localhost:9210/config/integrations",
        health_check_method="POST",
        ready_timeout=10,
        reset_endpoint="http://localhost:9210/config/reset",
    )


def test_service_health_check_rejects_incompatible_config_payload(monkeypatch):
    manager = ServiceManager([_config_service()])
    fake_response = _FakeResponse(
        200,
        {"text": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.invalid.mock.token"},
    )
    monkeypatch.setattr("httpx.Client", lambda **_kwargs: _FakeClient(fake_response))

    assert manager._is_healthy(_config_service()) is False


def test_service_health_check_accepts_mock_config_payload(monkeypatch):
    manager = ServiceManager([_config_service()])
    fake_response = _FakeResponse(200, {"integrations": [], "total": 0})
    monkeypatch.setattr("httpx.Client", lambda **_kwargs: _FakeClient(fake_response))

    assert manager._is_healthy(_config_service()) is True


def test_service_health_check_accepts_fastapi_validation_error(monkeypatch):
    svc = ServiceDef(
        name="kb",
        command="python mock_services/kb/server.py",
        port=9201,
        health_check="http://localhost:9201/kb/search",
        health_check_method="POST",
        ready_timeout=10,
        reset_endpoint="http://localhost:9201/kb/reset",
    )
    manager = ServiceManager([svc])
    fake_response = _FakeResponse(422, {"detail": [{"msg": "Field required"}]})
    monkeypatch.setattr("httpx.Client", lambda **_kwargs: _FakeClient(fake_response))

    assert manager._is_healthy(svc) is True


def test_service_manager_spawns_when_port_is_occupied_by_wrong_service(monkeypatch):
    svc = _config_service()
    manager = ServiceManager([svc])
    fake_response = _FakeResponse(
        200,
        {"text": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.invalid.mock.token"},
    )
    monkeypatch.setattr("httpx.Client", lambda **_kwargs: _FakeClient(fake_response))

    spawned: list[str] = []
    monkeypatch.setattr(manager, "_spawn", lambda service: spawned.append(service.name))

    manager.__enter__()
    try:
        assert spawned == ["config"]
    finally:
        manager.__exit__(None, None, None)
