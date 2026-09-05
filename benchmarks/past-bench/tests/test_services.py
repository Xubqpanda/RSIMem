from __future__ import annotations

import socket
import threading

import httpx
import pytest

from mock_services._base import fixture_identity
from past_bench.models.task import ServiceDef
from past_bench.runner.services import ServiceIdentityError, ServiceManager


class _FakeResponse:
    def __init__(self, status_code: int, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class _FakeClient:
    def __init__(self, response: _FakeResponse | dict[str, _FakeResponse]):
        self._response = response

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def _resolve(self, url):
        if isinstance(self._response, dict):
            return self._response[url]
        return self._response

    def get(self, url, **_kwargs):
        return self._resolve(url)

    def post(self, url, **_kwargs):
        return self._resolve(url)


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


def test_service_manager_rejects_port_occupied_by_wrong_service(monkeypatch):
    svc = _config_service()
    manager = ServiceManager([svc])
    fake_response = _FakeResponse(
        200,
        {"text": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.invalid.mock.token"},
    )
    monkeypatch.setattr("httpx.Client", lambda **_kwargs: _FakeClient(fake_response))

    monkeypatch.setattr(manager, "_identity_status", lambda _service: "mismatch")
    with pytest.raises(ServiceIdentityError, match="different fixture identity"):
        manager.__enter__()


def test_fixture_identity_changes_with_fixture_content(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text('[{"note_id":"one"}]', encoding="utf-8")
    second.write_text('[{"note_id":"two"}]', encoding="utf-8")
    one = fixture_identity(
        service_name="notes",
        environment={"NOTES_FIXTURES": str(first)},
    )
    two = fixture_identity(
        service_name="notes",
        environment={"NOTES_FIXTURES": str(second)},
    )
    assert one["schema"] == "past-bench-service-identity-v1"
    assert one["service"] == "notes"
    assert one["fixture_digest"] != two["fixture_digest"]


def test_manager_identity_matches_exact_fixture(tmp_path, monkeypatch):
    fixture = tmp_path / "notes.json"
    fixture.write_text('[{"note_id":"one"}]', encoding="utf-8")
    svc = ServiceDef(
        name="notes",
        command="python mock_services/notes/server.py",
        port=9105,
        health_check="http://localhost:9105/notes/list",
        env={"NOTES_FIXTURES": str(fixture)},
    )
    manager = ServiceManager([svc])
    expected = manager._expected_identity(svc)
    response = _FakeResponse(200, expected)
    monkeypatch.setattr("httpx.Client", lambda **_kwargs: _FakeClient(response))
    assert manager._identity_status(svc) == "match"

    fixture.write_text('[{"note_id":"changed"}]', encoding="utf-8")
    assert manager._identity_status(svc) == "mismatch"


def test_manager_starts_only_when_port_is_unreachable(monkeypatch):
    svc = _config_service()
    manager = ServiceManager([svc])
    monkeypatch.setattr(manager, "_identity_status", lambda _service: "unreachable")
    monkeypatch.setattr(manager, "_port_is_occupied", lambda _port: False)
    spawned: list[str] = []
    monkeypatch.setattr(manager, "_spawn", lambda service: spawned.append(service.name))
    monkeypatch.setattr(manager, "reset_all", lambda: None)
    monkeypatch.setattr(manager, "capture_identities", lambda: ())
    manager.__enter__()
    try:
        assert spawned == ["config"]
    finally:
        manager.__exit__(None, None, None)


def test_reset_failure_is_fatal(monkeypatch):
    svc = _config_service()
    manager = ServiceManager([svc])
    monkeypatch.setattr(
        "httpx.Client",
        lambda **_kwargs: _FakeClient(_FakeResponse(500, {"error": "reset failed"})),
    )
    with pytest.raises(ServiceIdentityError, match="reset returned HTTP 500"):
        manager.reset_all()


def test_enter_rejects_identity_drift_after_reset(monkeypatch):
    svc = _config_service()
    manager = ServiceManager([svc])
    monkeypatch.setattr(manager, "_identity_status", lambda _service: "match")
    monkeypatch.setattr(manager, "_is_healthy", lambda _service: True)
    monkeypatch.setattr(manager, "reset_all", lambda: None)
    monkeypatch.setattr(manager, "_identity_payload", lambda _service: {
        "schema": "past-bench-service-identity-v1",
        "service": "config",
        "fixture_digest": "0" * 64,
    })
    with pytest.raises(ServiceIdentityError, match="unavailable or changed"):
        manager.__enter__()


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def test_sequential_notes_fixtures_are_isolated_and_process_is_stopped(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(
        '[{"note_id":"FIRST","title":"one","created_at":"2026-01-01T00:00:00Z",'
        '"participants":[],"duration_minutes":1,"status":"active","tags":[]}]',
        encoding="utf-8",
    )
    second.write_text(
        '[{"note_id":"SECOND","title":"two","created_at":"2026-01-01T00:00:00Z",'
        '"participants":[],"duration_minutes":1,"status":"active","tags":[]}]',
        encoding="utf-8",
    )
    port = _free_port()

    def service(fixture):
        return ServiceDef(
            name="notes",
            command="python mock_services/notes/server.py",
            port=port,
            health_check=f"http://localhost:{port}/notes/list",
            health_check_method="POST",
            reset_endpoint=f"http://localhost:{port}/notes/reset",
            env={"NOTES_FIXTURES": str(fixture), "PORT": str(port)},
        )

    identities = []
    for fixture, expected in ((first, "FIRST"), (second, "SECOND")):
        svc = service(fixture)
        with ServiceManager([svc]) as manager:
            identities.append(manager._expected_identity(svc)["fixture_digest"])
            assert manager.verified_identities == ({
                "schema": "past-bench-verified-service-identity-v1",
                "service": "notes",
                "port": port,
                "fixture_digest": identities[-1],
            },)
            response = httpx.post(svc.health_check, json={}, timeout=3.0)
            assert [item["note_id"] for item in response.json()["notes"]] == [expected]
        assert not ServiceManager._port_is_occupied(port)

    assert identities[0] != identities[1]


def test_concurrent_notes_fixtures_use_distinct_ports_and_identities(tmp_path):
    fixture_data = {
        "FIRST": tmp_path / "first.json",
        "SECOND": tmp_path / "second.json",
    }
    for note_id, path in fixture_data.items():
        path.write_text(
            '[{"note_id":"' + note_id + '","title":"fixture","created_at":'
            '"2026-01-01T00:00:00Z","participants":[],"duration_minutes":1,'
            '"status":"active","tags":[]}]',
            encoding="utf-8",
        )
    ports = (_free_port(), _free_port())
    while ports[0] == ports[1]:
        ports = (ports[0], _free_port())
    barrier = threading.Barrier(2)
    observations: dict[str, tuple[str, str]] = {}

    def worker(note_id: str, fixture, port: int) -> None:
        svc = ServiceDef(
            name="notes", command="python mock_services/notes/server.py", port=port,
            health_check=f"http://localhost:{port}/notes/list",
            health_check_method="POST", reset_endpoint=f"http://localhost:{port}/notes/reset",
            env={"NOTES_FIXTURES": str(fixture), "PORT": str(port)},
        )
        with ServiceManager([svc]) as manager:
            barrier.wait(timeout=5)
            visible = httpx.post(svc.health_check, json={}, timeout=3.0).json()["notes"]
            identity = httpx.get(
                f"http://localhost:{port}/_past_bench/identity", timeout=3.0
            ).json()
            assert identity == manager._expected_identity(svc)
            observations[note_id] = (visible[0]["note_id"], identity["fixture_digest"])

    threads = [
        threading.Thread(target=worker, args=(note_id, fixture_data[note_id], port))
        for note_id, port in zip(("FIRST", "SECOND"), ports, strict=True)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=15)
        assert not thread.is_alive()

    assert observations["FIRST"][0] == "FIRST"
    assert observations["SECOND"][0] == "SECOND"
    assert observations["FIRST"][1] != observations["SECOND"][1]
    assert all(not ServiceManager._port_is_occupied(port) for port in ports)
