import runpy
from unittest.mock import patch

from fastapi.testclient import TestClient

from mock_services.x import server


def test_get_user_profile_accepts_tweet_author_id_in_username_field() -> None:
    server._users = {
        "U_RAVING": {
            "username": "superfan_liz",
            "followers": 3200,
            "verified": False,
        }
    }

    response = TestClient(server.app).post(
        "/x/action",
        json={"action": "getUserProfile", "username": "U_RAVING"},
    )

    assert response.json()["user"]["id"] == "U_RAVING"


def test_server_entrypoint_honors_common_port_offset_env(monkeypatch) -> None:
    monkeypatch.setenv("PORT", "34200")
    monkeypatch.setenv("X_PORT", "3300")
    with patch("uvicorn.run") as run:
        runpy.run_path(str(server.__file__), run_name="__main__")
    assert run.call_args.kwargs["port"] == 34200
