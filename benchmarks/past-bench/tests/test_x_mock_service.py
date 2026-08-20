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
