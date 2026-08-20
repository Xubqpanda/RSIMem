"""Mock X (Twitter) API service for agent evaluation (FastAPI on port 3300).

Supports timeline reads, mentions, DMs, tweets, retweets, likes, replies,
and analytics via a unified ``x_action`` tool endpoint.
Fixture data is loaded from a JSON file specified by X_FIXTURES env var.
"""

from __future__ import annotations

import json
import os
import copy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from fastapi import FastAPI, Request

app = FastAPI(title="Mock X API")

from mock_services._base import add_error_injection
add_error_injection(app)

FIXTURES_PATH = Path(os.environ.get(
    "X_FIXTURES",
    str(Path(__file__).resolve().parent / "default_fixtures.json"),
))

# In-memory state
_users: dict[str, dict] = {}
_tweets: dict[str, dict] = {}          # tweet_id -> tweet object
_timeline: list[str] = []              # ordered tweet_ids for home timeline
_mentions: list[str] = []              # tweet_ids that mention @agent
_dms: list[dict] = []                  # direct messages
_audit_log: list[dict] = []


def _load_fixtures() -> None:
    global _users, _tweets, _timeline, _mentions, _dms
    with open(FIXTURES_PATH) as f:
        data = json.load(f)
    _users = data.get("users", {})
    _tweets = data.get("tweets", {})
    _timeline = data.get("timeline", [])
    _mentions = data.get("mentions", [])
    _dms = data.get("dms", [])


_load_fixtures()


def _log(method: str, params: dict, response: Any) -> None:
    _audit_log.append({
        "method": method,
        "params": params,
        "response": response,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Action dispatcher ──────────────────────────────────────────────────────────

@app.post("/x/action")
async def x_action(request: Request):
    body = await request.json()
    action = body.get("action", "")

    # ── Read actions ────────────────────────────────────────────────────────── #

    if action == "getTimeline":
        count = int(body.get("count", 20))
        tweets = [copy.deepcopy(_tweets[tid]) for tid in _timeline[:count] if tid in _tweets]
        result = {"ok": True, "tweets": tweets}
        _log("timelines.homeTimeline", {"count": count}, result)
        return result

    elif action == "getMentions":
        count = int(body.get("count", 20))
        tweets = [copy.deepcopy(_tweets[tid]) for tid in _mentions[:count] if tid in _tweets]
        result = {"ok": True, "tweets": tweets}
        _log("statuses.mentionsTimeline", {"count": count}, result)
        return result

    elif action == "getTweet":
        tweet_id = body.get("tweetId", "")
        tweet = copy.deepcopy(_tweets.get(tweet_id))
        if tweet:
            result = {"ok": True, "tweet": tweet}
        else:
            result = {"ok": False, "error": "tweet_not_found"}
        _log("statuses.show", {"id": tweet_id}, result)
        return result

    elif action == "getThread":
        tweet_id = body.get("tweetId", "")
        root = _tweets.get(tweet_id)
        thread = []
        if root:
            thread = [copy.deepcopy(root)]
            # Collect replies in order
            for tid, tw in _tweets.items():
                if tw.get("reply_to") == tweet_id:
                    thread.append(copy.deepcopy(tw))
        result = {"ok": True, "thread": thread}
        _log("statuses.thread", {"id": tweet_id}, result)
        return result

    elif action == "getDMs":
        count = int(body.get("count", 20))
        result = {"ok": True, "dms": copy.deepcopy(_dms[:count])}
        _log("directMessages.list", {"count": count}, result)
        return result

    elif action == "searchTweets":
        query = body.get("query", "").lower()
        count = int(body.get("count", 20))
        matches = [
            copy.deepcopy(tw) for tw in _tweets.values()
            if query in tw.get("text", "").lower()
        ][:count]
        result = {"ok": True, "tweets": matches}
        _log("search.tweets", {"q": query, "count": count}, result)
        return result

    elif action == "getAnalytics":
        tweet_id = body.get("tweetId", "")
        period = body.get("period", "7d")
        tweet = _tweets.get(tweet_id)
        if tweet and "analytics" in tweet:
            result = {"ok": True, "tweet_id": tweet_id, "period": period, "analytics": tweet["analytics"]}
        else:
            # Return aggregate analytics
            analytics = {
                "impressions": sum(t.get("analytics", {}).get("impressions", 0) for t in _tweets.values()),
                "engagements": sum(t.get("analytics", {}).get("engagements", 0) for t in _tweets.values()),
                "likes": sum(t.get("metrics", {}).get("like_count", 0) for t in _tweets.values()),
                "retweets": sum(t.get("metrics", {}).get("retweet_count", 0) for t in _tweets.values()),
                "replies": sum(t.get("metrics", {}).get("reply_count", 0) for t in _tweets.values()),
                "period": period,
            }
            result = {"ok": True, "aggregate": analytics}
        _log("analytics.summary", {"tweet_id": tweet_id, "period": period}, result)
        return result

    elif action == "getUserProfile":
        username = body.get("username", "")
        user_id = body.get("userId", "")
        user = None
        for uid, u in _users.items():
            if u.get("username") == username or uid == user_id or uid == username:
                user = copy.deepcopy(u)
                user["id"] = uid
                break
        if user:
            result = {"ok": True, "user": user}
        else:
            result = {"ok": False, "error": "user_not_found"}
        _log("users.show", {"username": username}, result)
        return result

    # ── Write actions ───────────────────────────────────────────────────────── #

    elif action == "postTweet":
        text = body.get("text", "")
        reply_to = body.get("replyTo")
        media_ids = body.get("mediaIds", [])
        if len(text) > 280:
            result = {"ok": False, "error": "tweet_too_long", "detail": f"Tweet is {len(text)} chars, max 280"}
            _log("statuses.update", {"text": text, "reply_to": reply_to}, result)
            return result
        tweet_id = f"tw_{len(_tweets) + 1000:04d}"
        tweet = {
            "id": tweet_id,
            "text": text,
            "user_id": "U_AGENT",
            "created_at": _ts(),
            "metrics": {"like_count": 0, "retweet_count": 0, "reply_count": 0},
        }
        if reply_to:
            tweet["reply_to"] = reply_to
        if media_ids:
            tweet["media_ids"] = media_ids
        _tweets[tweet_id] = tweet
        _timeline.insert(0, tweet_id)
        result = {"ok": True, "tweet_id": tweet_id, "tweet": tweet}
        _log("statuses.update", {"text": text, "reply_to": reply_to, "media_ids": media_ids}, result)
        return result

    elif action == "replyToTweet":
        tweet_id = body.get("tweetId", "")
        text = body.get("text", "")
        if len(text) > 280:
            result = {"ok": False, "error": "tweet_too_long", "detail": f"Tweet is {len(text)} chars, max 280"}
            _log("statuses.update", {"text": text, "reply_to": tweet_id}, result)
            return result
        new_id = f"tw_{len(_tweets) + 1000:04d}"
        tweet = {
            "id": new_id,
            "text": text,
            "user_id": "U_AGENT",
            "reply_to": tweet_id,
            "created_at": _ts(),
            "metrics": {"like_count": 0, "retweet_count": 0, "reply_count": 0},
        }
        _tweets[new_id] = tweet
        _timeline.insert(0, new_id)
        original = _tweets.get(tweet_id)
        if original:
            original.setdefault("metrics", {})
            original["metrics"]["reply_count"] = original["metrics"].get("reply_count", 0) + 1
        result = {"ok": True, "tweet_id": new_id, "tweet": tweet}
        _log("statuses.update", {"text": text, "reply_to": tweet_id}, result)
        return result

    elif action == "quoteTweet":
        tweet_id = body.get("tweetId", "")
        text = body.get("text", "")
        if len(text) > 280:
            result = {"ok": False, "error": "tweet_too_long", "detail": f"Tweet is {len(text)} chars, max 280"}
            _log("statuses.quoteTweet", {"text": text, "quote_tweet_id": tweet_id}, result)
            return result
        new_id = f"tw_{len(_tweets) + 1000:04d}"
        tweet = {
            "id": new_id,
            "text": text,
            "user_id": "U_AGENT",
            "quote_tweet_id": tweet_id,
            "created_at": _ts(),
            "metrics": {"like_count": 0, "retweet_count": 0, "reply_count": 0},
        }
        _tweets[new_id] = tweet
        _timeline.insert(0, new_id)
        result = {"ok": True, "tweet_id": new_id, "tweet": tweet}
        _log("statuses.quoteTweet", {"text": text, "quote_tweet_id": tweet_id}, result)
        return result

    elif action == "likeTweet":
        tweet_id = body.get("tweetId", "")
        tweet = _tweets.get(tweet_id)
        if tweet:
            tweet.setdefault("metrics", {})
            tweet["metrics"]["like_count"] = tweet["metrics"].get("like_count", 0) + 1
            result = {"ok": True, "tweet_id": tweet_id}
        else:
            result = {"ok": False, "error": "tweet_not_found"}
        _log("favorites.create", {"id": tweet_id}, result)
        return result

    elif action == "retweetTweet":
        tweet_id = body.get("tweetId", "")
        tweet = _tweets.get(tweet_id)
        if tweet:
            tweet.setdefault("metrics", {})
            tweet["metrics"]["retweet_count"] = tweet["metrics"].get("retweet_count", 0) + 1
            result = {"ok": True, "tweet_id": tweet_id}
        else:
            result = {"ok": False, "error": "tweet_not_found"}
        _log("statuses.retweet", {"id": tweet_id}, result)
        return result

    elif action == "retweet":
        tweet_id = body.get("tweetId", "")
        tweet = _tweets.get(tweet_id)
        if tweet:
            tweet.setdefault("metrics", {})
            tweet["metrics"]["retweet_count"] = tweet["metrics"].get("retweet_count", 0) + 1
            result = {"ok": True, "tweet_id": tweet_id}
        else:
            result = {"ok": False, "error": "tweet_not_found"}
        _log("statuses.retweet", {"id": tweet_id}, result)
        return result

    elif action == "like":
        tweet_id = body.get("tweetId", "")
        tweet = _tweets.get(tweet_id)
        if tweet:
            tweet.setdefault("metrics", {})
            tweet["metrics"]["like_count"] = tweet["metrics"].get("like_count", 0) + 1
            result = {"ok": True, "tweet_id": tweet_id}
        else:
            result = {"ok": False, "error": "tweet_not_found"}
        _log("favorites.create", {"id": tweet_id}, result)
        return result

    elif action == "sendDM":
        recipient_id = body.get("recipientId", "")
        text = body.get("text", "")
        dm = {
            "id": f"dm_{len(_dms) + 1000:04d}",
            "sender_id": "U_AGENT",
            "recipient_id": recipient_id,
            "text": text,
            "created_at": _ts(),
        }
        _dms.append(dm)
        result = {"ok": True, "dm_id": dm["id"]}
        _log("directMessages.new", {"recipient_id": recipient_id, "text": text}, result)
        return result

    elif action == "deleteTweet":
        tweet_id = body.get("tweetId", "")
        if tweet_id in _tweets:
            del _tweets[tweet_id]
            if tweet_id in _timeline:
                _timeline.remove(tweet_id)
            result = {"ok": True, "tweet_id": tweet_id}
        else:
            result = {"ok": False, "error": "tweet_not_found"}
        _log("statuses.destroy", {"id": tweet_id}, result)
        return result

    elif action == "bookmarkTweet":
        tweet_id = body.get("tweetId", "")
        result = {"ok": True, "tweet_id": tweet_id, "bookmarked": True}
        _log("bookmarks.add", {"id": tweet_id}, result)
        return result

    elif action == "flagContent":
        tweet_id = body.get("tweetId", "")
        reason = body.get("reason", "")
        result = {"ok": True, "tweet_id": tweet_id, "reason": reason, "flagged": True}
        _log("moderation.flag", {"id": tweet_id, "reason": reason}, result)
        return result

    return {"ok": False, "error": f"unknown_action: {action}"}


# ── Management endpoints ───────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/x/reset")
async def reset():
    global _audit_log
    _audit_log = []
    _load_fixtures()
    return {"status": "reset"}


@app.get("/x/audit")
async def audit():
    return {"audit": _audit_log}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("X_PORT", "3300"))
    uvicorn.run(app, host="0.0.0.0", port=port)
