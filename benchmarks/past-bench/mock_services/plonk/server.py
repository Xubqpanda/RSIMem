"""Mock Plonk API service v2 for agent evaluation (FastAPI on port 3200).

Plonk is a synthetic social/communication platform with **fully obfuscated**
API method names, parameter names, AND response field names. This tests an
agent's ability to internalize a completely novel API from documentation alone.

Mapping of real operations to Plonk method names:
    send_message    -> zubefa   (xovik, belatu)
    list_channels   -> vimaku   ()
    read_messages   -> rolatu   (fenoka, drimpa)
    create_channel  -> gitefi   (moriku, tanevi)
    search_messages -> nexobi   (jurela)
    add_reaction    -> tuwela   (pikano, dravex)
    get_user_profile-> kopadi   (sulima)
    pin_message     -> felura   (pikano)
    edit_message    -> bavoni   (pikano, belatu)
    delete_message  -> sariku   (pikano)

DEAD / TRAP functions (return PlonkError):
    mokufa(xovik, belatu)  -- sounds like "post notification"
    volari(fenoka)         -- sounds like "fetch stream"
    plentu(jurela, fenoka) -- sounds like "deep search"
    ximako()               -- sounds like "enumerate nodes"
    doravi(pikano, dravex) -- sounds like "annotate entry"
    kaviru(fenoka, drimpa) -- sounds like "read messages" (alt)
    zolanti(xovik, belatu) -- sounds like "broadcast message"
    fenaldi(pikano)        -- sounds like "bookmark/pin" (alt)
    gremoku(sulima, xovik) -- sounds like "invite user to channel"
    torvani(pikano, belatu)-- sounds like "reply to message"
    wexidu(fenoka)         -- sounds like "archive channel"
    lunista(sulima)        -- sounds like "get user status"
    prikova(jurela)        -- sounds like "global search" (alt)
    darvoxi(pikano)        -- sounds like "mark as read"
    nubikov(moriku, tanevi)-- sounds like "create channel" (alt)

Response field mappings:
    ok             -> vorix
    messages       -> klenta
    channels       -> troveni
    profile        -> xelion
    matches        -> jurova
    id             -> tikano
    text           -> belatu
    user           -> sulima
    channel        -> xovik
    count          -> numera
    pinned         -> grevani
    reactions      -> draveni
    timestamp      -> kronex
    name           -> nomiku
    display_name   -> vizoku
    topic          -> tanevi
    message_count  -> numera_klenta
    channel_id     -> xovik_id
    message_id     -> tikano_ref
    deleted        -> rimova
    edited         -> modyfa
    error          -> faltex
    message (err)  -> detiku
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
from fastapi.responses import JSONResponse

app = FastAPI(title="Mock Plonk API v2")

from mock_services._base import add_error_injection
add_error_injection(app)

FIXTURES_PATH = Path(os.environ.get(
    "PLONK_FIXTURES",
    str(Path(__file__).resolve().parent / "default_fixtures.json"),
))

# ---------------------------------------------------------------------------
# In-memory state
# ---------------------------------------------------------------------------
_users: dict[str, dict[str, Any]] = {}
_channels: dict[str, dict[str, Any]] = {}
_audit_log: list[dict[str, Any]] = []

# Known parameter names per method (for validation)
_METHOD_PARAMS: dict[str, set[str]] = {
    # Live functions
    "zubefa": {"xovik", "belatu"},
    "vimaku": set(),
    "rolatu": {"fenoka", "drimpa"},
    "gitefi": {"moriku", "tanevi"},
    "nexobi": {"jurela"},
    "tuwela": {"pikano", "dravex"},
    "kopadi": {"sulima"},
    "felura": {"pikano"},
    "bavoni": {"pikano", "belatu"},
    "sariku": {"pikano"},
    # Dead / trap functions (15 total)
    "mokufa": {"xovik", "belatu"},
    "volari": {"fenoka"},
    "plentu": {"jurela", "fenoka"},
    "ximako": set(),
    "doravi": {"pikano", "dravex"},
    "kaviru": {"fenoka", "drimpa"},
    "zolanti": {"xovik", "belatu"},
    "fenaldi": {"pikano"},
    "gremoku": {"sulima", "xovik"},
    "torvani": {"pikano", "belatu"},
    "wexidu": {"fenoka"},
    "lunista": {"sulima"},
    "prikova": {"jurela"},
    "darvoxi": {"pikano"},
    "nubikov": {"moriku", "tanevi"},
}

# Common "wrong" parameter names that callers might mistakenly use
_FORBIDDEN_PARAMS: dict[str, str] = {
    "channel": "xovik or fenoka",
    "text": "belatu",
    "limit": "drimpa",
    "name": "moriku",
    "topic": "tanevi",
    "query": "jurela",
    "pattern": "jurela",
    "message_id": "pikano",
    "emoji": "dravex",
    "emoji_code": "dravex",
    "user_id": "sulima",
    "new_text": "belatu",
    "target": "xovik",
    "payload": "belatu",
    "space": "fenoka",
    "depth": "drimpa",
    "label": "moriku",
    "descriptor": "tanevi",
    "ref": "pikano",
    "glyph": "dravex",
    "entity": "sulima",
}

# Dead function set — these always return an error
_DEAD_FUNCTIONS = {
    "mokufa", "volari", "plentu", "ximako", "doravi",
    "kaviru", "zolanti", "fenaldi", "gremoku", "torvani",
    "wexidu", "lunista", "prikova", "darvoxi", "nubikov",
}


def _load_fixtures() -> None:
    global _users, _channels
    with open(FIXTURES_PATH) as f:
        data = json.load(f)
    _users = data.get("users", {})
    _channels = {}
    for ch_id, ch_data in data.get("channels", {}).items():
        _channels[ch_id] = {
            "topic": ch_data.get("topic", ""),
            "messages": ch_data.get("messages", []),
        }


_load_fixtures()


def _log_call(method: str, params: dict[str, Any], response: Any) -> None:
    _audit_log.append({
        "method": method,
        "params": params,
        "response": response,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


def _validate_params(method: str, body: dict[str, Any]) -> JSONResponse | None:
    """Check for forbidden (un-obfuscated) parameter names."""
    for key in body:
        if key in _FORBIDDEN_PARAMS:
            return JSONResponse(
                status_code=400,
                content={
                    "vorix": False,
                    "faltex": "invalid_parameter",
                    "detiku": (
                        f"Unknown parameter '{key}'. "
                        f"Did you mean '{_FORBIDDEN_PARAMS[key]}'? "
                        f"Valid parameters for {method}: {sorted(_METHOD_PARAMS.get(method, set()))}"
                    ),
                },
            )
    return None


def _find_message(message_id: str) -> tuple[str, int] | None:
    """Find a message by ID across all channels."""
    for ch_id, ch_data in _channels.items():
        for i, msg in enumerate(ch_data["messages"]):
            if msg.get("id") == message_id:
                return ch_id, i
    return None


def _now_ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def _obfuscate_message(msg: dict) -> dict:
    """Convert internal message format to obfuscated response format."""
    return {
        "tikano": msg.get("id", ""),
        "sulima": msg.get("user", ""),
        "belatu": msg.get("text", ""),
        "kronex": msg.get("ts", ""),
        "draveni": msg.get("reactions", []),
        "grevani": msg.get("pinned", False),
    }


def _obfuscate_channel(ch_id: str, ch_data: dict) -> dict:
    """Convert internal channel format to obfuscated response format."""
    return {
        "tikano": ch_id,
        "tanevi": ch_data.get("topic", ""),
        "numera_klenta": len(ch_data.get("messages", [])),
    }


def _obfuscate_profile(user_id: str, profile: dict) -> dict:
    """Convert internal profile format to obfuscated response format."""
    return {
        "tikano": user_id,
        "nomiku": profile.get("name", ""),
        "vizoku": profile.get("display_name", ""),
    }


# ---------------------------------------------------------------------------
# Dead function handler
# ---------------------------------------------------------------------------

def _dead_function_response(method: str, params: dict) -> JSONResponse:
    """Return a PlonkError for dead/trap functions."""
    result = {
        "vorix": False,
        "faltex": "deprecated_method",
        "detiku": (
            f"Method '{method}' is deprecated and no longer functional. "
            f"Consult the Plonk API documentation for the correct method."
        ),
    }
    _log_call(method, params, result)
    return JSONResponse(status_code=410, content=result)


# ---------------------------------------------------------------------------
# Action dispatch functions
# ---------------------------------------------------------------------------

def _action_zubefa(params: dict) -> Any:
    xovik = params.get("xovik", "")
    belatu = params.get("belatu", "")
    if xovik not in _channels:
        result = {"vorix": False, "faltex": "node_not_found", "detiku": f"No FAXI node '{xovik}'"}
        _log_call("zubefa", {"xovik": xovik, "belatu": belatu}, result)
        return JSONResponse(status_code=400, content=result)
    msg_id = f"m{len(_channels[xovik]['messages']) + 1:03d}"
    new_msg = {"id": msg_id, "user": "U_AGENT", "text": belatu, "ts": _now_ts(), "reactions": [], "pinned": False}
    _channels[xovik]["messages"].append(new_msg)
    result = {"vorix": True, "xovik": xovik, "tikano_ref": msg_id}
    _log_call("zubefa", {"xovik": xovik, "belatu": belatu}, result)
    return result


def _action_vimaku(params: dict) -> Any:
    channels_list = [_obfuscate_channel(ch_id, ch_data) for ch_id, ch_data in _channels.items()]
    result = {"vorix": True, "troveni": channels_list}
    _log_call("vimaku", {}, result)
    return result


def _action_rolatu(params: dict) -> Any:
    fenoka = params.get("fenoka", "")
    drimpa = params.get("drimpa", 50)
    if fenoka not in _channels:
        result = {"vorix": False, "faltex": "node_not_found", "detiku": f"No FAXI node '{fenoka}'"}
        _log_call("rolatu", {"fenoka": fenoka, "drimpa": drimpa}, result)
        return JSONResponse(status_code=400, content=result)
    messages = _channels[fenoka]["messages"][-drimpa:]
    result = {"vorix": True, "klenta": [_obfuscate_message(m) for m in messages]}
    _log_call("rolatu", {"fenoka": fenoka, "drimpa": drimpa}, result)
    return result


def _action_nexobi(params: dict) -> Any:
    jurela = params.get("jurela", "").lower()
    matches = []
    for ch_id, ch_data in _channels.items():
        for msg in ch_data["messages"]:
            if jurela in msg.get("text", "").lower():
                match = _obfuscate_message(msg)
                match["xovik"] = ch_id
                matches.append(match)
    result = {"vorix": True, "jurova": matches, "numera": len(matches)}
    _log_call("nexobi", {"jurela": jurela}, result)
    return result


def _action_tuwela(params: dict) -> Any:
    pikano = params.get("pikano", "")
    dravex = params.get("dravex", "")
    location = _find_message(pikano)
    if location is None:
        result = {"vorix": False, "faltex": "entry_not_found", "detiku": f"No entry with tikano '{pikano}'"}
        _log_call("tuwela", {"pikano": pikano, "dravex": dravex}, result)
        return JSONResponse(status_code=400, content=result)
    ch_id, idx = location
    _channels[ch_id]["messages"][idx]["reactions"].append(dravex)
    result = {"vorix": True, "pikano": pikano, "dravex": dravex}
    _log_call("tuwela", {"pikano": pikano, "dravex": dravex}, result)
    return result


def _action_felura(params: dict) -> Any:
    pikano = params.get("pikano", "")
    location = _find_message(pikano)
    if location is None:
        result = {"vorix": False, "faltex": "entry_not_found", "detiku": f"No entry with tikano '{pikano}'"}
        _log_call("felura", {"pikano": pikano}, result)
        return JSONResponse(status_code=400, content=result)
    ch_id, idx = location
    _channels[ch_id]["messages"][idx]["pinned"] = True
    result = {"vorix": True, "pikano": pikano, "grevani": True}
    _log_call("felura", {"pikano": pikano}, result)
    return result


def _action_gitefi(params: dict) -> Any:
    moriku = params.get("moriku", "")
    tanevi = params.get("tanevi", "")
    ch_id = f"plk-{moriku}"
    if ch_id in _channels:
        result = {"vorix": False, "faltex": "node_exists", "detiku": f"FAXI node '{ch_id}' already exists"}
        _log_call("gitefi", {"moriku": moriku, "tanevi": tanevi}, result)
        return JSONResponse(status_code=400, content=result)
    _channels[ch_id] = {"topic": tanevi, "messages": []}
    result = {"vorix": True, "xovik_id": ch_id}
    _log_call("gitefi", {"moriku": moriku, "tanevi": tanevi}, result)
    return result


def _action_kopadi(params: dict) -> Any:
    sulima = params.get("sulima", "")
    if sulima not in _users:
        result = {"vorix": False, "faltex": "venu_not_found", "detiku": f"No VENU with sulima '{sulima}'"}
        _log_call("kopadi", {"sulima": sulima}, result)
        return JSONResponse(status_code=400, content=result)
    profile = _obfuscate_profile(sulima, _users[sulima])
    result = {"vorix": True, "xelion": profile}
    _log_call("kopadi", {"sulima": sulima}, result)
    return result


def _action_bavoni(params: dict) -> Any:
    pikano = params.get("pikano", "")
    belatu = params.get("belatu", "")
    location = _find_message(pikano)
    if location is None:
        result = {"vorix": False, "faltex": "entry_not_found", "detiku": f"No entry with tikano '{pikano}'"}
        _log_call("bavoni", {"pikano": pikano, "belatu": belatu}, result)
        return JSONResponse(status_code=400, content=result)
    ch_id, idx = location
    _channels[ch_id]["messages"][idx]["text"] = belatu
    _channels[ch_id]["messages"][idx]["edited"] = True
    result = {"vorix": True, "pikano": pikano, "modyfa": True}
    _log_call("bavoni", {"pikano": pikano, "belatu": belatu}, result)
    return result


def _action_sariku(params: dict) -> Any:
    pikano = params.get("pikano", "")
    location = _find_message(pikano)
    if location is None:
        result = {"vorix": False, "faltex": "entry_not_found", "detiku": f"No entry with tikano '{pikano}'"}
        _log_call("sariku", {"pikano": pikano}, result)
        return JSONResponse(status_code=400, content=result)
    ch_id, idx = location
    _channels[ch_id]["messages"].pop(idx)
    result = {"vorix": True, "pikano": pikano, "rimova": True}
    _log_call("sariku", {"pikano": pikano}, result)
    return result


_ACTION_DISPATCH = {
    "zubefa": _action_zubefa,
    "vimaku": _action_vimaku,
    "rolatu": _action_rolatu,
    "gitefi": _action_gitefi,
    "nexobi": _action_nexobi,
    "tuwela": _action_tuwela,
    "kopadi": _action_kopadi,
    "felura": _action_felura,
    "bavoni": _action_bavoni,
    "sariku": _action_sariku,
}


# ---------------------------------------------------------------------------
# Individual method endpoints (for direct calls)
# ---------------------------------------------------------------------------

@app.post("/plonk/zubefa")
async def ep_zubefa(request: Request):
    body = await request.json()
    err = _validate_params("zubefa", body)
    if err:
        return err
    return _action_zubefa(body)


@app.post("/plonk/vimaku")
async def ep_vimaku(request: Request):
    body = await request.json()
    err = _validate_params("vimaku", body)
    if err:
        return err
    return _action_vimaku(body)


@app.post("/plonk/rolatu")
async def ep_rolatu(request: Request):
    body = await request.json()
    err = _validate_params("rolatu", body)
    if err:
        return err
    return _action_rolatu(body)


@app.post("/plonk/gitefi")
async def ep_gitefi(request: Request):
    body = await request.json()
    err = _validate_params("gitefi", body)
    if err:
        return err
    return _action_gitefi(body)


@app.post("/plonk/nexobi")
async def ep_nexobi(request: Request):
    body = await request.json()
    err = _validate_params("nexobi", body)
    if err:
        return err
    return _action_nexobi(body)


@app.post("/plonk/tuwela")
async def ep_tuwela(request: Request):
    body = await request.json()
    err = _validate_params("tuwela", body)
    if err:
        return err
    return _action_tuwela(body)


@app.post("/plonk/kopadi")
async def ep_kopadi(request: Request):
    body = await request.json()
    err = _validate_params("kopadi", body)
    if err:
        return err
    return _action_kopadi(body)


@app.post("/plonk/felura")
async def ep_felura(request: Request):
    body = await request.json()
    err = _validate_params("felura", body)
    if err:
        return err
    return _action_felura(body)


@app.post("/plonk/bavoni")
async def ep_bavoni(request: Request):
    body = await request.json()
    err = _validate_params("bavoni", body)
    if err:
        return err
    return _action_bavoni(body)


@app.post("/plonk/sariku")
async def ep_sariku(request: Request):
    body = await request.json()
    err = _validate_params("sariku", body)
    if err:
        return err
    return _action_sariku(body)


# Dead function endpoints
@app.post("/plonk/mokufa")
async def ep_mokufa(request: Request):
    body = await request.json()
    return _dead_function_response("mokufa", body)


@app.post("/plonk/volari")
async def ep_volari(request: Request):
    body = await request.json()
    return _dead_function_response("volari", body)


@app.post("/plonk/plentu")
async def ep_plentu(request: Request):
    body = await request.json()
    return _dead_function_response("plentu", body)


@app.post("/plonk/ximako")
async def ep_ximako(request: Request):
    body = await request.json()
    return _dead_function_response("ximako", body)


@app.post("/plonk/doravi")
async def ep_doravi(request: Request):
    body = await request.json()
    return _dead_function_response("doravi", body)


@app.post("/plonk/kaviru")
async def ep_kaviru(request: Request):
    body = await request.json()
    return _dead_function_response("kaviru", body)

@app.post("/plonk/zolanti")
async def ep_zolanti(request: Request):
    body = await request.json()
    return _dead_function_response("zolanti", body)

@app.post("/plonk/fenaldi")
async def ep_fenaldi(request: Request):
    body = await request.json()
    return _dead_function_response("fenaldi", body)

@app.post("/plonk/gremoku")
async def ep_gremoku(request: Request):
    body = await request.json()
    return _dead_function_response("gremoku", body)

@app.post("/plonk/torvani")
async def ep_torvani(request: Request):
    body = await request.json()
    return _dead_function_response("torvani", body)

@app.post("/plonk/wexidu")
async def ep_wexidu(request: Request):
    body = await request.json()
    return _dead_function_response("wexidu", body)

@app.post("/plonk/lunista")
async def ep_lunista(request: Request):
    body = await request.json()
    return _dead_function_response("lunista", body)

@app.post("/plonk/prikova")
async def ep_prikova(request: Request):
    body = await request.json()
    return _dead_function_response("prikova", body)

@app.post("/plonk/darvoxi")
async def ep_darvoxi(request: Request):
    body = await request.json()
    return _dead_function_response("darvoxi", body)

@app.post("/plonk/nubikov")
async def ep_nubikov(request: Request):
    body = await request.json()
    return _dead_function_response("nubikov", body)


# ---------------------------------------------------------------------------
# Unified plonk_action tool endpoint
# ---------------------------------------------------------------------------

@app.post("/plonk/action")
async def plonk_action(request: Request):
    """Unified tool endpoint: routes based on 'method' or 'action' field to the correct Plonk API."""
    body = await request.json()
    # Support both {"method": ..., "params": {...}} and flat {"action": ..., xovik: ..., ...} formats
    method = body.get("method", "") or body.get("action", "")
    if "params" in body and isinstance(body["params"], dict):
        params = body["params"]
    else:
        params = {k: v for k, v in body.items() if k not in ("method", "action")}

    if method not in _METHOD_PARAMS:
        return JSONResponse(
            status_code=400,
            content={"vorix": False, "faltex": "unknown_method", "detiku": f"Unknown method '{method}'"},
        )

    # Dead function check
    if method in _DEAD_FUNCTIONS:
        return _dead_function_response(method, params)

    # Validate params
    err = _validate_params(method, params)
    if err:
        return err

    handler = _ACTION_DISPATCH.get(method)
    if handler is None:
        return JSONResponse(
            status_code=400,
            content={"vorix": False, "faltex": "unknown_method", "detiku": f"Unknown method '{method}'"},
        )
    return handler(params)


# ---------------------------------------------------------------------------
# Management endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/plonk/reset")
async def reset():
    global _audit_log
    _audit_log = []
    _load_fixtures()
    return {"status": "reset"}


@app.get("/plonk/audit")
async def audit():
    return {"audit": _audit_log}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", os.environ.get("PLONK_PORT", "3200")))
    uvicorn.run(app, host="0.0.0.0", port=port)
