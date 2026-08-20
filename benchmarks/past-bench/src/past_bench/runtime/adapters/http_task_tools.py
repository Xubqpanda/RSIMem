"""Shared bench HTTP task-tool bridge for non-native agent runtimes."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, create_model

from ...models.tool import ToolEndpoint, ToolSpec

_BODY_METHODS = {"POST", "PUT", "PATCH"}


def invoke_http_tool(
    *,
    url: str,
    method: str,
    arguments: dict[str, Any] | None = None,
    timeout: int = 30,
) -> str:
    """Invoke one bench HTTP tool endpoint and return a text payload."""

    args = arguments or {}
    method_upper = method.upper()
    request_url = url
    data: bytes | None = None

    if method_upper in _BODY_METHODS:
        data = json.dumps(args, ensure_ascii=False).encode("utf-8")
    elif args:
        query = urllib.parse.urlencode(_flatten_query_args(args), doseq=True)
        request_url = f"{url}{'&' if '?' in url else '?'}{query}"

    req = urllib.request.Request(
        request_url,
        data=data,
        method=method_upper,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", errors="replace")
        except Exception:
            detail = ""
        return f"HTTP {exc.code}: {detail}"
    except Exception as exc:
        return f"Error: {type(exc).__name__}: {exc}"


def build_agent_zero_tool_module_source(*, url: str, method: str) -> str:
    """Return the source code for an Agent Zero task tool shim."""

    return f'''"""Auto-generated past_bench task tool — DO NOT EDIT.
Written by runtime adapter at session start; removed at close().
"""
import json
import urllib.error
import urllib.parse
import urllib.request

from helpers.tool import Response, Tool


_ENDPOINT_URL = {url!r}
_HTTP_METHOD = {method.upper()!r}
_BODY_METHODS = {sorted(_BODY_METHODS)!r}


def _flatten_query_args(arguments):
    items = []
    for key, value in (arguments or {{}}).items():
        if isinstance(value, list):
            for item in value:
                items.append((key, "" if item is None else str(item)))
        elif isinstance(value, dict):
            items.append((key, json.dumps(value, ensure_ascii=False)))
        elif value is not None:
            items.append((key, str(value)))
    return items


class PastBenchTaskTool(Tool):
    async def execute(self, **kwargs) -> Response:
        try:
            req_url = _ENDPOINT_URL
            data = None
            if _HTTP_METHOD in _BODY_METHODS:
                data = json.dumps(kwargs or {{}}, ensure_ascii=False).encode("utf-8")
            elif kwargs:
                query = urllib.parse.urlencode(_flatten_query_args(kwargs), doseq=True)
                req_url = f"{{_ENDPOINT_URL}}{{'&' if '?' in _ENDPOINT_URL else '?'}}{{query}}"
            req = urllib.request.Request(
                req_url,
                data=data,
                method=_HTTP_METHOD,
                headers={{"Content-Type": "application/json"}},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = resp.read().decode("utf-8", errors="replace")
            return Response(message=body, break_loop=False)
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", errors="replace")
            except Exception:
                detail = ""
            return Response(message=f"HTTP {{exc.code}}: {{detail}}", break_loop=False)
        except Exception as exc:
            return Response(message=f"Error: {{type(exc).__name__}}: {{exc}}", break_loop=False)
'''


def build_args_model(name: str, schema: dict[str, Any] | None) -> type[BaseModel]:
    """Build a permissive Pydantic args model from a JSON schema object."""

    schema = schema or {"type": "object", "properties": {}}
    fields: dict[str, tuple[Any, Any]] = {}
    properties = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    for field_name, field_schema in properties.items():
        annotation = _python_type_for_schema(field_schema)
        default = ... if field_name in required else field_schema.get("default", None)
        description = field_schema.get("description")
        field = Field(default, description=description) if description else default
        fields[field_name] = (annotation, field)
    model_name = "".join(part.capitalize() for part in name.replace("-", "_").split("_")) or "Args"
    return create_model(
        f"{model_name}Args",
        __config__=ConfigDict(extra="allow"),
        **fields,
    )


def endpoint_map(endpoints: list[ToolEndpoint]) -> dict[str, ToolEndpoint]:
    return {endpoint.tool_name: endpoint for endpoint in endpoints}


def _python_type_for_schema(schema: dict[str, Any] | None) -> Any:
    if not isinstance(schema, dict):
        return Any
    raw_type = schema.get("type")
    if isinstance(raw_type, list):
        raw_type = next((item for item in raw_type if item != "null"), None)
    if raw_type == "string":
        return str
    if raw_type == "integer":
        return int
    if raw_type == "number":
        return float
    if raw_type == "boolean":
        return bool
    if raw_type == "array":
        item_type = _python_type_for_schema(schema.get("items"))
        try:
            return list[item_type]
        except TypeError:
            return list[Any]
    if raw_type == "object":
        return dict[str, Any]
    return Any


def _flatten_query_args(arguments: dict[str, Any]) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    for key, value in (arguments or {}).items():
        if isinstance(value, list):
            for item in value:
                items.append((key, "" if item is None else str(item)))
        elif isinstance(value, dict):
            items.append((key, json.dumps(value, ensure_ascii=False)))
        elif value is not None:
            items.append((key, str(value)))
    return items


def matching_task_tools(
    tools: list[ToolSpec],
    endpoints: list[ToolEndpoint],
) -> list[tuple[ToolSpec, ToolEndpoint]]:
    """Return tools that have matching endpoint registrations."""

    endpoints_by_name = endpoint_map(endpoints)
    return [
        (tool, endpoints_by_name[tool.name])
        for tool in tools
        if tool.name in endpoints_by_name
    ]
