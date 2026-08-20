"""Runtime adapter that runs a ZeroClaw agent in-process."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ...models.content import TextBlock
from ...models.message import Message
from ...models.trace import TokenUsage
from ..protocol import StartSessionRequest, StepRequest, StepResponse
from ..registry import AgentSpec
from .base import RuntimeAdapter
from .cli_agent import _build_cli_prompt
from .http_task_tools import build_args_model, invoke_http_tool, matching_task_tools

_ZEROCLAW_ROOT = Path(__file__).resolve().parents[4] / "agents" / "zeroclaw" / "python"


def _normalize_memory_value(value: str) -> str:
    return " ".join(str(value or "").split()).strip()


class ZeroClawAdapter(RuntimeAdapter):
    """In-process adapter for the ZeroClaw LangGraph agent.

    Unlike the CLI adapters, this runs the zeroclaw Python agent directly
    in-process, keeping evaluation of the agent framework separate from
    task definitions.
    """

    def __init__(self, spec: AgentSpec, request: StartSessionRequest) -> None:
        super().__init__(spec, request)
        self._completed_response: StepResponse | None = None

    def step(self, request: StepRequest) -> StepResponse:
        if self._completed_response is not None:
            return self._completed_response

        if request.step_id > 0 or request.tool_results:
            return StepResponse(
                status="error",
                error="ZeroClaw adapter runs in one-shot mode; follow-up turns not supported.",
            )

        started = time.monotonic()
        try:
            response = self._run_agent()
        except Exception as exc:
            response = StepResponse(
                status="error",
                error=f"{type(exc).__name__}: {exc}",
            )
        response.model_time_s = time.monotonic() - started
        self._completed_response = response
        return response

    def _run_agent(self) -> StepResponse:
        zeroclaw_root = str(_ZEROCLAW_ROOT)
        if zeroclaw_root not in sys.path:
            sys.path.insert(0, zeroclaw_root)

        # Deferred import so the adapter module loads even when zeroclaw is not installed.
        from zeroclaw_tools import create_agent
        from zeroclaw_tools.tools.base import tool
        from langchain_core.messages import HumanMessage
        from langchain_core.tools import StructuredTool

        extra = self.request.model.extra_body or {}
        zeroclaw_cfg = extra.get("zeroclaw", {}) if isinstance(extra, dict) else {}
        recursion_limit = int(zeroclaw_cfg.get("recursion_limit", 100))
        persistence_enabled = bool(zeroclaw_cfg.get("persistence_enabled", True))
        memory_enabled = bool(zeroclaw_cfg.get("memory_enabled", persistence_enabled))
        session_search_enabled = bool(
            zeroclaw_cfg.get("session_search_enabled", persistence_enabled)
        )
        workspace_dir = str(zeroclaw_cfg.get("workspace_dir") or "").strip()
        workspace = Path(workspace_dir).expanduser().resolve() if workspace_dir else Path.cwd()
        workspace.mkdir(parents=True, exist_ok=True)
        home_dir_value = str(zeroclaw_cfg.get("home_dir") or "").strip()
        home_dir = Path(home_dir_value).expanduser().resolve() if home_dir_value else workspace / ".zeroclaw_home"
        home_dir.mkdir(parents=True, exist_ok=True)
        session_history_dir_value = str(zeroclaw_cfg.get("session_history_dir") or "").strip()
        session_history_dir = (
            Path(session_history_dir_value).expanduser().resolve()
            if session_history_dir_value
            else home_dir / "session_history"
        )
        session_history_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir = str(zeroclaw_cfg.get("artifacts_dir") or "").strip()
        transcript_path = (
            Path(artifacts_dir).expanduser().resolve() / "zeroclaw_transcript.json"
            if artifacts_dir
            else None
        )

        tools = [
            self._build_shell_tool(tool, workspace),
            self._build_file_read_tool(tool, workspace),
            self._build_file_write_tool(tool, workspace),
        ]
        if memory_enabled:
            tools.extend(
                [
                    self._build_memory_store_tool(tool, home_dir),
                    self._build_memory_recall_tool(tool, home_dir),
                ]
            )
        if session_search_enabled:
            tools.append(self._build_session_search_tool(tool, session_history_dir))
        tools.extend(self._build_task_tools(StructuredTool))

        with self._override_home(home_dir):
            agent = create_agent(
                tools=tools,
                model=self.request.model.model_id,
                api_key=self.request.model.api_key,
                base_url=self.request.model.base_url,
                temperature=float(self.request.runtime_config.temperature),
                system_prompt=zeroclaw_cfg.get("system_prompt"),
            )

            prompt = _build_cli_prompt(self.request)
            timeout = max(30, self.request.timeout_seconds)

            result = asyncio.run(
                asyncio.wait_for(
                    agent.ainvoke(
                        {"messages": [HumanMessage(content=prompt)]},
                        config={"recursion_limit": recursion_limit},
                    ),
                    timeout=timeout,
                )
            )

        self._capture_artifacts(
            workspace=workspace,
            home_dir=home_dir,
            transcript=result,
            transcript_path=transcript_path,
            session_history_dir=session_history_dir if persistence_enabled else None,
        )
        messages = result.get("messages", [])
        final_text = ""
        if messages:
            last = messages[-1]
            final_text = getattr(last, "content", "")
            if not isinstance(final_text, str):
                final_text = str(final_text)

        assistant = Message(role="assistant", content=[TextBlock(text=final_text)])
        return StepResponse(
            status="finished",
            assistant_message=assistant,
            usage=TokenUsage(),
            final_output=final_text,
        )

    def close(self, reason: str = "") -> None:
        pass

    @staticmethod
    def _resolve_workspace_path(workspace: Path, path: str) -> Path:
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = workspace / candidate
        resolved = candidate.resolve()
        try:
            resolved.relative_to(workspace.resolve())
        except ValueError as exc:
            raise PermissionError(f"Path {path} is outside workspace {workspace}") from exc
        return resolved

    @staticmethod
    def _build_shell_tool(tool_decorator: Any, workspace: Path) -> Any:
        @tool_decorator(name="shell")
        def _shell(command: str) -> str:
            """Execute a shell command inside the bench-managed workspace."""
            try:
                result = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=60,
                    cwd=workspace,
                )
                output = result.stdout
                if result.stderr:
                    output += f"\nSTDERR: {result.stderr}"
                if result.returncode != 0:
                    output += f"\nExit code: {result.returncode}"
                return output or "(no output)"
            except subprocess.TimeoutExpired:
                return "Error: Command timed out after 60 seconds"
            except Exception as exc:
                return f"Error: {exc}"

        return _shell

    @classmethod
    def _build_file_read_tool(cls, tool_decorator: Any, workspace: Path) -> Any:
        @tool_decorator(name="file_read")
        def _file_read(path: str) -> str:
            """Read a UTF-8 text file from the bench-managed workspace."""
            try:
                resolved = cls._resolve_workspace_path(workspace, path)
                content = resolved.read_text(encoding="utf-8", errors="replace")
                if len(content) > 100_000:
                    return content[:100_000] + f"\n... (truncated, {len(content)} bytes total)"
                return content
            except FileNotFoundError:
                return f"Error: File not found: {path}"
            except PermissionError:
                return f"Error: Permission denied: {path}"
            except Exception as exc:
                return f"Error: {exc}"

        return _file_read

    @classmethod
    def _build_file_write_tool(cls, tool_decorator: Any, workspace: Path) -> Any:
        @tool_decorator(name="file_write")
        def _file_write(path: str, content: str) -> str:
            """Write a UTF-8 text file inside the bench-managed workspace."""
            try:
                resolved = cls._resolve_workspace_path(workspace, path)
                resolved.parent.mkdir(parents=True, exist_ok=True)
                resolved.write_text(content, encoding="utf-8")
                return f"Successfully wrote {len(content)} bytes to {resolved}"
            except PermissionError:
                return f"Error: Permission denied: {path}"
            except Exception as exc:
                return f"Error: {exc}"

        return _file_write

    @staticmethod
    def _memory_store_path(home_dir: Path) -> Path:
        return home_dir / ".zeroclaw" / "memory_store.json"

    @classmethod
    def _load_memory_store(cls, home_dir: Path) -> dict[str, str]:
        path = cls._memory_store_path(home_dir)
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8")) or {}
        except Exception:
            return {}
        if not isinstance(payload, dict):
            return {}
        return {str(key): str(value) for key, value in payload.items() if value is not None}

    @classmethod
    def _save_memory_store(cls, home_dir: Path, data: dict[str, str]) -> None:
        path = cls._memory_store_path(home_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def _store_memory_value(cls, *, home_dir: Path, key: str, value: str) -> str:
        data = cls._load_memory_store(home_dir)
        normalized_value = _normalize_memory_value(value)
        if not normalized_value:
            return "Error: value must not be empty"
        existing_value = data.get(key)
        if existing_value is not None and _normalize_memory_value(existing_value) == normalized_value:
            return f"Already stored: {key}"
        for existing_key, existing_item in data.items():
            if _normalize_memory_value(existing_item) == normalized_value:
                return f"Already stored as {existing_key}"
        data[key] = value
        cls._save_memory_store(home_dir, data)
        return f"Stored: {key}"

    @classmethod
    def _recall_memory_values(cls, *, home_dir: Path, query: str) -> str:
        data = cls._load_memory_store(home_dir)
        if not data:
            return "No memories stored yet"
        query_lower = str(query or "").lower().strip()
        if not query_lower:
            return json.dumps(data, ensure_ascii=False, indent=2)
        matches = {
            key: value
            for key, value in data.items()
            if query_lower in key.lower() or query_lower in value.lower()
        }
        if not matches:
            return f"No matches for: {query}"
        return json.dumps(matches, ensure_ascii=False, indent=2)

    @staticmethod
    def _collect_session_history_records(session_history_dir: Path) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        if not session_history_dir.exists():
            return records
        for path in sorted(session_history_dir.iterdir()):
            if path.suffix not in {".json", ".jsonl"} or not path.is_file():
                continue
            messages: list[dict[str, Any]]
            title = path.stem
            source = "history"
            session_id = path.stem
            if path.suffix == ".jsonl":
                messages = []
                for line in path.read_text(encoding="utf-8").splitlines():
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    if isinstance(row, dict):
                        messages.append(row)
            else:
                try:
                    payload = json.loads(path.read_text(encoding="utf-8")) or {}
                except Exception:
                    continue
                if not isinstance(payload, dict) or not isinstance(payload.get("messages"), list):
                    continue
                if payload.get("messages") and isinstance(payload["messages"][0], dict) and "type" in payload["messages"][0]:
                    title = str(payload.get("title") or path.stem)
                    source = str(payload.get("source") or "zeroclaw_transcript")
                    messages = payload["messages"]
                else:
                    title = str(payload.get("title") or payload.get("id") or path.stem)
                    source = str(payload.get("source") or "seed")
                    session_id = str(payload.get("id") or path.stem)
                    messages = payload["messages"]
            text_parts: list[str] = []
            for message in messages:
                if not isinstance(message, dict):
                    continue
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    text_parts.append(content.strip())
            records.append(
                {
                    "session_id": session_id,
                    "title": title,
                    "source": source,
                    "path": str(path),
                    "text": "\n".join(text_parts),
                }
            )
        return records

    @classmethod
    def _search_session_history(cls, session_history_dir: Path, query: str, max_results: int = 3) -> dict[str, Any]:
        normalized_query = " ".join(str(query or "").split()).strip()
        if not normalized_query:
            return {"success": False, "query": query, "results": [], "error": "query must not be empty"}
        query_terms = [term for term in normalized_query.lower().split() if term]
        scored: list[tuple[int, dict[str, Any]]] = []
        for record in cls._collect_session_history_records(session_history_dir):
            haystack = f"{record['session_id']} {record['title']} {record['text']}".lower()
            score = sum(haystack.count(term) for term in query_terms)
            if score <= 0:
                continue
            excerpt = cls._build_session_excerpt(record["text"], query_terms)
            scored.append(
                (
                    score,
                    {
                        "session_id": record["session_id"],
                        "title": record["title"],
                        "source": record["source"],
                        "path": record["path"],
                        "excerpt": excerpt,
                    },
                )
            )
        scored.sort(key=lambda item: (-item[0], item[1]["session_id"]))
        return {
            "success": True,
            "query": normalized_query,
            "results": [item for _, item in scored[: max(1, max_results)]],
        }

    @staticmethod
    def _build_session_excerpt(text: str, query_terms: list[str], limit: int = 320) -> str:
        normalized = " ".join(str(text or "").split())
        if not normalized:
            return ""
        lower = normalized.lower()
        start = 0
        for term in query_terms:
            idx = lower.find(term)
            if idx >= 0:
                start = max(0, idx - 80)
                break
        excerpt = normalized[start : start + limit]
        if start > 0:
            excerpt = "..." + excerpt
        if start + limit < len(normalized):
            excerpt = excerpt.rstrip() + "..."
        return excerpt

    @classmethod
    def _build_memory_store_tool(cls, tool_decorator: Any, home_dir: Path) -> Any:
        @tool_decorator(name="memory_store")
        def _memory_store(key: str, value: str) -> str:
            """Store a key-value pair in persistent memory without duplicating entries."""
            return cls._store_memory_value(home_dir=home_dir, key=key, value=value)

        return _memory_store

    @classmethod
    def _build_memory_recall_tool(cls, tool_decorator: Any, home_dir: Path) -> Any:
        @tool_decorator(name="memory_recall")
        def _memory_recall(query: str) -> str:
            """Search persistent memory for entries matching the query."""
            return cls._recall_memory_values(home_dir=home_dir, query=query)

        return _memory_recall

    @classmethod
    def _build_session_search_tool(cls, tool_decorator: Any, session_history_dir: Path) -> Any:
        @tool_decorator(name="session_search")
        def _session_search(query: str, max_results: int = 3) -> str:
            """Search persisted session history and return the most relevant prior records."""
            result = cls._search_session_history(session_history_dir, query, max_results=max_results)
            return json.dumps(result, ensure_ascii=False, indent=2)

        return _session_search

    def _build_task_tools(self, structured_tool: type[Any]) -> list[Any]:
        tools: list[Any] = []
        for tool, endpoint in matching_task_tools(
            self.request.tools or [],
            self.request.tool_endpoints or [],
        ):
            args_model = build_args_model(tool.name, tool.input_schema)

            async def _coroutine(
                _url: str = endpoint.url,
                _method: str = endpoint.method or "POST",
                **kwargs: Any,
            ) -> str:
                return invoke_http_tool(
                    url=_url,
                    method=_method,
                    arguments=kwargs,
                    timeout=30,
                )

            def _function(
                _url: str = endpoint.url,
                _method: str = endpoint.method or "POST",
                **kwargs: Any,
            ) -> str:
                return invoke_http_tool(
                    url=_url,
                    method=_method,
                    arguments=kwargs,
                    timeout=30,
                )

            tools.append(
                structured_tool.from_function(
                    func=_function,
                    coroutine=_coroutine,
                    name=tool.name,
                    description=tool.description,
                    args_schema=args_model,
                )
            )
        return tools

    @staticmethod
    @contextmanager
    def _override_home(home_dir: Path):
        previous_home = os.environ.get("HOME")
        os.environ["HOME"] = str(home_dir)
        try:
            yield
        finally:
            if previous_home is None:
                os.environ.pop("HOME", None)
            else:
                os.environ["HOME"] = previous_home

    @staticmethod
    def _capture_artifacts(
        *,
        workspace: Path,
        home_dir: Path,
        transcript: dict[str, Any],
        transcript_path: Path | None,
        session_history_dir: Path | None,
    ) -> None:
        if transcript_path is None:
            return
        artifacts_dir = transcript_path.parent
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        serialized_transcript = ZeroClawAdapter._serialize_result(transcript)
        memory_store_path = home_dir / ".zeroclaw" / "memory_store.json"
        if memory_store_path.exists():
            target = artifacts_dir / ".zeroclaw" / "memory_store.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(memory_store_path, target)
        skills_dir = workspace / "skills"
        if skills_dir.exists():
            target_skills = artifacts_dir / "skills"
            if target_skills.exists():
                shutil.rmtree(target_skills)
            shutil.copytree(skills_dir, target_skills)
        transcript_path.write_text(json.dumps(serialized_transcript, ensure_ascii=False, indent=2), encoding="utf-8")
        if session_history_dir is not None:
            session_history_dir.mkdir(parents=True, exist_ok=True)
            session_id = transcript_path.parent.parent.name if transcript_path.parent.parent.name else transcript_path.stem
            history_target = session_history_dir / f"{session_id}.json"
            history_target.write_text(json.dumps(serialized_transcript, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _serialize_result(result: dict[str, Any]) -> dict[str, Any]:
        serialized: list[dict[str, Any]] = []
        for message in result.get("messages", []):
            tool_calls = []
            for tool_call in getattr(message, "tool_calls", []) or []:
                tool_calls.append({
                    "id": tool_call.get("id"),
                    "name": tool_call.get("name"),
                    "args": tool_call.get("args") or {},
                    "function": {
                        "name": tool_call.get("name"),
                        "arguments": ZeroClawAdapter._json_safe(tool_call.get("args") or {}),
                    },
                })
            role = ZeroClawAdapter._message_role(message)
            serialized.append({
                "type": type(message).__name__,
                "role": role,
                "content": ZeroClawAdapter._json_safe(getattr(message, "content", "")),
                "tool_calls": tool_calls,
                "tool_call_id": getattr(message, "tool_call_id", None),
                "name": getattr(message, "name", None),
            })
        return {"messages": serialized}

    @staticmethod
    def _message_role(message: Any) -> str:
        message_type = str(getattr(message, "type", "") or "").lower()
        if message_type == "human":
            return "user"
        if message_type == "ai":
            return "assistant"
        if message_type == "tool":
            return "tool"
        if message_type == "system":
            return "system"
        return message_type or type(message).__name__

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, list):
            return [ZeroClawAdapter._json_safe(item) for item in value]
        if isinstance(value, dict):
            return {str(key): ZeroClawAdapter._json_safe(item) for key, item in value.items()}
        return str(value)
