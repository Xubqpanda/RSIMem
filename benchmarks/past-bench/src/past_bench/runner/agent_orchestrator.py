"""Bench-owned orchestration loop for decoupled agent runtimes."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..config import Config, MediaConfig, ModelConfig, PromptConfig
from ..models.content import TextBlock, ToolResultBlock
from ..models.message import Message
from ..models.task import TaskDefinition
from ..models.tool import ToolEndpoint
from ..models.trace import AuditSnapshot, RuntimeRequest, RuntimeResponse, TokenUsage, TraceEnd, TraceMessage, TraceStart
from ..runtime.client import create_runtime_client
from ..runtime.protocol import BootstrapRequest, CloseSessionRequest, RuntimeConfigPayload, StartSessionRequest, StepRequest
from ..runtime.registry import get_agent_spec, load_agent_registry, resolve_model_config
from ..trace.writer import TraceWriter
from .dispatcher import ToolDispatcher
from .loop import _build_initial_user_content
from .sandbox_dispatcher import SandboxToolDispatcher
from .sandbox_tools import SANDBOX_TOOLS
from .system_prompt import build_system_prompt


def _log(msg: str) -> None:
    print(msg, flush=True)


def _fetch_acontext_block(cfg: Config) -> str | None:
    """Fetch Acontext skills and return a formatted block for the system prompt.

    Returns None if Acontext is disabled, misconfigured, or unavailable.
    """
    ac = cfg.acontext
    if not ac.enabled or not ac.inject_skills or not ac.space_id:
        return None
    try:
        from acontext import AcontextClient  # type: ignore[import]
        client = AcontextClient(api_key=ac.api_key, base_url=ac.base_url)
        skills = client.learning_spaces.list_skills(ac.space_id)
        if not skills:
            _log("[acontext] no skills in space yet (cold start)")
            return None
        lines = [
            "## Acontext Memory & Skills",
            f"Learning space: `{ac.space_id}`",
            "The following skills were distilled from previous agent sessions.",
            "Use `get_skill` or read the skill file when a skill is relevant.",
            "",
            "<acontext_skills>",
        ]
        for skill in skills:
            lines += [
                "  <skill>",
                f"    <name>{skill.name}</name>",
                f"    <description>{skill.description}</description>",
                f"    <id>{skill.id}</id>",
                "  </skill>",
            ]
        lines.append("</acontext_skills>")
        block = "\n".join(lines)
        _log(f"[acontext] injected {len(skills)} skill(s) into system prompt")
        return block
    except ImportError:
        _log("[acontext] package not installed — skipping skill injection")
        return None
    except Exception as e:
        _log(f"[acontext] skill fetch failed: {e}")
        return None


def _store_acontext_session(cfg: Config, messages: list, task_id: str) -> None:
    """After a task run, store messages in Acontext and trigger learning."""
    ac = cfg.acontext
    if not ac.enabled or not ac.learning_enabled or not ac.space_id:
        return
    try:
        from acontext import AcontextClient  # type: ignore[import]
        client = AcontextClient(api_key=ac.api_key, base_url=ac.base_url)
        session = client.sessions.create(configs={"task_id": task_id})
        for msg in messages:
            role = getattr(msg, "role", None) or msg.get("role", "user")
            content = getattr(msg, "text", None) or str(msg)
            client.sessions.store_message(
                session.id,
                blob={"role": role, "content": content},
                format="openai",
            )
        client.learning_spaces.learn(ac.space_id, session_id=session.id)
        _log(f"[acontext] session {session.id} submitted for learning")
    except Exception as e:
        _log(f"[acontext] post-task learning failed: {e}")


def _message_content_to_text(content: list[Any]) -> str:
    parts: list[str] = []
    for block in content:
        block_type = getattr(block, "type", None) or block.get("type")
        if block_type == "text":
            text = getattr(block, "text", None) or block.get("text", "")
            if text:
                parts.append(str(text))
            continue
        if block_type == "tool_use":
            payload = {
                "type": "tool_use",
                "name": getattr(block, "name", None) or block.get("name"),
                "input": getattr(block, "input", None) or block.get("input", {}),
            }
            parts.append(json.dumps(payload, ensure_ascii=False))
            continue
        if block_type == "tool_result":
            result_content = getattr(block, "content", None) or block.get("content", [])
            payload = {
                "type": "tool_result",
                "tool_use_id": getattr(block, "tool_use_id", None) or block.get("tool_use_id"),
                "is_error": getattr(block, "is_error", None) or block.get("is_error", False),
                "content": _message_content_to_text(result_content),
            }
            parts.append(json.dumps(payload, ensure_ascii=False))
            continue
    return "\n".join(part for part in parts if part).strip()


def _store_acontext_trace(cfg: Config, trace_path: Path, task_id: str) -> None:
    """Store the full trace transcript in Acontext and trigger learning."""
    ac = cfg.acontext
    if not ac.enabled or not ac.learning_enabled or not ac.space_id:
        return
    try:
        from acontext import AcontextClient  # type: ignore[import]
        from ..trace.reader import load_trace

        client = AcontextClient(api_key=ac.api_key, base_url=ac.base_url)
        session = client.sessions.create(configs={"task_id": task_id})
        _, trace_messages, _, _, _, _ = load_trace(trace_path)

        stored = 0
        for event in trace_messages:
            message = event.message
            content = _message_content_to_text(message.content)
            if not content:
                continue
            client.sessions.store_message(
                session.id,
                blob={"role": message.role, "content": content},
                format="openai",
            )
            stored += 1

        client.learning_spaces.learn(ac.space_id, session_id=session.id)
        _log(f"[acontext] submitted trace {trace_path.name} with {stored} message(s) for learning")
    except Exception as e:
        _log(f"[acontext] post-task trace learning failed: {e}")


# ── Graphiti integration ──────────────────────────────────────────────────── #

def _fetch_graphiti_block(cfg: Config, task_id: str) -> str | None:
    """Query the Graphiti knowledge graph and return a context block for the
    system prompt.

    Uses ``asyncio.run()`` because Graphiti's public API is fully async.
    Returns None if Graphiti is disabled, the graph DB is unreachable, or
    ``graphiti-core`` is not installed.
    """
    gc = cfg.graphiti
    if not gc.enabled or not gc.inject_graph:
        return None
    try:
        import asyncio
        from skills.Graphiti import get_graphiti, get_context_block  # type: ignore[import]

        async def _fetch() -> str:
            g = await get_graphiti(
                backend=gc.backend,
                uri=gc.neo4j_uri,
                user=gc.neo4j_user,
                password=gc.neo4j_password,
                openai_api_key=gc.openai_api_key,
            )
            try:
                # Build a richer query so FTS can match stored episode facts.
                # Stored episodes use phrasing like "Agent 'X' completed task 'T100'".
                query = f"{task_id} agent completed score"
                block = await get_context_block(g, query, group_id=gc.group_id)
            finally:
                await g.close()
            return block

        block = asyncio.run(_fetch())
        if block:
            _log(f"[graphiti] injected graph context for task {task_id}")
        else:
            _log(f"[graphiti] no graph context yet for task {task_id} (cold start)")
        return block or None
    except ImportError:
        _log("[graphiti] graphiti-core not installed — skipping graph injection")
        return None
    except Exception as e:
        _log(f"[graphiti] context fetch failed: {e}")
        return None


def _store_graphiti_trace(
    cfg: Config,
    trace_path: Path,
    task_id: str,
    agent_name: str,
    task_score: float,
) -> None:
    """Ingest the completed task run into the Graphiti knowledge graph.

    Stores two things:
      1. A compact outcome episode (task + agent + score) for quick fact lookup.
      2. The full message trace as sequential episodes so entity relationships
         can be extracted from the conversation.
    """
    gc = cfg.graphiti
    if not gc.enabled or not gc.learning_enabled:
        return
    try:
        import asyncio
        from ..trace.reader import load_trace
        from skills.Graphiti import get_graphiti, add_task_episode, add_trace_episodes  # type: ignore[import]

        _, trace_messages, _, _, _, _ = load_trace(trace_path)
        messages = [
            {"role": event.message.role, "content": _message_content_to_text(event.message.content)}
            for event in trace_messages
        ]

        async def _store() -> None:
            g = await get_graphiti(
                backend=gc.backend,
                uri=gc.neo4j_uri,
                user=gc.neo4j_user,
                password=gc.neo4j_password,
                openai_api_key=gc.openai_api_key,
            )
            try:
                await add_task_episode(
                    g,
                    task_id=task_id,
                    agent_name=agent_name,
                    score=task_score,
                    group_id=gc.group_id,
                )
                await add_trace_episodes(
                    g,
                    messages,
                    task_id=task_id,
                    agent_name=agent_name,
                    group_id=gc.group_id,
                )
            finally:
                await g.close()

        asyncio.run(_store())
        _log(f"[graphiti] ingested trace for {task_id} (score={task_score:.3f})")
    except ImportError:
        _log("[graphiti] graphiti-core not installed — skipping graph ingestion")
    except Exception as e:
        _log(f"[graphiti] post-task ingestion failed: {e}")


# ── Letta integration ─────────────────────────────────────────────────────── #

def _fetch_letta_block(cfg: Config) -> str | None:
    """Fetch the PAST-Bench memory agent's MemoryBlocks and return a formatted
    block for the system prompt.

    Returns None if Letta is disabled, the server is unreachable, or
    letta-client is not installed.
    """
    lc = cfg.letta
    if not lc.enabled or not lc.inject_blocks:
        return None
    try:
        from skills.Letta import get_client, ensure_memory_agent, get_context_block  # type: ignore[import]

        client = get_client(api_key=lc.api_key, base_url=lc.base_url)
        agent_id = lc.agent_id or ensure_memory_agent(client)
        block = get_context_block(client, agent_id)
        if block:
            _log("[letta] injected MemoryBlocks into system prompt")
        else:
            _log("[letta] no memory blocks yet (cold start)")
        return block or None
    except ImportError:
        _log("[letta] letta-client not installed — skipping block injection")
        return None
    except Exception as e:
        _log(f"[letta] block fetch failed: {e}")
        return None


def _store_letta_outcome(
    cfg: Config,
    task_id: str,
    agent_name: str,
    task_score: float,
) -> None:
    """Send the completed task outcome to the Letta memory agent so it can
    self-edit its MemoryBlocks with new patterns and insights.
    """
    lc = cfg.letta
    if not lc.enabled or not lc.learning_enabled:
        return
    try:
        from skills.Letta import get_client, ensure_memory_agent  # type: ignore[import]
        from skills.Letta.agent import send_task_outcome  # type: ignore[import]

        client = get_client(api_key=lc.api_key, base_url=lc.base_url)
        agent_id = lc.agent_id or ensure_memory_agent(client)
        send_task_outcome(
            client,
            agent_id,
            task_id=task_id,
            agent_name=agent_name,
            score=task_score,
        )
        _log(f"[letta] sent outcome for {task_id} (score={task_score:.3f}) to memory agent")
    except ImportError:
        _log("[letta] letta-client not installed — skipping outcome storage")
    except Exception as e:
        _log(f"[letta] outcome storage failed: {e}")


# ── mem0 integration ──────────────────────────────────────────────────────── #

def _fetch_mem0_block(cfg: Config, task_id: str) -> str | None:
    """Search mem0 memories relevant to task_id and return a context block.

    Returns None if mem0 is disabled, unavailable, or no memories exist yet.
    """
    mc = cfg.mem0
    if not mc.enabled or not mc.inject_memories:
        return None
    try:
        from skills.mem0 import get_memory, MemoryStore  # type: ignore[import]

        memory = get_memory(api_key=mc.api_key, host=mc.host)
        store = MemoryStore(memory, user_id=mc.user_id, agent_id=mc.agent_id)
        query = f"{task_id} agent task memory"
        block = store.get_context_block(query, limit=mc.search_limit)
        if block:
            _log(f"[mem0] injected memory context for task {task_id}")
        else:
            _log(f"[mem0] no memories yet for task {task_id} (cold start)")
        return block or None
    except ImportError:
        _log("[mem0] mem0ai not installed — skipping memory injection")
        return None
    except Exception as e:
        _log(f"[mem0] memory fetch failed: {e}")
        return None


def _store_mem0_trace(
    cfg: Config,
    trace_path: Path,
    task_id: str,
) -> None:
    """Extract and store salient facts from the completed task trace into mem0."""
    mc = cfg.mem0
    if not mc.enabled or not mc.learning_enabled:
        return
    try:
        from ..trace.reader import load_trace
        from skills.mem0 import get_memory, MemoryStore  # type: ignore[import]

        _, trace_messages, _, _, _, _ = load_trace(trace_path)
        messages = [
            {"role": event.message.role, "content": _message_content_to_text(event.message.content)}
            for event in trace_messages
        ]

        memory = get_memory(api_key=mc.api_key, host=mc.host)
        store = MemoryStore(memory, user_id=mc.user_id, agent_id=mc.agent_id)
        store.add_from_trace(messages, task_id=task_id)
        _log(f"[mem0] stored trace memories for {task_id}")
    except ImportError:
        _log("[mem0] mem0ai not installed — skipping memory storage")
    except Exception as e:
        _log(f"[mem0] trace storage failed: {e}")


def _merge_dicts(base: dict[str, Any] | None, override: dict[str, Any] | None) -> dict[str, Any] | None:
    if base is None and override is None:
        return None
    if base is None:
        return dict(override or {})
    if override is None:
        return dict(base)

    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _merge_dicts(current, value)
        else:
            merged[key] = value
    return merged


def _collect_audit_snapshots(writer: TraceWriter, trace_id: str, task: TaskDefinition) -> None:
    import httpx

    for svc in task.services:
        if not svc.reset_endpoint:
            continue
        audit_url = svc.reset_endpoint.rsplit("/reset", 1)[0] + "/audit"
        try:
            resp = httpx.get(audit_url, timeout=5)
            writer.write_event(AuditSnapshot(
                trace_id=trace_id,
                service_name=svc.name,
                audit_url=audit_url,
                audit_data=resp.json(),
            ))
        except Exception:
            pass


_SANDBOX_ENDPOINT_PATHS: dict[str, str] = {
    "sandbox_shell_exec": "/exec",
    "sandbox_file_read": "/read",
    "sandbox_file_write": "/write",
    "sandbox_browser_screenshot": "/screenshot",
}


def _runtime_host_url(url: str, *, runtime_mode: str) -> str:
    if runtime_mode != "container":
        return url
    url = url.replace("://localhost", "://host.docker.internal")
    return url.replace("://127.0.0.1", "://host.docker.internal")


def _runtime_tool_endpoints(
    task: TaskDefinition,
    *,
    runtime_mode: str,
    sandbox_tools: bool = False,
    sandbox_url: str | None = None,
) -> list[ToolEndpoint]:
    endpoints: list[ToolEndpoint] = []
    for endpoint in task.tool_endpoints:
        endpoints.append(endpoint.model_copy(update={"url": _runtime_host_url(endpoint.url, runtime_mode=runtime_mode)}))
    if sandbox_tools and sandbox_url:
        runtime_sandbox_url = _runtime_host_url(sandbox_url, runtime_mode=runtime_mode)
        for tool_name, path in _SANDBOX_ENDPOINT_PATHS.items():
            endpoints.append(
                ToolEndpoint(
                    tool_name=tool_name,
                    url=f"{runtime_sandbox_url}{path}",
                    method="POST",
                )
            )
    return endpoints


def run_task_via_agent(
    task: TaskDefinition,
    *,
    agent_name: str,
    agent_profile: str | None = None,
    cfg: Config,
    trace_dir: str | Path = "traces",
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    runtime_mode: str = "local",
    runtime_image: str | None = None,
    registry_path: str | None = None,
    model_extra_body: dict[str, Any] | None = None,
    sandbox_tools: bool = False,
    sandbox_url: str | None = None,
    prompt_cfg: PromptConfig | None = None,
    model_cfg: ModelConfig | None = None,
    media_cfg: MediaConfig | None = None,
    runtime_temperature: float = 0.0,
    task_timeout_override: int | None = None,
) -> Path:
    """Run one task by delegating model turns to a decoupled agent runtime."""

    effective_timeout_seconds = (
        task_timeout_override
        if task_timeout_override is not None
        else task.environment.timeout_seconds
    )

    trace_id = str(uuid4())
    trace_path = Path(trace_dir) / f"{task.task_id}_{trace_id[:8]}.jsonl"
    registry = load_agent_registry(registry_path or cfg.runtime.registry_path or cfg.defaults.agent_registry)
    agent_spec = get_agent_spec(agent_name, registry)
    resolved_model = resolve_model_config(
        agent_spec,
        model=model,
        api_key=api_key,
        base_url=base_url,
        profile=agent_profile,
    )
    resolved_model = resolved_model.model_copy(
        update={"extra_body": _merge_dicts(resolved_model.extra_body, model_extra_body)}
    )

    endpoint_map = task.get_endpoint_map()
    http_dispatcher = ToolDispatcher(endpoint_map)
    if sandbox_tools:
        existing_names = {tool.name for tool in task.tools}
        sandbox_tool_list = [tool for tool in SANDBOX_TOOLS if tool.name not in existing_names]
        task_tools = list(task.tools) + sandbox_tool_list
        dispatcher = SandboxToolDispatcher(http_dispatcher, sandbox_url=sandbox_url)
    else:
        sandbox_tool_list = None
        task_tools = list(task.tools)
        dispatcher = http_dispatcher

    client = create_runtime_client(
        cfg.runtime,
        mode=runtime_mode,
        registry_path=registry_path or cfg.runtime.registry_path or cfg.defaults.agent_registry,
        image=runtime_image or agent_spec.runtime_image,
    )

    total_usage = TokenUsage()
    tool_time_s = 0.0
    model_time_s = 0.0
    wall_start = time.monotonic()
    turn_count = 0
    pending_tool_results: list[ToolResultBlock] = []
    session_id = trace_id

    _log(f"[start] task={task.task_id} agent={agent_name} model={resolved_model.model_id} trace={trace_path.name}")
    _log(
        "[config] "
        f"max_turns={task.environment.max_turns} "
        f"timeout={effective_timeout_seconds}s"
        + (f" (override; task default {task.environment.timeout_seconds}s)" if task_timeout_override is not None else "")
        + f" sandbox_tools={sandbox_tools} runtime={runtime_mode} "
        f"temperature={runtime_temperature}"
    )

    with TraceWriter(trace_path) as writer:
        writer.write_event(TraceStart(
            trace_id=trace_id,
            task_id=task.task_id,
            model=resolved_model.model_id,
            persona=agent_name,
            runtime_temperature=runtime_temperature,
        ))

        acontext_block = _fetch_acontext_block(cfg)
        graphiti_block = _fetch_graphiti_block(cfg, task.task_id)
        letta_block = _fetch_letta_block(cfg)
        mem0_block = _fetch_mem0_block(cfg, task.task_id)
        system_prompt = build_system_prompt(
            task, prompt_cfg,
            extra_tools=sandbox_tool_list,
            acontext_skills_block=acontext_block,
            graphiti_context_block=graphiti_block,
            letta_context_block=letta_block,
            mem0_context_block=mem0_block,
        )
        if model_cfg and model_cfg.system_prompt_prefix:
            system_prompt = model_cfg.system_prompt_prefix + "\n\n" + system_prompt
        user_content = _build_initial_user_content(
            task,
            trace_id=trace_id,
            writer=writer,
            model_cfg=model_cfg,
            media_cfg=media_cfg,
        )
        initial_messages = [
            Message(role="system", content=[TextBlock(text=system_prompt)]),
            Message(role="user", content=user_content),
        ]
        writer.write_event(TraceMessage(trace_id=trace_id, message=initial_messages[-1]))

        loop_error: str | None = None
        loop_exc: Exception | None = None
        try:
            client.start(run_id=trace_id[:8], agent_spec=agent_spec)
            client.bootstrap(BootstrapRequest(agent_name=agent_name))
            client.start_session(StartSessionRequest(
                session_id=session_id,
                agent_name=agent_name,
                task_id=task.task_id,
                task_name=task.task_name,
                max_turns=task.environment.max_turns,
                timeout_seconds=effective_timeout_seconds,
                initial_messages=initial_messages,
                tools=task_tools,
                tool_endpoints=_runtime_tool_endpoints(
                    task,
                    runtime_mode=runtime_mode,
                    sandbox_tools=sandbox_tools,
                    sandbox_url=sandbox_url,
                ),
                model=resolved_model,
                runtime_config=RuntimeConfigPayload(temperature=runtime_temperature),
            ))

            while turn_count < task.environment.max_turns:
                elapsed = time.monotonic() - wall_start
                if elapsed > effective_timeout_seconds:
                    _log(f"[timeout] {elapsed:.1f}s exceeded limit {effective_timeout_seconds}s")
                    break

                step_request = StepRequest(
                    session_id=session_id,
                    step_id=turn_count,
                    tool_results=pending_tool_results,
                )
                writer.write_event(RuntimeRequest(
                    trace_id=trace_id,
                    session_id=session_id,
                    step_id=turn_count,
                    payload=step_request.model_dump(mode="json"),
                ))
                _log(f"[turn {turn_count + 1}/{task.environment.max_turns}] calling runtime ...")
                response = client.step(step_request)
                model_time_s += response.model_time_s
                total_usage.input_tokens += response.usage.input_tokens
                total_usage.output_tokens += response.usage.output_tokens
                writer.write_event(RuntimeResponse(
                    trace_id=trace_id,
                    session_id=session_id,
                    step_id=turn_count,
                    status=response.status,
                    payload=response.model_dump(mode="json"),
                ))

                assistant_message = response.assistant_message
                if assistant_message is None and response.final_output is not None:
                    assistant_message = Message(
                        role="assistant",
                        content=[TextBlock(text=response.final_output)],
                    )
                if assistant_message is not None:
                    writer.write_event(TraceMessage(
                        trace_id=trace_id,
                        message=assistant_message,
                        usage=response.usage,
                    ))

                turn_count += 1
                if response.status == "error":
                    raise RuntimeError(response.error or "runtime returned error")

                if response.status == "finished":
                    _log(f"[done] runtime finished at turn {turn_count}")
                    break

                result_blocks: list[ToolResultBlock] = []
                for tool_call in response.tool_calls:
                    tool_use = next(
                        (
                            block for block in (assistant_message.content if assistant_message else [])
                            if block.type == "tool_use" and block.id == tool_call.tool_use_id
                        ),
                        None,
                    )
                    if tool_use is None:
                        raise RuntimeError(f"Runtime omitted tool_use block for {tool_call.name}")
                    _log(f"  -> tool: {tool_call.name}")
                    result, dispatch_event = dispatcher.dispatch(tool_use, trace_id)
                    writer.write_event(dispatch_event)
                    result_blocks.append(result)
                    tool_time_s += dispatch_event.latency_ms / 1000.0
                    status_tag = "OK" if not result.is_error else "ERR"
                    _log(f"  <- {tool_call.name}: {status_tag} ({dispatch_event.latency_ms:.0f}ms)")

                pending_tool_results = result_blocks
                if result_blocks:
                    tool_msg = Message(role="user", content=result_blocks)
                    writer.write_event(TraceMessage(trace_id=trace_id, message=tool_msg))
                else:
                    pending_tool_results = []
        except Exception as exc:
            loop_error = f"{type(exc).__name__}: {exc}"
            loop_exc = exc
            _log(f"[error] runtime loop failed: {loop_error}")
        finally:
            try:
                client.close_session(CloseSessionRequest(session_id=session_id, reason=loop_error or "completed"))
            except Exception:
                pass
            client.stop()
            dispatcher.close()
            _collect_audit_snapshots(writer, trace_id, task)
            wall_time = time.monotonic() - wall_start
            input_tok = total_usage.input_tokens
            output_tok = total_usage.output_tokens
            total_tok = input_tok + output_tok
            other_time_s = max(0.0, wall_time - model_time_s - tool_time_s)
            writer.write_event(TraceEnd(
                trace_id=trace_id,
                total_turns=turn_count,
                model_input_tokens=input_tok,
                model_output_tokens=output_tok,
                input_tokens=input_tok,
                output_tokens=output_tok,
                total_tokens=total_tok,
                model_time_s=round(model_time_s, 2),
                tool_time_s=round(tool_time_s, 2),
                other_time_s=round(other_time_s, 2),
                wall_time_s=round(wall_time, 2),
                failure_modes=[loop_error] if loop_error else [],
            ))

        if loop_error:
            raise loop_exc

    _log(
        f"[end] turns={turn_count} tokens={total_usage.input_tokens + total_usage.output_tokens} "
        f"({total_usage.input_tokens}in/{total_usage.output_tokens}out) "
        f"time=model {model_time_s:.1f}s tool {tool_time_s:.1f}s wall {time.monotonic() - wall_start:.1f}s"
    )
    _store_acontext_trace(cfg, trace_path, task.task_id)
    _store_graphiti_trace(cfg, trace_path, task.task_id, agent_name, task_score=0.0)
    _store_letta_outcome(cfg, task.task_id, agent_name, task_score=0.0)
    _store_mem0_trace(cfg, trace_path, task.task_id)
    return trace_path
