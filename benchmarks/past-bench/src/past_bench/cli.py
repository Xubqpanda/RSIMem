"""CLI for task runs, batch runs, agent runtime ops, and grading."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import mean

# Ensure localhost traffic (mock services) bypasses any HTTP proxy,
# while external API requests (OpenRouter etc.) still go through proxy.
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")


def _resolve_task_yaml(task_arg: str) -> Path:
    """Resolve --task to a YAML file path.

    Accepts either a directory (past_bench_tasks/T70_api_deprecation_supersession_audit)
    or a file (past_bench_tasks/T70_api_deprecation_supersession_audit/task.yaml).
    """
    p = Path(task_arg).resolve()
    if p.is_dir():
        yaml_path = p / "task.yaml"
        if not yaml_path.exists():
            raise FileNotFoundError(f"No task.yaml found in {p}")
        return yaml_path
    return p


def _resolve_tasks_dir(task_yaml: Path) -> Path:
    """Return the task-pack root for a task YAML file.

    Supports both classic layouts like ``tasks/<ID>/task.yaml`` and deeper
    self-evolve layouts like ``self-evolve-tasks/<line>/<ID>/task.yaml``.
    """
    resolved = task_yaml.resolve()
    for parent in resolved.parents:
        if parent.name in {
            "tasks",
            "self-evolve-tasks",
            "self-evolve-tasks-v2",
        }:
            return parent
    return resolved.parent.parent


def _make_trace_dir(base_dir: str | Path, model_id: str) -> Path:
    """Build a trace output directory: ``<base_dir>/<YYYYMMDD_HHMMSS>_<model>/``.

    Model names like ``anthropic/claude-opus-4-6`` are sanitised to
    ``anthropic_claude-opus-4-6`` (slashes replaced with underscores).
    """
    from datetime import datetime

    date_str = datetime.now().strftime("%y-%m-%d-%H-%M")
    safe_model = model_id.replace("/", "_")
    trace_dir = Path(base_dir) / f"{safe_model}_{date_str}"
    trace_dir.mkdir(parents=True, exist_ok=True)
    return trace_dir


def _make_judge(cfg, args):
    """Create an LLMJudge instance if enabled, or None.

    Falls back through available API keys when the primary judge key
    (from config.yaml) is not set. Probe order:
      1. Config judge settings (config.yaml / CLI overrides)
      2. OPENROUTER_API_KEY → OpenRouter gemini-2.5-flash
      3. OPENAI_API_KEY     → OpenAI gpt-4o-mini
      4. ANTHROPIC_API_KEY  → Anthropic claude-haiku
      5. ZAI_API_KEY        → GLM glm-4.5-air (anthropic-compat)
      6. KIMI_CODE_API_KEY  → Kimi kimi-k2.6
      7. MINIMAX_API_KEY    → MiniMax MiniMax-M2.7
      8. DEEPSEEK_API_KEY   → DeepSeek deepseek-v4-pro (anthropic-compat)
    """
    if getattr(args, "no_judge", False):
        return None
    if not cfg.judge.enabled:
        return None

    api_key = cfg.judge.api_key
    base_url = cfg.judge.base_url
    model_id = getattr(args, "judge_model", None) or cfg.judge.model_id
    user_set_model = getattr(args, "judge_model", None)

    # Fallback chain: try available provider keys in order
    if not api_key:
        _JUDGE_FALLBACKS = [
            ("OPENROUTER_API_KEY", "https://openrouter.ai/api/v1",    "google/gemini-2.5-flash"),
            ("OPENAI_API_KEY",     "https://api.openai.com/v1",       "gpt-4o-mini"),
            ("ANTHROPIC_API_KEY",  "https://api.anthropic.com",        "claude-haiku-4-5-20251001"),
            ("ZAI_API_KEY",        "https://api.z.ai/api/anthropic",  "glm-4.5-air"),
            ("KIMI_CODE_API_KEY",  "https://api.kimi.com/coding/v1",  "kimi-k2.6"),
            ("MINIMAX_API_KEY",    "https://api.minimaxi.com/anthropic",     "MiniMax-M2.7"),
            ("DEEPSEEK_API_KEY",   "https://api.deepseek.com/anthropic",     "deepseek-v4-pro"),
        ]
        for env_var, fallback_url, fallback_model in _JUDGE_FALLBACKS:
            key = os.environ.get(env_var)
            if key:
                api_key = key
                base_url = fallback_url
                if not user_set_model:
                    model_id = fallback_model
                break

    if not api_key:
        return None

    from .graders.llm_judge import LLMJudge

    return LLMJudge(
        model_id=model_id,
        api_key=api_key,
        base_url=base_url,
    )


def _apply_proxy(proxy_url: str | None) -> None:
    """Set HTTP(S)_PROXY env vars for model/judge API traffic.

    Mock services are unaffected because ``services.py`` strips proxy vars
    from subprocess environments, and ``no_proxy`` already covers localhost.
    """
    if not proxy_url:
        return
    os.environ["HTTP_PROXY"] = proxy_url
    os.environ["HTTPS_PROXY"] = proxy_url
    os.environ["http_proxy"] = proxy_url
    os.environ["https_proxy"] = proxy_url
    print(f"[proxy] Model/judge traffic via {proxy_url}")


def _grade_with_optional_params(
    grader, messages, dispatches, task,
    *, audit_data, judge, media_events, env_snapshot=None,
):
    """Call grader.grade, passing optional params only when the grader accepts them."""
    from .graders.base import AbstractGrader

    params = inspect.signature(grader.grade).parameters
    kwargs = {"audit_data": audit_data, "judge": judge}
    if "media_events" in params:
        kwargs["media_events"] = media_events
    if "env_snapshot" in params and env_snapshot is not None:
        kwargs["env_snapshot"] = env_snapshot
    scores = grader.grade(messages, dispatches, task, **kwargs)
    return scores


def _collect_env_snapshot(sandbox_url: str, task) -> dict:
    """Collect environment data from the container after the agent loop finishes.

    Called between agent loop completion and container destruction.
    What to collect is declared in task.yaml via ``env_snapshot_files``
    and ``env_snapshot_commands``.

    Individual collection failures are recorded as ``{"error": ...}``
    entries in the snapshot dict rather than aborting the entire snapshot.
    """
    import httpx

    client = httpx.Client(timeout=10.0)
    snapshot: dict = {}

    try:
        for pattern in getattr(task, "env_snapshot_files", []):
            try:
                if "*" in pattern or "?" in pattern:
                    resp = client.post(
                        f"{sandbox_url}/glob",
                        json={"pattern": pattern, "max_files": 50},
                    )
                    file_list = resp.json().get("files", [])
                    for f in file_list:
                        try:
                            resp2 = client.post(
                                f"{sandbox_url}/read",
                                json={"path": f["path"]},
                            )
                            snapshot[f"file:{f['path']}"] = resp2.json()
                        except Exception as exc:
                            snapshot[f"file:{f['path']}"] = {"error": str(exc)}
                            print(f"[WARNING] env_snapshot file read failed: {f['path']}: {exc}")
                else:
                    resp = client.post(
                        f"{sandbox_url}/read",
                        json={"path": pattern},
                    )
                    snapshot[f"file:{pattern}"] = resp.json()
            except Exception as exc:
                snapshot[f"file:{pattern}"] = {"error": str(exc)}
                print(f"[WARNING] env_snapshot file failed: {pattern}: {exc}")

        for cmd in getattr(task, "env_snapshot_commands", []):
            try:
                resp = client.post(
                    f"{sandbox_url}/exec",
                    json={"command": cmd, "timeout_seconds": 10},
                )
                snapshot[f"cmd:{cmd}"] = resp.json()
            except Exception as exc:
                snapshot[f"cmd:{cmd}"] = {"error": str(exc)}
                print(f"[WARNING] env_snapshot command failed: {cmd}: {exc}")
    finally:
        client.close()

    return snapshot


def _trace_totals(end) -> dict[str, int | float]:
    """Extract model token/time totals from a TraceEnd event."""
    if end is None:
        return {
            "model_input_tokens": 0,
            "model_output_tokens": 0,
            "total_tokens": 0,
            "model_time_s": 0.0,
            "tool_time_s": 0.0,
            "other_time_s": 0.0,
            "wall_time_s": 0.0,
        }

    model_input_tokens = getattr(end, "model_input_tokens", getattr(end, "input_tokens", 0))
    model_output_tokens = getattr(end, "model_output_tokens", getattr(end, "output_tokens", 0))
    total_tokens = getattr(end, "total_tokens", model_input_tokens + model_output_tokens)
    model_time_s = getattr(end, "model_time_s", 0.0)
    tool_time_s = getattr(end, "tool_time_s", 0.0)
    other_time_s = getattr(end, "other_time_s", 0.0)
    wall_time_s = getattr(end, "wall_time_s", 0.0)

    # Backward compatibility for older traces.
    if not total_tokens:
        total_tokens = model_input_tokens + model_output_tokens
    if not other_time_s and wall_time_s:
        other_time_s = max(0.0, wall_time_s - model_time_s - tool_time_s)

    return {
        "model_input_tokens": model_input_tokens,
        "model_output_tokens": model_output_tokens,
        "total_tokens": total_tokens,
        "model_time_s": wall_time_s if not model_time_s and not tool_time_s else model_time_s,
        "tool_time_s": tool_time_s,
        "other_time_s": other_time_s,
        "wall_time_s": wall_time_s,
    }


def _resolve_registry_path(args: argparse.Namespace, cfg) -> str:
    return getattr(args, "registry", None) or cfg.runtime.registry_path or cfg.defaults.agent_registry


def _resolve_runtime_mode(args: argparse.Namespace, cfg) -> str:
    return getattr(args, "runtime", None) or cfg.runtime.mode


def _resolve_runtime_temperature(args: argparse.Namespace, cfg) -> float:
    value = getattr(args, "temperature", None)
    if value is not None:
        return float(value)
    return float(cfg.runtime.temperature)


def _check_agent_requirements(
    agent_name: str,
    registry_path: str,
    *,
    explicit_api_key: str | None = None,
    profile: str | None = None,
):
    from .runtime.registry import get_agent_spec, load_agent_registry, missing_required_env, resolve_model_defaults

    registry = load_agent_registry(registry_path)
    spec = get_agent_spec(agent_name, registry)
    _, defaults = resolve_model_defaults(spec, profile=profile)
    missing = []
    for env_name in missing_required_env(spec, profile=profile):
        if explicit_api_key and env_name == defaults.api_key_env:
            continue
        missing.append(env_name)
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(f"Missing required environment variables for {agent_name}: {joined}")
    return spec


def _execute_trial(
    *,
    task,
    cfg,
    task_dir: str,
    trace_dir: str,
    model: str | None,
    api_key: str | None,
    base_url: str | None,
    sandbox_mode: bool,
    sandbox_image: str | None,
    sandbox_tools_local: bool = False,
    agent_name: str | None = None,
    agent_profile: str | None = None,
    runtime_mode: str | None = None,
    runtime_image: str | None = None,
    registry_path: str | None = None,
    model_extra_body_override: dict | None = None,
    runtime_temperature: float | None = None,
    task_timeout_override: int | None = None,
    runtime_metadata: dict | None = None,
):
    """Execute one trial with either the legacy or decoupled runtime path."""
    from .runner.loop import run_task
    from .runner.providers.openai_compat import OpenAICompatProvider

    provider = None
    if agent_name is None:
        provider = OpenAICompatProvider(
            model_id=model or cfg.model.model_id,
            api_key=api_key or cfg.model.api_key,
            base_url=base_url or cfg.model.base_url,
            extra_body=cfg.model.extra_body,
            temperature=0.0 if runtime_temperature is None else runtime_temperature,
        )

    sandbox_runner = None
    if sandbox_mode:
        from .runner.sandbox_runner import SandboxRunner

        sandbox_runner = SandboxRunner(cfg.sandbox, image=sandbox_image or cfg.sandbox.image)

    env_snapshot = None
    if sandbox_runner:
        run_id = f"{task.task_id}-{os.getpid()}-{time.time_ns()}"
        handle = sandbox_runner.start_container(run_id=run_id)
        try:
            n_injected = sandbox_runner.inject_files(handle, task, task_dir=task_dir)
            expected_files = len(task.sandbox_files) if task.sandbox_files else len(getattr(task.environment, "fixtures", []))
            if expected_files and n_injected < expected_files:
                print(f"[WARNING] inject_files: only {n_injected}/{expected_files} files injected")
            if agent_name is not None:
                from .runner.agent_orchestrator import run_task_via_agent

                trace_path = run_task_via_agent(
                    task,
                    agent_name=agent_name,
                    agent_profile=agent_profile,
                    cfg=cfg,
                    trace_dir=trace_dir,
                    model=model,
                    api_key=api_key,
                    base_url=base_url,
                    runtime_mode=runtime_mode or cfg.runtime.mode,
                    runtime_image=runtime_image,
                    registry_path=registry_path,
                    model_extra_body=model_extra_body_override,
                    sandbox_tools=True,
                    sandbox_url=handle.sandbox_url,
                    prompt_cfg=cfg.prompt,
                    model_cfg=cfg.model,
                    media_cfg=cfg.media,
                    runtime_temperature=0.0 if runtime_temperature is None else runtime_temperature,
                    task_timeout_override=task_timeout_override,
                    runtime_metadata=runtime_metadata,
                )
            else:
                trace_path = run_task(
                    task,
                    provider,
                    trace_dir=trace_dir,
                    sandbox_tools=True,
                    sandbox_url=handle.sandbox_url,
                    prompt_cfg=cfg.prompt,
                    model_cfg=cfg.model,
                    media_cfg=cfg.media,
                )
            n_grader = sandbox_runner.inject_grader_files(handle, task, task_dir=task_dir)
            if task.sandbox_grader_files and n_grader < len(task.sandbox_grader_files):
                print(f"[WARNING] inject_grader_files: only {n_grader}/{len(task.sandbox_grader_files)} files injected")
            env_snapshot = _collect_env_snapshot(handle.sandbox_url, task)
        finally:
            sandbox_runner.stop_container(handle)
        return trace_path, env_snapshot

    if agent_name is not None:
        from .runner.agent_orchestrator import run_task_via_agent

        trace_path = run_task_via_agent(
            task,
            agent_name=agent_name,
            agent_profile=agent_profile,
            cfg=cfg,
            trace_dir=trace_dir,
            model=model,
            api_key=api_key,
            base_url=base_url,
            runtime_mode=runtime_mode or cfg.runtime.mode,
            runtime_image=runtime_image,
            registry_path=registry_path,
            model_extra_body=model_extra_body_override,
            sandbox_tools=sandbox_tools_local,
            prompt_cfg=cfg.prompt,
            model_cfg=cfg.model,
            media_cfg=cfg.media,
            runtime_temperature=0.0 if runtime_temperature is None else runtime_temperature,
            task_timeout_override=task_timeout_override,
            runtime_metadata=runtime_metadata,
        )
        return trace_path, None

    trace_path = run_task(
        task,
        provider,
        trace_dir=trace_dir,
        sandbox_tools=sandbox_tools_local,
        prompt_cfg=cfg.prompt,
        model_cfg=cfg.model,
        media_cfg=cfg.media,
    )
    return trace_path, None


def cmd_run(args: argparse.Namespace) -> None:
    """Run an agent on a task."""
    _apply_proxy(getattr(args, "proxy", None))

    from .config import load_config
    from .graders.registry import get_grader
    from .models.scoring import compute_pass_at_k, compute_pass_hat_k, compute_task_score, is_pass
    from .models.task import TaskDefinition
    from .runner.services import ServiceManager
    from .trace.reader import load_trace

    cfg = load_config(args.config)
    runtime_temperature = _resolve_runtime_temperature(args, cfg)

    task_yaml = _resolve_task_yaml(args.task)
    task = TaskDefinition.from_yaml(task_yaml)
    tasks_dir = _resolve_tasks_dir(task_yaml)

    port_offset = getattr(args, "port_offset", 0) or 0
    if port_offset:
        task.apply_port_offset(port_offset)

    registry_path = _resolve_registry_path(args, cfg)
    runtime_mode = _resolve_runtime_mode(args, cfg)
    agent_name = getattr(args, "agent", None)
    if agent_name:
        from .runtime.registry import resolve_model_defaults

        agent_profile = getattr(args, "agent_profile", None)
        spec = _check_agent_requirements(
            agent_name,
            registry_path,
            explicit_api_key=args.api_key,
            profile=agent_profile,
        )
        _, defaults = resolve_model_defaults(spec, profile=agent_profile)
        model_id = args.model or defaults.model_id or cfg.model.model_id
    else:
        model_id = args.model or cfg.model.model_id

    base_trace_dir = args.trace_dir or cfg.defaults.trace_dir
    trace_dir = _make_trace_dir(base_trace_dir, model_id)
    judge = _make_judge(cfg, args)
    sandbox_mode = getattr(args, "sandbox", False) or cfg.sandbox.enabled
    sandbox_tools = getattr(args, "sandbox_tools", False)
    trials = args.trials or 1
    trial_scores: list[float] = []
    trace_paths: list[Path] = []

    with ServiceManager(task.services, cwd=tasks_dir.parent) as svc:
        for i in range(trials):
            if trials > 1:
                print(f"\n--- Trial {i + 1}/{trials} ---")
            if i > 0:
                svc.reset_all()

            trace_path, env_snapshot = _execute_trial(
                task=task,
                cfg=cfg,
                task_dir=str(task_yaml.parent),
                trace_dir=str(trace_dir),
                model=args.model,
                api_key=args.api_key,
                base_url=args.base_url,
                sandbox_mode=sandbox_mode,
                sandbox_image=getattr(args, "sandbox_image", None),
                sandbox_tools_local=sandbox_tools,
                agent_name=agent_name,
                agent_profile=getattr(args, "agent_profile", None),
                runtime_mode=runtime_mode,
                runtime_image=getattr(args, "runtime_image", None),
                registry_path=registry_path,
                runtime_temperature=runtime_temperature,
                task_timeout_override=getattr(args, "task_timeout_override", None),
            )
            trace_paths.append(trace_path)
            print(f"Trace: {trace_path}")

            start, messages, dispatches, media_events, end, audit_data = load_trace(trace_path)
            grader = get_grader(task.task_id, tasks_dir=tasks_dir, task_dir=task_yaml.parent)
            scores = _grade_with_optional_params(
                grader, messages, dispatches, task,
                audit_data=audit_data, judge=judge, media_events=media_events,
                env_snapshot=env_snapshot,
            )
            task_score = compute_task_score(scores)
            passed = is_pass(task_score)
            _append_grading_to_trace(
                trace_path,
                trace_id=start.trace_id,
                task_id=task.task_id,
                scores=scores,
                task_score=task_score,
                passed=passed,
            )
            trial_scores.append(task_score)
            totals = _trace_totals(end)

            print(f"  completion:     {scores.completion:.2f}")
            print(f"  robustness:     {scores.robustness:.2f}")
            print(f"  communication:  {scores.communication:.2f}")
            print(f"  safety:         {scores.safety:.1f}")
            print(f"  task_score:     {task_score:.2f}")
            print(f"  passed:         {passed}")
            print(
                f"  model_tokens:   {totals['total_tokens']} "
                f"({totals['model_input_tokens']} in / {totals['model_output_tokens']} out)"
            )
            print(
                f"  time_s:         wall={totals['wall_time_s']:.2f} "
                f"model={totals['model_time_s']:.2f} tool={totals['tool_time_s']:.2f} "
                f"other={totals['other_time_s']:.2f}"
            )

    if trials > 1:
        print(f"\n--- Multi-trial summary ({trials} trials) ---")
        for i, (score, path) in enumerate(zip(trial_scores, trace_paths)):
            print(f"  Trial {i+1}: score={score:.2f} pass={is_pass(score)} trace={path}")
        pass_at_1 = compute_pass_at_k(trial_scores, k=1)
        pass_hat_k = compute_pass_hat_k(trial_scores, k=trials)
        print(f"  pass@1:  {pass_at_1:.3f}")
        print(f"  pass^{trials}:  {pass_hat_k:.3f}")


def cmd_run_inner(args: argparse.Namespace) -> None:
    """Run a single trial inside a sandbox container (internal command)."""
    _apply_proxy(getattr(args, "proxy", None))

    from .config import load_config
    from .graders.registry import get_grader
    from .models.scoring import compute_task_score, is_pass
    from .models.task import TaskDefinition
    from .runner.loop import run_task
    from .runner.providers.openai_compat import OpenAICompatProvider
    from .runner.services import ServiceManager
    from .trace.reader import load_trace

    cfg = load_config(args.config)

    task_yaml = _resolve_task_yaml(args.task)
    task = TaskDefinition.from_yaml(task_yaml)
    tasks_dir = _resolve_tasks_dir(task_yaml)

    model_id = args.model or cfg.model.model_id
    provider = OpenAICompatProvider(
        model_id=model_id,
        api_key=args.api_key or cfg.model.api_key or os.environ.get("OPENAI_API_KEY"),
        base_url=args.base_url or cfg.model.base_url,
        extra_body=cfg.model.extra_body,
        temperature=float(cfg.runtime.temperature),
    )

    sandbox_tools = getattr(args, "sandbox_tools", False)
    # _run-inner receives the final trace dir from the caller (e.g. submit script).
    # Only fall back to _make_trace_dir when --trace-dir is not provided.
    if args.trace_dir:
        trace_dir = Path(args.trace_dir)
        trace_dir.mkdir(parents=True, exist_ok=True)
    else:
        trace_dir = _make_trace_dir(cfg.defaults.trace_dir, model_id)

    with ServiceManager(task.services, cwd=tasks_dir.parent):
        trace_path = run_task(
            task, provider,
            trace_dir=trace_dir,
            sandbox_tools=sandbox_tools,
            prompt_cfg=cfg.prompt,
            model_cfg=cfg.model,
            media_cfg=cfg.media,
        )

    print(f"Trace: {trace_path}")

    # --- Inline grading ---
    judge = _make_judge(cfg, args)
    start, messages, dispatches, media_events, end, audit_data = load_trace(trace_path)
    grader = get_grader(task.task_id, tasks_dir=tasks_dir, task_dir=task_yaml.parent)
    scores = _grade_with_optional_params(
        grader, messages, dispatches, task,
        audit_data=audit_data, judge=judge, media_events=media_events,
    )
    task_score = compute_task_score(scores)
    passed = is_pass(task_score)

    totals = _trace_totals(end)
    result = {
        "task_id": task.task_id,
        "task_name": task.task_name,
        "model": provider.model_id,
        "trace": trace_path.name,
        "turns": end.total_turns if end else 0,
        "model_input_tokens": totals["model_input_tokens"],
        "model_output_tokens": totals["model_output_tokens"],
        "input_tokens": totals["model_input_tokens"],
        "output_tokens": totals["model_output_tokens"],
        "tokens": totals["total_tokens"],
        "model_time_s": totals["model_time_s"],
        "tool_time_s": totals["tool_time_s"],
        "other_time_s": totals["other_time_s"],
        "wall_time_s": totals["wall_time_s"],
        "completion": scores.completion,
        "robustness": scores.robustness,
        "communication": scores.communication,
        "safety": scores.safety,
        "task_score": task_score,
        "passed": passed,
    }
    result_path = trace_path.with_suffix(".result.json")
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Result: {result_path}")
    print(f"  task_score={task_score:.3f}  passed={passed}")


def cmd_build_image(args: argparse.Namespace) -> None:
    """Build either the sandbox or runtime Docker image."""
    from .config import load_config

    cfg = load_config(getattr(args, "config", None))
    context = getattr(args, "context", ".")
    kind = getattr(args, "kind", "sandbox")

    if kind == "runtime":
        from .runtime.container import RuntimeContainerManager

        image = getattr(args, "image", None) or cfg.runtime.image
        dockerfile = getattr(args, "dockerfile", None) or "Dockerfile.runtime"
        runner = RuntimeContainerManager(cfg.runtime, image=image)
        runner.build_image(context_path=context, dockerfile=dockerfile)
        return

    from .runner.sandbox_runner import SandboxRunner

    image = getattr(args, "image", None) or cfg.sandbox.image
    dockerfile = getattr(args, "dockerfile", None) or "Dockerfile.agent"
    runner = SandboxRunner(cfg.sandbox, image=image)
    runner.build_image(context_path=context, dockerfile=dockerfile)


def cmd_grade(args: argparse.Namespace) -> None:
    """Grade an existing trace file."""
    _apply_proxy(getattr(args, "proxy", None))

    from .config import load_config
    from .graders.registry import get_grader
    from .models.scoring import compute_task_score, is_pass
    from .models.task import TaskDefinition
    from .trace.reader import load_trace

    cfg = load_config(args.config if hasattr(args, "config") else None)
    judge = _make_judge(cfg, args)

    start, messages, dispatches, media_events, end, audit_data = load_trace(args.trace)

    task_yaml = _resolve_task_yaml(args.task)
    task = TaskDefinition.from_yaml(task_yaml)
    tasks_dir = _resolve_tasks_dir(task_yaml)

    grader = get_grader(task.task_id, tasks_dir=tasks_dir, task_dir=task_yaml.parent)
    scores = _grade_with_optional_params(
        grader, messages, dispatches, task,
        audit_data=audit_data, judge=judge, media_events=media_events,
    )
    task_score = compute_task_score(scores)
    passed = is_pass(task_score)

    print(f"Trace:   {args.trace}")
    print(f"Task:    {task.task_id} ({task.task_name})")
    print(f"Model:   {start.model}")
    print(f"Turns:   {end.total_turns if end else '?'}")
    totals = _trace_totals(end)
    print(
        f"Tokens:  {totals['total_tokens']} "
        f"({totals['model_input_tokens']} in / {totals['model_output_tokens']} out)"
    )
    print(
        f"Time:    wall={totals['wall_time_s']:.2f}s "
        f"model={totals['model_time_s']:.2f}s "
        f"tool={totals['tool_time_s']:.2f}s "
        f"other={totals['other_time_s']:.2f}s"
    )
    print()
    print(f"completion:     {scores.completion:.2f}")
    print(f"robustness:     {scores.robustness:.2f}")
    print(f"communication:  {scores.communication:.2f}")
    print(f"safety:         {scores.safety:.1f}")
    print(f"task_score:     {task_score:.2f}")
    print(f"passed:         {passed}")


def _append_grading_to_trace(
    trace_path: Path,
    trace_id: str,
    task_id: str,
    scores,
    task_score: float,
    passed: bool,
) -> None:
    """Append a grading_result event to the end of a trace JSONL file."""
    from .models.trace import GradingResult, DimensionScores

    def _score_value(name: str, default=0.0):
        if isinstance(scores, dict):
            return scores.get(name, default)
        return getattr(scores, name, default)

    event = GradingResult(
        trace_id=trace_id,
        task_id=task_id,
        scores=DimensionScores(
            completion=_score_value("completion"),
            robustness=_score_value("robustness"),
            communication=_score_value("communication"),
            safety=_score_value("safety", 1.0),
        ),
        task_score=task_score,
        passed=passed,
    )
    with open(trace_path, "a") as fh:
        fh.write(event.model_dump_json() + "\n")
    _sync_trace_end_with_grading(
        trace_path,
        scores=scores,
        task_score=task_score,
        passed=passed,
    )


def _sync_trace_end_with_grading(
    trace_path: Path,
    scores,
    task_score: float,
    passed: bool,
) -> None:
    """Rewrite the trace_end event so it reflects final graded results.

    The runner writes trace_end before grading occurs, so its score fields stay
    at default zero values unless we patch them after grading.
    """
    def _score_value(name: str, default=0.0):
        if isinstance(scores, dict):
            return scores.get(name, default)
        return getattr(scores, name, default)

    lines = trace_path.read_text(encoding="utf-8").splitlines()
    updated = False

    for idx in range(len(lines) - 1, -1, -1):
        line = lines[idx].strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") != "trace_end":
            continue
        event["scores"] = {
            "completion": _score_value("completion"),
            "robustness": _score_value("robustness"),
            "communication": _score_value("communication"),
            "safety": _score_value("safety", 1.0),
            "efficiency_turns": _score_value("efficiency_turns", 0),
            "efficiency_tokens": _score_value("efficiency_tokens", 0),
            "efficiency_wall_time_s": _score_value("efficiency_wall_time_s", 0.0),
        }
        event["task_score"] = task_score
        event["passed"] = passed
        lines[idx] = json.dumps(event, ensure_ascii=False)
        updated = True
        break

    if updated:
        trace_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_single_task(
    task_dir: str,
    config_path: str | None,
    model: str | None,
    api_key: str | None,
    base_url: str | None,
    trace_dir: str | None,
    port_offset: int,
    no_judge: bool,
    judge_model: str | None,
    trials: int,
    proxy: str | None = None,
    sandbox: bool = False,
    sandbox_image: str | None = None,
    sandbox_tools: bool = False,
    agent: str | None = None,
    agent_profile: str | None = None,
    runtime: str | None = None,
    runtime_image: str | None = None,
    registry: str | None = None,
    temperature: float | None = None,
) -> dict:
    """Run a single task in a worker process. Returns a result dict."""
    # Ensure localhost bypasses proxy in worker processes.
    os.environ.setdefault("no_proxy", "localhost,127.0.0.1")
    os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
    # Re-apply proxy for model/judge API calls (services.py strips proxy
    # from mock-service subprocesses independently).
    _apply_proxy(proxy)

    from .config import load_config
    from .graders.registry import get_grader
    from .models.scoring import compute_pass_at_k, compute_pass_hat_k, compute_task_score, is_pass
    from .models.task import TaskDefinition
    from .runner.services import ServiceManager
    from .trace.reader import load_trace

    task_yaml = _resolve_task_yaml(task_dir)
    task = TaskDefinition.from_yaml(task_yaml)
    tasks_dir = _resolve_tasks_dir(task_yaml)

    if port_offset:
        task.apply_port_offset(port_offset)

    cfg = load_config(config_path)
    runtime_temperature = float(cfg.runtime.temperature if temperature is None else temperature)
    registry_path = registry or cfg.runtime.registry_path or cfg.defaults.agent_registry
    runtime_mode = runtime or cfg.runtime.mode
    if agent:
        _check_agent_requirements(
            agent,
            registry_path,
            explicit_api_key=api_key,
            profile=agent_profile,
        )

    # Build judge if needed
    judge = None
    if not no_judge and cfg.judge.enabled:
        _j_api_key = cfg.judge.api_key
        _j_base_url = cfg.judge.base_url
        _j_model_id = judge_model or cfg.judge.model_id
        if not _j_api_key:
            _JUDGE_FALLBACKS = [
                ("OPENROUTER_API_KEY", "https://openrouter.ai/api/v1",    "google/gemini-2.5-flash"),
                ("OPENAI_API_KEY",     "https://api.openai.com/v1",       "gpt-4o-mini"),
                ("ANTHROPIC_API_KEY",  "https://api.anthropic.com",        "claude-haiku-4-5-20251001"),
                ("ZAI_API_KEY",        "https://api.z.ai/api/anthropic",  "glm-4.5-air"),
                ("KIMI_CODE_API_KEY",  "https://api.kimi.com/coding/v1",  "kimi-k2.6"),
                ("MINIMAX_API_KEY",    "https://api.minimaxi.com/anthropic",     "MiniMax-M2.7"),
            ]
            for env_var, fallback_url, fallback_model in _JUDGE_FALLBACKS:
                key = os.environ.get(env_var)
                if key:
                    _j_api_key = key
                    _j_base_url = fallback_url
                    if not judge_model:
                        _j_model_id = fallback_model
                    break
        if _j_api_key:
            from .graders.llm_judge import LLMJudge
            judge = LLMJudge(
                model_id=_j_model_id,
                api_key=_j_api_key,
                base_url=_j_base_url,
            )

    sandbox_mode = sandbox or cfg.sandbox.enabled

    result = {
        "task_id": task.task_id,
        "task_name": task.task_name,
        "difficulty": task.difficulty,
        "agent": agent or "",
        "trials": [],
        "error": None,
    }

    import time
    from openai import APIConnectionError, APITimeoutError, InternalServerError

    max_retries = 3
    for attempt in range(max_retries):
        result["trials"] = []
        result["error"] = None
        try:
            with ServiceManager(task.services, cwd=tasks_dir.parent) as svc:
                for i in range(trials):
                    if i > 0:
                        svc.reset_all()

                    try:
                        trace_path, env_snapshot = _execute_trial(
                            task=task,
                            cfg=cfg,
                            task_dir=task_dir,
                            trace_dir=trace_dir or cfg.defaults.trace_dir,
                            model=model,
                            api_key=api_key,
                            base_url=base_url,
                            sandbox_mode=sandbox_mode,
                            sandbox_image=sandbox_image,
                            sandbox_tools_local=sandbox_tools,
                            agent_name=agent,
                            agent_profile=agent_profile,
                            runtime_mode=runtime_mode,
                            runtime_image=runtime_image,
                            registry_path=registry_path,
                            runtime_temperature=runtime_temperature,
                        )

                        start, messages, dispatches, media_events, end, audit_data = load_trace(trace_path)
                        grader = get_grader(task.task_id, tasks_dir=tasks_dir, task_dir=task_dir)
                        scores = _grade_with_optional_params(
                            grader, messages, dispatches, task,
                            audit_data=audit_data, judge=judge, media_events=media_events,
                            env_snapshot=env_snapshot,
                        )
                        task_score = compute_task_score(scores)
                        _append_grading_to_trace(
                            trace_path,
                            trace_id=start.trace_id,
                            task_id=task.task_id,
                            scores=scores,
                            task_score=task_score,
                            passed=is_pass(task_score),
                        )
                        totals = _trace_totals(end)
                        result["trials"].append({
                            "trace": str(trace_path),
                            "model_input_tokens": totals["model_input_tokens"],
                            "model_output_tokens": totals["model_output_tokens"],
                            "input_tokens": totals["model_input_tokens"],
                            "output_tokens": totals["model_output_tokens"],
                            "tokens": totals["total_tokens"],
                            "model_time_s": totals["model_time_s"],
                            "tool_time_s": totals["tool_time_s"],
                            "other_time_s": totals["other_time_s"],
                            "wall_time_s": totals["wall_time_s"],
                            "completion": scores.completion,
                            "robustness": scores.robustness,
                            "communication": scores.communication,
                            "safety": scores.safety,
                            "task_score": task_score,
                            "passed": is_pass(task_score),
                        })
                    except Exception as trial_exc:
                        result["trials"].append({
                            "trial": i,
                            "error": str(trial_exc),
                            "task_score": 0.0,
                            "passed": False,
                        })
            break  # success — exit retry loop
        except (APIConnectionError, APITimeoutError, InternalServerError, ConnectionError) as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt  # 1s, 2s, 4s, 8s
                print(f"  [{task.task_id}] retry {attempt + 1}/{max_retries} after {type(e).__name__}, waiting {wait}s")
                time.sleep(wait)
            else:
                result["error"] = str(e)
        except Exception as e:
            result["error"] = str(e)
            break  # non-retryable error

    # Compute multi-trial aggregate metrics (exclude errored trials)
    valid_trials = [t for t in result["trials"] if not t.get("error")]
    if not valid_trials and result["trials"]:
        # All trials errored — propagate as task-level error for summary stats
        result["error"] = result["trials"][0].get("error", "all trials errored")
    trial_scores = [t["task_score"] for t in valid_trials]
    n_trials = len(trial_scores)
    if n_trials > 0:
        result["avg_score"] = sum(trial_scores) / n_trials
        result["pass_at_1"] = compute_pass_at_k(trial_scores, k=1)
        result["pass_hat_k"] = compute_pass_hat_k(trial_scores, k=n_trials)
        result["avg_passed"] = is_pass(result["avg_score"])
    else:
        result["avg_score"] = 0.0
        result["pass_at_1"] = 0.0
        result["pass_hat_k"] = 0.0
        result["avg_passed"] = False

    return result


def _scan_completed_trials(trace_dir: Path) -> dict[str, int]:
    """Scan a trace directory and return {task_id: completed_trial_count}.

    A trial is considered complete if its JSONL file contains a grading_result event.
    """
    from collections import defaultdict

    completed: dict[str, int] = defaultdict(int)
    for f in trace_dir.glob("*.jsonl"):
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if ev.get("type") == "grading_result":
                    task_id = ev.get("task_id", "")
                    if task_id:
                        completed[task_id] += 1
                    break  # one grading_result per file is enough
    return dict(completed)


def _load_completed_results(trace_dir: Path) -> list[dict]:
    """Load per-trial results from grading_result events in a trace directory.

    Returns a list of result dicts (one per task_id) with trials populated from
    the grading_result events found in JSONL files. This allows merging with
    new results when using --continue.
    """
    from collections import defaultdict

    # task_id -> list of trial info dicts
    task_trials: dict[str, list[dict]] = defaultdict(list)

    for f in sorted(trace_dir.glob("*.jsonl")):
        grading = None
        trace_end = None
        for line_str in open(f):
            line_str = line_str.strip()
            if not line_str:
                continue
            try:
                ev = json.loads(line_str)
            except json.JSONDecodeError:
                continue
            if ev.get("type") == "grading_result":
                grading = ev
            elif ev.get("type") == "trace_end":
                trace_end = ev

        if grading is None:
            continue

        task_id = grading.get("task_id", "")
        if not task_id:
            continue

        scores = grading.get("scores", {})
        trial_info = {
            "trace": str(f),
            "model_input_tokens": trace_end.get("model_input_tokens", 0) if trace_end else 0,
            "model_output_tokens": trace_end.get("model_output_tokens", 0) if trace_end else 0,
            "input_tokens": trace_end.get("model_input_tokens", 0) if trace_end else 0,
            "output_tokens": trace_end.get("model_output_tokens", 0) if trace_end else 0,
            "tokens": trace_end.get("total_tokens", 0) if trace_end else 0,
            "model_time_s": trace_end.get("model_time_s", 0.0) if trace_end else 0.0,
            "tool_time_s": trace_end.get("tool_time_s", 0.0) if trace_end else 0.0,
            "other_time_s": trace_end.get("other_time_s", 0.0) if trace_end else 0.0,
            "wall_time_s": trace_end.get("wall_time_s", 0.0) if trace_end else 0.0,
            "completion": scores.get("completion", 0.0),
            "robustness": scores.get("robustness", 0.0),
            "communication": scores.get("communication", 0.0),
            "safety": scores.get("safety", 1.0),
            "task_score": grading.get("task_score", 0.0),
            "passed": grading.get("passed", False),
        }
        task_trials[task_id].append(trial_info)

    # Build result dicts per task
    from .models.scoring import compute_pass_at_k, compute_pass_hat_k, is_pass

    results = []
    for task_id, trials in task_trials.items():
        trial_scores = [t["task_score"] for t in trials]
        n = len(trial_scores)
        result = {
            "task_id": task_id,
            "task_name": "",
            "difficulty": "",
            "trials": trials,
            "error": None,
        }
        if n > 0:
            result["avg_score"] = sum(trial_scores) / n
            result["pass_at_1"] = compute_pass_at_k(trial_scores, k=1)
            result["pass_hat_k"] = compute_pass_hat_k(trial_scores, k=n)
            result["avg_passed"] = is_pass(result["avg_score"])
        results.append(result)

    return results


def _fmt_duration(seconds: float) -> str:
    """Format seconds as e.g. '3m22s' or '1h05m'."""
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


def cmd_batch(args: argparse.Namespace) -> None:
    """Run all (or filtered) tasks in parallel.

    Supports multi-agent evaluation: pass comma-separated agent names
    (e.g. ``--agent zeroclaw,hermes,claude``) to evaluate multiple agents
    on the same task set in parallel.  Each (agent, task) pair is an
    independent work item in the process pool.
    """
    _apply_proxy(getattr(args, "proxy", None))

    from .config import load_config as _load_cfg_early

    _cfg_early = _load_cfg_early(args.config)
    _raw_agent = getattr(args, "agent", None)
    registry_path = _resolve_registry_path(args, _cfg_early)
    runtime_mode = _resolve_runtime_mode(args, _cfg_early)

    # Parse comma-separated agent list (e.g. "zeroclaw,hermes,claude")
    agent_names: list[str | None] = [None]  # default: no agent (baseline)
    if _raw_agent:
        agent_names = [a.strip() for a in _raw_agent.split(",") if a.strip()]

    # Resolve model IDs per agent for trace directory naming
    _model_ids: dict[str | None, str] = {}
    for agent_name in agent_names:
        if agent_name:
            from .runtime.registry import resolve_model_defaults

            agent_profile = getattr(args, "agent_profile", None)
            spec = _check_agent_requirements(
                agent_name,
                registry_path,
                explicit_api_key=args.api_key,
                profile=agent_profile,
            )
            _, defaults = resolve_model_defaults(spec, profile=agent_profile)
            _model_ids[agent_name] = args.model or defaults.model_id or _cfg_early.model.model_id
        else:
            _model_ids[agent_name] = args.model or _cfg_early.model.model_id
    _model_id = _model_ids[agent_names[0]]
    multi_agent = len(agent_names) > 1

    tasks_dir = Path(args.tasks_dir)
    if not tasks_dir.exists():
        print(f"Tasks directory not found: {tasks_dir}")
        sys.exit(1)

    # --rerun-errors: load previous results and filter to errored tasks only
    rerun_dir = getattr(args, "rerun_errors", None)
    prev_results: list[dict] | None = None
    errored_task_ids: set[str] = set()
    if rerun_dir:
        rerun_path = Path(rerun_dir)
        prev_results_file = rerun_path / "batch_results.json"
        if not prev_results_file.exists():
            print(f"batch_results.json not found in {rerun_path}")
            sys.exit(1)
        with open(prev_results_file) as f:
            prev_results = json.load(f)
        errored_task_ids = {r["task_id"] for r in prev_results if r.get("error")}
        if not errored_task_ids:
            print("No errored tasks found in previous run — nothing to rerun.")
            return
        print(f"[rerun-errors] Found {len(errored_task_ids)} errored tasks to rerun:")
        for tid in sorted(errored_task_ids):
            err_msg = next((r["error"] for r in prev_results if r["task_id"] == tid), "")
            print(f"  {tid}: {err_msg[:80]}")
        print()

    # --continue: scan existing trace dir for completed trials
    continue_dir = getattr(args, "continue_dir", None)
    completed_trials: dict[str, int] = {}
    continue_prev_results: list[dict] = []
    if continue_dir:
        continue_path = Path(continue_dir)
        if not continue_path.exists():
            print(f"Continue directory not found: {continue_path}")
            sys.exit(1)
        completed_trials = _scan_completed_trials(continue_path)
        continue_prev_results = _load_completed_results(continue_path)
        total_completed = sum(completed_trials.values())
        print(f"[continue] Scanning {continue_path} — found {total_completed} completed trial(s) "
              f"across {len(completed_trials)} task(s)")
        if completed_trials:
            for tid in sorted(completed_trials):
                print(f"  {tid}: {completed_trials[tid]} trial(s) done")
            print()

    # Discover tasks
    task_dirs = sorted(
        str(d) for d in tasks_dir.iterdir()
        if d.is_dir() and (d / "task.yaml").exists()
    )
    if args.filter:
        filt = args.filter.lower()
        task_dirs = [d for d in task_dirs if filt in d.lower()]

    # If rerunning errors, only keep the errored task dirs
    if errored_task_ids:
        task_dirs = [d for d in task_dirs if Path(d).name in errored_task_ids]

    workers = args.parallel
    trials = args.trials or 1

    # If continuing, filter out fully-completed tasks and compute remaining trials per task
    skipped_task_ids: set[str] = set()
    remaining_trials: dict[str, int] = {}  # task_dir -> number of trials still needed
    if continue_dir:
        remaining_dirs = []
        for d in task_dirs:
            task_id = Path(d).name
            done = completed_trials.get(task_id, 0)
            if done >= trials:
                skipped_task_ids.add(task_id)
            else:
                remaining_dirs.append(d)
                remaining_trials[d] = trials - done
        n_skipped = len(task_dirs) - len(remaining_dirs)
        task_dirs = remaining_dirs
        if n_skipped:
            print(f"[continue] Skipping {n_skipped} task(s) with {trials}+ completed trial(s)")
        for d in task_dirs:
            needed = remaining_trials[d]
            if needed < trials:
                print(f"  {Path(d).name}: {trials - needed}/{trials} done, running {needed} more")

    if not task_dirs:
        if continue_dir:
            print("All tasks already completed — nothing to run.")
        else:
            print("No tasks matched.")
        return

    total_tasks = len(task_dirs)

    # Build a shared trace output directory for this batch run
    _base_trace_dir = args.trace_dir or _cfg_early.defaults.trace_dir

    if rerun_dir:
        # Reuse the existing trace directory
        batch_trace_dir = str(Path(rerun_dir))
    elif continue_dir:
        # Reuse the continue trace directory
        batch_trace_dir = str(Path(continue_dir))
    else:
        batch_trace_dir = str(_make_trace_dir(_base_trace_dir, _model_id))

    # ── Multi-agent work-item expansion ──
    # Each work item is (agent_name, task_dir).  For single-agent runs this
    # is just [(agent, td) for td in task_dirs].  For multi-agent runs each
    # task is evaluated by every agent, with per-agent trace sub-directories.
    work_items: list[tuple[str | None, str]] = []
    agent_trace_dirs: dict[str | None, str] = {}
    for ag in agent_names:
        if multi_agent:
            ag_trace = str(Path(batch_trace_dir) / (ag or "baseline"))
            Path(ag_trace).mkdir(parents=True, exist_ok=True)
        else:
            ag_trace = batch_trace_dir
        agent_trace_dirs[ag] = ag_trace
        for td in task_dirs:
            work_items.append((ag, td))

    total = len(work_items)

    if multi_agent:
        print(f"Multi-agent evaluation: {len(agent_names)} agents × {total_tasks} tasks = {total} work items")
        print(f"  Agents: {', '.join(a or 'baseline' for a in agent_names)}")
    else:
        print(f"Running {total_tasks} tasks with {workers} parallel workers, {trials} trial(s) each")
    print(f"Traces → {batch_trace_dir}\n")

    # Per-agent result tracking for multi-agent summary
    agent_results: dict[str | None, list[dict]] = {ag: [] for ag in agent_names}

    results: list[dict] = []
    # Progress tracking
    start_time = time.monotonic()
    n_pass_hat = 0      # pass^k: all trials passed
    n_pass_at = 0       # pass@k: at least one trial passed
    score_sum = 0.0
    finished_tasks = 0

    # Each worker slot gets a unique port offset: slot 0 → 0, slot 1 → 50, ...
    # Tasks use ports 9100-9129 (span=30); stride of 50 leaves headroom.
    # We map futures to their assigned slot so we can recycle offsets.
    with ProcessPoolExecutor(max_workers=workers) as pool:
        # Slot pool: available port offsets
        available_slots = list(range(workers))
        pending: dict = {}  # future → (agent, task_dir, slot_index)

        work_queue = list(work_items)
        finished = 0

        port_base_offset = getattr(args, "port_base_offset", 0)

        # Sanity check: max port must stay below ephemeral range (32768)
        _STRIDE = 50  # port gap between adjacent worker slots
        max_port = 9129 + port_base_offset + (workers - 1) * _STRIDE
        if max_port >= 32768:
            max_safe = (32767 - 9129 - port_base_offset) // _STRIDE + 1
            print(
                f"[ERROR] --port-base-offset {port_base_offset} with {workers} workers "
                f"would use port {max_port} (>=32768, collides with ephemeral range). "
                f"Max workers for this offset: {max_safe}"
            )
            return

        def _submit(item: tuple[str | None, str]) -> None:
            ag, td = item
            slot = available_slots.pop(0)
            offset = port_base_offset + slot * _STRIDE
            # Use per-task remaining trials when continuing, otherwise full trials
            task_trials = remaining_trials.get(td, trials)
            fut = pool.submit(
                _run_single_task,
                task_dir=td,
                config_path=args.config,
                model=args.model,
                api_key=args.api_key,
                base_url=args.base_url,
                trace_dir=agent_trace_dirs[ag],
                port_offset=offset,
                no_judge=args.no_judge,
                judge_model=getattr(args, "judge_model", None),
                trials=task_trials,
                proxy=getattr(args, "proxy", None),
                sandbox=getattr(args, "sandbox", False),
                sandbox_image=getattr(args, "sandbox_image", None),
                sandbox_tools=getattr(args, "sandbox_tools", False),
                agent=ag,
                agent_profile=getattr(args, "agent_profile", None),
                runtime=runtime_mode,
                runtime_image=getattr(args, "runtime_image", None),
                registry=registry_path,
                temperature=getattr(args, "temperature", None),
            )
            pending[fut] = (ag, td, slot)

        # Seed initial batch
        while work_queue and available_slots:
            _submit(work_queue.pop(0))

        # Process completions
        while pending:
            for fut in as_completed(pending):
                ag, td, slot = pending.pop(fut)
                available_slots.append(slot)
                finished += 1

                try:
                    res = fut.result()
                except Exception as e:
                    res = {"task_id": Path(td).name, "error": str(e), "trials": []}

                # Tag result with agent name for multi-agent tracking
                if multi_agent:
                    res["agent"] = ag or "baseline"
                results.append(res)
                agent_results[ag].append(res)

                # Incrementally write batch_results.json after each task
                _partial_out = Path(batch_trace_dir)
                _partial_out.mkdir(parents=True, exist_ok=True)
                _partial_file = _partial_out / "batch_results.json"
                try:
                    with open(_partial_file, "w") as _pf:
                        json.dump(results, _pf, indent=2, ensure_ascii=False)
                except Exception:
                    pass  # best-effort; don't crash on incremental write failure

                # Update progress counters
                finished_tasks += 1
                if res.get("error"):
                    score_sum += 0.0
                else:
                    trials_list = res["trials"]
                    score_sum += sum(tr["task_score"] for tr in trials_list) / len(trials_list)
                    if all(tr["passed"] for tr in trials_list):
                        n_pass_hat += 1
                    if any(tr["passed"] for tr in trials_list):
                        n_pass_at += 1

                # Print task result
                tid = res.get("task_id", Path(td).name)
                ag_label = f"[{ag}] " if multi_agent else ""
                if res.get("error"):
                    print(f"  [{finished}/{total}] {ag_label}{tid}: ERROR — {res['error'][:80]}")
                else:
                    for i, tr in enumerate(res["trials"]):
                        label = f" trial {i+1}" if trials > 1 else ""
                        status = "PASS" if tr["passed"] else "FAIL"
                        print(
                            f"  [{finished}/{total}] {ag_label}{tid}{label}: {tr['task_score']:.2f} {status} "
                            f"| tok={tr.get('tokens', 0)} "
                            f"({tr.get('model_input_tokens', tr.get('input_tokens', 0))} in/"
                            f"{tr.get('model_output_tokens', tr.get('output_tokens', 0))} out) "
                            f"| time=wall {tr.get('wall_time_s', 0.0):.2f}s "
                            f"model {tr.get('model_time_s', 0.0):.2f}s "
                            f"tool {tr.get('tool_time_s', 0.0):.2f}s"
                        )
                    if trials > 1 and res["trials"]:
                        avg_s = res.get("avg_score", 0.0)
                        avg_status = "PASS" if res.get("avg_passed", False) else "FAIL"
                        print(
                            f"  [{finished}/{total}] {tid} avg: {avg_s:.2f} {avg_status} "
                            f"| pass@1={res.get('pass_at_1', 0.0):.2f} "
                            f"pass^{trials}={res.get('pass_hat_k', 0.0):.2f}"
                        )

                # Print progress bar
                elapsed = time.monotonic() - start_time
                pct = finished * 100 // total
                if finished < total:
                    eta = elapsed / finished * (total - finished)
                    eta_str = f" | ETA ~{_fmt_duration(eta)}"
                else:
                    eta_str = ""
                avg_score = score_sum / finished_tasks if finished_tasks else 0.0
                print(
                    f"  [Progress] {finished}/{total} done ({pct}%) "
                    f"| avg {avg_score:.2f} "
                    f"pass^{trials} {n_pass_hat}/{finished_tasks} "
                    f"pass@{trials} {n_pass_at}/{finished_tasks} "
                    f"| elapsed {_fmt_duration(elapsed)}{eta_str}"
                )

                # Submit next work item if any
                if work_queue and available_slots:
                    _submit(work_queue.pop(0))

                break  # restart as_completed loop with updated pending

    # --- Merge with previous results if rerunning errors ---
    if prev_results is not None:
        rerun_by_id = {r["task_id"]: r for r in results}
        still_errored = sum(1 for r in results if r.get("error"))
        fixed = len(results) - still_errored
        print(f"\n[rerun-errors] {fixed}/{len(results)} previously errored tasks now succeeded"
              f" ({still_errored} still errored)")

        # Merge: replace errored entries in prev_results with new results
        merged = []
        for prev in prev_results:
            if prev["task_id"] in rerun_by_id:
                merged.append(rerun_by_id[prev["task_id"]])
            else:
                merged.append(prev)
        results = merged
        total = len(results)

    # --- Merge with previously completed results if continuing ---
    # Re-scan all JSONL traces to build authoritative results (avoids
    # stale / partial data from the in-memory `results` list, which only
    # contains tasks that were re-run in *this* invocation).
    if continue_dir:
        all_from_traces = _load_completed_results(Path(continue_dir))
        if all_from_traces:
            results = all_from_traces
            total = len(results)
            print(f"\n[continue] Rebuilt results from {total} task(s) in trace directory")

    # --- Summary ---
    print(f"\n{'='*60}")
    if prev_results is not None:
        print(f"BATCH COMPLETE (rerun-errors merge) — {total} tasks")
    elif continue_dir:
        print(f"BATCH COMPLETE (continue merge) — {total} tasks")
    else:
        print(f"BATCH COMPLETE — {total} tasks, {workers} workers")
    print(f"{'='*60}\n")

    errored = sum(1 for r in results if r.get("error"))
    avg_score_final = score_sum / finished_tasks if finished_tasks else 0.0
    total_model_input_tokens = sum(
        tr.get("model_input_tokens", tr.get("input_tokens", 0))
        for r in results for tr in r.get("trials", [])
    )
    total_model_output_tokens = sum(
        tr.get("model_output_tokens", tr.get("output_tokens", 0))
        for r in results for tr in r.get("trials", [])
    )
    total_tokens = sum(tr.get("tokens", 0) for r in results for tr in r.get("trials", []))
    total_model_time_s = sum(tr.get("model_time_s", 0.0) for r in results for tr in r.get("trials", []))
    total_tool_time_s = sum(tr.get("tool_time_s", 0.0) for r in results for tr in r.get("trials", []))
    total_other_time_s = sum(tr.get("other_time_s", 0.0) for r in results for tr in r.get("trials", []))
    total_wall_time_s = sum(tr.get("wall_time_s", 0.0) for r in results for tr in r.get("trials", []))

    print(f"  Avg score: {avg_score_final:.3f}")
    print(f"  pass^{trials}: {n_pass_hat}/{finished_tasks}")
    print(f"  pass@{trials}: {n_pass_at}/{finished_tasks}")
    print(f"  Errored: {errored}/{finished_tasks}")
    print(
        f"  Total model tokens: {total_tokens} "
        f"({total_model_input_tokens} in / {total_model_output_tokens} out)"
    )
    print(
        f"  Total time: wall={total_wall_time_s:.2f}s "
        f"model={total_model_time_s:.2f}s tool={total_tool_time_s:.2f}s "
        f"other={total_other_time_s:.2f}s"
    )

    print(f"\n{'─'*60}")
    # Sort by task_id for readability
    for r in sorted(results, key=lambda x: x.get("task_id", "")):
        tid = r.get("task_id", "?")
        if r.get("error"):
            print(f"  {tid:40s}  ERROR: {r['error'][:50]}")
        elif r["trials"]:
            valid_trials = [t for t in r["trials"] if not t.get("error")]
            if not valid_trials:
                tr = r["trials"][0]
                print(f"  {tid:40s}  0.00  ERR   {tr.get('error', 'unknown')[:60]}")
            elif len(valid_trials) == 1:
                # Single trial: show as before
                tr = valid_trials[0]
                status = "PASS" if tr["passed"] else "FAIL"
                print(f"  {tid:40s}  {tr['task_score']:.2f}  {status}  "
                      f"C={tr['completion']:.2f} R={tr['robustness']:.2f} "
                      f"M={tr['communication']:.2f} S={tr['safety']:.0f} "
                      f"TOK={tr.get('tokens', 0)} "
                      f"({tr.get('model_input_tokens', tr.get('input_tokens', 0))}in/"
                      f"{tr.get('model_output_tokens', tr.get('output_tokens', 0))}out) "
                      f"TIME=wall {tr.get('wall_time_s', 0.0):.2f}s "
                      f"model {tr.get('model_time_s', 0.0):.2f}s "
                      f"tool {tr.get('tool_time_s', 0.0):.2f}s")
            else:
                # Multi-trial: show avg score + per-trial scores + pass^k/pass@k
                tl = r["trials"]
                avg_sc = sum(tr["task_score"] for tr in tl) / len(tl)
                trial_strs = "/".join(f"{t['task_score']:.2f}" for t in tl)
                p_hat = "Y" if all(tr["passed"] for tr in tl) else "N"
                p_at = "Y" if any(tr["passed"] for tr in tl) else "N"
                total_tok = sum(t.get("tokens", 0) for t in tl)
                total_in = sum(t.get("model_input_tokens", t.get("input_tokens", 0)) for t in tl)
                total_out = sum(t.get("model_output_tokens", t.get("output_tokens", 0)) for t in tl)
                total_wall = sum(t.get("wall_time_s", 0.0) for t in tl)
                total_model = sum(t.get("model_time_s", 0.0) for t in tl)
                total_tool = sum(t.get("tool_time_s", 0.0) for t in tl)
                print(f"  {tid:40s}  {avg_sc:.2f}  "
                      f"trials=[{trial_strs}] "
                      f"pass^{len(tl)}={p_hat} pass@{len(tl)}={p_at} "
                      f"TOK={total_tok} ({total_in}in/{total_out}out) "
                      f"TIME=wall {total_wall:.2f}s "
                      f"model {total_model:.2f}s "
                      f"tool {total_tool:.2f}s")

    # ── Multi-agent comparison table ──
    if multi_agent:
        print(f"\n{'='*80}")
        print("MULTI-AGENT COMPARISON")
        print(f"{'='*80}")
        # Header
        print(f"  {'Agent':<20s} {'Avg':>6s} {'Pass':>5s} {'Fail':>5s} {'Err':>4s} {'Tokens':>10s} {'Wall(s)':>8s}")
        print(f"  {'─'*20} {'─'*6} {'─'*5} {'─'*5} {'─'*4} {'─'*10} {'─'*8}")
        per_agent_summaries = {}
        for ag in agent_names:
            ag_res = agent_results[ag]
            ag_label = ag or "baseline"
            ag_n = len(ag_res)
            ag_err = sum(1 for r in ag_res if r.get("error"))
            ag_pass = sum(
                1 for r in ag_res
                if not r.get("error") and r.get("trials")
                and any(t.get("passed") for t in r["trials"])
            )
            ag_fail = ag_n - ag_err - ag_pass
            ag_scores = [
                sum(t["task_score"] for t in r["trials"]) / len(r["trials"])
                for r in ag_res if not r.get("error") and r.get("trials")
            ]
            ag_avg = sum(ag_scores) / len(ag_scores) if ag_scores else 0.0
            ag_tok = sum(t.get("tokens", 0) for r in ag_res for t in r.get("trials", []))
            ag_wall = sum(t.get("wall_time_s", 0.0) for r in ag_res for t in r.get("trials", []))
            print(f"  {ag_label:<20s} {ag_avg:>6.3f} {ag_pass:>5d} {ag_fail:>5d} {ag_err:>4d} {ag_tok:>10d} {ag_wall:>8.1f}")
            per_agent_summaries[ag_label] = {
                "agent": ag_label,
                "tasks": ag_n,
                "avg_score": ag_avg,
                "passed": ag_pass,
                "failed": ag_fail,
                "errored": ag_err,
                "total_tokens": ag_tok,
                "total_wall_time_s": ag_wall,
            }
        print()

        # Per-task agent comparison matrix
        task_ids = sorted({Path(td).name for _, td in work_items})
        print(f"  {'Task':<40s}", end="")
        for ag in agent_names:
            print(f"  {(ag or 'baseline'):>12s}", end="")
        print()
        print(f"  {'─'*40}", end="")
        for _ in agent_names:
            print(f"  {'─'*12}", end="")
        print()
        for tid in task_ids:
            print(f"  {tid:<40s}", end="")
            for ag in agent_names:
                r_match = [r for r in agent_results[ag] if r.get("task_id") == tid]
                if r_match:
                    r = r_match[0]
                    if r.get("error"):
                        print(f"  {'ERR':>12s}", end="")
                    elif r.get("trials"):
                        sc = sum(t["task_score"] for t in r["trials"]) / len(r["trials"])
                        status = "P" if any(t.get("passed") for t in r["trials"]) else "F"
                        print(f"  {f'{sc:.2f} {status}':>12s}", end="")
                    else:
                        print(f"  {'—':>12s}", end="")
                else:
                    print(f"  {'—':>12s}", end="")
            print()
        print()

    # Write JSON results into the same trace subdir
    out_dir = Path(batch_trace_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_file = out_dir / "batch_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    summary_file = out_dir / "batch_summary.json"
    summary_data = {
        "tasks": total,
        "agents": [a or "baseline" for a in agent_names] if multi_agent else None,
        "trials_per_task": trials,
        f"pass_hat_{trials}": n_pass_hat,
        f"pass_at_{trials}": n_pass_at,
        "errored": errored,
        "avg_score": avg_score_final,
        "total_model_input_tokens": total_model_input_tokens,
        "total_model_output_tokens": total_model_output_tokens,
        "total_input_tokens": total_model_input_tokens,
        "total_output_tokens": total_model_output_tokens,
        "total_tokens": total_tokens,
        "total_model_time_s": total_model_time_s,
        "total_tool_time_s": total_tool_time_s,
        "total_other_time_s": total_other_time_s,
        "total_wall_time_s": total_wall_time_s,
    }
    if multi_agent:
        summary_data["per_agent"] = per_agent_summaries
    with open(summary_file, "w") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    # Write per-agent results for multi-agent runs
    if multi_agent:
        for ag in agent_names:
            ag_label = ag or "baseline"
            ag_dir = Path(agent_trace_dirs[ag])
            ag_dir.mkdir(parents=True, exist_ok=True)
            ag_file = ag_dir / "batch_results.json"
            with open(ag_file, "w") as f:
                json.dump(agent_results[ag], f, indent=2, ensure_ascii=False)

    print(f"\n  Results saved to {results_file}")
    print(f"  Summary saved to {summary_file}")


def cmd_cleanup(args: argparse.Namespace) -> None:
    """Remove all PAST-Bench Docker containers, including runtime containers."""
    from .config import load_config

    cfg = load_config(getattr(args, "config", None))

    from .runner.sandbox_runner import SandboxRunner

    runner = SandboxRunner(cfg.sandbox, image=cfg.sandbox.image)
    count = runner.cleanup_all()
    if count:
        print(f"Removed {count} PAST-Bench container(s).")
    else:
        print("No PAST-Bench containers found.")


def _safe_label(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value).strip("_") or "episode"


def _empty_artifact_summary(artifacts_dir: Path) -> dict:
    return {
        "artifacts_dir": str(artifacts_dir),
        "memory_file_exists": False,
        "user_file_exists": False,
        "memory_chars": 0,
        "user_chars": 0,
        "memory_entries": [],
        "user_entries": [],
        "skill_count": 0,
        "skill_names": [],
        "skill_docs": {},
        "internal_tools": {
            "session_file": None,
            "tool_call_counts": {},
            "memory_calls": 0,
            "memory_action_counts": {},
            "memory_write_count": 0,
            "memory_read_count": 0,
            "skill_manage_calls": 0,
            "skill_manage_action_counts": {},
            "skill_create_count": 0,
            "skill_update_count": 0,
            "session_search_calls": 0,
            "skill_view_calls": 0,
            "skills_list_calls": 0,
            "skill_read_count": 0,
            "calls": [],
        },
    }


def _build_failed_episode_result(*, exc: Exception, task, episode, index: int) -> dict:
    """Synthesize a zero-score, infra-blocked episode_result for a hard failure
    (timeout, runtime crash) so the evolve sequence can continue past one bad
    episode instead of aborting the whole run. Schema mirrors what
    grade_episode produces — every key downstream summarize_sequence reads
    must be present, even if zero/empty."""
    reason = f"{type(exc).__name__}: {exc}"
    return {
        "trace": "",
        "trace_id": "",
        "task_id": task.task_id,
        "task_name": task.task_name,
        "final_response_text": "",
        "scores": {
            "completion": 0.0,
            "robustness": 0.0,
            "communication": 0.0,
            "safety": 1.0,
            "efficiency_turns": 0,
            "efficiency_tokens": 0,
            "efficiency_wall_time_s": 0.0,
        },
        "judge_score": 0.0,
        "task_score": 0.0,
        "task_components": {},
        "hard_pass": False,
        "passed": False,
        "total_turns": 0,
        "tool_dispatch_count": 0,
        "token_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        "timing": {},
        # Mirror _empty_artifact_summary schema so summarize_sequence can
        # walk these keys without KeyError.
        "artifacts": {
            "artifacts_dir": "",
            "memory_file_exists": False,
            "user_file_exists": False,
            "memory_chars": 0,
            "user_chars": 0,
            "memory_entries": [],
            "user_entries": [],
            "skill_count": 0,
            "skill_chars": 0,
            "skill_names": [],
            "skill_docs": {},
        },
        "artifact_diff": {},
        "retrieval_signals": {},
        "mechanism_scores": {},
        "internal_tools": {
            "memory_calls": 0,
            "memory_action_counts": {},
            "memory_write_count": 0,
            "memory_read_count": 0,
            "skill_manage_calls": 0,
            "skill_manage_action_counts": {},
            "skill_create_count": 0,
            "skill_update_count": 0,
            "skill_view_calls": 0,
            "skills_list_calls": 0,
            "skill_read_count": 0,
            "session_search_calls": 0,
            "tool_call_counts": {},
            "calls": [],
        },
        "env_snapshot_present": False,
        "infra_blocked": True,
        "infra_block_reason": reason,
    }


def _is_timeout_failure(exc: Exception) -> bool:
    msg = str(exc)
    return "TimeoutError" in type(exc).__name__ or "TimeoutError" in msg or "did not complete within" in msg


def _reset_runtime_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _clone_runtime_dir(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    if src.exists():
        shutil.copytree(src, dst)
    else:
        dst.mkdir(parents=True, exist_ok=True)


def _family_runtime_root(variant_dir: Path, family_id: str) -> Path:
    return variant_dir / "family_homes" / family_id


def _family_runtime_paths(variant_dir: Path, family_id: str) -> tuple[Path, Path]:
    family_root = _family_runtime_root(variant_dir, family_id)
    return family_root / "hermes_home", family_root / "history_anchors"


def _prepare_episode_history(
    *,
    hermes_home: Path,
    episode,
    history_anchors: dict[str, Path],
) -> None:
    """Materialize the runtime home for an episode from its history contract."""
    if episode.history_mode == "continue":
        return
    if episode.history_mode == "fresh":
        _reset_runtime_dir(hermes_home)
        return
    if episode.history_mode == "from_anchor":
        anchor_name = episode.history_load_anchor
        anchor_home = history_anchors.get(anchor_name)
        if anchor_home is None:
            raise RuntimeError(
                f"missing history anchor {anchor_name!r} for episode {episode.label or episode.task}"
            )
        _clone_runtime_dir(anchor_home, hermes_home)
        return
    raise RuntimeError(f"unsupported history_mode {episode.history_mode!r}")


def _save_episode_history_anchor(
    *,
    hermes_home: Path,
    episode,
    anchors_dir: Path,
    history_anchors: dict[str, Path],
) -> None:
    """Snapshot the current runtime home for later branch reuse."""
    if not episode.history_save_anchor:
        return
    anchors_dir.mkdir(parents=True, exist_ok=True)
    anchor_home = anchors_dir / episode.history_save_anchor
    _clone_runtime_dir(hermes_home, anchor_home)
    history_anchors[episode.history_save_anchor] = anchor_home


def _resolve_episode_preseed_dir(sequence, episode) -> Path | None:
    """Prefer an episode-level preseed overlay over the sequence default."""
    preseed_value = getattr(episode, "preseed_artifacts_dir", "") or sequence.hermes.preseed_artifacts_dir
    if not preseed_value:
        return None
    path = Path(preseed_value).expanduser()
    if not path.is_absolute():
        path = (sequence.manifest_path.parent / path).resolve()
    return path


def _resolve_episode_initial_home_fixture_dir(sequence, episode) -> Path | None:
    """Prefer an episode-level native Hermes home fixture over the sequence default."""
    fixture_value = (
        getattr(episode, "initial_home_fixture_dir", "")
        or sequence.hermes.initial_home_fixture_dir
    )
    if not fixture_value:
        return None
    path = Path(fixture_value).expanduser()
    if not path.is_absolute():
        path = (sequence.manifest_path.parent / path).resolve()
    return path


def _copy_named_home_entries(src_dir: Path | None, dst_dir: Path, names: tuple[str, ...]) -> None:
    if src_dir is None or not src_dir.exists():
        return
    for name in names:
        src = src_dir / name
        dst = dst_dir / name
        if not src.exists() or dst.exists():
            continue
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def _overlay_path(src: Path, dst: Path) -> None:
    if src.is_dir():
        dst.mkdir(parents=True, exist_ok=True)
        for child in src.iterdir():
            _overlay_path(child, dst / child.name)
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _overlay_named_home_entries(src_dir: Path | None, dst_dir: Path, names: tuple[str, ...]) -> None:
    if src_dir is None or not src_dir.exists():
        return
    for name in names:
        src = src_dir / name
        dst = dst_dir / name
        if not src.exists():
            continue
        _overlay_path(src, dst)


def _materialize_episode_home_inputs(
    *,
    hermes_home: Path,
    initial_home_fixture_dir: Path | None,
    preseed_artifacts_dir: Path | None,
) -> None:
    """Materialize benchmark-provided Hermes home state before artifact snapshots."""
    _copy_named_home_entries(
        initial_home_fixture_dir,
        hermes_home,
        ("memories", "skills", "sessions", "state.db", "state.db-wal", "state.db-shm"),
    )
    _overlay_named_home_entries(
        preseed_artifacts_dir,
        hermes_home,
        ("memories", "skills"),
    )


def _print_sequence_bucket_summary(label: str, summary: dict) -> None:
    print(
        f"  {label}: score={summary['avg_task_score']:.3f} pass_rate={summary['pass_rate']:.3f} "
        f"dispatches={summary['avg_tool_dispatch_count']:.2f} "
        f"memory_calls={summary['memory_calls']} skill_calls={summary['skill_manage_calls']} "
        f"search_calls={summary['session_search_calls']}"
    )


def _apply_rsimem_execution_overrides(sequence, args: argparse.Namespace) -> None:
    mode = getattr(args, "rsimem_mode", None)
    failure_policy = getattr(args, "rsimem_adapter_failure_policy", None)
    verify_projection = getattr(args, "rsimem_verify_native_projection", False)
    lifecycle_mode = getattr(args, "rsimem_lifecycle_evaluator_mode", None)
    lifecycle_policy = getattr(args, "rsimem_lifecycle_policy_version", None)
    lifecycle_compiler = getattr(args, "rsimem_lifecycle_compiler_version", None)
    lifecycle_timeout = getattr(args, "rsimem_lifecycle_timeout_seconds", None)
    lifecycle_max_tokens = getattr(args, "rsimem_lifecycle_max_output_tokens", None)
    semantic_mode = getattr(args, "rsimem_semantic_writeback_mode", None)
    semantic_timeout = getattr(args, "rsimem_semantic_writeback_timeout_seconds", None)
    semantic_max_tokens = getattr(args, "rsimem_semantic_writeback_max_output_tokens", None)
    semantic_feedback = getattr(args, "rsimem_semantic_feedback_contract", None)
    adaptive_config_path = getattr(args, "rsimem_adaptive_config", None)
    extraction_trial_path = getattr(
        args,
        "rsimem_extraction_trial_config",
        None,
    )
    extraction_offline_path = getattr(
        args,
        "rsimem_extraction_offline_config",
        None,
    )
    revocation_registry_path = getattr(
        args,
        "rsimem_revocation_registry",
        None,
    )
    revocation_registry = None
    if revocation_registry_path is not None:
        from rsimem.memory.revocation import JsonRevocationRegistry

        path = Path(revocation_registry_path).expanduser().resolve()
        if not path.is_file() or path.is_symlink():
            raise SystemExit(
                "invalid RSIMem revocation registry: expected an existing regular file"
            )
        revocation_registry = JsonRevocationRegistry(path)
    if all(value is None for value in (
        mode,
        failure_policy,
        lifecycle_mode,
        lifecycle_policy,
        lifecycle_compiler,
        lifecycle_timeout,
        lifecycle_max_tokens,
        semantic_mode,
        semantic_timeout,
        semantic_max_tokens,
        semantic_feedback,
        adaptive_config_path,
        extraction_trial_path,
        extraction_offline_path,
        revocation_registry_path,
    )) and not verify_projection:
        return
    if not str(args.agent).startswith("hermes"):
        raise SystemExit("RSIMem execution overrides require a Hermes agent")
    if mode is not None:
        sequence.hermes.rsimem_mode = mode
    if failure_policy is not None:
        sequence.hermes.rsimem_adapter_failure_policy = failure_policy
    if verify_projection:
        sequence.hermes.rsimem_verify_native_projection = True
    if lifecycle_mode is not None:
        sequence.hermes.rsimem_lifecycle_evaluator_mode = lifecycle_mode
    if lifecycle_policy is not None:
        sequence.hermes.rsimem_lifecycle_policy_version = lifecycle_policy
    if lifecycle_compiler is not None:
        sequence.hermes.rsimem_lifecycle_compiler_version = lifecycle_compiler
    if lifecycle_timeout is not None:
        sequence.hermes.rsimem_lifecycle_timeout_seconds = lifecycle_timeout
    if lifecycle_max_tokens is not None:
        sequence.hermes.rsimem_lifecycle_max_output_tokens = lifecycle_max_tokens
    if semantic_mode is not None:
        sequence.hermes.rsimem_semantic_writeback_mode = semantic_mode
    if semantic_timeout is not None:
        sequence.hermes.rsimem_semantic_writeback_timeout_seconds = semantic_timeout
    if semantic_max_tokens is not None:
        sequence.hermes.rsimem_semantic_writeback_max_output_tokens = semantic_max_tokens
    if semantic_feedback is not None:
        sequence.hermes.rsimem_semantic_feedback_contract = semantic_feedback
    if adaptive_config_path is not None:
        from .models.self_evolve import RSIMemAdaptiveWritebackConfig

        path = Path(adaptive_config_path).expanduser().resolve()
        try:
            serialized = path.read_text(encoding="utf-8")
            sequence.hermes.rsimem_adaptive_config = (
                RSIMemAdaptiveWritebackConfig.model_validate_json(serialized)
            )
            source = (
                path.parent
                / sequence.hermes.rsimem_adaptive_config.prepared_policy_store_file
            ).resolve()
            if not source.is_file() or not source.is_relative_to(path.parent):
                raise ValueError(
                    "prepared adaptive policy store is missing or escapes config directory"
                )
            sequence.hermes.rsimem_adaptive_policy_source_path = str(source)
        except (OSError, ValueError) as exc:
            raise SystemExit(f"invalid RSIMem adaptive config: {exc}") from exc
    if extraction_trial_path is not None:
        from .models.self_evolve import RSIMemExtractionTrialProfile
        from rsimem.extraction_validation_runtime import (
            load_extraction_matched_trial_profile,
        )

        path = Path(extraction_trial_path).expanduser().resolve()
        try:
            resolved = (
                load_extraction_matched_trial_profile(path)
                if revocation_registry is None
                else load_extraction_matched_trial_profile(
                    path,
                    revocation_registry=revocation_registry,
                    require_revocation_registry=True,
                )
            )
            sequence.hermes.rsimem_extraction_trial_profile = (
                RSIMemExtractionTrialProfile.model_validate(resolved.profile())
            )
            sequence.hermes.rsimem_extraction_trial_source_path = str(path)
        except (OSError, ValueError) as exc:
            raise SystemExit(
                f"invalid RSIMem extraction trial config: {exc}"
            ) from exc
    if extraction_offline_path is not None:
        from .models.self_evolve import RSIMemExtractionOfflineValidationProfile
        from rsimem.extraction_validation_runtime import (
            load_extraction_offline_validation_profile,
        )
        path = Path(extraction_offline_path).expanduser().resolve()
        try:
            resolved = (
                load_extraction_offline_validation_profile(path)
                if revocation_registry is None
                else load_extraction_offline_validation_profile(
                    path,
                    revocation_registry=revocation_registry,
                    require_revocation_registry=True,
                )
            )
            sequence.hermes.rsimem_extraction_offline_profile = (
                RSIMemExtractionOfflineValidationProfile.model_validate(resolved.profile())
            )
            sequence.hermes.rsimem_extraction_offline_source_path = str(path)
        except (OSError, ValueError) as exc:
            raise SystemExit(f"invalid RSIMem extraction offline config: {exc}") from exc
    if revocation_registry_path is not None:
        sequence.hermes.rsimem_revocation_registry_path = str(
            revocation_registry.path
        )
    adaptive_selected = (
        sequence.hermes.rsimem_semantic_writeback_mode == "adaptive_utility"
    )
    if adaptive_selected and sequence.hermes.rsimem_adaptive_config is None:
        raise SystemExit("adaptive semantic writeback requires adaptive config")
    if not adaptive_selected and sequence.hermes.rsimem_adaptive_config is not None:
        raise SystemExit("adaptive config requires adaptive_utility mode")
    extraction_selected = (
        sequence.hermes.rsimem_extraction_trial_profile is not None
    )
    offline_selected = sequence.hermes.rsimem_extraction_offline_profile is not None
    if extraction_selected and offline_selected:
        raise SystemExit("extraction trial and offline profile are mutually exclusive")
    if extraction_selected and (
        sequence.hermes.rsimem_semantic_writeback_mode != "static"
    ):
        raise SystemExit(
            "extraction matched trial requires static semantic writeback"
        )
    if extraction_selected != bool(
        sequence.hermes.rsimem_extraction_trial_source_path
    ):
        raise SystemExit(
            "extraction trial profile and source path must be configured together"
        )
    if extraction_selected and sequence.hermes.rsimem_adaptive_config is not None:
        raise SystemExit(
            "extraction trial cannot use legacy adaptive utility config"
        )


def _print_episode_result_summary(result: dict) -> None:
    print(
        "  task: "
        f"index={result['index']} "
        f"task_id={result['task_id']} "
        f"bucket={result.get('bucket', '')} "
        f"score={result['task_score']:.3f} "
        f"passed={result['passed']} "
        f"search_calls={result['internal_tools'].get('session_search_calls', 0)}"
    )


def cmd_evolve(args: argparse.Namespace) -> None:
    """Run a sequence benchmark for cross-task self-evolve evaluation."""
    _apply_proxy(getattr(args, "proxy", None))

    method_task_id = getattr(args, "rsimem_method_task_id", None)
    if method_task_id is not None and (
        not isinstance(method_task_id, str) or not method_task_id.strip()
    ):
        raise SystemExit("--rsimem-method-task-id must be non-empty text")

    if not getattr(args, "agent", None):
        raise SystemExit("--agent is required for evolve runs")

    from .config import load_config
    from .graders.registry import get_grader
    from .models.self_evolve import SelfEvolveSequenceDefinition
    from .models.task import TaskDefinition
    from .runner.self_evolve import (
        build_hermes_extra_body,
        build_past_bench_application_opportunity_schema,
        build_nanobot_extra_body,
        build_zeroclaw_extra_body,
        build_reflection_prompt,
        choose_calibration_candidate,
        diff_artifact_snapshots,
        grade_episode,
        make_persistence_backend,
        make_reflection_task,
        materialize_task_hermes_seed,
        resolve_episode_tool_config,
        snapshot_hermes_artifacts,
        snapshot_hermes_home,
        summarize_reflection_episode,
        summarize_single_task_comparison,
        summarize_single_task_sequence,
        summarize_comparison,
        summarize_sequence,
        write_json,
    )
    from .runner.services import ServiceManager

    cfg = load_config(args.config)
    port_offset = getattr(args, "port_offset", 0) or 0

    # §4 family.yaml → runner source of truth: `--family <ability/family_id>`
    # and `--v2-family <ability/family_id>` both regenerate a temp manifest
    # from the same generator on demand.
    sequence_path = args.sequence
    family_rel = None
    if getattr(args, "v2_family", None):
        if sequence_path or getattr(args, "family", None):
            raise SystemExit("--v2-family is mutually exclusive with --sequence and --family")
        family_rel = args.v2_family
    elif getattr(args, "family", None):
        if sequence_path:
            raise SystemExit("--family and --sequence are mutually exclusive")
        family_rel = args.family

    if family_rel:
        import tempfile
        from .self_evolve_v2 import generate_manifest as _gen_manifest

        tmp_dir = Path(tempfile.mkdtemp(prefix="v2_fam_"))
        fam_id = family_rel.split("/")[-1].lower()
        tmp_manifest = tmp_dir / f"hermes_self_evolve_v2_{fam_id}_only.yaml"
        _gen_manifest(family_rel, out_path=tmp_manifest)
        sequence_path = str(tmp_manifest)
        print(f"[v2] generated manifest from family.yaml -> {sequence_path}")
    if not sequence_path:
        raise SystemExit("either --sequence, --family, or --v2-family is required")

    sequence = SelfEvolveSequenceDefinition.from_yaml(sequence_path)
    _apply_rsimem_execution_overrides(sequence, args)
    judge = _make_judge(cfg, args)
    runtime_mode = _resolve_runtime_mode(args, cfg)
    runtime_temperature = _resolve_runtime_temperature(args, cfg)
    registry_path = _resolve_registry_path(args, cfg)
    persistence_backend = None
    if sequence.mode == "episodes":
        if args.agent.startswith("hermes") or args.agent in {"nanobot", "zeroclaw"}:
            persistence_backend = make_persistence_backend(args.agent)
    persistence_variant = getattr(args, "persistence_variant", None)
    if (
        getattr(args, "compare_no_persistence", False)
        and persistence_variant not in {None, "paired"}
    ):
        raise SystemExit(
            "--compare-no-persistence conflicts with an unpaired --persistence-variant"
        )
    if persistence_variant == "paired":
        args.compare_no_persistence = True
    needs_persistence_backend = (
        getattr(args, "compare_no_persistence", False)
        or persistence_variant == "without_persistence"
    )
    if needs_persistence_backend and persistence_backend is None:
        raise SystemExit(
            "persistence variants are only supported for hermes/nanobot/zeroclaw episode runs"
        )
    if persistence_backend is not None and runtime_mode == "container" and sequence.mode == "episodes":
        raise SystemExit(
            f"{args.agent} self-evolve episode sequences are invalid with --runtime container right now; "
            "use --runtime local."
        )
    _check_agent_requirements(
        args.agent,
        registry_path,
        explicit_api_key=args.api_key,
        profile=getattr(args, "agent_profile", None),
    )

    if args.trace_dir:
        trace_root = Path(args.trace_dir)
        trace_root.mkdir(parents=True, exist_ok=True)
    else:
        trace_root = _make_trace_dir(cfg.defaults.trace_dir, f"{args.agent}_{sequence.name}")

    variants = (
        [("without_persistence", False)]
        if persistence_variant == "without_persistence"
        else [("with_persistence", True)]
    )
    if getattr(args, "compare_no_persistence", False):
        variants.append(("without_persistence", False))
    shared_cold_variant = (
        variants[0][0] if len(variants) == 1 else "with_persistence"
    )

    # §15 reflection_off control: flip the sequence's reflection_enabled flag
    # for this run so PC04 / failure_reflection families can measure
    # reflection's marginal contribution.
    if getattr(args, "reflection_off", False):
        sequence.hermes.reflection_enabled = False

    # Disable post-task learning integrations for the entire evolve run.
    # These can block indefinitely (e.g. kuzu/graphiti GIL contention) and
    # self-evolve is a benchmark — post-task graph ingestion is not needed.
    if getattr(cfg, "graphiti", None) is not None and hasattr(cfg.graphiti, "learning_enabled"):
        cfg.graphiti.learning_enabled = False
    if getattr(cfg, "letta", None) is not None and hasattr(cfg.letta, "enabled"):
        cfg.letta.enabled = False
    if getattr(cfg, "mem0", None) is not None and hasattr(cfg.mem0, "enabled"):
        cfg.mem0.enabled = False

    variant_summaries: dict[str, dict] = {}

    if sequence.mode == "single_task":
        single = sequence.single_task
        assert single is not None

        def _run_episode(
            *,
            variant_dir: Path,
            hermes_home: Path,
            index: int,
            label: str,
            bucket: str,
            episode_kind: str,
            task: TaskDefinition,
            persistence_enabled: bool,
            persistence_allowed: bool,
            experiment_variant: str,
        ) -> dict:
            episode_slug = _safe_label(label or task.task_id)
            episode_dir = variant_dir / f"{index:02d}_{episode_slug}"
            episode_dir.mkdir(parents=True, exist_ok=True)
            artifacts_dir = episode_dir / "artifacts"
            if args.agent.startswith("hermes"):
                _materialize_episode_home_inputs(
                    hermes_home=hermes_home,
                    initial_home_fixture_dir=Path(sequence.hermes.initial_home_fixture_dir).expanduser()
                    if sequence.hermes.initial_home_fixture_dir
                    else None,
                    preseed_artifacts_dir=Path(sequence.hermes.preseed_artifacts_dir).expanduser()
                    if sequence.hermes.preseed_artifacts_dir
                    else None,
                )
            artifact_before = (
                snapshot_hermes_home(hermes_home, include_contents=True)
                if args.agent.startswith("hermes")
                else _empty_artifact_summary(hermes_home)
            )

            model_extra_body_override = None
            if args.agent.startswith("hermes"):
                tool_config = resolve_episode_tool_config(
                    persistence_enabled=persistence_enabled and persistence_allowed,
                    expected_signal=single.mechanism,
                    memory_enabled=sequence.hermes.memory_enabled,
                    user_profile_enabled=sequence.hermes.user_profile_enabled,
                    skills_enabled=sequence.hermes.skills_enabled,
                    session_search_enabled=sequence.hermes.session_search_enabled,
                )
                tool_config["application_opportunity_schema"] = (
                    build_past_bench_application_opportunity_schema(task)
                )
                review_wait_s = (
                    args.background_review_wait_s
                    if args.background_review_wait_s is not None
                    else sequence.hermes.background_review_wait_s
                )
                preseed_dir = materialize_task_hermes_seed(
                    task=task,
                    target_dir=episode_dir / "preseed",
                    base_preseed_dir=Path(sequence.hermes.preseed_artifacts_dir).expanduser()
                    if sequence.hermes.preseed_artifacts_dir
                    else None,
                )
                model_extra_body_override = build_hermes_extra_body(
                    home_dir=hermes_home,
                    artifacts_dir=artifacts_dir,
                    persistence_enabled=persistence_enabled and persistence_allowed,
                    memory_enabled=tool_config["memory_enabled"],
                    user_profile_enabled=tool_config["user_profile_enabled"],
                    skills_enabled=tool_config["skills_enabled"],
                    session_search_enabled=tool_config["session_search_enabled"],
                    memory_nudge_interval=sequence.hermes.memory_nudge_interval,
                    memory_flush_min_turns=sequence.hermes.memory_flush_min_turns,
                    skill_creation_nudge_interval=sequence.hermes.skill_creation_nudge_interval,
                    background_review_wait_s=float(review_wait_s),
                    initial_home_fixture_dir=Path(sequence.hermes.initial_home_fixture_dir).expanduser()
                    if sequence.hermes.initial_home_fixture_dir
                    else None,
                    preseed_artifacts_dir=preseed_dir,
                    rsimem_mode=sequence.hermes.rsimem_mode,
                    rsimem_adapter_failure_policy=(
                        sequence.hermes.rsimem_adapter_failure_policy
                    ),
                    rsimem_lifecycle_evaluator_mode=(
                        sequence.hermes.rsimem_lifecycle_evaluator_mode
                    ),
                    rsimem_lifecycle_policy_version=(
                        sequence.hermes.rsimem_lifecycle_policy_version
                    ),
                    rsimem_lifecycle_compiler_version=(
                        sequence.hermes.rsimem_lifecycle_compiler_version
                    ),
                    rsimem_lifecycle_timeout_seconds=(
                        sequence.hermes.rsimem_lifecycle_timeout_seconds
                    ),
                    rsimem_lifecycle_max_output_tokens=(
                        sequence.hermes.rsimem_lifecycle_max_output_tokens
                    ),
                    rsimem_semantic_writeback_mode=(
                        sequence.hermes.rsimem_semantic_writeback_mode
                    ),
                    rsimem_semantic_writeback_timeout_seconds=(
                        sequence.hermes.rsimem_semantic_writeback_timeout_seconds
                    ),
                    rsimem_semantic_writeback_max_output_tokens=(
                        sequence.hermes.rsimem_semantic_writeback_max_output_tokens
                    ),
                    rsimem_semantic_feedback_contract=(
                        sequence.hermes.rsimem_semantic_feedback_contract
                    ),
                    rsimem_adaptive_config=sequence.hermes.rsimem_adaptive_config,
                    rsimem_adaptive_policy_source_path=(
                        sequence.hermes.rsimem_adaptive_policy_source_path
                    ),
                    rsimem_extraction_trial_profile=(
                        sequence.hermes.rsimem_extraction_trial_profile
                    ),
                    rsimem_extraction_trial_source_path=(
                        sequence.hermes.rsimem_extraction_trial_source_path
                    ),
                    rsimem_extraction_offline_profile=(
                        sequence.hermes.rsimem_extraction_offline_profile
                    ),
                    rsimem_extraction_offline_source_path=(
                        sequence.hermes.rsimem_extraction_offline_source_path
                    ),
                    rsimem_application_opportunity_schema=(
                        tool_config.get("application_opportunity_schema")
                    ),
                )

            print(
                f"\n[{index}] kind={episode_kind} bucket={bucket} task={task.task_id} label={label or task.task_name}"
            )

            _svc_cwd = (
                _resolve_tasks_dir(Path(task.task_file)).parent
                if task.task_file
                else sequence.manifest_path.parent
            )
            with ServiceManager(task.services, cwd=_svc_cwd):
                trace_path, env_snapshot = _execute_trial(
                    task=task,
                    cfg=cfg,
                    task_dir=str(Path(task.task_file).parent) if task.task_file else str(sequence.manifest_path.parent),
                    trace_dir=str(episode_dir),
                    model=args.model,
                    api_key=args.api_key,
                    base_url=args.base_url,
                    sandbox_mode=getattr(args, "sandbox", False) or cfg.sandbox.enabled,
                    sandbox_image=getattr(args, "sandbox_image", None),
                    sandbox_tools_local=getattr(args, "sandbox_tools", False),
                    agent_name=args.agent,
                    agent_profile=getattr(args, "agent_profile", None),
                    runtime_mode=runtime_mode,
                    runtime_image=getattr(args, "runtime_image", None),
                    registry_path=registry_path,
                    model_extra_body_override=model_extra_body_override,
                    runtime_temperature=runtime_temperature,
                    runtime_metadata={
                        "run_id": trace_root.name,
                        "episode_id": episode_dir.name,
                        "family_id": sequence.name,
                        "stage": bucket,
                        "experiment_variant": experiment_variant,
                        **(
                            {"rsimem_method_task_id": method_task_id}
                            if method_task_id is not None
                            else {}
                        ),
                    },
                )

            artifact_summary = (
                snapshot_hermes_artifacts(artifacts_dir)
                if args.agent.startswith("hermes")
                else _empty_artifact_summary(artifacts_dir)
            )

            if episode_kind == "reflection":
                result = summarize_reflection_episode(
                    trace_path=trace_path,
                    artifact_summary=artifact_summary,
                    task_id=task.task_id,
                    label=label or task.task_name,
                )
            else:
                grader = get_grader(task.task_id, tasks_dir=_resolve_tasks_dir(Path(task.task_file)), task_dir=Path(task.task_file).parent)
                result = grade_episode(
                    trace_path=trace_path,
                    task=task,
                    tasks_dir=_resolve_tasks_dir(Path(task.task_file)),
                    task_dir=Path(task.task_file).parent,
                    grader=grader,
                    judge=judge,
                    env_snapshot=env_snapshot,
                    artifact_before=artifact_before,
                    artifact_summary=artifact_summary,
                    expected_persistence_signal=single.mechanism,
                )
                _append_grading_to_trace(
                    trace_path=trace_path,
                    trace_id=result["trace_id"],
                    task_id=task.task_id,
                    scores=result["scores"],
                    task_score=result["task_score"],
                    passed=result["passed"],
                )
            result.update({
                "index": index,
                "label": label or task.task_name,
                "bucket": bucket,
                "episode_kind": episode_kind,
                "family_id": "single_task_evolve",
                "mechanism": single.mechanism,
                "expected_persistence_signal": single.mechanism,
                "requires_fresh_session": True,
                "persistence_allowed": persistence_allowed,
            })
            return result

        for variant_label, persistence_enabled in variants:
            variant_dir = trace_root / variant_label if len(variants) > 1 else trace_root
            variant_dir.mkdir(parents=True, exist_ok=True)
            hermes_home = variant_dir / "hermes_home"
            episode_results: list[dict] = []

            print(f"\n=== Sequence: {sequence.name} [{variant_label}] ===")
            print(sequence.description or "(no description)")

            candidate_runs: list[dict] = []
            global_index = 1

            for candidate in single.candidates:
                task_yaml = sequence.resolve_task_yaml(candidate.task)
                task = TaskDefinition.from_yaml(task_yaml)
                if port_offset:
                    task.apply_port_offset(port_offset)
                baseline_runs: list[dict] = []
                print(f"\nCalibration candidate: {candidate.label or task.task_id}")
                for repeat in range(1, single.baseline_repeats + 1):
                    _reset_runtime_dir(hermes_home)
                    result = _run_episode(
                        variant_dir=variant_dir,
                        hermes_home=hermes_home,
                        index=global_index,
                        label=f"{candidate.label or task.task_id}_baseline_{repeat}",
                        bucket="baseline",
                        episode_kind="attempt",
                        task=task,
                        persistence_enabled=persistence_enabled,
                        persistence_allowed=False,
                        experiment_variant=variant_label,
                    )
                    global_index += 1
                    baseline_runs.append(result)
                    print(
                        f"  baseline[{repeat}/{single.baseline_repeats}] score={result['task_score']:.3f} passed={result['passed']}"
                    )

                baseline_summary = {
                    "pass_count": sum(1 for item in baseline_runs if item["passed"]),
                    "avg_task_score": mean(item["task_score"] for item in baseline_runs),
                }
                candidate_runs.append({
                    "label": candidate.label or task.task_id,
                    "task": str(task_yaml),
                    "baseline_summary": baseline_summary,
                    "baseline_runs": baseline_runs,
                })

            selected = choose_calibration_candidate(
                target_pass_count=single.target_baseline_pass_count,
                score_min=single.calibration_score_min,
                score_max=single.calibration_score_max,
                candidates=candidate_runs,
            )
            selected_task = TaskDefinition.from_yaml(Path(selected["task"]))
            if port_offset:
                selected_task.apply_port_offset(port_offset)
            print(
                f"\nSelected candidate: {selected['label']} "
                f"(pass_count={selected['baseline_summary']['pass_count']}/{single.baseline_repeats}, "
                f"avg_score={selected['baseline_summary']['avg_task_score']:.3f})"
            )
            episode_results.extend(selected["baseline_runs"])

            if persistence_enabled:
                _reset_runtime_dir(hermes_home)

            for round_idx in range(1, single.teach_rounds + 1):
                if not persistence_enabled:
                    _reset_runtime_dir(hermes_home)
                attempt = _run_episode(
                    variant_dir=variant_dir,
                    hermes_home=hermes_home,
                    index=global_index,
                    label=f"{selected['label']}_teach_{round_idx}",
                    bucket="teach_attempt",
                    episode_kind="attempt",
                    task=selected_task,
                    persistence_enabled=persistence_enabled,
                    persistence_allowed=persistence_enabled,
                    experiment_variant=variant_label,
                )
                global_index += 1
                episode_results.append(attempt)
                print(f"  teach[{round_idx}] score={attempt['task_score']:.3f} passed={attempt['passed']}")

                if sequence.hermes.reflection_enabled:
                    reflection_prompt = build_reflection_prompt(
                        trace_path=Path(attempt["trace"]),
                        task=selected_task,
                        attempt_result=attempt,
                    )
                    reflection_task = make_reflection_task(
                        task_id=f"{selected_task.task_id}_REFLECT_{round_idx}",
                        task_name=f"{selected_task.task_name} Reflection {round_idx}",
                        prompt_text=reflection_prompt,
                    )
                    reflection = _run_episode(
                        variant_dir=variant_dir,
                        hermes_home=hermes_home,
                        index=global_index,
                        label=f"{selected['label']}_reflect_{round_idx}",
                        bucket="reflection",
                        episode_kind="reflection",
                        task=reflection_task,
                        persistence_enabled=persistence_enabled,
                        persistence_allowed=persistence_enabled,
                        experiment_variant=variant_label,
                    )
                    global_index += 1
                    episode_results.append(reflection)
                    print(
                        "  reflection "
                        f"memory={reflection['artifacts']['memory_chars']} "
                        f"skills={reflection['artifacts']['skill_count']} "
                        f"internal(memory={reflection['internal_tools'].get('memory_calls', 0)}, "
                        f"skill={reflection['internal_tools'].get('skill_manage_calls', 0)}, "
                        f"search={reflection['internal_tools'].get('session_search_calls', 0)})"
                    )

            frozen_eval_home = variant_dir / "hermes_home_eval_seed"
            if persistence_enabled:
                _clone_runtime_dir(hermes_home, frozen_eval_home)

            for repeat in range(1, single.retention_repeats + 1):
                episode_home = hermes_home
                if not persistence_enabled:
                    _reset_runtime_dir(hermes_home)
                else:
                    episode_home = variant_dir / f"retention_home_{repeat:02d}"
                    _clone_runtime_dir(frozen_eval_home, episode_home)
                result = _run_episode(
                    variant_dir=variant_dir,
                    hermes_home=episode_home,
                    index=global_index,
                    label=f"{selected['label']}_retention_{repeat}",
                    bucket="retention",
                    episode_kind="attempt",
                    task=selected_task,
                    persistence_enabled=persistence_enabled,
                    persistence_allowed=persistence_enabled,
                    experiment_variant=variant_label,
                )
                global_index += 1
                episode_results.append(result)
                print(f"  retention[{repeat}/{single.retention_repeats}] score={result['task_score']:.3f} passed={result['passed']}")

            for transfer_idx, transfer_ref in enumerate(single.transfer_tasks, start=1):
                episode_home = hermes_home
                if not persistence_enabled:
                    _reset_runtime_dir(hermes_home)
                else:
                    episode_home = variant_dir / f"transfer_home_{transfer_idx:02d}"
                    _clone_runtime_dir(frozen_eval_home, episode_home)
                transfer_task = TaskDefinition.from_yaml(sequence.resolve_task_yaml(transfer_ref.task))
                if port_offset:
                    transfer_task.apply_port_offset(port_offset)
                result = _run_episode(
                    variant_dir=variant_dir,
                    hermes_home=episode_home,
                    index=global_index,
                    label=transfer_ref.label or f"transfer_{transfer_idx}",
                    bucket="transfer",
                    episode_kind="attempt",
                    task=transfer_task,
                    persistence_enabled=persistence_enabled,
                    persistence_allowed=persistence_enabled,
                    experiment_variant=variant_label,
                )
                global_index += 1
                episode_results.append(result)
                print(f"  transfer[{transfer_idx}/{len(single.transfer_tasks)}] score={result['task_score']:.3f} passed={result['passed']}")

            summary = summarize_single_task_sequence(
                sequence_name=sequence.name,
                variant=variant_label,
                selected_candidate=selected,
                episodes=episode_results,
            )
            write_json(variant_dir / "sequence_results.json", {
                "sequence": sequence.name,
                "variant": variant_label,
                "episodes": episode_results,
            })
            write_json(variant_dir / "sequence_summary.json", summary)
            variant_summaries[variant_label] = summary

            print(f"\nSummary [{variant_label}]")
            for bucket_name, bucket_summary in summary["bucket_summary"].items():
                _print_sequence_bucket_summary(bucket_name, bucket_summary)
            print(
                "  reflection: "
                f"memory_calls={summary['reflection_summary']['memory_calls']} "
                f"skill_calls={summary['reflection_summary']['skill_manage_calls']} "
                f"search_calls={summary['reflection_summary']['session_search_calls']}"
            )
            print(
                "  deltas: "
                f"retention_score={summary['benchmark_signal']['retention_score_delta']:.3f} "
                f"retention_pass={summary['benchmark_signal']['retention_pass_delta']:.3f} "
                f"transfer_score={summary['benchmark_signal']['transfer_score_delta_vs_baseline']:.3f} "
                f"transfer_pass={summary['benchmark_signal']['transfer_pass_delta_vs_baseline']:.3f}"
            )

        if len(variants) > 1:
            comparison = summarize_single_task_comparison(
                with_persistence=variant_summaries["with_persistence"],
                without_persistence=variant_summaries["without_persistence"],
            )
            write_json(trace_root / "sequence_comparison.json", comparison)
            print("\nComparison")
            print(
                "  retention delta: "
                f"score={comparison['delta']['retention_avg_score']:.3f} "
                f"pass_rate={comparison['delta']['retention_pass_rate']:.3f}"
            )
            print(
                "  transfer delta: "
                f"score={comparison['delta']['transfer_avg_score']:.3f} "
                f"pass_rate={comparison['delta']['transfer_pass_rate']:.3f}"
            )
            print(
                "  gain delta: "
                f"retention_score={comparison['delta']['retention_score_delta']:.3f} "
                f"retention_pass={comparison['delta']['retention_pass_delta']:.3f}"
            )
        return

    # ── Shared cold baseline pre-pass ───────────────────────────────────────────
    # Episodes marked shared_cold_run=True are executed ONCE under a neutral
    # (no-persistence) config before the variant loop.  Both variants reuse the
    # identical trace and seed their agent state from the same post-I01 state,
    # ensuring the cold baseline is not a confound.
    shared_cold_homes: dict[str, Path] = {}          # family_id → saved runtime state root
    shared_cold_episode_results: dict[int, dict] = {}  # episode index → graded result
    shared_cold_trace_paths: dict[int, Path] = {}      # episode index → .jsonl path

    if persistence_backend is not None and any(ep.shared_cold_run for ep in sequence.episodes):
        cold_base_dir = trace_root / "shared_cold"
        _reset_runtime_dir(cold_base_dir)
        print("\n=== Shared Cold Baseline Pre-pass ===")


        for _sc_index, _sc_episode in enumerate(sequence.episodes, start=1):
            if not _sc_episode.shared_cold_run:
                continue

            _sc_task_yaml = sequence.resolve_task_yaml(_sc_episode.task)
            _sc_task = TaskDefinition.from_yaml(_sc_task_yaml)
            if port_offset:
                _sc_task.apply_port_offset(port_offset)
            _sc_tasks_dir = _resolve_tasks_dir(_sc_task_yaml)

            _sc_slug = _safe_label(_sc_episode.label or _sc_task.task_id)
            _sc_episode_dir = cold_base_dir / f"{_sc_index:02d}_{_sc_slug}"
            _sc_episode_dir.mkdir(parents=True, exist_ok=True)
            _sc_artifacts_dir = _sc_episode_dir / "artifacts"
            _sc_state_root, _ = persistence_backend.family_paths(cold_base_dir, _sc_episode.family_id)
            persistence_backend.reset_state(_sc_state_root)
            persistence_backend.materialize_inputs(
                state_root=_sc_state_root,
                initial_home_fixture_dir=_resolve_episode_initial_home_fixture_dir(sequence, _sc_episode),
                preseed_artifacts_dir=_resolve_episode_preseed_dir(sequence, _sc_episode),
            )
            _sc_artifact_before = persistence_backend.snapshot_before(_sc_state_root, include_contents=True)

            _sc_review_wait = (
                args.background_review_wait_s
                if args.background_review_wait_s is not None
                else sequence.hermes.background_review_wait_s
            )
            _sc_tool_config = resolve_episode_tool_config(
                persistence_enabled=False,
                expected_signal=_sc_episode.mechanism,
                memory_enabled=sequence.hermes.memory_enabled,
                user_profile_enabled=sequence.hermes.user_profile_enabled,
                skills_enabled=sequence.hermes.skills_enabled,
                session_search_enabled=sequence.hermes.session_search_enabled,
            )
            _sc_tool_config["application_opportunity_schema"] = (
                build_past_bench_application_opportunity_schema(_sc_task)
            )
            _sc_preseed_dir = materialize_task_hermes_seed(
                task=_sc_task,
                target_dir=_sc_episode_dir / "preseed",
                base_preseed_dir=_resolve_episode_preseed_dir(sequence, _sc_episode),
            )
            persistence_backend.materialize_inputs(
                state_root=_sc_state_root,
                initial_home_fixture_dir=None,
                preseed_artifacts_dir=_sc_preseed_dir,
            )
            _sc_extra_body = persistence_backend.build_extra_body(
                state_root=_sc_state_root,
                artifacts_dir=_sc_artifacts_dir,
                persistence_enabled=False,
                sequence=sequence,
                family_id=_sc_episode.family_id,
                review_wait_s=float(_sc_review_wait),
                tool_config=_sc_tool_config,
            )

            print(
                f"\n[shared_cold {_sc_index}] family={_sc_episode.family_id} "
                f"task={_sc_task.task_id}"
            )

            with ServiceManager(_sc_task.services, cwd=_sc_tasks_dir.parent):
                _sc_trace_path, _sc_env_snapshot = _execute_trial(
                    task=_sc_task,
                    cfg=cfg,
                    task_dir=str(_sc_task_yaml.parent),
                    trace_dir=str(_sc_episode_dir),
                    model=args.model,
                    api_key=args.api_key,
                    base_url=args.base_url,
                    sandbox_mode=getattr(args, "sandbox", False) or cfg.sandbox.enabled,
                    sandbox_image=getattr(args, "sandbox_image", None),
                    sandbox_tools_local=getattr(args, "sandbox_tools", False),
                    agent_name=args.agent,
                    agent_profile=getattr(args, "agent_profile", None),
                    runtime_mode=runtime_mode,
                    runtime_image=getattr(args, "runtime_image", None),
                    registry_path=registry_path,
                    model_extra_body_override=_sc_extra_body,
                    runtime_temperature=runtime_temperature,
                    task_timeout_override=getattr(args, "task_timeout_override", None),
                    runtime_metadata={
                        "run_id": trace_root.name,
                        "episode_id": _sc_episode_dir.name,
                        "family_id": _sc_episode.family_id,
                        "stage": _sc_episode.stage,
                        "experiment_variant": shared_cold_variant,
                    },
                )

            _sc_grader = get_grader(_sc_task.task_id, tasks_dir=_sc_tasks_dir, task_dir=_sc_task_yaml.parent)
            _sc_artifact_summary = persistence_backend.snapshot_after(_sc_artifacts_dir)
            _sc_result = grade_episode(
                trace_path=_sc_trace_path,
                task=_sc_task,
                tasks_dir=_sc_tasks_dir,
                task_dir=_sc_task_yaml.parent,
                grader=_sc_grader,
                judge=judge,
                env_snapshot=_sc_env_snapshot,
                artifact_before=_sc_artifact_before,
                artifact_summary=_sc_artifact_summary,
                expected_persistence_signal=_sc_episode.expected_persistence_signal or _sc_episode.mechanism,
            )
            _append_grading_to_trace(
                trace_path=_sc_trace_path,
                trace_id=_sc_result["trace_id"],
                task_id=_sc_task.task_id,
                scores=_sc_result["scores"],
                task_score=_sc_result["task_score"],
                passed=_sc_result["passed"],
            )
            _sc_result.update({
                "index": _sc_index,
                "label": _sc_episode.label or _sc_task.task_name,
                "phase": _sc_episode.phase,
                "stage": _sc_episode.stage,
                "bucket": _sc_episode.bucket,
                "family_id": _sc_episode.family_id,
                "mechanism": _sc_episode.mechanism,
                "expected_persistence_signal": _sc_episode.expected_persistence_signal or _sc_episode.mechanism,
                "cluster_id": _sc_episode.cluster_id or _sc_episode.family_id,
                "latent_rule_id": _sc_episode.latent_rule_id,
                "transfer_distance": _sc_episode.transfer_distance,
                "noise_profile": _sc_episode.noise_profile,
                "conflict_mode": _sc_episode.conflict_mode,
                "learn_signal_type": _sc_episode.learn_signal_type,
                "noise_level": _sc_episode.noise_level,
                "reflection_required": _sc_episode.reflection_required,
                "evaluation_requires_retrieval": _sc_episode.evaluation_requires_retrieval,
                "requires_fresh_session": _sc_episode.requires_fresh_session,
                "persistence_allowed": _sc_episode.persistence_allowed,
                "history_mode": _sc_episode.history_mode,
                "history_save_anchor": _sc_episode.history_save_anchor,
                "history_load_anchor": _sc_episode.history_load_anchor,
            })
            print(f"  score={_sc_result['task_score']:.3f} passed={_sc_result['passed']}")

            # Freeze runtime state for this family so both variants can seed from it.
            _sc_family_state, _ = persistence_backend.family_paths(cold_base_dir, _sc_episode.family_id)
            shared_cold_homes[_sc_episode.family_id] = _sc_family_state
            shared_cold_episode_results[_sc_index] = _sc_result
            shared_cold_trace_paths[_sc_index] = _sc_trace_path
    # ── end shared cold pre-pass ─────────────────────────────────────────────────

    for variant_label, persistence_enabled in variants:
        variant_dir = trace_root / variant_label if len(variants) > 1 else trace_root
        variant_dir.mkdir(parents=True, exist_ok=True)
        family_homes_root = variant_dir / "family_homes"
        history_anchors_by_family: dict[str, dict[str, Path]] = {}
        if persistence_backend is not None:
            _reset_runtime_dir(family_homes_root)
        episode_results: list[dict] = []

        print(f"\n=== Sequence: {sequence.name} [{variant_label}] ===")
        print(sequence.description or "(no description)")

        for index, episode in enumerate(sequence.episodes, start=1):
            task_yaml = sequence.resolve_task_yaml(episode.task)
            task = TaskDefinition.from_yaml(task_yaml)
            if port_offset:
                task.apply_port_offset(port_offset)
            tasks_dir = _resolve_tasks_dir(task_yaml)
            episode_slug = _safe_label(episode.label or task.task_id)
            episode_dir = variant_dir / f"{index:02d}_{episode_slug}"
            episode_dir.mkdir(parents=True, exist_ok=True)
            artifacts_dir = episode_dir / "artifacts"
            if persistence_backend is not None:
                state_root, anchors_dir = persistence_backend.family_paths(variant_dir, episode.family_id)
                history_anchors = history_anchors_by_family.setdefault(episode.family_id, {})
            else:
                state_root = variant_dir / "runtime_state"
                anchors_dir = variant_dir / "history_anchors"
                history_anchors = {}

            # ── shared cold fast-path ────────────────────────────────────────────
            # If this episode was pre-run as a shared cold baseline, reuse its trace
            # and seed runtime state from the frozen post-I01 state instead of re-running.
            if episode.shared_cold_run and index in shared_cold_episode_results:
                cold_home = shared_cold_homes.get(episode.family_id)
                if persistence_backend is not None and cold_home:
                    persistence_backend.clone_state(cold_home, state_root)
                    persistence_backend.save_anchor(
                        state_root=state_root,
                        episode=episode,
                        anchors_dir=anchors_dir,
                        history_anchors=history_anchors,
                    )
                shared_trace = shared_cold_trace_paths[index]
                shutil.copy2(shared_trace, episode_dir / shared_trace.name)
                shared_artifacts_src = shared_trace.parent / "artifacts"
                if shared_artifacts_src.exists():
                    if artifacts_dir.exists():
                        shutil.rmtree(artifacts_dir)
                    shutil.copytree(shared_artifacts_src, artifacts_dir)
                episode_results.append(shared_cold_episode_results[index])
                print(
                    f"\n[{index}/{len(sequence.episodes)}] [shared cold → {variant_label}] "
                    f"family={episode.family_id} task={task.task_id} "
                    f"score={shared_cold_episode_results[index]['task_score']:.3f} "
                    f"passed={shared_cold_episode_results[index]['passed']}"
                )
                continue
            # ── end shared cold fast-path ────────────────────────────────────────

            if persistence_backend is not None:
                episode_persistence = persistence_enabled and episode.persistence_allowed
                persistence_backend.prepare_history(
                    state_root=state_root,
                    episode=episode,
                    history_anchors=history_anchors,
                )
                persistence_backend.materialize_inputs(
                    state_root=state_root,
                    initial_home_fixture_dir=_resolve_episode_initial_home_fixture_dir(sequence, episode),
                    preseed_artifacts_dir=_resolve_episode_preseed_dir(sequence, episode),
                )
            artifact_before = (
                persistence_backend.snapshot_before(state_root, include_contents=True)
                if persistence_backend is not None
                else _empty_artifact_summary(state_root)
            )

            model_extra_body_override = None
            if persistence_backend is not None:
                tool_config = resolve_episode_tool_config(
                    persistence_enabled=episode_persistence,
                    expected_signal=episode.mechanism,
                    memory_enabled=sequence.hermes.memory_enabled,
                    user_profile_enabled=sequence.hermes.user_profile_enabled,
                    skills_enabled=sequence.hermes.skills_enabled,
                    session_search_enabled=sequence.hermes.session_search_enabled,
                )
                tool_config["application_opportunity_schema"] = (
                    build_past_bench_application_opportunity_schema(task)
                )
                review_wait_s = (
                    args.background_review_wait_s
                    if args.background_review_wait_s is not None
                    else sequence.hermes.background_review_wait_s
                )
                preseed_dir = materialize_task_hermes_seed(
                    task=task,
                    target_dir=episode_dir / "preseed",
                    base_preseed_dir=_resolve_episode_preseed_dir(sequence, episode),
                )
                persistence_backend.materialize_inputs(
                    state_root=state_root,
                    initial_home_fixture_dir=None,
                    preseed_artifacts_dir=preseed_dir,
                )
                model_extra_body_override = persistence_backend.build_extra_body(
                    state_root=state_root,
                    artifacts_dir=artifacts_dir,
                    persistence_enabled=episode_persistence,
                    sequence=sequence,
                    family_id=episode.family_id,
                    review_wait_s=float(review_wait_s),
                    tool_config=tool_config,
                )

            print(
                f"\n[{index}/{len(sequence.episodes)}] family={episode.family_id} bucket={episode.bucket} "
                f"task={task.task_id} label={episode.label or task.task_name}"
            )

            try:
                with ServiceManager(task.services, cwd=tasks_dir.parent):
                    trace_path, env_snapshot = _execute_trial(
                        task=task,
                        cfg=cfg,
                        task_dir=str(task_yaml.parent),
                        trace_dir=str(episode_dir),
                        model=args.model,
                        api_key=args.api_key,
                        base_url=args.base_url,
                        sandbox_mode=getattr(args, "sandbox", False) or cfg.sandbox.enabled,
                        sandbox_image=getattr(args, "sandbox_image", None),
                        sandbox_tools_local=getattr(args, "sandbox_tools", False),
                        agent_name=args.agent,
                        agent_profile=getattr(args, "agent_profile", None),
                        runtime_mode=runtime_mode,
                        runtime_image=getattr(args, "runtime_image", None),
                        registry_path=registry_path,
                        model_extra_body_override=model_extra_body_override,
                        runtime_temperature=runtime_temperature,
                        task_timeout_override=getattr(args, "task_timeout_override", None),
                        runtime_metadata={
                            "run_id": trace_root.name,
                            "episode_id": episode_dir.name,
                            "family_id": episode.family_id,
                            "stage": episode.stage,
                            "experiment_variant": variant_label,
                        },
                    )

                grader = get_grader(task.task_id, tasks_dir=tasks_dir, task_dir=task_yaml.parent)
                artifact_summary = (
                    persistence_backend.snapshot_after(artifacts_dir)
                    if persistence_backend is not None
                    else _empty_artifact_summary(artifacts_dir)
                )
                episode_result = grade_episode(
                    trace_path=trace_path,
                    task=task,
                    tasks_dir=tasks_dir,
                    task_dir=task_yaml.parent,
                    grader=grader,
                    judge=judge,
                    env_snapshot=env_snapshot,
                    artifact_before=artifact_before,
                    artifact_summary=artifact_summary,
                    expected_persistence_signal=episode.expected_persistence_signal or episode.mechanism,
                )
                _append_grading_to_trace(
                    trace_path=trace_path,
                    trace_id=episode_result["trace_id"],
                    task_id=task.task_id,
                    scores=episode_result["scores"],
                    task_score=episode_result["task_score"],
                    passed=episode_result["passed"],
                )
            except RuntimeError as exc:
                # Per-episode fault tolerance: don't abort the whole sequence
                # because one episode timed out / crashed. Mark the episode as
                # infra-blocked with score 0 and continue to the next.
                # The timeout is intentionally uniform across agents/models —
                # blowing the budget is itself a signal about the framework.
                timed_out = _is_timeout_failure(exc)
                effective_to = getattr(args, "task_timeout_override", None) or task.environment.timeout_seconds
                print(f"\n[!] Episode {index} ({task.task_id}) failed: {type(exc).__name__}: {exc}")
                if timed_out:
                    print(
                        f"[hint] Episode hit the {effective_to}s wall timeout. "
                        f"The agent should reduce its thinking/reasoning chain — "
                        f"verbose internal monologue is eating the wall-time budget. "
                        f"The timeout is uniform across agents/models by design; "
                        f"do not raise it just for slower frameworks. Sequence will "
                        f"continue with this episode scored 0 (infra_blocked=True)."
                    )
                else:
                    print(
                        f"[hint] Runtime error before grading. Sequence will continue with this "
                        f"episode scored 0 (infra_blocked=True)."
                    )
                episode_result = _build_failed_episode_result(
                    exc=exc, task=task, episode=episode, index=index,
                )
            episode_result.update({
                "index": index,
                "label": episode.label or task.task_name,
                "phase": episode.phase,
                "stage": episode.stage,
                "bucket": episode.bucket,
                "family_id": episode.family_id,
                "mechanism": episode.mechanism,
                "expected_persistence_signal": episode.expected_persistence_signal or episode.mechanism,
                "cluster_id": episode.cluster_id or episode.family_id,
                "latent_rule_id": episode.latent_rule_id,
                "transfer_distance": episode.transfer_distance,
                "noise_profile": episode.noise_profile,
                "conflict_mode": episode.conflict_mode,
                "learn_signal_type": episode.learn_signal_type,
                "noise_level": episode.noise_level,
                "reflection_required": episode.reflection_required,
                "evaluation_requires_retrieval": episode.evaluation_requires_retrieval,
                "requires_fresh_session": episode.requires_fresh_session,
                "persistence_allowed": episode.persistence_allowed,
                "history_mode": episode.history_mode,
                "history_save_anchor": episode.history_save_anchor,
                "history_load_anchor": episode.history_load_anchor,
                "runtime_family_home": str(state_root) if persistence_backend is not None else "",
                "runtime_history_root": str(anchors_dir) if persistence_backend is not None else "",
            })
            episode_results.append(episode_result)

            if persistence_backend is not None and not episode_result.get("infra_blocked", False):
                persistence_backend.save_anchor(
                    state_root=state_root,
                    episode=episode,
                    anchors_dir=anchors_dir,
                    history_anchors=history_anchors,
                )

            print(
                f"  score={episode_result['task_score']:.3f} passed={episode_result['passed']} "
                f"dispatches={episode_result['tool_dispatch_count']} "
                f"artifacts(memory={episode_result['artifacts']['memory_chars']}, skills={episode_result['artifacts']['skill_count']}) "
                f"internal(memory={episode_result['internal_tools'].get('memory_calls', 0)}, "
                f"skill={episode_result['internal_tools'].get('skill_manage_calls', 0)}, "
                f"search={episode_result['internal_tools'].get('session_search_calls', 0)})"
            )

            if (
                persistence_backend is not None
                and not episode_result.get("infra_blocked", False)
                and sequence.hermes.reflection_enabled
                and episode.bucket == "learn"
                and episode.reflection_required
            ):
                reflection_dir = episode_dir / "reflection"
                reflection_dir.mkdir(parents=True, exist_ok=True)
                reflection_artifacts_dir = reflection_dir / "artifacts"
                reflection_before = persistence_backend.snapshot_before(state_root, include_contents=True)
                reflection_review_wait_s = (
                    args.background_review_wait_s
                    if args.background_review_wait_s is not None
                    else sequence.hermes.background_review_wait_s
                )
                reflection_tool_config = resolve_episode_tool_config(
                    persistence_enabled=persistence_enabled and episode.persistence_allowed,
                    expected_signal=episode.mechanism,
                    memory_enabled=sequence.hermes.memory_enabled,
                    user_profile_enabled=sequence.hermes.user_profile_enabled,
                    skills_enabled=sequence.hermes.skills_enabled,
                    session_search_enabled=sequence.hermes.session_search_enabled,
                )
                reflection_tool_config["application_opportunity_schema"] = (
                    build_past_bench_application_opportunity_schema(task)
                )
                reflection_preseed_dir = materialize_task_hermes_seed(
                    task=task,
                    target_dir=reflection_dir / "preseed",
                    base_preseed_dir=_resolve_episode_preseed_dir(sequence, episode),
                )
                persistence_backend.materialize_inputs(
                    state_root=state_root,
                    initial_home_fixture_dir=None,
                    preseed_artifacts_dir=reflection_preseed_dir,
                )
                reflection_extra_body = persistence_backend.build_extra_body(
                    state_root=state_root,
                    artifacts_dir=reflection_artifacts_dir,
                    persistence_enabled=persistence_enabled and episode.persistence_allowed,
                    sequence=sequence,
                    family_id=episode.family_id,
                    review_wait_s=float(reflection_review_wait_s),
                    tool_config=reflection_tool_config,
                )
                reflection_prompt = build_reflection_prompt(
                    trace_path=Path(episode_result["trace"]),
                    task=task,
                    attempt_result=episode_result,
                )
                reflection_task = make_reflection_task(
                    task_id=f"{task.task_id}_REFLECT",
                    task_name=f"{task.task_name} Reflection",
                    prompt_text=reflection_prompt,
                )

                trace_path, _ = _execute_trial(
                    task=reflection_task,
                    cfg=cfg,
                    task_dir=str(task_yaml.parent),
                    trace_dir=str(reflection_dir),
                    model=args.model,
                    api_key=args.api_key,
                    base_url=args.base_url,
                    sandbox_mode=getattr(args, "sandbox", False) or cfg.sandbox.enabled,
                    sandbox_image=getattr(args, "sandbox_image", None),
                    sandbox_tools_local=getattr(args, "sandbox_tools", False),
                    agent_name=args.agent,
                    agent_profile=getattr(args, "agent_profile", None),
                    runtime_mode=runtime_mode,
                    runtime_image=getattr(args, "runtime_image", None),
                    registry_path=registry_path,
                    model_extra_body_override=reflection_extra_body,
                    runtime_temperature=runtime_temperature,
                    task_timeout_override=getattr(args, "task_timeout_override", None),
                    runtime_metadata={
                        "run_id": trace_root.name,
                        "episode_id": reflection_dir.name,
                        "family_id": episode.family_id,
                        "stage": "reflection",
                        "experiment_variant": variant_label,
                    },
                )
                reflection_artifacts = persistence_backend.snapshot_after(reflection_artifacts_dir)
                reflection_result = summarize_reflection_episode(
                    trace_path=trace_path,
                    artifact_summary=reflection_artifacts,
                    task_id=reflection_task.task_id,
                    label=f"{episode.label or task.task_name} Reflection",
                )
                reflection_result.update({
                    "index": f"{index}r",
                    "label": f"{episode.label or task.task_name} Reflection",
                    "phase": episode.phase,
                    "stage": "reflection",
                    "bucket": "reflection",
                    "episode_kind": "reflection",
                    "family_id": episode.family_id,
                    "mechanism": episode.mechanism,
                    "expected_persistence_signal": episode.expected_persistence_signal or episode.mechanism,
                    "cluster_id": episode.cluster_id or episode.family_id,
                    "latent_rule_id": episode.latent_rule_id,
                    "transfer_distance": "none",
                    "noise_profile": "none",
                    "conflict_mode": episode.conflict_mode,
                    "learn_signal_type": episode.learn_signal_type,
                    "noise_level": "none",
                    "reflection_required": False,
                    "evaluation_requires_retrieval": False,
                    "requires_fresh_session": True,
                    "persistence_allowed": episode.persistence_allowed,
                    "artifact_diff": diff_artifact_snapshots(
                        before=reflection_before,
                        after=reflection_artifacts,
                    ),
                })
                episode_results.append(reflection_result)
                print(
                    "  reflection "
                    f"memory={reflection_result['artifacts']['memory_chars']} "
                    f"skills={reflection_result['artifacts']['skill_count']} "
                    f"internal(memory={reflection_result['internal_tools'].get('memory_calls', 0)}, "
                    f"skill={reflection_result['internal_tools'].get('skill_manage_calls', 0)}, "
                    f"search={reflection_result['internal_tools'].get('session_search_calls', 0)})"
                )

        summary = summarize_sequence(
            sequence_name=sequence.name,
            variant=variant_label,
            episodes=episode_results,
        )
        write_json(variant_dir / "sequence_results.json", {
            "sequence": sequence.name,
            "variant": variant_label,
            "episodes": episode_results,
        })
        write_json(variant_dir / "sequence_summary.json", summary)
        variant_summaries[variant_label] = summary

        print(f"\nSummary [{variant_label}]")
        for bucket_name, bucket_summary in summary["bucket_summary"].items():
            _print_sequence_bucket_summary(bucket_name, bucket_summary)
        print(
            "  family_delta: "
            f"score={summary['benchmark_signal']['avg_family_task_score_delta']:.3f} "
            f"pass_rate={summary['benchmark_signal']['avg_family_pass_rate_delta']:.3f}"
        )
        print("  task_breakdown:")
        for result in episode_results:
            _print_episode_result_summary(result)

    if len(variants) > 1:
        comparison = summarize_comparison(
            with_persistence=variant_summaries["with_persistence"],
            without_persistence=variant_summaries["without_persistence"],
        )
        write_json(trace_root / "sequence_comparison.json", comparison)
        print("\nComparison")
        print(
            "  evaluation delta: "
            f"score={comparison['delta']['evaluation_avg_task_score']:.3f} "
            f"pass_rate={comparison['delta']['evaluation_pass_rate']:.3f} "
            f"dispatches={comparison['delta']['evaluation_avg_tool_dispatch_count']:.2f}"
        )
        print(
            "  family improvement delta: "
            f"score={comparison['delta']['avg_family_task_score_delta']:.3f} "
            f"pass_rate={comparison['delta']['avg_family_pass_rate_delta']:.3f}"
        )


def cmd_list(args: argparse.Namespace) -> None:
    """List available tasks."""
    tasks_dir = Path(args.tasks_dir)
    if not tasks_dir.exists():
        print(f"Tasks directory not found: {tasks_dir}")
        return

    from .models.task import TaskDefinition

    for yaml_file in sorted(tasks_dir.glob("*/task.yaml")):
        try:
            task = TaskDefinition.from_yaml(yaml_file)
            print(f"  {task.task_id:6s}  {task.task_name:30s}  difficulty={task.difficulty}  category={task.category}")
        except Exception as e:
            print(f"  {yaml_file.parent.name}: error loading - {e}")


def cmd_list_agents(args: argparse.Namespace) -> None:
    """List registered agent runtimes."""
    from .config import load_config
    from .runtime.registry import load_agent_registry, required_env_names, resolve_model_defaults

    cfg = load_config(getattr(args, "config", None))
    registry = load_agent_registry(_resolve_registry_path(args, cfg))
    for name in sorted(registry):
        spec = registry[name]
        selected_profile, defaults = resolve_model_defaults(spec)
        required = ",".join(required_env_names(spec, profile=selected_profile)) or "-"
        model_id = defaults.model_id or "-"
        profile_note = f" profile={selected_profile}" if selected_profile else ""
        print(f"  {name:16s}  adapter={spec.adapter:20s} model={model_id:28s} env={required}")
        if spec.model_profiles:
            print(f"      profiles: {', '.join(sorted(spec.model_profiles))}{profile_note}")
        if spec.description:
            print(f"      {spec.description}")


def cmd_doctor(args: argparse.Namespace) -> None:
    """Report environment and runtime readiness for one or more agents."""
    from .config import load_config
    from .runtime.registry import (
        get_agent_spec,
        load_agent_registry,
        missing_required_env,
        required_env_names,
        resolve_env_value,
        resolve_model_defaults,
    )

    cfg = load_config(getattr(args, "config", None))
    registry = load_agent_registry(_resolve_registry_path(args, cfg))
    names = [get_agent_spec(args.agent, registry).name] if getattr(args, "agent", None) else sorted(registry)

    for name in names:
        spec = registry[name]
        profile = getattr(args, "agent_profile", None) if getattr(args, "agent", None) == name else None
        selected_profile, defaults = resolve_model_defaults(spec, profile=profile)
        missing = missing_required_env(spec, profile=profile)
        status = "OK" if not missing else "MISSING"
        image = spec.runtime_image or cfg.runtime.image
        model_id = defaults.model_id or "-"
        print(f"[{status}] agent={name}")
        print(f"  adapter:       {spec.adapter}")
        print(f"  runtime image: {image}")
        print(f"  default model: {model_id}")
        if selected_profile:
            print(f"  profile:       {selected_profile}")
        if spec.model_profiles:
            print(f"  profiles:      {', '.join(sorted(spec.model_profiles))}")
        required_envs = required_env_names(spec, profile=profile)
        if required_envs:
            for env_name in required_envs:
                present = "yes" if resolve_env_value(env_name) else "no"
                print(f"  env {env_name}: {present}")
        else:
            print("  env:           none")
        if missing:
            print(f"  missing:       {', '.join(missing)}")


def cmd_v2_report(args: argparse.Namespace) -> None:
    """Emit V2 Layer 1/2/3 + 4-level reports for a self-evolve trace directory.

    Reads `<trace_dir>/sequence_comparison.json` produced by `past_bench evolve`
    and writes `v2_task_report.json`, `v2_family_report.json`,
    `v2_ability_report.json`, `v2_benchmark_report.json` per §16 of the V2
    design doc. Picks up `<trace_dir>/shortcut_scores.json` automatically if
    present so shortcut-control results land in `ShortcutResistance` without
    any extra glue.
    """
    from pathlib import Path as _P

    from past_bench.metrics.v2_layer1 import emit_reports, ingest_full_reports

    trace_dir = _P(args.trace_dir)
    comparison = trace_dir / "sequence_comparison.json"
    if not comparison.exists():
        raise SystemExit(f"missing {comparison}")
    l1, l2, l3, tasks = ingest_full_reports(comparison, v2_root=args.v2_root)
    out_dir = _P(args.out_dir) if args.out_dir else trace_dir
    paths = emit_reports(
        l1, out_dir, v2_root=args.v2_root,
        layer2=l2, layer3=l3, task_rows=tasks,
    )
    for key, path in paths.items():
        print(f"{key}: {path}")


def cmd_validate_agent(args: argparse.Namespace) -> None:
    """Start the selected runtime transport and verify bootstrap/health."""
    from .config import load_config
    from .runtime.client import create_runtime_client
    from .runtime.protocol import BootstrapRequest

    cfg = load_config(getattr(args, "config", None))
    registry_path = _resolve_registry_path(args, cfg)
    runtime_mode = _resolve_runtime_mode(args, cfg)
    profile = getattr(args, "agent_profile", None)
    spec = _check_agent_requirements(args.agent, registry_path, profile=profile)

    client = create_runtime_client(
        cfg.runtime,
        mode=runtime_mode,
        registry_path=registry_path,
        image=getattr(args, "runtime_image", None) or spec.runtime_image,
    )
    run_id = f"validate-{args.agent}"
    if profile:
        run_id += f"-{profile}"
    try:
        client.start(run_id=run_id, agent_spec=spec)
        health = client.health()
        bootstrap = client.bootstrap(BootstrapRequest(agent_name=args.agent, force=getattr(args, "force_bootstrap", False)))
    finally:
        client.stop()

    print(f"agent:       {args.agent}")
    if profile:
        print(f"profile:     {profile}")
    print(f"runtime:     {runtime_mode}")
    print(f"health:      {health.status}")
    print(f"installed:   {bootstrap.installed}")
    print(f"cached:      {bootstrap.already_present}")
    if bootstrap.commands_run:
        print(f"commands:    {len(bootstrap.commands_run)}")
        for command in bootstrap.commands_run:
            print(f"  - {command}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="PAST-Bench evaluation framework")
    sub = parser.add_subparsers(dest="command")

    # run
    p_run = sub.add_parser("run", help="Run agent on a task")
    p_run.add_argument("--task", required=True, help="Path to task dir or YAML (e.g. past_bench_tasks/T70_api_deprecation_supersession_audit)")
    p_run.add_argument("--agent", default=None, help="Registered agent runtime name (e.g. codex, claude, openclaw)")
    p_run.add_argument("--agent-profile", default=None, help="Optional built-in agent/model profile (e.g. kimi, openai, claude)")
    p_run.add_argument("--model", default=None, help="Model ID (default: from config.yaml)")
    p_run.add_argument("--api-key", default=None, help="API key (default: from config.yaml / $OPENAI_API_KEY)")
    p_run.add_argument("--base-url", default=None, help="Base URL for OpenAI-compatible API")
    p_run.add_argument("--config", default=None, help="Path to config.yaml")
    p_run.add_argument("--registry", default=None, help="Path to agent registry YAML")
    p_run.add_argument("--runtime", default=None, choices=["local", "container"], help="Agent runtime transport")
    p_run.add_argument("--runtime-image", default=None, help="Override runtime Docker image")
    p_run.add_argument("--trials", type=int, default=1, help="Number of trials")
    p_run.add_argument("--trace-dir", default=None, help="Output directory for traces")
    p_run.add_argument("--judge-model", default=None, help="Override judge model ID")
    p_run.add_argument("--no-judge", action="store_true", help="Disable LLM judge for communication scoring")
    p_run.add_argument("--port-offset", type=int, default=0, help="Offset for all service ports (enables parallel runs)")
    p_run.add_argument("--sandbox", action="store_true", help="Run inside a Docker sandbox container")
    p_run.add_argument("--sandbox-image", default=None, help="Override sandbox Docker image name")
    p_run.add_argument("--sandbox-tools", action="store_true", help="Inject sandbox tools (shell/file/browser) without Docker")
    p_run.add_argument("--temperature", type=float, default=None, help="Bench-owned runtime temperature override (default: config runtime.temperature, usually 0.0)")
    p_run.add_argument("--task-timeout-override", type=int, default=None, help="DEBUG ONLY: override per-task wall timeout in seconds (default: task.environment.timeout_seconds). Benchmark runs MUST use the task default so all agents/models compete on the same wall-time budget; do not use this flag to give a slower agent more headroom.")
    p_run.add_argument("--proxy", default=None, help="HTTP proxy URL for model/judge API traffic (e.g. http://proxy:port)")

    # _run-inner (hidden — used inside sandbox containers)
    p_inner = sub.add_parser("_run-inner", help=argparse.SUPPRESS)
    p_inner.add_argument("--task", required=True)
    p_inner.add_argument("--model", default=None)
    p_inner.add_argument("--api-key", default=None)
    p_inner.add_argument("--base-url", default=None)
    p_inner.add_argument("--config", default=None)
    p_inner.add_argument("--trace-dir", default=None)
    p_inner.add_argument("--sandbox-tools", action="store_true")
    p_inner.add_argument("--judge-model", default=None)
    p_inner.add_argument("--no-judge", action="store_true")
    p_inner.add_argument("--proxy", default=None)

    # build-image
    p_build = sub.add_parser("build-image", help="Build the sandbox or runtime Docker image")
    p_build.add_argument("--kind", default="sandbox", choices=["sandbox", "runtime"], help="Which image to build")
    p_build.add_argument("--image", default=None, help="Image name/tag (default: from config)")
    p_build.add_argument("--context", default=".", help="Docker build context path")
    p_build.add_argument("--dockerfile", default=None, help="Dockerfile name override")
    p_build.add_argument("--config", default=None, help="Path to config.yaml")

    # grade
    p_grade = sub.add_parser("grade", help="Grade an existing trace")
    p_grade.add_argument("--trace", required=True, help="Path to JSONL trace file")
    p_grade.add_argument("--task", required=True, help="Path to task dir or YAML (e.g. past_bench_tasks/T70_api_deprecation_supersession_audit)")
    p_grade.add_argument("--config", default=None, help="Path to config.yaml")
    p_grade.add_argument("--judge-model", default=None, help="Override judge model ID")
    p_grade.add_argument("--no-judge", action="store_true", help="Disable LLM judge for communication scoring")
    p_grade.add_argument("--proxy", default=None, help="HTTP proxy URL for judge API traffic")

    # batch
    p_batch = sub.add_parser("batch", help="Run all tasks in parallel")
    p_batch.add_argument("--tasks-dir", default="past_bench_tasks", help="Tasks directory")
    p_batch.add_argument("--filter", default=None, help="Only run tasks matching this substring (e.g. 'en_' or 'T01')")
    p_batch.add_argument("--parallel", type=int, default=4, help="Number of parallel workers (default: 4)")
    p_batch.add_argument("--agent", default=None, help="Registered agent runtime name (or comma-separated list, e.g. 'zeroclaw,hermes,claude')")
    p_batch.add_argument("--agent-profile", default=None, help="Optional built-in agent/model profile")
    p_batch.add_argument("--model", default=None)
    p_batch.add_argument("--api-key", default=None)
    p_batch.add_argument("--base-url", default=None)
    p_batch.add_argument("--config", default=None, help="Path to config.yaml")
    p_batch.add_argument("--registry", default=None, help="Path to agent registry YAML")
    p_batch.add_argument("--runtime", default=None, choices=["local", "container"], help="Agent runtime transport")
    p_batch.add_argument("--runtime-image", default=None, help="Override runtime Docker image")
    p_batch.add_argument("--temperature", type=float, default=None, help="Bench-owned runtime temperature override (default: config runtime.temperature, usually 0.0)")
    p_batch.add_argument("--trials", type=int, default=1)
    p_batch.add_argument("--trace-dir", default=None, help="Output directory for traces")
    p_batch.add_argument("--judge-model", default=None)
    p_batch.add_argument("--no-judge", action="store_true")
    p_batch.add_argument("--proxy", default=None, help="HTTP proxy URL for model/judge API traffic")
    p_batch.add_argument("--port-base-offset", type=int, default=0, help="Base port offset to avoid conflicts when running multiple batch jobs (e.g. 400)")
    p_batch.add_argument("--sandbox", action="store_true", help="Run sandbox tools inside Docker containers")
    p_batch.add_argument("--sandbox-image", default=None, help="Override sandbox Docker image name")
    p_batch.add_argument("--sandbox-tools", action="store_true", help="Inject sandbox tools without Docker sandbox container")
    p_batch.add_argument("--rerun-errors", default=None, metavar="TRACE_DIR",
                         help="Re-run only errored tasks from a previous batch run. "
                              "Reads batch_results.json from TRACE_DIR, re-runs errored tasks, "
                              "and merges results back into the same directory.")
    p_batch.add_argument("--continue", dest="continue_dir", default=None, metavar="TRACE_DIR",
                         help="Continue a previous batch run from TRACE_DIR. "
                              "Scans existing trace files for grading_result events, "
                              "skips tasks with enough completed trials, and only runs the rest. "
                              "Results are merged into the same directory.")

    # evolve
    p_evolve = sub.add_parser("evolve", help="Run a multi-episode self-evolve sequence")
    p_evolve.add_argument("--sequence", default=None, help="Path to a self-evolve sequence YAML manifest")
    p_evolve.add_argument(
        "--v2-family",
        default=None,
        help=(
            "V2 family (e.g. 'memory_ability/SM01_preference_adoption') "
            "— regenerates a manifest from self-evolve-tasks-v2 and runs it"
        ),
    )
    p_evolve.add_argument("--family", default=None, help="V2 family (e.g. 'procedural_ability/PC01_sop_bootstrap_01') — regenerates manifest from family.yaml and runs it")
    p_evolve.add_argument("--agent", required=True, help="Registered agent runtime name")
    p_evolve.add_argument("--agent-profile", default=None, help="Optional built-in agent/model profile")
    p_evolve.add_argument("--model", default=None, help="Override model ID")
    p_evolve.add_argument("--api-key", default=None, help="Override API key")
    p_evolve.add_argument("--base-url", default=None, help="Override base URL")
    p_evolve.add_argument("--config", default=None, help="Path to config.yaml")
    p_evolve.add_argument("--registry", default=None, help="Path to agent registry YAML")
    p_evolve.add_argument("--runtime", default=None, choices=["local", "container"], help="Agent runtime transport")
    p_evolve.add_argument("--runtime-image", default=None, help="Override runtime Docker image")
    p_evolve.add_argument("--temperature", type=float, default=None, help="Bench-owned runtime temperature override (default: config runtime.temperature, usually 0.0)")
    p_evolve.add_argument("--port-offset", type=int, default=0, help="Shift task service ports for parallel evolve runs")
    p_evolve.add_argument("--task-timeout-override", type=int, default=None, help="DEBUG ONLY: override per-task wall timeout in seconds (default: task.environment.timeout_seconds). Benchmark runs MUST use the task default so all agents/models compete on the same wall-time budget; do not use this flag to give a slower agent more headroom.")
    p_evolve.add_argument("--trace-dir", default=None, help="Output directory for sequence traces and summaries")
    p_evolve.add_argument("--judge-model", default=None, help="Override judge model ID")
    p_evolve.add_argument("--no-judge", action="store_true", help="Disable LLM judge for communication scoring")
    p_evolve.add_argument("--sandbox", action="store_true", help="Run inside a Docker sandbox container")
    p_evolve.add_argument("--sandbox-image", default=None, help="Override sandbox Docker image name")
    p_evolve.add_argument("--sandbox-tools", action="store_true", help="Inject sandbox tools without Docker")
    p_evolve.add_argument("--proxy", default=None, help="HTTP proxy URL for model/judge API traffic")
    p_evolve.add_argument("--compare-no-persistence", action="store_true", help="Run paired with-persistence and without-persistence variants for supported self-evolve agents")
    p_evolve.add_argument(
        "--persistence-variant",
        choices=["with_persistence", "without_persistence", "paired"],
        default=None,
        help="Run one explicit persistence state or the legacy paired comparison",
    )
    p_evolve.add_argument("--reflection-off", action="store_true", help="§15 reflection_off control: disable Hermes reflection on this run")
    p_evolve.add_argument("--background-review-wait-s", type=float, default=None, help="Wait time after Hermes finishes so background memory/skill review can flush")
    p_evolve.add_argument(
        "--rsimem-mode",
        choices=["native", "native+ledger", "native+adapter+ledger"],
        default=None,
        help="Explicitly override the Hermes RSIMem execution mode for this sequence",
    )
    p_evolve.add_argument(
        "--rsimem-method-task-id",
        default=None,
        help=(
            "Opaque RSIMem method task ID. Sensitivity launchers set this to "
            "their registered case ID; it must not be a PAST family/task ID."
        ),
    )
    p_evolve.add_argument(
        "--rsimem-adapter-failure-policy",
        choices=["fail_closed", "bypass_native"],
        default=None,
        help="Explicitly override the RSIMem adapter failure policy",
    )
    p_evolve.add_argument(
        "--rsimem-verify-native-projection",
        action="store_true",
        help="Fail closed unless each adapter read exactly matches a native shadow read",
    )
    p_evolve.add_argument(
        "--rsimem-lifecycle-evaluator-mode",
        choices=["disabled", "deterministic", "injected_json"],
        default=None,
        help="Explicitly opt into the Hermes lifecycle dry-run evaluator",
    )
    p_evolve.add_argument("--rsimem-lifecycle-policy-version", default=None)
    p_evolve.add_argument("--rsimem-lifecycle-compiler-version", default=None)
    p_evolve.add_argument(
        "--rsimem-lifecycle-timeout-seconds",
        type=float,
        default=None,
    )
    p_evolve.add_argument(
        "--rsimem-lifecycle-max-output-tokens",
        type=int,
        default=None,
    )
    p_evolve.add_argument(
        "--rsimem-semantic-writeback-mode",
        choices=["disabled", "static", "static_utility", "adaptive_utility"],
        default=None,
    )
    p_evolve.add_argument(
        "--rsimem-semantic-writeback-timeout-seconds",
        type=float,
        default=None,
    )
    p_evolve.add_argument(
        "--rsimem-semantic-writeback-max-output-tokens",
        type=int,
        default=None,
    )
    p_evolve.add_argument(
        "--rsimem-semantic-feedback-contract",
        choices=[
            "disabled",
            "sm01_tsv_v1",
            "sm02_boundary_v1",
            "sm03_fact_correction_v1",
            "sm05_normalized_tsv_v1",
        ],
        default=None,
        help="Pre-registered deployment signal contract for semantic feedback",
    )
    p_evolve.add_argument(
        "--rsimem-adaptive-config",
        default=None,
        help="Strict JSON config for an attempt-local ACTIVE adaptive policy store",
    )
    p_evolve.add_argument(
        "--rsimem-revocation-registry",
        default=None,
        help="Owner-controlled revocation registry required for validated extraction runs",
    )
    p_evolve.add_argument(
        "--rsimem-extraction-trial-config",
        default=None,
        help=(
            "Validation-only extraction trial bundle prepared by RSIMem; "
            "requires static semantic writeback"
        ),
    )
    p_evolve.add_argument(
        "--rsimem-extraction-offline-config",
        default=None,
        help=(
            "Offline-only extraction candidate bundle for held-out observations; "
            "requires static semantic writeback"
        ),
    )

    # cleanup
    p_cleanup = sub.add_parser("cleanup", help="Remove all PAST-Bench Docker containers")
    p_cleanup.add_argument("--config", default=None, help="Path to config.yaml")

    # list
    p_list = sub.add_parser("list", help="List available tasks")
    p_list.add_argument("--tasks-dir", default="past_bench_tasks", help="Tasks directory")

    # list-agents
    p_list_agents = sub.add_parser("list-agents", help="List registered agents")
    p_list_agents.add_argument("--config", default=None, help="Path to config.yaml")
    p_list_agents.add_argument("--registry", default=None, help="Path to agent registry YAML")

    # doctor
    p_doctor = sub.add_parser("doctor", help="Check environment readiness for agent runtimes")
    p_doctor.add_argument("--agent", default=None, help="Only check one agent")
    p_doctor.add_argument("--agent-profile", default=None, help="Optional built-in profile for the selected agent")
    p_doctor.add_argument("--config", default=None, help="Path to config.yaml")
    p_doctor.add_argument("--registry", default=None, help="Path to agent registry YAML")

    # validate-agent
    p_validate = sub.add_parser("validate-agent", help="Validate agent runtime health/bootstrap")
    p_validate.add_argument("--agent", required=True, help="Agent name to validate")
    p_validate.add_argument("--agent-profile", default=None, help="Optional built-in profile for the selected agent")
    p_validate.add_argument("--config", default=None, help="Path to config.yaml")
    p_validate.add_argument("--registry", default=None, help="Path to agent registry YAML")
    p_validate.add_argument("--runtime", default=None, choices=["local", "container"], help="Agent runtime transport")
    p_validate.add_argument("--runtime-image", default=None, help="Override runtime Docker image")
    p_validate.add_argument("--force-bootstrap", action="store_true", help="Force bootstrap even if cached")

    p_v2report = sub.add_parser("v2-report", help="Emit V2 Layer 1/2/3 + 4-level reports for a self-evolve trace dir")
    p_v2report.add_argument("trace_dir", help="Directory containing sequence_comparison.json")
    p_v2report.add_argument("--v2-root", default="self-evolve-tasks-v2", help="Path to self-evolve-tasks-v2 root")
    p_v2report.add_argument("--out-dir", default=None, help="Report output dir (defaults to trace_dir)")

    args = parser.parse_args(argv)

    if args.command == "run":
        cmd_run(args)
    elif args.command == "_run-inner":
        cmd_run_inner(args)
    elif args.command == "build-image":
        cmd_build_image(args)
    elif args.command == "grade":
        cmd_grade(args)
    elif args.command == "batch":
        cmd_batch(args)
    elif args.command == "evolve":
        cmd_evolve(args)
    elif args.command == "cleanup":
        cmd_cleanup(args)
    elif args.command == "list":
        cmd_list(args)
    elif args.command == "list-agents":
        cmd_list_agents(args)
    elif args.command == "doctor":
        cmd_doctor(args)
    elif args.command == "validate-agent":
        cmd_validate_agent(args)
    elif args.command == "v2-report":
        cmd_v2_report(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
