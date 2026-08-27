import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from past_bench.models.content import TextBlock
from past_bench.models.message import Message
from past_bench.models.self_evolve import (
    RSIMemAdaptiveWritebackConfig,
    SelfEvolveEpisode,
    SelfEvolveSequenceDefinition,
)
from past_bench.models.tool import ToolEndpoint, ToolSpec
from past_bench.models.trace import TraceMessage
from past_bench.graders.self_evolve_helpers import compute_self_evolve_mechanism_scores
from past_bench.runner.self_evolve import (
    _resolve_effective_final_text,
    build_hermes_extra_body,
    choose_calibration_candidate,
    compute_retrieval_signals,
    diff_artifact_snapshots,
    resolve_episode_tool_config,
    snapshot_hermes_artifacts,
    snapshot_hermes_home,
    summarize_comparison,
    summarize_sequence,
    summarize_single_task_sequence,
)
from past_bench.cli import _apply_rsimem_execution_overrides, _save_episode_history_anchor
from past_bench.runtime.adapters.hermes import (
    HermesAdapter,
    _RecordedHermesCompletionClient,
    _build_hermes_prompt,
)
from past_bench.runtime.protocol import RuntimeConfigPayload, RuntimeModelConfig, StartSessionRequest
from past_bench.runtime.registry import AgentSpec

_MISSING_TOOL = object()


def _adaptive_config() -> dict:
    return {
        "schema_version": 1,
        "adaptive_policy_store_path": ".rsimem/adaptive-policies.json",
        "adaptive_trusted_roots": ["mem0-flat.static-policy-v1"],
        "adaptive_parameters": [{
            "parameter_id": "parameter.retrieval",
            "name": "retrieval_accept_threshold",
            "prompt_ref": "mem0-flat.retrieval",
            "baseline_value": 0.35,
        }],
    }


def _rsimem_adapter_request(tmp_path: Path, rsimem: dict) -> StartSessionRequest:
    return StartSessionRequest(
        session_id="session-rsimem",
        agent_name="hermes",
        task_id="T_rsimem",
        task_name="RSIMem bridge",
        max_turns=4,
        timeout_seconds=60,
        initial_messages=[],
        model=RuntimeModelConfig(
            model_id="fixture-model",
            extra_body={"hermes": {
                "home_dir": str(tmp_path / "home"),
                "capture_artifacts_dir": str(tmp_path / "artifacts"),
                "rsimem": rsimem,
            }},
        ),
        runtime_config=RuntimeConfigPayload(metadata={
            "run_id": "run-rsimem",
            "trace_id": "trace-rsimem",
            "episode_id": "episode-rsimem",
            "family_id": "SM01",
            "stage": "learn",
            "experiment_variant": "with_persistence",
        }),
    )


def test_self_evolve_sequence_resolves_relative_task_dirs(tmp_path: Path):
    manifest = tmp_path / "sequence.yaml"
    task_dir = tmp_path / "tasks" / "T_demo"
    task_dir.mkdir(parents=True)
    (task_dir / "task.yaml").write_text("task_id: demo\ntask_name: Demo\nprompt:\n  text: hi\n", encoding="utf-8")
    manifest.write_text(
        yaml.safe_dump(
            {
                "name": "demo-seq",
                "episodes": [
                    {
                        "task": "tasks/T_demo",
                        "cluster_id": "cluster-a",
                        "phase": "teach",
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    sequence = SelfEvolveSequenceDefinition.from_yaml(manifest)

    assert sequence.resolve_task_yaml("tasks/T_demo") == task_dir / "task.yaml"
    assert sequence.episodes[0].family_id == "cluster-a"
    assert sequence.episodes[0].bucket == "baseline"


def test_build_hermes_extra_body_contains_persistence_overrides(tmp_path: Path):
    payload = build_hermes_extra_body(
        home_dir=tmp_path / "home",
        artifacts_dir=tmp_path / "artifacts",
        persistence_enabled=True,
        memory_enabled=True,
        user_profile_enabled=False,
        skills_enabled=True,
        session_search_enabled=False,
        memory_nudge_interval=2,
        memory_flush_min_turns=3,
        skill_creation_nudge_interval=4,
        background_review_wait_s=1.5,
    )

    hermes_cfg = payload["hermes"]
    assert hermes_cfg["persistence_enabled"] is True
    assert hermes_cfg["enabled_toolsets"] == ["memory", "skills"]
    assert hermes_cfg["config_overrides"]["memory"]["user_profile_enabled"] is False
    assert hermes_cfg["config_overrides"]["memory"]["nudge_interval"] == 2
    assert hermes_cfg["config_overrides"]["skills"]["creation_nudge_interval"] == 4
    assert hermes_cfg["background_review_wait_s"] == 1.5
    assert hermes_cfg["initial_home_fixture_dir"] == ""
    assert hermes_cfg["rsimem"] == {
        "mode": "native",
        "adapter_failure_policy": "fail_closed",
        "verify_native_projection": False,
        "evidence_path": str(tmp_path / "artifacts" / "rsimem_memory_events.jsonl"),
        "lifecycle": {
            "evaluator_mode": "disabled",
            "policy_version": "phase1-dry-run-v1",
            "compiler_version": "uncompiled-v0",
            "timeout_seconds": 30.0,
            "max_output_tokens": 4096,
        },
        "semantic_writeback": {
            "mode": "disabled",
            "timeout_seconds": 30.0,
            "max_output_tokens": 4096,
        },
    }


def test_static_semantic_writeback_isolated_from_native_and_no_persistence(
    tmp_path: Path,
) -> None:
    common = {
        "home_dir": tmp_path / "home",
        "artifacts_dir": tmp_path / "artifacts",
        "memory_enabled": True,
        "user_profile_enabled": True,
        "skills_enabled": True,
        "session_search_enabled": True,
        "memory_nudge_interval": 1,
        "memory_flush_min_turns": 1,
        "skill_creation_nudge_interval": 1,
        "background_review_wait_s": 0.0,
        "rsimem_mode": "native+ledger",
        "rsimem_lifecycle_evaluator_mode": "deterministic",
        "rsimem_semantic_writeback_mode": "static_utility",
    }
    enabled = build_hermes_extra_body(persistence_enabled=True, **common)["hermes"]
    assert "memory" not in enabled["enabled_toolsets"]
    assert enabled["config_overrides"]["memory"] == {
        "memory_enabled": True,
        "user_profile_enabled": True,
        "nudge_interval": 1,
        "flush_min_turns": 1,
    }
    assert enabled["rsimem"]["semantic_writeback"]["mode"] == "static_utility"

    disabled = build_hermes_extra_body(
        persistence_enabled=False,
        **common,
    )["hermes"]
    assert disabled["rsimem"]["semantic_writeback"]["mode"] == "disabled"

    with pytest.raises(ValueError, match=r"native\+ledger"):
        build_hermes_extra_body(
            persistence_enabled=True,
            **{**common, "rsimem_mode": "native+adapter+ledger"},
        )
    with pytest.raises(ValueError, match="requires lifecycle"):
        build_hermes_extra_body(
            persistence_enabled=True,
            **{**common, "rsimem_lifecycle_evaluator_mode": "disabled"},
        )


def test_adaptive_semantic_writeback_transport_is_strict_and_fail_closed(
    tmp_path: Path,
) -> None:
    common = {
        "home_dir": tmp_path / "home",
        "artifacts_dir": tmp_path / "artifacts",
        "memory_enabled": True,
        "user_profile_enabled": True,
        "skills_enabled": True,
        "session_search_enabled": True,
        "memory_nudge_interval": 1,
        "memory_flush_min_turns": 1,
        "skill_creation_nudge_interval": 1,
        "background_review_wait_s": 0.0,
        "rsimem_mode": "native+ledger",
        "rsimem_lifecycle_evaluator_mode": "deterministic",
        "rsimem_semantic_writeback_mode": "adaptive_utility",
    }
    enabled = build_hermes_extra_body(
        persistence_enabled=True,
        rsimem_adaptive_config=_adaptive_config(),
        **common,
    )["hermes"]
    writeback = enabled["rsimem"]["semantic_writeback"]
    assert writeback["mode"] == "adaptive_utility"
    assert writeback["adaptive_policy_store_path"] == (
        ".rsimem/adaptive-policies.json"
    )
    assert writeback["adaptive_trusted_roots"] == ["mem0-flat.static-policy-v1"]
    assert writeback["adaptive_parameters"] == _adaptive_config()[
        "adaptive_parameters"
    ]
    assert "schema_version" not in writeback
    assert "memory" not in enabled["enabled_toolsets"]

    disabled = build_hermes_extra_body(
        persistence_enabled=False,
        rsimem_adaptive_config=_adaptive_config(),
        **common,
    )["hermes"]["rsimem"]["semantic_writeback"]
    assert disabled == {
        "mode": "disabled",
        "timeout_seconds": 30.0,
        "max_output_tokens": 4096,
    }

    with pytest.raises(ValueError, match="requires adaptive config"):
        build_hermes_extra_body(persistence_enabled=True, **common)
    with pytest.raises(ValueError, match=r"native\+ledger"):
        build_hermes_extra_body(
            persistence_enabled=True,
            rsimem_adaptive_config=_adaptive_config(),
            **{**common, "rsimem_mode": "native+adapter+ledger"},
        )
    with pytest.raises(ValueError, match="requires lifecycle"):
        build_hermes_extra_body(
            persistence_enabled=True,
            rsimem_adaptive_config=_adaptive_config(),
            **{**common, "rsimem_lifecycle_evaluator_mode": "disabled"},
        )
    with pytest.raises(ValueError, match="requires adaptive_utility"):
        build_hermes_extra_body(
            persistence_enabled=True,
            rsimem_adaptive_config=_adaptive_config(),
            **{**common, "rsimem_semantic_writeback_mode": "static_utility"},
        )

    malformed = {**_adaptive_config(), "unknown": True}
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        RSIMemAdaptiveWritebackConfig.model_validate(malformed)
    escaped = {
        **_adaptive_config(),
        "adaptive_policy_store_path": "../adaptive-policies.json",
    }
    with pytest.raises(ValueError, match="relative to Hermes home"):
        RSIMemAdaptiveWritebackConfig.model_validate(escaped)


def test_sequence_validates_rsimem_execution_config(tmp_path: Path):
    manifest = tmp_path / "sequence.yaml"
    task_dir = tmp_path / "tasks" / "T_demo"
    task_dir.mkdir(parents=True)
    (task_dir / "task.yaml").write_text(
        "task_id: demo\ntask_name: Demo\nprompt:\n  text: hi\n",
        encoding="utf-8",
    )
    manifest.write_text(
        yaml.safe_dump({
            "name": "rsimem-mode",
            "hermes": {
                "rsimem_mode": "native+adapter+ledger",
                "rsimem_adapter_failure_policy": "bypass_native",
            },
            "episodes": [{"task": "tasks/T_demo", "cluster_id": "cluster-a"}],
        }),
        encoding="utf-8",
    )

    sequence = SelfEvolveSequenceDefinition.from_yaml(manifest)

    assert sequence.hermes.rsimem_mode == "native+adapter+ledger"
    assert sequence.hermes.rsimem_adapter_failure_policy == "bypass_native"

    invalid = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    invalid["hermes"]["rsimem_mode"] = "silent-adapter"
    manifest.write_text(yaml.safe_dump(invalid), encoding="utf-8")
    with pytest.raises(ValueError, match="rsimem_mode"):
        SelfEvolveSequenceDefinition.from_yaml(manifest)

    adaptive = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    adaptive["hermes"]["rsimem_mode"] = "native+ledger"
    adaptive["hermes"]["rsimem_semantic_writeback_mode"] = "adaptive_utility"
    adaptive["hermes"]["rsimem_adaptive_config"] = _adaptive_config()
    manifest.write_text(yaml.safe_dump(adaptive), encoding="utf-8")
    sequence = SelfEvolveSequenceDefinition.from_yaml(manifest)
    assert sequence.hermes.rsimem_adaptive_config is not None

    adaptive["hermes"].pop("rsimem_adaptive_config")
    manifest.write_text(yaml.safe_dump(adaptive), encoding="utf-8")
    with pytest.raises(ValueError, match="requires adaptive config"):
        SelfEvolveSequenceDefinition.from_yaml(manifest)

    adaptive["hermes"]["rsimem_semantic_writeback_mode"] = "static_utility"
    adaptive["hermes"]["rsimem_adaptive_config"] = _adaptive_config()
    manifest.write_text(yaml.safe_dump(adaptive), encoding="utf-8")
    with pytest.raises(ValueError, match="requires adaptive_utility"):
        SelfEvolveSequenceDefinition.from_yaml(manifest)


def test_cli_rsimem_override_is_explicit_and_hermes_only(tmp_path: Path) -> None:
    manifest = tmp_path / "sequence.yaml"
    task_dir = tmp_path / "tasks" / "T_demo"
    task_dir.mkdir(parents=True)
    (task_dir / "task.yaml").write_text(
        "task_id: demo\ntask_name: Demo\nprompt:\n  text: hi\n",
        encoding="utf-8",
    )
    manifest.write_text(
        yaml.safe_dump({
            "name": "rsimem-cli-mode",
            "episodes": [{"task": "tasks/T_demo", "cluster_id": "cluster-a"}],
        }),
        encoding="utf-8",
    )
    sequence = SelfEvolveSequenceDefinition.from_yaml(manifest)
    adaptive_config = tmp_path / "adaptive.json"
    adaptive_config.write_text(
        json.dumps(_adaptive_config()),
        encoding="utf-8",
    )

    _apply_rsimem_execution_overrides(sequence, SimpleNamespace(
        agent="hermes-luna",
        rsimem_mode="native+adapter+ledger",
        rsimem_adapter_failure_policy="fail_closed",
        rsimem_verify_native_projection=True,
        rsimem_lifecycle_evaluator_mode="deterministic",
        rsimem_lifecycle_policy_version="phase1-acceptance-v1",
        rsimem_lifecycle_compiler_version="uncompiled-v0",
        rsimem_lifecycle_timeout_seconds=15.0,
        rsimem_lifecycle_max_output_tokens=2048,
        rsimem_semantic_writeback_mode="adaptive_utility",
        rsimem_semantic_writeback_timeout_seconds=20.0,
        rsimem_semantic_writeback_max_output_tokens=1024,
        rsimem_adaptive_config=str(adaptive_config),
    ))

    assert sequence.hermes.rsimem_mode == "native+adapter+ledger"
    assert sequence.hermes.rsimem_adapter_failure_policy == "fail_closed"
    assert sequence.hermes.rsimem_verify_native_projection is True
    assert sequence.hermes.rsimem_lifecycle_evaluator_mode == "deterministic"
    assert sequence.hermes.rsimem_lifecycle_policy_version == "phase1-acceptance-v1"
    assert sequence.hermes.rsimem_lifecycle_compiler_version == "uncompiled-v0"
    assert sequence.hermes.rsimem_lifecycle_timeout_seconds == 15.0
    assert sequence.hermes.rsimem_lifecycle_max_output_tokens == 2048
    assert sequence.hermes.rsimem_semantic_writeback_mode == "adaptive_utility"
    assert sequence.hermes.rsimem_semantic_writeback_timeout_seconds == 20.0
    assert sequence.hermes.rsimem_semantic_writeback_max_output_tokens == 1024
    assert sequence.hermes.rsimem_adaptive_config is not None
    assert sequence.hermes.rsimem_adaptive_config.adaptive_policy_store_path == (
        ".rsimem/adaptive-policies.json"
    )

    with pytest.raises(SystemExit, match="Hermes agent"):
        _apply_rsimem_execution_overrides(sequence, SimpleNamespace(
            agent="nanobot",
            rsimem_mode="native+ledger",
            rsimem_adapter_failure_policy=None,
        ))


def test_cli_adaptive_override_rejects_missing_or_mismatched_config(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "sequence.yaml"
    task_dir = tmp_path / "tasks" / "T_demo"
    task_dir.mkdir(parents=True)
    (task_dir / "task.yaml").write_text(
        "task_id: demo\ntask_name: Demo\nprompt:\n  text: hi\n",
        encoding="utf-8",
    )
    manifest.write_text(
        yaml.safe_dump({
            "name": "rsimem-adaptive-cli",
            "episodes": [{"task": "tasks/T_demo", "cluster_id": "cluster-a"}],
        }),
        encoding="utf-8",
    )
    sequence = SelfEvolveSequenceDefinition.from_yaml(manifest)
    with pytest.raises(SystemExit, match="requires adaptive config"):
        _apply_rsimem_execution_overrides(sequence, SimpleNamespace(
            agent="hermes-luna",
            rsimem_semantic_writeback_mode="adaptive_utility",
        ))

    config_path = tmp_path / "adaptive.json"
    config_path.write_text(json.dumps(_adaptive_config()), encoding="utf-8")
    sequence = SelfEvolveSequenceDefinition.from_yaml(manifest)
    with pytest.raises(SystemExit, match="requires adaptive_utility"):
        _apply_rsimem_execution_overrides(sequence, SimpleNamespace(
            agent="hermes-luna",
            rsimem_semantic_writeback_mode="static_utility",
            rsimem_adaptive_config=str(config_path),
        ))

    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text('{"schema_version": 1}', encoding="utf-8")
    sequence = SelfEvolveSequenceDefinition.from_yaml(manifest)
    with pytest.raises(SystemExit, match="invalid RSIMem adaptive config"):
        _apply_rsimem_execution_overrides(sequence, SimpleNamespace(
            agent="hermes-luna",
            rsimem_semantic_writeback_mode="adaptive_utility",
            rsimem_adaptive_config=str(invalid_path),
        ))


def test_hermes_adapter_activates_and_closes_opt_in_rsimem_bridge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import rsimem.hermes_past_bridge as bridge_module

    captured = {}

    class Bridge:
        def __init__(self, home, config, **kwargs):
            captured.update(home=home, config=config, kwargs=kwargs)
            self.closed = False

        def attach(self, agent):
            captured["agent"] = agent

        def close(self):
            self.closed = True
            captured["closed"] = True

    monkeypatch.setattr(bridge_module, "HermesPastBenchBridge", Bridge)
    request = _rsimem_adapter_request(tmp_path, {
        "mode": "native+adapter+ledger",
        "adapter_failure_policy": "bypass_native",
        "evidence_path": str(tmp_path / "artifacts" / "events.jsonl"),
    })
    adapter = HermesAdapter(AgentSpec(name="hermes", adapter="hermes"), request)
    agent = object()

    adapter._activate_rsimem_bridge(
        agent,
        request.model.extra_body["hermes"],
        tmp_path / "home",
    )
    adapter.close("test")

    assert captured["home"] == tmp_path / "home"
    assert captured["config"].mode.value == "native+adapter+ledger"
    assert captured["config"].adapter_failure_policy.value == "bypass_native"
    assert captured["kwargs"]["evidence_path"] == (
        tmp_path / "artifacts" / "events.jsonl"
    )
    assert captured["kwargs"]["run_id"] == "run-rsimem"
    assert captured["kwargs"]["family_id"] == "SM01"
    assert captured["kwargs"]["experiment_variant"] == "with_persistence"
    assert captured["agent"] is agent
    assert captured["closed"] is True


def test_hermes_adapter_parses_adaptive_writeback_transport(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import rsimem.hermes_past_bridge as bridge_module

    captured = {}

    class Bridge:
        def __init__(self, home, config, **kwargs):
            captured.update(home=home, config=config, kwargs=kwargs)

        def attach(self, agent):
            captured["agent"] = agent

        def close(self):
            return None

    monkeypatch.setattr(bridge_module, "HermesPastBenchBridge", Bridge)
    request = _rsimem_adapter_request(tmp_path, {
        "mode": "native+ledger",
        "evidence_path": str(tmp_path / "artifacts" / "events.jsonl"),
        "lifecycle": {
            "evaluator_mode": "deterministic",
            "policy_version": "adaptive-fixture-v1",
            "compiler_version": "uncompiled-v0",
        },
        "semantic_writeback": {
            "mode": "adaptive_utility",
            "timeout_seconds": 30.0,
            "max_output_tokens": 4096,
            **{
                key: value
                for key, value in _adaptive_config().items()
                if key != "schema_version"
            },
        },
    })
    adapter = HermesAdapter(AgentSpec(name="hermes", adapter="hermes"), request)
    try:
        agent = object()
        adapter._activate_rsimem_bridge(
            agent,
            request.model.extra_body["hermes"],
            tmp_path / "home",
        )
    finally:
        adapter.close("test")

    writeback = captured["kwargs"]["static_writeback_config"]
    assert writeback.adaptive_enabled is True
    assert writeback.adaptive_policy_store_path == (
        ".rsimem/adaptive-policies.json"
    )
    assert writeback.adaptive_trusted_roots == ("mem0-flat.static-policy-v1",)
    assert writeback.adaptive_parameters[0].parameter_id == "parameter.retrieval"
    assert captured["agent"] is agent


def test_hermes_adapter_keeps_native_default_and_rejects_evidence_escape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import rsimem.hermes_past_bridge as bridge_module

    monkeypatch.setattr(
        bridge_module,
        "HermesPastBenchBridge",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("native mode must not construct the bridge")
        ),
    )
    native_request = _rsimem_adapter_request(tmp_path, {"mode": "native"})
    native = HermesAdapter(
        AgentSpec(name="hermes", adapter="hermes"),
        native_request,
    )
    try:
        native._activate_rsimem_bridge(
            object(),
            native_request.model.extra_body["hermes"],
            tmp_path / "home",
        )
    finally:
        native.close("test")

    escaped_request = _rsimem_adapter_request(tmp_path, {
        "mode": "native+ledger",
        "evidence_path": str(tmp_path / "outside.jsonl"),
    })
    escaped = HermesAdapter(
        AgentSpec(name="hermes", adapter="hermes"),
        escaped_request,
    )
    try:
        with pytest.raises(ValueError, match="evidence_path"):
            escaped._activate_rsimem_bridge(
                object(),
                escaped_request.model.extra_body["hermes"],
                tmp_path / "home",
            )
    finally:
        escaped.close("test")


def test_injected_lifecycle_uses_recorded_hermes_model_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import rsimem.hermes_past_bridge as bridge_module
    from agent import auxiliary_client

    captured: dict[str, object] = {}

    class Bridge:
        def __init__(self, home, config, **kwargs):
            captured["complete"] = kwargs["lifecycle_complete"]

        def attach(self, agent):
            return None

        def close(self):
            return None

    class Agent:
        def _execute_recorded_model_call(self, request, **kwargs):
            captured["recorded"] = kwargs
            return request()

    def call_llm(**kwargs):
        captured["call"] = kwargs
        response = SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content='{"signals": []}'),
            )],
        )
        return kwargs["request_executor"](
            lambda: response,
            attempt=1,
            purpose="rsimem_lifecycle",
            provider="custom",
            model="fixture-model",
            api_mode="chat_completions",
        )

    monkeypatch.setattr(bridge_module, "HermesPastBenchBridge", Bridge)
    monkeypatch.setattr(auxiliary_client, "call_llm", call_llm)
    request = _rsimem_adapter_request(tmp_path, {
        "mode": "native+adapter+ledger",
        "evidence_path": str(tmp_path / "artifacts" / "events.jsonl"),
        "lifecycle": {
            "evaluator_mode": "injected_json",
            "policy_version": "host-policy-v2",
            "compiler_version": "uncompiled-v0",
            "timeout_seconds": 12.5,
            "max_output_tokens": 2048,
        },
    })
    request.model.api_key = "fixture-key"
    request.model.base_url = "https://fixture.invalid/v1"
    adapter = HermesAdapter(AgentSpec(name="hermes", adapter="hermes"), request)
    try:
        adapter._activate_rsimem_bridge(
            Agent(),
            request.model.extra_body["hermes"],
            tmp_path / "home",
        )
        complete = captured["complete"]
        assert callable(complete)
        assert complete("fixture prompt") == '{"signals": []}'
    finally:
        adapter.close("test")

    call = captured["call"]
    assert call["task"] == "rsimem_lifecycle"
    assert call["timeout"] == 12.5
    assert call["max_tokens"] == 2048
    assert call["base_url"] == "https://fixture.invalid/v1"
    assert call["api_key"] == "fixture-key"
    assert captured["recorded"]["component"] == "lifecycle_evaluator"
    assert captured["recorded"]["purpose"] == "rsimem_lifecycle"


def test_static_completion_uses_hermes_accounting_and_raw_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent import auxiliary_client
    from rsimem.memory_systems.mem0_flat import POLICY_FACT_EXTRACTION_PROMPT

    captured: dict[str, object] = {}

    class Agent:
        def __init__(self):
            self.model_call_usage_records = []

        def _execute_recorded_model_call(self, request, **kwargs):
            captured["recorded"] = kwargs
            response = request()
            self.model_call_usage_records.append({
                "duration_ms": 12.4,
                "usage": {
                    "input_tokens": 90,
                    "output_tokens": 15,
                    "cache_read_tokens": 30,
                    "cache_write_tokens": 2,
                    "reasoning_tokens": 4,
                    "retry_count": 1,
                },
            })
            return response

    def call_llm(**kwargs):
        captured["call"] = kwargs
        response = SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content='{"facts": ["fixture fact"]}'),
        )])
        return kwargs["request_executor"](
            lambda: response,
            attempt=2,
            purpose=kwargs["task"],
            provider="custom",
            model="fixture-model",
            api_mode="chat_completions",
        )

    monkeypatch.setattr(auxiliary_client, "call_llm", call_llm)
    agent = Agent()
    client = _RecordedHermesCompletionClient(
        agent,
        model="fixture-model",
        base_url="https://fixture.invalid/v1",
        api_key="fixture-key",
        timeout_seconds=8.0,
        max_output_tokens=512,
    )
    prompt = POLICY_FACT_EXTRACTION_PROMPT.render({
        "source_messages": [],
        "exit_evidence": {},
    })
    result = client.complete(prompt)

    assert result.output_text == '{"facts": ["fixture fact"]}'
    assert result.usage.to_dict() == {
        "schema_version": 1,
        "input_tokens": 90,
        "output_tokens": 15,
        "cache_read_tokens": 30,
        "cache_write_tokens": 2,
        "reasoning_tokens": 4,
        "model_requests": 1,
        "retry_count": 1,
        "duration_ms": 12,
        "storage_bytes": 0,
    }
    assert captured["call"]["task"] == "semantic_fact_extraction"
    assert captured["call"]["timeout"] == 8.0
    assert captured["call"]["max_tokens"] == 512
    assert captured["recorded"]["component"] == "semantic_fact_extraction"

    agent.model_call_usage_records.clear()
    monkeypatch.setattr(
        auxiliary_client,
        "call_llm",
        lambda **kwargs: SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content='{"facts": []}'),
        )]),
    )
    with pytest.raises(ValueError, match="bypassed request accounting"):
        client.complete(prompt)


def test_resolve_episode_tool_config_isolates_expected_mechanism():
    skill = resolve_episode_tool_config(
        persistence_enabled=True,
        expected_signal="skill",
        memory_enabled=True,
        user_profile_enabled=True,
        skills_enabled=True,
        session_search_enabled=True,
    )
    memory = resolve_episode_tool_config(
        persistence_enabled=True,
        expected_signal="memory",
        memory_enabled=True,
        user_profile_enabled=True,
        skills_enabled=True,
        session_search_enabled=True,
    )
    recall = resolve_episode_tool_config(
        persistence_enabled=True,
        expected_signal="session_search",
        memory_enabled=True,
        user_profile_enabled=True,
        skills_enabled=True,
        session_search_enabled=True,
    )
    disabled = resolve_episode_tool_config(
        persistence_enabled=False,
        expected_signal="mixed",
        memory_enabled=True,
        user_profile_enabled=True,
        skills_enabled=True,
        session_search_enabled=True,
    )

    assert skill == {
        "memory_enabled": False,
        "user_profile_enabled": False,
        "skills_enabled": True,
        "session_search_enabled": False,
    }
    assert memory == {
        "memory_enabled": True,
        "user_profile_enabled": True,
        "skills_enabled": False,
        "session_search_enabled": False,
    }
    assert recall == {
        "memory_enabled": False,
        "user_profile_enabled": False,
        "skills_enabled": False,
        "session_search_enabled": True,
    }
    assert disabled == {
        "memory_enabled": False,
        "user_profile_enabled": False,
        "skills_enabled": False,
        "session_search_enabled": False,
    }


def test_self_evolve_episode_v2_metadata_backfills(tmp_path: Path):
    manifest = tmp_path / "sequence.yaml"
    task_dir = tmp_path / "tasks" / "T_demo"
    task_dir.mkdir(parents=True)
    (task_dir / "task.yaml").write_text("task_id: demo\ntask_name: Demo\nprompt:\n  text: hi\n", encoding="utf-8")
    manifest.write_text(
        yaml.safe_dump(
            {
                "name": "demo-seq",
                "episodes": [
                    {
                        "task": "tasks/T_demo",
                        "family_id": "F01",
                        "mechanism": "skill",
                        "bucket": "learn",
                        "noise_profile": "stale_conflict",
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    sequence = SelfEvolveSequenceDefinition.from_yaml(manifest)
    episode = sequence.episodes[0]

    assert episode.expected_persistence_signal == "skill"
    assert episode.transfer_distance == "none"
    assert episode.reflection_required is True
    assert episode.conflict_mode == "stale_conflict"
    assert episode.noise_level == "heavy"


def test_resolve_effective_final_text_recovers_pre_reply_content(tmp_path: Path):
    session_file = tmp_path / "session_latest.json"
    session_file.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "assistant", "content": "Maya Ortiz | RISK-CANON | close the weekly risk digest | 2026-06-23"},
                    {"role": "assistant", "content": "[[reply_to_current]]"},
                ]
            }
        ),
        encoding="utf-8",
    )

    messages = [
        TraceMessage(
            trace_id="trace-inline",
            message=Message(role="assistant", content="[[reply_to_current]]"),
        )
    ]

    assert _resolve_effective_final_text(
        messages,
        {"internal_tools": {"session_file": str(session_file)}},
    ) == "Maya Ortiz | RISK-CANON | close the weekly risk digest | 2026-06-23"


def test_single_task_manifest_parses(tmp_path: Path):
    task_dir = tmp_path / "tasks" / "T_demo"
    task_dir.mkdir(parents=True)
    (task_dir / "task.yaml").write_text("task_id: demo\ntask_name: Demo\nprompt:\n  text: hi\n", encoding="utf-8")
    transfer_dir = tmp_path / "tasks" / "T_transfer"
    transfer_dir.mkdir(parents=True)
    (transfer_dir / "task.yaml").write_text("task_id: transfer\ntask_name: Transfer\nprompt:\n  text: hi\n", encoding="utf-8")
    manifest = tmp_path / "single.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "name": "single-demo",
                "mode": "single_task",
                "single_task": {
                    "candidates": [{"task": "tasks/T_demo"}],
                    "transfer_tasks": [{"task": "tasks/T_transfer"}],
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    sequence = SelfEvolveSequenceDefinition.from_yaml(manifest)
    assert sequence.mode == "single_task"
    assert sequence.single_task is not None
    assert sequence.resolve_task_yaml(sequence.single_task.candidates[0].task) == task_dir / "task.yaml"


def test_hermes_adapter_prepares_home_and_captures_artifacts(tmp_path: Path):
    home_dir = tmp_path / "home"
    capture_dir = tmp_path / "capture"
    fixture_dir = tmp_path / "initial_home_fixture"
    fixture_memories = fixture_dir / "memories"
    fixture_memories.mkdir(parents=True)
    (fixture_memories / "MEMORY.md").write_text("native home rule", encoding="utf-8")
    fixture_sessions = fixture_dir / "sessions"
    fixture_sessions.mkdir(parents=True)
    (fixture_sessions / "session_20260329_000000_fixture.json").write_text("{}", encoding="utf-8")
    (fixture_dir / "state.db").write_text("sqlite placeholder", encoding="utf-8")
    preseed_dir = tmp_path / "preseed"
    preseed_skills = preseed_dir / "skills" / "seeded-skill"
    preseed_skills.mkdir(parents=True)
    (preseed_skills / "SKILL.md").write_text("---\nname: seeded-skill\n---\nUse seeded.\n", encoding="utf-8")

    prepared = HermesAdapter._prepare_hermes_home(
        {
            "home_dir": str(home_dir),
            "config_overrides": {
                "memory": {"memory_enabled": True, "nudge_interval": 1},
                "skills": {"creation_nudge_interval": 1},
            },
            "initial_home_fixture_dir": str(fixture_dir),
            "preseed_artifacts_dir": str(preseed_dir),
        }
    )

    assert prepared == home_dir
    config = yaml.safe_load((home_dir / "config.yaml").read_text(encoding="utf-8"))
    assert config["memory"]["memory_enabled"] is True
    assert config["skills"]["creation_nudge_interval"] == 1
    assert (home_dir / "memories" / "MEMORY.md").read_text(encoding="utf-8") == "native home rule"
    assert (home_dir / "skills" / "seeded-skill" / "SKILL.md").exists()
    assert (home_dir / "sessions" / "session_20260329_000000_fixture.json").exists()
    assert (home_dir / "state.db").exists()

    memories_dir = home_dir / "memories"
    memories_dir.mkdir(parents=True, exist_ok=True)
    (memories_dir / "MEMORY.md").write_text("learned rule", encoding="utf-8")
    skills_dir = home_dir / "skills" / "demo-skill"
    skills_dir.mkdir(parents=True)
    (skills_dir / "SKILL.md").write_text("---\nname: demo-skill\ndescription: demo\n---\nUse demo.\n", encoding="utf-8")
    sessions_dir = home_dir / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    (sessions_dir / "session_20260330_000000_demo.json").write_text(
        json.dumps(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "timestamp": "2026-03-30T00:00:01Z",
                        "tool_calls": [
                            {"function": {"name": "session_search"}},
                            {"function": {"name": "skill_manage", "arguments": "{\"action\":\"patch\"}"}},
                            {"function": {"name": "memory", "arguments": "{\"action\":\"add\"}"}},
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    current_session = sessions_dir / "session_20260330_000000_demo.json"
    HermesAdapter._capture_hermes_artifacts(
        {"capture_artifacts_dir": str(capture_dir)},
        home_dir,
        session_log_file=current_session,
    )

    assert (capture_dir / "config.yaml").exists()
    assert (capture_dir / "memories" / "MEMORY.md").read_text(encoding="utf-8") == "learned rule"
    assert (capture_dir / "skills" / "demo-skill" / "SKILL.md").exists()
    assert (capture_dir / "session_current.json").exists()
    assert (capture_dir / "session_latest.json").exists()

    summary = snapshot_hermes_artifacts(capture_dir)
    assert summary["internal_tools"]["session_search_calls"] == 1
    assert summary["internal_tools"]["skill_manage_calls"] == 1
    assert summary["internal_tools"]["skill_update_count"] == 1
    assert summary["internal_tools"]["memory_write_count"] == 1


def test_save_episode_history_anchor_registers_anchor_mapping(tmp_path: Path):
    hermes_home = tmp_path / "hermes_home"
    hermes_home.mkdir(parents=True)
    (hermes_home / "memories").mkdir()
    (hermes_home / "memories" / "MEMORY.md").write_text("seeded", encoding="utf-8")
    anchors_dir = tmp_path / "anchors"
    history_anchors: dict[str, Path] = {}
    episode = SelfEvolveEpisode(
        task="tasks/T_demo",
        family_id="EP02_exception_list_recall",
        mechanism="session_search",
        bucket="learn",
        history_save_anchor="ep02_post_learn",
        history_mode="continue",
    )

    _save_episode_history_anchor(
        hermes_home=hermes_home,
        episode=episode,
        anchors_dir=anchors_dir,
        history_anchors=history_anchors,
    )

    assert history_anchors["ep02_post_learn"] == anchors_dir / "ep02_post_learn"
    assert (history_anchors["ep02_post_learn"] / "memories" / "MEMORY.md").read_text(encoding="utf-8") == "seeded"


def test_artifact_diff_and_mechanism_scores_use_rule_keywords(tmp_path: Path):
    home_before = tmp_path / "before"
    home_after = tmp_path / "after"
    (home_before / "memories").mkdir(parents=True)
    (home_after / "memories").mkdir(parents=True)
    (home_before / "skills" / "triage-sop").mkdir(parents=True)
    (home_after / "skills" / "triage-sop").mkdir(parents=True)

    (home_before / "memories" / "MEMORY.md").write_text("old rule", encoding="utf-8")
    (home_after / "memories" / "MEMORY.md").write_text(
        "rule=shared root cause\n§\ntrigger=urgent incident\ndo not close",
        encoding="utf-8",
    )
    (home_before / "skills" / "triage-sop" / "SKILL.md").write_text(
        "---\nname: triage-sop\ndescription: demo\n---\nUse v1.\n",
        encoding="utf-8",
    )
    (home_after / "skills" / "triage-sop" / "SKILL.md").write_text(
        "---\nname: triage-sop\ndescription: demo\n---\nUse shared root cause incident, do not close.\n",
        encoding="utf-8",
    )

    before = snapshot_hermes_home(home_before, include_contents=True)
    after = snapshot_hermes_home(home_after, include_contents=True)
    after["internal_tools"]["skill_update_count"] = 1

    diff = diff_artifact_snapshots(
        before=before,
        after=after,
        rule_keywords=["shared root cause", "urgent", "do not close"],
    )
    retrieval = compute_retrieval_signals(
        dispatches=[],
        artifact_before=before,
        internal_tools=after["internal_tools"],
        expected_signal="memory",
    )
    scores = compute_self_evolve_mechanism_scores(
        expectations={
            "expected_mechanism": "memory",
            "artifact_contract": {
                "type": "memory",
                "require_rule_keywords": ["shared root cause", "urgent", "do not close"],
                "min_count_delta": 1,
            },
            "retrieval_contract": {"min_memory_injections": 1},
        },
        artifact_before=before,
        artifact_after=after,
        artifact_diff=diff,
        retrieval_signals=retrieval,
    )

    assert retrieval["memory_read_count"] == 0
    assert retrieval["memory_injection_count"] == 1
    assert "shared root cause" in diff["rule_keyword_hits"]["hit_keywords"]
    assert diff["updated_rules"]
    assert scores["artifact_quality_score"] > 0.5
    assert scores["mechanism_confidence"] > 0.5


def test_memory_injection_counts_as_reuse_signal_for_memory_tasks():
    memory_signals = compute_retrieval_signals(
        dispatches=[],
        artifact_before={"memory_file_exists": True, "user_file_exists": False},
        internal_tools={
            "memory_read_count": 0,
            "skill_read_count": 0,
            "session_search_calls": 0,
            "calls": [],
        },
        expected_signal="memory",
    )
    mixed_signals = compute_retrieval_signals(
        dispatches=[],
        artifact_before={"memory_file_exists": True, "user_file_exists": False},
        internal_tools={
            "memory_read_count": 0,
            "skill_read_count": 0,
            "session_search_calls": 0,
            "calls": [],
        },
        expected_signal="mixed",
    )

    assert memory_signals["memory_injection_count"] == 1
    assert memory_signals["used_expected_signal"] is True
    assert memory_signals["retrieval_before_first_update"] is True
    assert memory_signals["first_retrieval_at"] is None

    assert mixed_signals["memory_injection_count"] == 1
    assert mixed_signals["retrieval_signal_count"] == 1
    assert mixed_signals["used_expected_signal"] is True
    assert mixed_signals["retrieval_before_first_update"] is True


def test_mechanism_scores_preserve_retrieval_before_update_signal():
    scores = compute_self_evolve_mechanism_scores(
        expectations={
            "expected_mechanism": "session_search",
            "artifact_contract": {"type": "session", "min_count_delta": 0},
            "retrieval_contract": {"min_session_search_calls": 1},
        },
        artifact_before={},
        artifact_after={"internal_tools": {}},
        artifact_diff={"rule_keyword_hits": {}, "updated_rules": [], "changed_skill_names": [], "added_rules": []},
        retrieval_signals={
            "memory_read_count": 0,
            "memory_injection_count": 0,
            "skill_read_count": 0,
            "session_search_count": 1,
            "retrieval_signal_count": 1,
            "used_expected_signal": True,
            "retrieval_before_first_update": True,
            "retrieval_before_final_response": True,
        },
    )

    assert scores["retrieval_before_first_update"] is True
    assert scores["retrieval_before_final_response"] is True


def test_mechanism_scores_do_not_award_artifact_presence_without_contract_hits():
    scores = compute_self_evolve_mechanism_scores(
        expectations={
            "expected_mechanism": "mixed",
            "artifact_contract": {
                "type": "mixed",
                "require_rule_keywords": ["missing rule"],
                "min_count_delta": 0,
            },
            "retrieval_contract": {"evaluation_only": True},
        },
        artifact_before={"memory_file_exists": True, "user_file_exists": False, "skill_count": 0},
        artifact_after={
            "memory_file_exists": True,
            "user_file_exists": False,
            "skill_count": 0,
            "internal_tools": {
                "memory_write_count": 0,
                "skill_create_count": 0,
                "skill_update_count": 0,
            },
        },
        artifact_diff={
            "memory_entry_delta": 0,
            "skill_count_delta": 0,
            "rule_keyword_hits": {
                "required_keywords": ["missing rule"],
                "hit_keywords": [],
                "hit_count": 0,
                "hit_rate": 0.0,
            },
            "updated_rules": [],
            "changed_skill_names": [],
            "added_rules": [],
        },
        retrieval_signals={
            "memory_read_count": 0,
            "memory_injection_count": 0,
            "skill_read_count": 0,
            "session_search_count": 0,
            "retrieval_signal_count": 0,
            "used_expected_signal": False,
            "retrieval_before_first_update": False,
        },
    )

    assert scores["artifact_quality_score"] == 0.0
    assert scores["mechanism_confidence"] == 0.0


def test_build_hermes_prompt_uses_native_tool_call_instructions():
    prompt = _build_hermes_prompt(
        StartSessionRequest(
            session_id="sess-1",
            agent_name="hermes",
            task_id="T_demo",
            task_name="Demo Task",
            max_turns=4,
            timeout_seconds=60,
            initial_messages=[
                Message(role="system", content=[TextBlock(text="System text")]),
                Message(role="user", content=[TextBlock(text="User text")]),
            ],
            model=RuntimeModelConfig(model_id="demo-model"),
            runtime_config=RuntimeConfigPayload(),
        )
    )

    assert "You MUST use tool calls to actually perform actions." in prompt
    assert "call them yourself from Bash" not in prompt


def test_hermes_adapter_registers_past_bench_http_tools(monkeypatch):
    captured = {}

    class _Response:
        status_code = 200
        text = '{"ok": true}'

        @staticmethod
        def json():
            return {"ok": True}

    def fake_request(method, url, **kwargs):
        captured["method"] = method
        captured["url"] = url
        captured["kwargs"] = kwargs
        return _Response()

    monkeypatch.setattr("httpx.request", fake_request)

    request = StartSessionRequest(
        session_id="sess-1",
        agent_name="hermes",
        task_id="T_demo",
        task_name="Demo Task",
        max_turns=4,
        timeout_seconds=60,
        initial_messages=[
            Message(role="system", content=[TextBlock(text="System text")]),
            Message(role="user", content=[TextBlock(text="User text")]),
        ],
        tools=[
            ToolSpec(
                name="sandbox_shell_exec",
                description="Run shell commands in sandbox.",
                input_schema={"type": "object", "properties": {"command": {"type": "string"}}},
            )
        ],
        tool_endpoints=[
            ToolEndpoint(
                tool_name="sandbox_shell_exec",
                url="http://host.docker.internal:18080/exec",
                method="POST",
            )
        ],
        model=RuntimeModelConfig(model_id="demo-model"),
        runtime_config=RuntimeConfigPayload(),
    )

    adapter = HermesAdapter(AgentSpec(name="hermes", adapter="hermes"), request)
    adapter._register_past_bench_tools()

    import sys

    hermes_root = str(Path(__file__).resolve().parents[1] / "agents" / "hermes-agent")
    if hermes_root not in sys.path:
        sys.path.insert(0, hermes_root)
    from tools.registry import registry

    payload = registry.dispatch("sandbox_shell_exec", {"command": "pwd"})
    decoded = json.loads(payload)

    assert registry.get_toolset_for_tool("sandbox_shell_exec") == "past_bench_runtime"
    assert decoded == {"ok": True}
    assert captured["method"] == "POST"
    assert captured["url"] == "http://host.docker.internal:18080/exec"
    assert captured["kwargs"]["json"] == {"command": "pwd"}


def test_hermes_adapter_overrides_conflicting_tool_registration_and_restores_after_close(monkeypatch):
    request = StartSessionRequest(
        session_id="sess-1",
        agent_name="hermes",
        task_id="T_demo",
        task_name="Demo Task",
        max_turns=4,
        timeout_seconds=60,
        initial_messages=[],
        tools=[
            ToolSpec(
                name="config_list_integrations",
                description="List integrations.",
                input_schema={"type": "object", "properties": {}},
            )
        ],
        tool_endpoints=[
            ToolEndpoint(
                tool_name="config_list_integrations",
                url="http://localhost:9210/config/integrations",
                method="POST",
            )
        ],
        model=RuntimeModelConfig(model_id="demo-model"),
        runtime_config=RuntimeConfigPayload(),
    )

    adapter = HermesAdapter(AgentSpec(name="hermes", adapter="hermes"), request)

    import sys

    hermes_root = str(Path(__file__).resolve().parents[1] / "agents" / "hermes-agent")
    if hermes_root not in sys.path:
        sys.path.insert(0, hermes_root)
    from tools.registry import registry

    original_entry = registry._tools.get("config_list_integrations", _MISSING_TOOL)

    registry.register(
        name="config_list_integrations",
        toolset="preexisting_toolset",
        schema={"name": "config_list_integrations", "parameters": {"type": "object"}},
        handler=lambda args, **kwargs: json.dumps({"source": "preexisting"}),
        check_fn=lambda: True,
    )

    captured = {}

    class _Response:
        status_code = 200

        def json(self):
            return {"source": "past_bench"}

    def fake_request(method, url, **kwargs):
        captured["method"] = method
        captured["url"] = url
        captured["kwargs"] = kwargs
        return _Response()

    monkeypatch.setattr("httpx.request", fake_request)

    try:
        adapter._register_past_bench_tools()
        payload = registry.dispatch("config_list_integrations", {})
        decoded = json.loads(payload)

        assert registry.get_toolset_for_tool("config_list_integrations") == "past_bench_runtime"
        assert decoded == {"source": "past_bench"}
        assert captured["url"] == "http://localhost:9210/config/integrations"
    finally:
        adapter.close("test")
        if original_entry is _MISSING_TOOL:
            registry._tools.pop("config_list_integrations", None)
        else:
            registry._tools["config_list_integrations"] = original_entry

    restored = registry.get_toolset_for_tool("config_list_integrations")
    if original_entry is _MISSING_TOOL:
        assert restored is None
    else:
        assert restored == original_entry.toolset


def test_summarize_sequence_computes_family_deltas():
    summary = summarize_sequence(
        sequence_name="demo",
        variant="with_persistence",
        episodes=[
            {
                "family_id": "F01",
                "mechanism": "skill",
                "bucket": "baseline",
                "task_score": 0.2,
                "passed": False,
                "tool_dispatch_count": 6,
                "token_usage": {"total_tokens": 100},
                "timing": {"wall_time_s": 10.0},
                "artifacts": {
                    "memory_file_exists": False,
                    "user_file_exists": False,
                    "skill_count": 0,
                },
                "internal_tools": {"memory_calls": 0, "skill_manage_calls": 0, "session_search_calls": 0},
            },
            {
                "family_id": "F01",
                "mechanism": "skill",
                "bucket": "evaluation",
                "task_score": 0.8,
                "passed": True,
                "tool_dispatch_count": 3,
                "token_usage": {"total_tokens": 60},
                "timing": {"wall_time_s": 6.0},
                "artifacts": {
                    "memory_file_exists": False,
                    "user_file_exists": False,
                    "skill_count": 1,
                },
                "internal_tools": {"memory_calls": 0, "skill_manage_calls": 1, "session_search_calls": 0},
            },
        ],
    )

    assert summary["family_summary"]["F01"]["bucket_summary"]["baseline"]["avg_task_score"] == 0.2
    assert summary["family_summary"]["F01"]["bucket_summary"]["evaluation"]["avg_task_score"] == 0.8
    assert summary["family_summary"]["F01"]["improvement"]["task_score_delta_eval_minus_baseline"] == 0.6
    assert summary["benchmark_signal"]["families_with_skill_artifacts"] == 1


def test_choose_calibration_candidate_prefers_target_pass_count():
    selected = choose_calibration_candidate(
        target_pass_count=1,
        score_min=0.45,
        score_max=0.82,
        candidates=[
            {"label": "too_easy", "baseline_summary": {"pass_count": 3, "avg_task_score": 0.93}},
            {"label": "target", "baseline_summary": {"pass_count": 1, "avg_task_score": 0.62}},
            {"label": "too_hard", "baseline_summary": {"pass_count": 0, "avg_task_score": 0.31}},
        ],
    )
    assert selected["label"] == "target"


def test_summarize_single_task_sequence_computes_retention_and_transfer():
    summary = summarize_single_task_sequence(
        sequence_name="single-demo",
        variant="with_persistence",
        selected_candidate={"label": "candidate_primary", "baseline_summary": {"pass_count": 1, "avg_task_score": 0.55}},
        episodes=[
            {
                "bucket": "baseline",
                "episode_kind": "attempt",
                "task_score": 0.5,
                "passed": False,
                "tool_dispatch_count": 1,
                "token_usage": {"total_tokens": 10},
                "timing": {"wall_time_s": 1.0},
                "artifacts": {"memory_file_exists": False, "user_file_exists": False, "skill_count": 0},
                "internal_tools": {"memory_calls": 0, "skill_manage_calls": 0, "session_search_calls": 0},
            },
            {
                "bucket": "retention",
                "episode_kind": "attempt",
                "task_score": 0.9,
                "passed": True,
                "tool_dispatch_count": 1,
                "token_usage": {"total_tokens": 10},
                "timing": {"wall_time_s": 1.0},
                "artifacts": {"memory_file_exists": False, "user_file_exists": False, "skill_count": 0},
                "internal_tools": {"memory_calls": 0, "skill_manage_calls": 0, "session_search_calls": 0},
            },
            {
                "bucket": "transfer",
                "episode_kind": "attempt",
                "task_score": 0.8,
                "passed": True,
                "tool_dispatch_count": 1,
                "token_usage": {"total_tokens": 10},
                "timing": {"wall_time_s": 1.0},
                "artifacts": {"memory_file_exists": False, "user_file_exists": False, "skill_count": 0},
                "internal_tools": {"memory_calls": 0, "skill_manage_calls": 0, "session_search_calls": 0},
            },
            {
                "bucket": "reflection",
                "episode_kind": "reflection",
                "task_score": 0.0,
                "passed": False,
                "tool_dispatch_count": 0,
                "token_usage": {"total_tokens": 10},
                "timing": {"wall_time_s": 1.0},
                "artifacts": {"memory_file_exists": True, "user_file_exists": False, "skill_count": 1},
                "internal_tools": {"memory_calls": 1, "skill_manage_calls": 1, "session_search_calls": 0},
            },
        ],
    )
    assert summary["benchmark_signal"]["retention_score_delta"] == 0.4
    assert summary["benchmark_signal"]["transfer_score_delta_vs_baseline"] == 0.3
    assert summary["reflection_summary"]["memory_calls"] == 1


def test_summarize_sequence_exposes_memory_trigger_family_metrics():
    summary = summarize_sequence(
        sequence_name="memory-trigger-demo",
        variant="with_persistence",
        episodes=[
            {
                "family_id": "F06_stale_conflict_update",
                "mechanism": "memory",
                "bucket": "baseline",
                "stage": "cold",
                "expected_persistence_signal": "memory",
                "task_score": 0.4,
                "passed": False,
                "tool_dispatch_count": 1,
                "token_usage": {"total_tokens": 10},
                "timing": {"wall_time_s": 1.0},
                "artifacts": {"memory_file_exists": False, "user_file_exists": False, "skill_count": 0},
                "internal_tools": {"memory_calls": 0, "skill_manage_calls": 0, "session_search_calls": 0},
                "retrieval_signals": {"used_expected_signal": False, "retrieval_before_first_update": False},
                "mechanism_scores": {"artifact_quality_score": 0.0, "mechanism_confidence": 0.0, "transfer_quality": 0.0, "shortcut_resistance": 0.5},
            },
            {
                "family_id": "F06_stale_conflict_update",
                "mechanism": "memory",
                "bucket": "learn",
                "stage": "learn_a",
                "expected_persistence_signal": "memory",
                "task_score": 0.6,
                "passed": False,
                "tool_dispatch_count": 1,
                "token_usage": {"total_tokens": 10},
                "timing": {"wall_time_s": 1.0},
                "artifacts": {"memory_file_exists": True, "user_file_exists": False, "skill_count": 0},
                "internal_tools": {"memory_calls": 1, "skill_manage_calls": 0, "session_search_calls": 0},
                "retrieval_signals": {"used_expected_signal": False, "retrieval_before_first_update": False},
                "mechanism_scores": {"artifact_quality_score": 0.6, "mechanism_confidence": 0.6, "transfer_quality": 0.6, "shortcut_resistance": 0.7},
            },
            {
                "family_id": "F06_stale_conflict_update",
                "mechanism": "memory",
                "bucket": "learn",
                "stage": "learn_b",
                "expected_persistence_signal": "memory",
                "task_score": 0.8,
                "passed": True,
                "tool_dispatch_count": 1,
                "token_usage": {"total_tokens": 10},
                "timing": {"wall_time_s": 1.0},
                "artifacts": {"memory_file_exists": True, "user_file_exists": False, "skill_count": 0},
                "internal_tools": {"memory_calls": 1, "skill_manage_calls": 0, "session_search_calls": 0},
                "retrieval_signals": {"used_expected_signal": False, "retrieval_before_first_update": False},
                "mechanism_scores": {"artifact_quality_score": 0.8, "mechanism_confidence": 0.8, "transfer_quality": 0.8, "shortcut_resistance": 0.8},
            },
            {
                "family_id": "F06_stale_conflict_update",
                "mechanism": "memory",
                "bucket": "evaluation",
                "stage": "eval_near",
                "expected_persistence_signal": "memory",
                "task_score": 0.85,
                "passed": True,
                "tool_dispatch_count": 1,
                "token_usage": {"total_tokens": 10},
                "timing": {"wall_time_s": 1.0},
                "artifacts": {"memory_file_exists": True, "user_file_exists": False, "skill_count": 0},
                "internal_tools": {"memory_calls": 0, "skill_manage_calls": 0, "session_search_calls": 0},
                "retrieval_signals": {"used_expected_signal": True, "retrieval_before_first_update": True},
                "mechanism_scores": {"artifact_quality_score": 0.8, "mechanism_confidence": 0.9, "transfer_quality": 0.9, "shortcut_resistance": 0.9},
            },
            {
                "family_id": "F06_stale_conflict_update",
                "mechanism": "memory",
                "bucket": "evaluation",
                "stage": "eval_far",
                "expected_persistence_signal": "memory",
                "task_score": 0.75,
                "passed": True,
                "tool_dispatch_count": 1,
                "token_usage": {"total_tokens": 10},
                "timing": {"wall_time_s": 1.0},
                "artifacts": {"memory_file_exists": True, "user_file_exists": False, "skill_count": 0},
                "internal_tools": {"memory_calls": 0, "skill_manage_calls": 0, "session_search_calls": 0},
                "retrieval_signals": {"used_expected_signal": True, "retrieval_before_first_update": True},
                "mechanism_scores": {"artifact_quality_score": 0.7, "mechanism_confidence": 0.85, "transfer_quality": 0.85, "shortcut_resistance": 0.85},
            },
        ],
    )

    metrics = summary["family_summary"]["F06_stale_conflict_update"]["metrics"]
    assert metrics["write_precision"] == 0.7
    assert metrics["recall_accuracy"] == 0.8
    assert metrics["update_correctness"] == 0.783333
    assert metrics["retention_horizon"] == 0.882353
    assert metrics["retention_horizon_score_delta"] == -0.1
    assert metrics["memory_pollution_rate"] == 0.3

    evolve = summary["family_summary"]["F06_stale_conflict_update"]["evolve"]
    assert evolve["outcome_delta"] == 0.4
    assert evolve["mechanism_score"] == 0.773137
    assert evolve["evolve_score"] == 0.309255

    assert summary["benchmark_signal"]["avg_write_precision"] == 0.7
    assert summary["benchmark_signal"]["avg_recall_accuracy"] == 0.8
    assert summary["benchmark_signal"]["avg_update_correctness"] == 0.783333
    assert summary["benchmark_signal"]["avg_retention_horizon"] == 0.882353
    assert summary["benchmark_signal"]["avg_memory_pollution_rate"] == 0.3
    assert summary["benchmark_signal"]["avg_outcome_delta"] == 0.4
    assert summary["benchmark_signal"]["avg_mechanism_score"] == 0.773137
    assert summary["benchmark_signal"]["avg_evolve_score"] == 0.309255
    assert summary["benchmark_signal"]["avg_evolve_index"] == 0.309255


def test_summarize_comparison_reports_ablation_evolve_score():
    with_persistence = {
        "bucket_summary": {
            "evaluation": {
                "avg_task_score": 0.8,
                "pass_rate": 0.5,
                "avg_tool_dispatch_count": 3.0,
            }
        },
        "benchmark_signal": {
            "avg_family_task_score_delta": -0.2,
            "avg_family_pass_rate_delta": -0.5,
            "avg_outcome_delta": -0.2,
            "avg_mechanism_score": 0.75,
            "avg_evolve_score": 0.0,
        },
    }
    without_persistence = {
        "bucket_summary": {
            "evaluation": {
                "avg_task_score": 0.6,
                "pass_rate": 0.0,
                "avg_tool_dispatch_count": 2.0,
            }
        },
        "benchmark_signal": {
            "avg_family_task_score_delta": -0.4,
            "avg_family_pass_rate_delta": -1.0,
            "avg_outcome_delta": -0.4,
            "avg_mechanism_score": 0.25,
            "avg_evolve_score": 0.0,
        },
    }

    comparison = summarize_comparison(
        with_persistence=with_persistence,
        without_persistence=without_persistence,
    )

    assert comparison["delta"]["ablation_outcome_delta"] == 0.2
    assert comparison["delta"]["ablation_evolve_score"] == 0.15
