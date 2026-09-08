from pathlib import Path

import pytest

from rsimem.adamem_adapter import AdaMemFeedbackView, AdaMemPolicy, update_policy
from rsimem.adamem_experiment import AdaMemCondition, AdaMemRunSpec
from rsimem.adamem_launcher import (
    FROZEN_MODEL_ID,
    _past_command,
    _require_accepted_phase,
    compare_run_manifests,
    run_trajectory,
)
from rsimem.adamem_runtime import (
    AdaMemPolicyReceipt,
    build_pure_process_feedback,
    materialize_phase_manifest,
    split_family_manifest,
)


def _source() -> dict[str, object]:
    return {
        "name": "fixture",
        "hermes": {},
        "episodes": [
            {"label": "learn-a", "bucket": "learn", "family_id": "SM01"},
            {"label": "learn-b", "bucket": "learn", "family_id": "SM01"},
            {"label": "eval-n1", "bucket": "evaluation", "family_id": "SM01"},
            {"label": "control", "bucket": "control", "family_id": "SM01"},
        ],
    }


def test_split_requires_explicit_learning_cutover_and_filters_controls() -> None:
    split = split_family_manifest(_source(), cutover_label="learn-a")
    assert [item["label"] for item in split.prefix_episodes] == ["learn-a"]
    assert [item["label"] for item in split.suffix_episodes] == ["learn-b", "eval-n1"]
    suffix = materialize_phase_manifest(_source(), split=split, phase="suffix", initial_home_fixture_dir="seed")
    assert suffix["episodes"][0]["initial_home_fixture_dir"] == "seed"
    assert suffix["hermes"]["reasoning_effort"] == "none"
    with pytest.raises(ValueError, match="cutover"):
        split_family_manifest(_source(), cutover_label="control")


def test_feedback_excludes_grading_and_forbidden_fields(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        '{"type":"message","message":{"role":"user","content":[{"type":"text","text":"hello"}]}}\n'
        '{"type":"message","message":{"role":"assistant","content":[{"type":"text","text":"visible"}]}}\n'
        '{"type":"grading_result","scores":{"official_score":1}}\n',
        encoding="utf-8",
    )
    feedback = build_pure_process_feedback(
        feedback_view=AdaMemFeedbackView.FULL_TRAJECTORY,
        trace_paths=[trace],
    )
    assert feedback["visible_output"] == ["visible"]
    assert "grading_result" not in str(feedback)
    operation = tmp_path / "operation.jsonl"
    operation.write_text(
        '{"evidenceKind":"operation","payload":{"task_score":1}}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="forbidden"):
        build_pure_process_feedback(
            feedback_view=AdaMemFeedbackView.FULL_TRAJECTORY,
            trace_paths=[trace],
            operation_paths=[operation],
        )


def test_policy_receipt_binds_update_to_split() -> None:
    split = split_family_manifest(_source(), cutover_label="learn-a")
    parent = AdaMemPolicy.root()
    result = update_policy(
        parent=parent,
        feedback_view=AdaMemFeedbackView.TERMINAL,
        feedback={"outcome": "completed"},
        reflect=lambda _: '{"general_policy":"Keep durable preferences."}',
    )
    receipt = AdaMemPolicyReceipt.from_update(split=split, result=result)
    assert receipt.split_id == split.split_id
    assert receipt.activation == "activated"
    assert receipt.payload()["schema"] == "rsimem-adamem-runtime-v1"


def test_launcher_rejects_incomplete_usage_and_preserves_port_isolation(tmp_path: Path) -> None:
    root = tmp_path / "phase"
    root.mkdir()
    (root / "sequence_results.json").write_text(
        '{"episodes":[{"task_id":"SM01","token_usage":{"model_usage_complete":true}}]}',
        encoding="utf-8",
    )
    _require_accepted_phase(root)
    (root / "sequence_results.json").write_text(
        '{"episodes":[{"task_id":"SM01","token_usage":{"model_usage_complete":false}}]}',
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="incomplete model usage for SM01"):
        _require_accepted_phase(root)
    command = _past_command(
        past_bin=Path("past-bench"), past_root=Path("past-root"),
        sequence=Path("phase.yaml"), trace_dir=Path("trace"), config=Path("config"),
        registry=Path("registry"), model="model", base_url="https://example.test/v1",
        policy_path=None, port_offset=1000,
    )
    assert command[command.index("--agent") + 1] == "hermes"
    assert command[command.index("--port-offset") + 1] == "1000"


def test_launcher_rejects_model_drift_before_execution(tmp_path: Path) -> None:
    sequence = tmp_path / "sequence.yaml"
    sequence.write_text("name: fixture\nepisodes: []\n", encoding="utf-8")
    with pytest.raises(ValueError, match=FROZEN_MODEL_ID):
        run_trajectory(
            source_sequence=sequence,
            run=AdaMemRunSpec(
                run_id="run", condition=AdaMemCondition.MEM0_STATIC, replicate=1,
                state_directory="state", trace_directory="trace", artifact_directory="artifacts",
                mem0_collection="collection",
            ),
            condition=AdaMemCondition.MEM0_STATIC, cutover_label="learn", output_root=tmp_path,
            past_bin=tmp_path / "past", past_root=tmp_path, config=tmp_path / "config",
            registry=tmp_path / "registry", base_model="gpt-5.4",
            meta_agent_model="gpt-5.4", base_url="https://example.test/v1", api_key=None,
            update_budget=1, temperature=0.0, dry_run=True,
        )


def test_no_update_retains_mem0_root_binding(tmp_path: Path) -> None:
    split = split_family_manifest(_source(), cutover_label="learn-a")
    parent = AdaMemPolicy.root()
    result = update_policy(
        parent=parent, feedback_view=AdaMemFeedbackView.TERMINAL,
        feedback={"outcome": "completed"}, reflect=lambda _: "{}",
    )
    receipt = AdaMemPolicyReceipt.from_update(split=split, result=result)
    assert receipt.outcome == "no_update"
    assert receipt.candidate_policy_version == receipt.parent_policy_version


def test_run_manifest_allows_only_condition_identity_differences(tmp_path: Path) -> None:
    common = {
        "schema": "rsimem-adamem-run-manifest-v1", "protocol_id": "adamem-trajectory-baseline-v1",
        "run_id": "a", "condition": "B0_mem0_static", "feedback_view": None,
        "source_sequence_digest": "a" * 64, "split_id": "split", "base_model": "gpt-5.6-luna",
        "meta_agent_model": "gpt-5.6-luna", "temperature": 0.0, "update_budget": 1,
        "policy_update_space": "versioned_semantic_extraction_policy_only",
        "mem0_backend": "mem0-flat-hermes-v1", "past_bin_digest": "b" * 64,
        "config_digest": "c" * 64, "registry_digest": "d" * 64, "port_offset": 1000,
    }
    left = tmp_path / "left.json"; right = tmp_path / "right.json"
    left.write_text(__import__("json").dumps(common), encoding="utf-8")
    other = dict(common); other.update(run_id="b", condition="B1_mem0_adamem_terminal", feedback_view="terminal", port_offset=1100)
    right.write_text(__import__("json").dumps(other), encoding="utf-8")
    assert set(compare_run_manifests(left, right)) == {"run_id", "condition", "feedback_view", "port_offset"}
    other["base_model"] = "gpt-5.4"
    right.write_text(__import__("json").dumps(other), encoding="utf-8")
    with pytest.raises(ValueError, match="identity drift"):
        compare_run_manifests(left, right)
