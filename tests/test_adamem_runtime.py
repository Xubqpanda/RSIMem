from pathlib import Path

import pytest

from rsimem.adamem_adapter import AdaMemFeedbackView, AdaMemPolicy, update_policy
from rsimem.adamem_launcher import _past_command, _require_accepted_phase
from rsimem.adamem_runtime import (
    AdaMemPolicyReceipt,
    build_pure_process_feedback,
    materialize_phase_manifest,
    split_family_manifest,
)


def _source() -> dict[str, object]:
    return {
        "name": "fixture",
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
    assert command[command.index("--port-offset") + 1] == "1000"
