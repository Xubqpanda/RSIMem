from __future__ import annotations

import json
from pathlib import Path

from past_bench.graders.self_evolve_helpers import SelfEvolveTaskGrader
from past_bench.models.message import Message
from past_bench.models.scoring import compute_task_score
from past_bench.models.task import Prompt, TaskDefinition
from past_bench.models.trace import DimensionScores, TraceMessage


PG01_ROOT = (
    Path(__file__).resolve().parent.parent
    / "self-evolve-tasks-v2"
    / "proactive_information_gathering"
    / "PG01_release_decision_followup"
)


def _task(task_id: str) -> TaskDefinition:
    return TaskDefinition(
        task_id=task_id,
        task_name=task_id,
        prompt=Prompt(text=task_id),
    )


def _messages(text: str) -> list[TraceMessage]:
    return [
        TraceMessage(
            trace_id="trace-pg01",
            message=Message(role="assistant", content=text),
        )
    ]


def _expectations(episode: str) -> dict:
    return json.loads((PG01_ROOT / episode / "expectations.json").read_text(encoding="utf-8"))


def test_pg01_reflection_perfect_three_line_output_can_pass(monkeypatch) -> None:
    grader = SelfEvolveTaskGrader()
    expectations = _expectations("reflection_gap_review")
    final_text = (
        "ERROR_CLASS|answered from fuzzy prior-decision recall instead of grounding on exact earlier record\n"
        "MISSED_CLUE|exact release reference plus environment/expiry fields\n"
        "NEXT_CHECK|before sharing release guidance, search session history for the matching release review "
        "and verify the exact reference, allowed environment, and expiry"
    )
    monkeypatch.setattr(grader, "_judge_score", lambda *args, **kwargs: 1.0)

    scores = grader._grade_proactive_retrieval(
        messages=_messages(final_text),
        dispatches=[],
        audit_data={},
        judge=None,
        task=_task("PG01_REFLECTION_001"),
        expectations=expectations,
    )

    assert scores.completion == 1.0
    assert compute_task_score(scores) == 1.0


def test_pg01_update_full_artifact_contract_uses_judge_blend(monkeypatch) -> None:
    grader = SelfEvolveTaskGrader()
    expectations = _expectations("update_or_stabilize_policy_note")
    expectations["_internal_tools"] = {
        "skill_manage_calls": 3,
        "skill_create_count": 1,
        "skill_update_count": 2,
        "skill_view_calls": 1,
        "skill_read_count": 1,
    }
    expectations["_mechanism_metrics"] = {
        "artifact_quality_score": 1.0,
        "skill_read_count": 1,
        "retrieval_signal_count": 1,
        "explicit_retrieval_count": 1,
        "used_expected_signal": True,
        "skill_create_count": 1,
        "skill_update_count": 2,
    }
    monkeypatch.setattr(grader, "_judge_score", lambda *args, **kwargs: 0.4)

    scores = grader._grade_proactive_retrieval(
        messages=_messages("POLICY_UPDATED|release_followup_lookup"),
        dispatches=[],
        audit_data={},
        judge=None,
        task=_task("PG01_UPDATE_001"),
        expectations=expectations,
    )

    assert scores.communication == 0.4
    assert scores.completion == 0.91
    assert compute_task_score(scores) == 0.928


def test_pg01_attempt_missing_exact_reference_cannot_hard_pass(monkeypatch) -> None:
    grader = SelfEvolveTaskGrader()
    expectations = _expectations("attempt_probe_pre")
    expectations["_internal_tools"] = {"session_search_calls": 1}
    expectations["_mechanism_metrics"] = {
        "explicit_retrieval_count": 1,
        "explicit_retrieval_before_first_update": True,
        "session_search_count": 1,
        "used_expected_signal": True,
    }
    notes_audit = {
        "notes": {
            "shared": [
                {
                    "note_id": "RELNOTE-PG01-A",
                    "recipients": ["release-ops@corp.com"],
                }
            ]
        }
    }
    monkeypatch.setattr(grader, "_judge_score", lambda *args, **kwargs: 0.7)

    scores = grader._grade_proactive_retrieval(
        messages=_messages(
            "RELEASE|Falcon Edge connector|not approved for production|staging-only validation|2026-05-31"
        ),
        dispatches=[],
        audit_data=notes_audit,
        judge=None,
        task=_task("PG01_ATTEMPT_001"),
        expectations=expectations,
    )

    assert scores.completion < 1.0
    assert compute_task_score(scores) < 1.0


def test_pg01_reflection_uses_judge_blend(monkeypatch) -> None:
    grader = SelfEvolveTaskGrader()
    expectations = _expectations("reflection_gap_review")
    final_text = (
        "ERROR_CLASS|answered from fuzzy prior-decision recall instead of grounding on exact earlier record\n"
        "MISSED_CLUE|exact release reference plus environment/expiry fields\n"
        "NEXT_CHECK|before sharing release guidance, search session history for the matching release review "
        "and verify the exact reference, allowed environment, and expiry"
    )
    monkeypatch.setattr(grader, "_judge_score", lambda *args, **kwargs: 0.4)

    scores = grader._grade_proactive_retrieval(
        messages=_messages(final_text),
        dispatches=[],
        audit_data={},
        judge=None,
        task=_task("PG01_REFLECTION_001"),
        expectations=expectations,
    )

    assert scores.communication == 0.4
    assert scores.completion == 0.91
    assert compute_task_score(scores) == 0.928
