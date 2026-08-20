from __future__ import annotations

from past_bench.graders.self_evolve_helpers import SelfEvolveTaskGrader
from past_bench.models.task import Prompt, TaskDefinition


def _task() -> TaskDefinition:
    return TaskDefinition(
        task_id="EP02_TEST",
        task_name="EP02 Test",
        prompt=Prompt(text="test"),
    )


def _audit_data() -> dict:
    return {
        "config": {
            "calls": [
                {
                    "endpoint": "/config/integrations",
                    "request_body": {"status": "suspended"},
                    "response_body": {"integrations": [], "total": 4},
                }
            ],
            "updates": [
                {
                    "integration_id": "INTG-EU-002",
                    "changes": {"status": "active", "exception_approved": True},
                },
                {
                    "integration_id": "INTG-US-004",
                    "changes": {"status": "active", "exception_approved": True},
                },
                {
                    "integration_id": "INTG-AP-006",
                    "changes": {"status": "active", "exception_approved": True},
                },
            ],
            "notifications": [],
        }
    }


def _expectations(*, inspection_optional: bool) -> dict:
    return {
        "mode": "config_recall",
        "inspected_ids": ["INTG-EU-002", "INTG-US-004", "INTG-AP-006", "INTG-EU-009"],
        "inspection_optional": inspection_optional,
        "required_update_ids": ["INTG-EU-002", "INTG-US-004", "INTG-AP-006"],
        "forbidden_update_ids": ["INTG-EU-009"],
        "update_requirements": {
            "INTG-EU-002": {"status": "active", "exception_approved": True},
            "INTG-US-004": {"status": "active", "exception_approved": True},
            "INTG-AP-006": {"status": "active", "exception_approved": True},
        },
        "required_summary_keywords": [
            "INTG-EU-002",
            "INTG-US-004",
            "INTG-AP-006",
            "INTG-EU-009",
        ],
        "min_session_search_calls": 1,
        "session_search_weight": 0.1,
        "_internal_tools": {"session_search_calls": 1},
        "_mechanism_metrics": {},
    }


def test_config_recall_keeps_strict_cap_when_explicit_inspection_is_required(monkeypatch) -> None:
    grader = SelfEvolveTaskGrader()
    monkeypatch.setattr(
        grader,
        "_get_final_assistant_text",
        lambda _messages: "BATCH|INTG-EU-002,INTG-US-004,INTG-AP-006|INTG-EU-009",
    )

    scores = grader._grade_config_recall(
        messages=[],
        dispatches=[],
        audit_data=_audit_data(),
        judge=None,
        task=_task(),
        expectations=_expectations(inspection_optional=False),
    )

    assert scores.completion == 0.2975


def test_config_recall_allows_grounded_update_only_runs_when_inspection_is_optional(monkeypatch) -> None:
    grader = SelfEvolveTaskGrader()
    monkeypatch.setattr(
        grader,
        "_get_final_assistant_text",
        lambda _messages: "BATCH|INTG-EU-002,INTG-US-004,INTG-AP-006|INTG-EU-009",
    )

    scores = grader._grade_config_recall(
        messages=[],
        dispatches=[],
        audit_data=_audit_data(),
        judge=None,
        task=_task(),
        expectations=_expectations(inspection_optional=True),
    )

    assert scores.completion == 0.6375


def test_config_recall_rewards_guarded_no_update_controls(monkeypatch) -> None:
    grader = SelfEvolveTaskGrader()
    monkeypatch.setattr(
        grader,
        "_get_final_assistant_text",
        lambda _messages: "BATCH|none|INTG-EU-002,INTG-US-004,INTG-AP-006",
    )

    scores = grader._grade_config_recall(
        messages=[],
        dispatches=[],
        audit_data={
            "config": {
                "calls": [
                    {
                        "endpoint": "/config/integrations",
                        "request_body": {"status": "suspended"},
                        "response_body": {"integrations": [], "total": 6},
                    }
                ],
                "updates": [],
                "notifications": [],
            }
        },
        judge=None,
        task=_task(),
        expectations={
            "mode": "config_recall",
            "inspected_ids": ["INTG-EU-002", "INTG-US-004"],
            "inspection_optional": True,
            "required_update_ids": [],
            "forbidden_update_ids": ["INTG-EU-002"],
            "require_no_updates": True,
            "no_update_weight": 0.2,
            "required_summary_keywords": ["BATCH", "none", "INTG-EU-002"],
            "min_session_search_calls": 0,
            "session_search_weight": 0.0,
            "_internal_tools": {"session_search_calls": 0},
            "_mechanism_metrics": {},
        },
    )

    assert scores.completion == 0.255
