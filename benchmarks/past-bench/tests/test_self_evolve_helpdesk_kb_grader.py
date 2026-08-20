from pathlib import Path

from past_bench.graders.self_evolve_helpers import SelfEvolveTaskGrader
from past_bench.models.message import Message
from past_bench.models.task import TaskDefinition
from past_bench.models.trace import TraceMessage


ROOT = Path(__file__).resolve().parents[1]


def _assistant_message(text: str) -> list[TraceMessage]:
    return [
        TraceMessage(
            trace_id="t",
            message=Message(role="assistant", content=text),
        )
    ]


def _audit(updated_tickets: list[dict]) -> dict:
    return {
        "helpdesk": {
            "updated_tickets": updated_tickets,
            "calls": [
                {"endpoint": "/helpdesk/tickets/get", "request_body": {"ticket_id": "TK-FM03-009"}},
                {"endpoint": "/helpdesk/tickets/get", "request_body": {"ticket_id": "TK-FM03-010"}},
            ],
        },
        "kb": {
            "calls": [
                {"endpoint": "/kb/search", "request_body": {}},
                {"endpoint": "/kb/articles/get", "request_body": {}},
            ]
        },
    }


def test_helpdesk_kb_supports_per_ticket_expected_updates() -> None:
    task = TaskDefinition.from_yaml(
        ROOT
        / "self-evolve-tasks-v2/procedural_ability/PC01_sop_bootstrap_04/"
        / "FM03_C02_control_shortcut/task.yaml"
    )
    grader = SelfEvolveTaskGrader()
    messages = _assistant_message("TK-FM03-009\nTK-FM03-010")

    correct = grader.grade(
        messages,
        [],
        task,
        audit_data=_audit(
            [
                {
                    "ticket_id": "TK-FM03-009",
                    "priority": "low",
                    "category": "data-export",
                    "tags": ["data-approved", "export-tool-approved"],
                },
                {
                    "ticket_id": "TK-FM03-010",
                    "priority": "medium",
                    "category": "data-export",
                    "tags": ["data-share-pending"],
                    "internal_note": "Awaiting DPO approval before export",
                },
            ]
        ),
        env_snapshot={
            "_self_evolve_mechanism_metrics": {
                "used_expected_signal": True,
                "skill_read_count": 1,
            }
        },
    )
    stale_all_approved = grader.grade(
        messages,
        [],
        task,
        audit_data=_audit(
            [
                {
                    "ticket_id": "TK-FM03-009",
                    "priority": "low",
                    "category": "data-export",
                    "tags": ["data-approved", "export-tool-approved"],
                },
                {
                    "ticket_id": "TK-FM03-010",
                    "priority": "low",
                    "category": "data-export",
                    "tags": ["data-approved"],
                },
            ]
        ),
        env_snapshot={
            "_self_evolve_mechanism_metrics": {
                "used_expected_signal": True,
                "skill_read_count": 1,
            }
        },
    )

    assert correct.completion > stale_all_approved.completion


def test_helpdesk_kb_penalizes_stale_tags_even_when_required_tags_are_present() -> None:
    task = TaskDefinition.from_yaml(
        ROOT
        / "self-evolve-tasks-v2/update_ability/PC02_sop_patch_02/"
        / "FC04_I03_errcorr_eval_near/task.yaml"
    )
    grader = SelfEvolveTaskGrader()
    messages = _assistant_message("TK-E401\nTK-E402\nTK-E403")

    correct = grader.grade(
        messages,
        [],
        task,
        audit_data=_audit(
            [
                {
                    "ticket_id": "TK-E401",
                    "priority": "medium",
                    "category": "code-review",
                    "tags": ["deprecated-api", "needs-migration"],
                },
                {
                    "ticket_id": "TK-E403",
                    "priority": "medium",
                    "category": "code-review",
                    "tags": ["deprecated-api", "needs-migration"],
                },
            ]
        ),
        env_snapshot={
            "_self_evolve_mechanism_metrics": {
                "used_expected_signal": True,
                "skill_read_count": 1,
            }
        },
    )
    stale = grader.grade(
        messages,
        [],
        task,
        audit_data=_audit(
            [
                {
                    "ticket_id": "TK-E401",
                    "priority": "medium",
                    "category": "code-review",
                    "tags": ["deprecated-api", "needs-migration", "api-compliant"],
                },
                {
                    "ticket_id": "TK-E403",
                    "priority": "medium",
                    "category": "code-review",
                    "tags": ["deprecated-api", "needs-migration", "api-compliant"],
                },
            ]
        ),
        env_snapshot={
            "_self_evolve_mechanism_metrics": {
                "used_expected_signal": True,
                "skill_read_count": 1,
            }
        },
    )

    assert correct.completion > stale.completion
