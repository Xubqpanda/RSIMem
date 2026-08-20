from __future__ import annotations

from past_bench.graders.self_evolve_helpers import SelfEvolveTaskGrader
from past_bench.models.message import Message
from past_bench.models.task import Prompt, TaskDefinition
from past_bench.models.trace import ToolDispatch, TraceMessage


def _task() -> TaskDefinition:
    return TaskDefinition(
        task_id="EP03_TEST",
        task_name="EP03 Test",
        prompt=Prompt(text="test"),
    )


def _messages(text: str) -> list[TraceMessage]:
    return [
        TraceMessage(
            trace_id="trace-ep03",
            message=Message(role="assistant", content=text),
        )
    ]


def _dispatch(*, response_body: object, timestamp: str = "2026-04-17T10:00:00Z") -> ToolDispatch:
    return ToolDispatch(
        trace_id="trace-ep03",
        tool_use_id="tool-1",
        tool_name="session_search",
        endpoint_url="internal://session_search",
        response_body=response_body,
        timestamp=timestamp,
    )


def _notes_audit(
    *,
    updated_note_id: str,
    updated_content: str = (
        "ONBOARDING PLAYBOOK v1. Step 1: HR emails welcome packet. Step 2: manager books kickoff call. "
        "Step 3: IT provisions laptop. Step 4: mentor assigned within 24h. Owner: Carol. Status: active."
    ),
) -> dict:
    shell_notes = [
        {
            "note_id": "DOC-721",
            "title": "Onboarding Playbook",
            "participants": ["alice@corp.com", "bob@corp.com", "carol@corp.com"],
            "tags": ["approved", "artifact-shell"],
            "status": "active",
        },
        {
            "note_id": "DOC-722",
            "title": "Onboarding Playbook Draft",
            "participants": ["alice@corp.com", "bob@corp.com", "carol@corp.com"],
            "tags": ["approved", "artifact-shell"],
            "status": "active",
        },
        {
            "note_id": "DOC-723",
            "title": "Onboarding Playbook",
            "participants": ["alice@corp.com", "people-ops@corp.com"],
            "tags": ["candidate", "artifact-shell"],
            "status": "active",
        },
    ]
    updated_note = {
        "note_id": updated_note_id,
        "title": "Onboarding Playbook" if updated_note_id != "DOC-722" else "Onboarding Playbook Draft",
        "participants": (
            ["alice@corp.com", "bob@corp.com", "carol@corp.com"]
            if updated_note_id != "DOC-723"
            else ["alice@corp.com", "people-ops@corp.com"]
        ),
        "tags": ["approved", "artifact-shell"] if updated_note_id != "DOC-723" else ["candidate", "artifact-shell"],
        "content": updated_content,
    }
    return {
        "notes": {
            "calls": [
                {
                    "endpoint": "/notes/list",
                    "response_body": {"notes": shell_notes, "total": len(shell_notes)},
                },
                {
                    "endpoint": "/notes/update",
                    "response_body": {"note": updated_note},
                },
            ],
            "updates": [
                {
                    "note_id": updated_note_id,
                    "payload": {"content": updated_note["content"]},
                    "changed_fields": ["content"],
                }
            ],
        }
    }


def _expectations() -> dict:
    return {
        "expected_mechanism": "session_search",
        "bucket_role": "evaluation",
        "mode": "notes_session_recall",
        "required_prior_note_id": "DOC-431",
        "required_updated_note_id": "DOC-721",
        "required_output_keywords": ["PLAYBOOK", "mentor_sla_24h"],
        "target_modify_field": "content",
        "preserved_fields": ["title", "participants", "tags"],
        "required_shell_metadata": {
            "title": "Onboarding Playbook",
            "participants": ["alice@corp.com", "bob@corp.com", "carol@corp.com"],
            "tags": ["approved", "artifact-shell"],
        },
        "shell_match_fields": ["title", "participants", "tags"],
        "require_updated_note_differs_from_prior": True,
        "require_retrieval_before_update": True,
        "retrieval_snippet_source": "session_search_response_only",
        "min_session_search_calls": 1,
        "session_search_weight": 0.1,
        "required_content_replacements": [
            {
                "old": "mentor assigned within 48h",
                "new": "mentor assigned within 24h",
            }
        ],
        "forbidden_content_regressions": [
            "Step 1: HR emails welcome packet.",
            "Step 2: manager books kickoff call.",
            "Step 3: IT provisions laptop.",
            "Owner: Carol.",
            "Status: active.",
        ],
        "require_retrieved_prior_snippets": ["DOC-431", "mentor assigned within 48h"],
        "max_without_grounding": 0.25,
        "max_without_target_update": 0.25,
        "max_without_required_replacements": 0.25,
        "retrieval_contract": {"evaluation_only": True, "min_skill_reads": 0},
        "artifact_contract": {
            "type": "session",
            "require_rule_keywords": ["onboarding", "playbook"],
            "min_count_delta": 0,
        },
        "skill_artifact_weight": 0.0,
        "_internal_tools": {"session_search_calls": 1},
        "_mechanism_metrics": {
            "memory_read_count": 0,
            "retrieval_before_first_update": True,
        },
    }


def test_ep03_grader_rejects_memory_only_retrieval(monkeypatch) -> None:
    grader = SelfEvolveTaskGrader()
    expectations = _expectations()
    expectations["_internal_tools"] = {"session_search_calls": 0}
    expectations["_mechanism_metrics"] = {
        "memory_read_count": 1,
        "retrieval_before_first_update": True,
    }
    monkeypatch.setattr(grader, "_get_final_assistant_text", lambda _messages: "PLAYBOOK|mentor_sla_24h|DOC-431")

    scores = grader._grade_notes_session_recall(
        messages=_messages("PLAYBOOK|mentor_sla_24h|DOC-431"),
        dispatches=[],
        audit_data=_notes_audit(updated_note_id="DOC-721"),
        judge=None,
        task=_task(),
        expectations=expectations,
    )

    assert scores.completion <= 0.25


def test_ep03_grader_caps_when_retrieval_happens_after_update(monkeypatch) -> None:
    grader = SelfEvolveTaskGrader()
    expectations = _expectations()
    expectations["_mechanism_metrics"] = {
        "memory_read_count": 0,
        "retrieval_before_first_update": False,
    }
    monkeypatch.setattr(grader, "_get_final_assistant_text", lambda _messages: "PLAYBOOK|mentor_sla_24h|DOC-431")

    scores = grader._grade_notes_session_recall(
        messages=_messages("PLAYBOOK|mentor_sla_24h|DOC-431"),
        dispatches=[_dispatch(response_body={"results": [{"summary": "DOC-431 mentor assigned within 48h"}]})],
        audit_data=_notes_audit(updated_note_id="DOC-721"),
        judge=None,
        task=_task(),
        expectations=expectations,
    )

    assert scores.completion <= 0.25


def test_ep03_grader_ignores_snippets_that_only_appear_in_final_text(monkeypatch) -> None:
    grader = SelfEvolveTaskGrader()
    expectations = _expectations()
    monkeypatch.setattr(
        grader,
        "_get_final_assistant_text",
        lambda _messages: "PLAYBOOK|mentor_sla_24h|DOC-431|mentor assigned within 48h",
    )

    scores = grader._grade_notes_session_recall(
        messages=_messages("PLAYBOOK|mentor_sla_24h|DOC-431|mentor assigned within 48h"),
        dispatches=[_dispatch(response_body={"results": [{"summary": "Recovered onboarding artifact without note id."}]})],
        audit_data=_notes_audit(updated_note_id="DOC-721"),
        judge=None,
        task=_task(),
        expectations=expectations,
    )

    assert scores.completion <= 0.25


def test_ep03_grader_uses_internal_session_search_responses(tmp_path, monkeypatch) -> None:
    session_file = tmp_path / "session_latest.json"
    session_file.write_text(
        """
{
  "messages": [
    {
      "role": "assistant",
      "tool_calls": [
        {
          "id": "call-session-search",
          "type": "function",
          "function": {
            "name": "session_search",
            "arguments": "{\\"query\\": \\"onboarding playbook\\"}"
          }
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "call-session-search",
      "content": "{\\"results\\": [{\\"summary\\": \\"Recovered DOC-431 with mentor assigned within 48h.\\"}]}"
    }
  ]
}
""",
        encoding="utf-8",
    )

    grader = SelfEvolveTaskGrader()
    expectations = _expectations()
    expectations["_internal_tools"] = {
        "session_search_calls": 1,
        "session_file": str(session_file),
    }
    monkeypatch.setattr(grader, "_get_final_assistant_text", lambda _messages: "PLAYBOOK|mentor_sla_24h|DOC-431")

    scores = grader._grade_notes_session_recall(
        messages=_messages("PLAYBOOK|mentor_sla_24h|DOC-431"),
        dispatches=[],
        audit_data=_notes_audit(updated_note_id="DOC-721"),
        judge=None,
        task=_task(),
        expectations=expectations,
    )

    assert scores.completion > 0.25


def test_ep03_grader_caps_when_wrong_shell_is_updated(monkeypatch) -> None:
    grader = SelfEvolveTaskGrader()
    expectations = _expectations()
    monkeypatch.setattr(grader, "_get_final_assistant_text", lambda _messages: "PLAYBOOK|mentor_sla_24h|DOC-431")

    scores = grader._grade_notes_session_recall(
        messages=_messages("PLAYBOOK|mentor_sla_24h|DOC-431"),
        dispatches=[_dispatch(response_body={"results": [{"summary": "DOC-431 mentor assigned within 48h"}]})],
        audit_data=_notes_audit(updated_note_id="DOC-722"),
        judge=None,
        task=_task(),
        expectations=expectations,
    )

    assert scores.completion <= 0.25


def test_ep03_grader_caps_when_required_replacement_is_missing(monkeypatch) -> None:
    grader = SelfEvolveTaskGrader()
    expectations = _expectations()
    stale_content = (
        "ONBOARDING PLAYBOOK v1. Step 1: HR emails welcome packet. Step 2: manager books kickoff call. "
        "Step 3: IT provisions laptop. Step 4: mentor assigned within 48h. Owner: Carol. Status: active."
    )
    monkeypatch.setattr(grader, "_get_final_assistant_text", lambda _messages: "PLAYBOOK")

    scores = grader._grade_notes_session_recall(
        messages=_messages("PLAYBOOK"),
        dispatches=[_dispatch(response_body={"results": [{"summary": "DOC-431 mentor assigned within 48h"}]})],
        audit_data=_notes_audit(updated_note_id="DOC-721", updated_content=stale_content),
        judge=None,
        task=_task(),
        expectations=expectations,
    )

    assert scores.completion <= 0.25
