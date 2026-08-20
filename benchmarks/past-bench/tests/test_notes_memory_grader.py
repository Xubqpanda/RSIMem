import pytest

from past_bench.graders.self_evolve_helpers import (
    compute_notes_memory_completion,
    compute_notes_memory_hard_pass,
    SelfEvolveTaskGrader,
    compute_notes_memory_task_components,
)
from past_bench.models.message import Message
from past_bench.models.task import Prompt, TaskDefinition
from past_bench.models.trace import TraceMessage


class InlineNotesMemoryGrader(SelfEvolveTaskGrader):
    def __init__(self, expectations: dict):
        self._expectations = expectations

    def _load_expectations(self, task: TaskDefinition) -> dict:
        return dict(self._expectations)


def _task() -> TaskDefinition:
    return TaskDefinition(
        task_id="SE_TEST",
        task_name="Inline Notes Memory Test",
        prompt=Prompt(text="demo"),
    )


def _messages(text: str) -> list[TraceMessage]:
    return [
        TraceMessage(
            trace_id="trace-inline",
            message=Message(role="assistant", content=text),
        )
    ]


def _audit(recipients: list[str], note_id: str | None = None) -> dict:
    get_call = {"endpoint": "/notes/get"}
    share_record = {"recipients": recipients}
    if note_id:
        get_call["request_body"] = {"note_id": note_id}
        share_record["note_id"] = note_id
    return {
        "notes": {
            "calls": [
                {"endpoint": "/notes/list"},
                get_call,
            ],
            "shared": [share_record],
        }
    }


def _env(*, used_signal: bool, memory_injections: int = 0) -> dict:
    return {
        "_self_evolve_internal_tools": {},
        "_self_evolve_skill_count": 0,
        "_self_evolve_has_memory_artifact": False,
        "_self_evolve_mechanism_metrics": {
            "used_expected_signal": used_signal,
            "memory_injection_count": memory_injections,
            "memory_read_count": 0,
            "skill_read_count": 0,
            "session_search_count": 0,
            "retrieval_signal_count": memory_injections,
            "retrieval_before_first_update": used_signal,
        },
    }


def test_notes_memory_grader_falls_back_to_required_summary_keywords():
    grader = InlineNotesMemoryGrader(
        {
            "mode": "notes_memory",
            "bucket_role": "baseline",
            "required_recipients": ["Kevin Wang", "Sarah Zhang"],
            "required_summary_keywords": ["50,000", "Engineering", "budget"],
        }
    )

    scores = grader.grade(
        messages=_messages("Engineering budget confirmed at 50,000 and shared with Kevin Wang and Sarah Zhang."),
        dispatches=[],
        task=_task(),
        audit_data=_audit(["Kevin Wang", "Sarah Zhang"]),
        env_snapshot=_env(used_signal=False),
    )

    assert scores.completion == 1.0


def test_notes_memory_grader_penalizes_forbidden_output_keywords():
    grader = InlineNotesMemoryGrader(
        {
            "mode": "notes_memory",
            "bucket_role": "baseline",
            "required_recipients": ["Sarah Zhang", "Leo Li"],
            "required_output_keywords": ["75,000", "Engineering", "budget"],
            "forbidden_output_keywords": ["50,000"],
        }
    )

    scores = grader.grade(
        messages=_messages("Engineering budget updated to 75,000 but the old 50,000 estimate is still shown."),
        dispatches=[],
        task=_task(),
        audit_data=_audit(["Sarah Zhang", "Leo Li"]),
        env_snapshot=_env(used_signal=False),
    )

    assert scores.completion == 0.5


def test_notes_memory_task_components_require_grounding_content_and_share():
    task_components = compute_notes_memory_task_components(
        expectations={
            "required_recipients": ["Kevin Wang", "Leo Li"],
            "required_output_keywords": ["Kevin Wang", "Leo Li"],
            "forbidden_output_keywords": ["Sarah Zhang"],
        },
        final_text="Shared the update with Kevin Wang and Leo Li.",
        notes={
            "calls": [
                {"endpoint": "/notes/list"},
                {"endpoint": "/notes/get"},
            ],
            "shared": [{"recipients": ["Kevin Wang", "Leo Li"]}],
        },
    )

    assert task_components == {
        "grounding": 1.0,
        "content_correctness": 1.0,
        "share_correctness": 1.0,
    }
    assert compute_notes_memory_completion(task_components) == 1.0
    assert compute_notes_memory_hard_pass(task_components) is True


def test_notes_memory_required_note_id_controls_grounding_and_share():
    task_components = compute_notes_memory_task_components(
        expectations={
            "required_note_id": "note-source",
            "required_recipients": ["Kevin Wang"],
            "required_output_keywords": ["approved"],
        },
        final_text="The update is approved.",
        notes={
            "calls": [
                {"endpoint": "/notes/list"},
                {"endpoint": "/notes/get", "request_body": {"note_id": "note-other"}},
            ],
            "shared": [{"note_id": "note-other", "recipients": ["Kevin Wang"]}],
        },
    )

    assert task_components == {
        "grounding": 0.5,
        "content_correctness": 1.0,
        "share_correctness": 0.0,
    }


def test_notes_memory_keywords_are_case_and_hyphen_insensitive():
    task_components = compute_notes_memory_task_components(
        expectations={
            "required_output_keywords": ["4-space", "double quotes", "Black"],
            "required_recipients": ["Kevin Wang"],
        },
        final_text="Use 4 spaces, Double Quotes, and black formatting.",
        notes={
            "calls": [
                {"endpoint": "/notes/list"},
                {"endpoint": "/notes/get"},
            ],
            "shared": [{"recipients": ["Kevin Wang"]}],
        },
    )

    assert task_components["content_correctness"] == 1.0


def test_notes_memory_keywords_ignore_alignment_whitespace():
    task_components = compute_notes_memory_task_components(
        expectations={
            "required_output_keywords": ["Security | Renee Shah | SEC-GAMMA"],
            "required_recipients": ["Renee Shah"],
        },
        final_text="Security | Renee Shah   | SEC-GAMMA | rotate the privileged credential chain | 2026-07-09",
        notes={
            "calls": [
                {"endpoint": "/notes/list"},
                {"endpoint": "/notes/get"},
            ],
            "shared": [{"recipients": ["Renee Shah"]}],
        },
    )

    assert task_components["content_correctness"] == 1.0


def test_notes_memory_require_tsv_output_zeroes_non_tsv_content():
    task_components = compute_notes_memory_task_components(
        expectations={
            "required_output_keywords": ["Kevin Wang", "Leo Li", "High", "2026/05/30"],
            "required_recipients": ["Kevin Wang", "Leo Li"],
            "require_tsv_output": True,
        },
        final_text="Kevin Wang High follow up 2026/05/30\nLeo Li Medium review 2026/05/31",
        notes={
            "calls": [
                {"endpoint": "/notes/list"},
                {"endpoint": "/notes/get"},
            ],
            "shared": [{"recipients": ["Kevin Wang", "Leo Li"]}],
        },
    )

    assert task_components["content_correctness"] == 0.0


def test_notes_memory_reports_slot_checks_for_trigger_attribution():
    task_components = compute_notes_memory_task_components(
        expectations={
            "required_output_keywords": ["Kevin Wang", "Leo Li"],
            "required_recipients": ["Kevin Wang", "Leo Li"],
            "slot_checks": {
                "r1_tsv_structure": {"require_tsv_output": True, "min_tsv_rows": 2},
                "r2_priority_normalization": {
                    "require_tsv_output": True,
                    "tsv_column_index": 1,
                    "required_tsv_column_values": ["High", "Medium"],
                    "forbidden_tsv_column_values": ["urgent", "normal priority"],
                },
                "r3_date_normalization": {
                    "require_tsv_output": True,
                    "tsv_column_index": 3,
                    "required_tsv_column_values": ["2026/05/30", "2026/05/31"],
                    "forbidden_tsv_column_values": ["May 30", "May 31"],
                },
            },
        },
        final_text=(
            "Kevin Wang\turgent\tfollow up\tMay 30\n"
            "Leo Li\tnormal priority\treview\tMay 31"
        ),
        notes={
            "calls": [
                {"endpoint": "/notes/list"},
                {"endpoint": "/notes/get"},
            ],
            "shared": [{"recipients": ["Kevin Wang", "Leo Li"]}],
        },
    )

    assert task_components["r1_tsv_structure"] == 1.0
    assert task_components["r2_priority_normalization"] == 0.0
    assert task_components["r3_date_normalization"] == 0.0
    assert task_components["content_correctness"] == 0.333333
    assert compute_notes_memory_hard_pass(
        task_components,
        expectations={
            "slot_checks": {
                "r1_tsv_structure": {},
                "r2_priority_normalization": {},
                "r3_date_normalization": {},
            },
        },
    ) is False


def test_notes_memory_allows_forbidden_keyword_in_correction_context():
    task_components = compute_notes_memory_task_components(
        expectations={
            "required_output_keywords": ["75,000", "Engineering", "budget"],
            "forbidden_output_keywords": ["50,000"],
            "forbidden_output_context_exceptions": [
                {
                    "keyword": "50,000",
                    "required_context_keywords": ["75,000", "correct"],
                }
            ],
        },
        final_text="The corrected Engineering budget is 75,000, corrected from the old 50,000 estimate.",
        notes={"calls": [], "shared": []},
    )

    assert task_components["content_correctness"] == 1.0


def test_notes_memory_hard_pass_requires_all_components():
    task_components = {
        "grounding": 1.0,
        "content_correctness": 1.0,
        "share_correctness": 0.5,
    }

    assert compute_notes_memory_completion(task_components) == 0.85
    assert compute_notes_memory_hard_pass(task_components) is False


def test_notes_memory_hard_pass_requires_expected_memory_write_and_artifact():
    task_components = {
        "grounding": 1.0,
        "content_correctness": 1.0,
        "share_correctness": 1.0,
    }

    assert compute_notes_memory_hard_pass(
        task_components,
        expectations={
            "min_memory_calls": 1,
            "require_memory_artifact": True,
            "require_rule_artifact": ["TSV"],
        },
        artifact_summary={
            "memory_file_exists": False,
            "user_file_exists": False,
            "internal_tools": {"memory_write_count": 0},
        },
        artifact_diff={"rule_keyword_hits": {"hit_keywords": []}},
        retrieval_signals={},
    ) is False

    assert compute_notes_memory_hard_pass(
        task_components,
        expectations={
            "min_memory_calls": 1,
            "require_memory_artifact": True,
            "require_rule_artifact": ["TSV"],
        },
        artifact_summary={
            "memory_file_exists": True,
            "user_file_exists": False,
            "internal_tools": {"memory_write_count": 1},
        },
        artifact_diff={"rule_keyword_hits": {"hit_keywords": ["TSV"]}},
        retrieval_signals={},
    ) is True


def test_notes_memory_hard_pass_requires_expected_memory_injection():
    task_components = {
        "grounding": 1.0,
        "content_correctness": 1.0,
        "share_correctness": 1.0,
    }

    assert compute_notes_memory_hard_pass(
        task_components,
        expectations={"min_memory_injections": 1},
        artifact_summary={"internal_tools": {}},
        artifact_diff={},
        retrieval_signals={"memory_injection_count": 0},
    ) is False

    assert compute_notes_memory_hard_pass(
        task_components,
        expectations={"min_memory_injections": 1},
        artifact_summary={"internal_tools": {}},
        artifact_diff={},
        retrieval_signals={"memory_injection_count": 1},
    ) is True


def test_notes_memory_grader_evaluation_does_not_cap_without_retrieval_signal():
    grader = InlineNotesMemoryGrader(
        {
            "mode": "notes_memory",
            "bucket_role": "evaluation",
            "required_recipients": ["Kevin Wang", "Leo Li"],
            "required_output_keywords": ["Kevin Wang", "Leo Li"],
        }
    )

    scores = grader.grade(
        messages=_messages("Shared the update with Kevin Wang and Leo Li."),
        dispatches=[],
        task=_task(),
        audit_data=_audit(["Kevin Wang", "Leo Li"]),
        env_snapshot=_env(used_signal=False, memory_injections=0),
    )

    assert scores.completion == 1.0
