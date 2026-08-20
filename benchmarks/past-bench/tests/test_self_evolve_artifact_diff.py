from pathlib import Path

from past_bench.models.trace import ToolDispatch
from past_bench.runner.self_evolve import (
    compute_retrieval_signals,
    diff_artifact_snapshots,
    snapshot_hermes_artifacts,
    snapshot_hermes_home,
)


def test_artifact_diff_reports_changed_skill_and_memory_entries(tmp_path: Path):
    before_dir = tmp_path / "before"
    after_dir = tmp_path / "after"
    (before_dir / "memories").mkdir(parents=True)
    (after_dir / "memories").mkdir(parents=True)
    (before_dir / "skills" / "triage").mkdir(parents=True)
    (after_dir / "skills" / "triage").mkdir(parents=True)

    (before_dir / "memories" / "MEMORY.md").write_text("old rule", encoding="utf-8")
    (after_dir / "memories" / "MEMORY.md").write_text(
        "release hash match\n§\ndeployment event same\n§\ndo not close",
        encoding="utf-8",
    )
    (before_dir / "skills" / "triage" / "SKILL.md").write_text("Use v1.\n", encoding="utf-8")
    (after_dir / "skills" / "triage" / "SKILL.md").write_text("Use release hash match.\n", encoding="utf-8")

    diff = diff_artifact_snapshots(
        before=snapshot_hermes_home(before_dir, include_contents=True),
        after=snapshot_hermes_home(after_dir, include_contents=True),
        rule_keywords=["release hash", "deployment event", "do not close"],
    )

    assert diff["added_rules"]
    assert "triage" in diff["changed_skill_names"]
    assert diff["rule_keyword_hits"]["hit_rate"] == 1.0


def test_retrieval_signals_track_retrieval_before_write():
    dispatches = [
        ToolDispatch(
            trace_id="trace-1",
            tool_use_id="u1",
            tool_name="helpdesk_update_ticket",
            endpoint_url="http://localhost/helpdesk/tickets/update",
            timestamp="2026-03-20T08:02:00Z",
        )
    ]
    internal_tools = {
        "skill_read_count": 1,
        "session_search_calls": 0,
        "calls": [
            {
                "name": "skill_view",
                "timestamp": "2026-03-20T08:01:00Z",
                "args": {},
            }
        ],
    }

    signals = compute_retrieval_signals(
        dispatches=dispatches,
        artifact_before={"memory_file_exists": False, "user_file_exists": False},
        internal_tools=internal_tools,
        expected_signal="skill",
    )

    assert signals["used_expected_signal"] is True
    assert signals["retrieval_before_first_update"] is True
    assert signals["first_write_endpoint"].endswith("/helpdesk/tickets/update")


def test_skills_list_does_not_count_as_skill_read(tmp_path: Path):
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    (artifacts_dir / "session_latest.json").write_text(
        """
        {
          "messages": [
            {
              "tool_calls": [
                {"function": {"name": "skills_list", "arguments": "{}"}}
              ]
            }
          ]
        }
        """,
        encoding="utf-8",
    )

    internal_tools = snapshot_hermes_artifacts(artifacts_dir)["internal_tools"]

    assert internal_tools["skills_list_calls"] == 1
    assert internal_tools["skill_view_calls"] == 0
    assert internal_tools["skill_read_count"] == 0

    signals = compute_retrieval_signals(
        dispatches=[],
        artifact_before={"memory_file_exists": False, "user_file_exists": False},
        internal_tools=internal_tools,
        expected_signal="skill",
    )

    assert signals["used_expected_signal"] is False
