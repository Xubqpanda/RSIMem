from __future__ import annotations

import json
from pathlib import Path
import sys


HERMES_PLUS_ROOT = Path(__file__).resolve().parents[1] / "agents" / "hermes-plus"
if str(HERMES_PLUS_ROOT) not in sys.path:
    sys.path.insert(0, str(HERMES_PLUS_ROOT))

from agent.evolve_controller import (  # noqa: E402
    build_action_gate_guidance,
    build_closeout_guidance,
    build_evolve_plan,
    build_turn_guidance,
    encode_memory_entry,
    parse_memory_entry,
    render_memory_block,
)
from tools.memory_tool import MEMORY_SCHEMA, memory_tool  # noqa: E402
from tools import skills_tool  # noqa: E402


class _FakeStore:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def add(self, target: str, content: str):
        self.calls.append((target, content))
        return {"success": True, "target": target, "entries": [content]}

    def replace(self, target: str, old_text: str, content: str):
        self.calls.append((target, content))
        return {"success": True, "target": target, "entries": [content], "old_text": old_text}

    def remove(self, target: str, old_text: str):
        return {"success": True, "target": target, "entries": [], "old_text": old_text}


def test_memory_tool_encodes_typed_metadata_for_add():
    store = _FakeStore()

    payload = json.loads(
        memory_tool(
            action="add",
            target="memory",
            content="Use TSV with four columns.",
            tags="family=SM01, type=format_rule",
            kind="format",
            scope="action_items",
            entity="summary_output",
            binding=True,
            supersedes="markdown_table_default",
            example="owner\tpriority\ttask\tdue_date",
            store=store,
        )
    )

    assert payload["success"] is True
    assert store.calls
    saved = store.calls[0][1]
    assert "[tags: family=SM01, type=format_rule]" in saved
    assert "kind=format" in saved
    assert "scope=action_items" in saved
    assert "binding=true" in saved
    assert "supersedes=markdown_table_default" in saved
    assert "Example:\nowner\tpriority\ttask\tdue_date" in saved


def test_memory_schema_exposes_typed_fields():
    properties = MEMORY_SCHEMA["parameters"]["properties"]

    for field in (
        "kind",
        "scope",
        "entity",
        "source",
        "confidence",
        "binding",
        "expires_at",
        "supersedes",
        "example",
    ):
        assert field in properties


def test_parse_and_render_memory_block_splits_binding_and_archival_entries():
    binding_entry = encode_memory_entry(
        content="Use TSV, not markdown tables.",
        tags="family=SM01",
        kind="format",
        scope="output",
        binding=True,
    )
    archival_entry = encode_memory_entry(
        content="Vendor X staging exception was approved through May 31, 2026.",
        tags="family=EP01",
        kind="episodic_case",
        entity="Vendor X",
    )
    expired_entry = encode_memory_entry(
        content="Temporary waiver for Maya only.",
        tags="family=SM06",
        kind="temporary_exception",
        entity="Maya",
        expires_at="2000-01-01T00:00:00+00:00",
    )

    parsed = parse_memory_entry(binding_entry)
    assert parsed.binding is True
    assert parsed.kind == "format"
    assert parsed.scope == "output"

    block = render_memory_block(
        "memory",
        [binding_entry, archival_entry, expired_entry],
        current=300,
        limit=2200,
    )

    assert "BINDING RULES" in block
    assert "ARCHIVAL CONTEXT" in block
    assert "Use TSV, not markdown tables." in block
    assert "Vendor X staging exception was approved through May 31, 2026." in block
    assert "Temporary waiver for Maya only." not in block
    assert "expired memory entr" in block


def test_turn_guidance_prioritizes_session_search_and_safe_updates():
    memory_entries = [
        encode_memory_entry(
            content="Use TSV with exactly four columns.",
            tags="family=SM01",
            kind="format",
            scope="output",
            entity="action-item summaries",
            binding=True,
            example="owner\tpriority\ttask\tdue_date",
        )
    ]

    guidance = build_turn_guidance(
        user_message=(
            "Task: PG05 Change Freeze Followup\n"
            "Before updating the freeze note, recover the latest packet and exact exception list from the previous session. "
            "Return the result in our action-item summary format."
        ),
        memory_entries=memory_entries,
        available_tools=["memory", "session_search", "skill_manage"],
    )

    assert "TURN-SPECIFIC EVOLVE PLAN" in guidance
    assert "session_search" in guidance
    assert "latest" in guidance.lower()
    assert "Replace or supersede stale state" in guidance
    assert "Use TSV with exactly four columns." in guidance


def test_action_gate_uses_ability_signals_not_task_ids():
    guidance = build_action_gate_guidance(
        user_message=(
            "Before sending the customer escalation note, recover the exact approved "
            "version and exception list from our last session."
        ),
        memory_entries=[],
        available_tools=["session_search", "skills_list", "skill_view"],
        skill_names=[],
        completed_tool_names=[],
    )

    assert "EVOLVE ACTION GATE" in guidance
    assert "session_search" in guidance
    assert "ability signals" in guidance


def test_action_gate_clears_after_required_retrieval():
    guidance = build_action_gate_guidance(
        user_message="Recover the exact approved version from our previous session.",
        memory_entries=[],
        available_tools=["session_search"],
        completed_tool_names=["session_search"],
    )

    assert guidance == ""


def test_evolve_plan_detects_generic_procedure_without_family_name():
    plan = build_evolve_plan(
        user_message="Revise the vendor escalation workflow and preserve the unchanged approval checks.",
        memory_entries=[],
        available_tools=["skills_list", "skill_view", "skill_manage"],
        skill_names=["vendor-escalation-workflow"],
    )

    assert plan.procedure_task is True
    assert plan.update_task is True
    assert plan.needs_skill_lookup is True
    assert plan.relevant_skills == ("vendor-escalation-workflow",)


def test_turn_guidance_surfaces_relevant_skills_for_procedure_tasks():
    guidance = build_turn_guidance(
        user_message=(
            "Task: PC02 Billing Incident Followup\n"
            "Patch the billing incident SOP before handling the next escalation workflow."
        ),
        memory_entries=[],
        available_tools=["skills_list", "skill_view", "skill_manage"],
        skill_names=["billing_incident_sop", "release_followup_lookup"],
    )

    assert "Existing skills worth checking first" in guidance
    assert "billing_incident_sop" in guidance
    assert "patch it" in guidance.lower()


def test_closeout_guidance_prefers_patching_existing_skill():
    memory_entries = [
        encode_memory_entry(
            content="Billing incident response workflow must confirm the tenant before escalation.",
            kind="procedure",
            scope="billing",
            entity="incident response",
        )
    ]

    guidance = build_closeout_guidance(
        user_message="Patch the billing incident SOP and migrate the escalation workflow.",
        memory_entries=memory_entries,
        available_tools=["skill_manage"],
        skill_names=["billing_incident_sop"],
    )

    assert "TASK CLOSEOUT EVOLVE CHECK" in guidance
    assert "billing_incident_sop" in guidance
    assert "patch the existing skill" in guidance.lower()


def test_closeout_guidance_skips_plain_exact_recall_tasks():
    guidance = build_closeout_guidance(
        user_message="Recover the latest approval ID from the previous session.",
        memory_entries=[],
        available_tools=["skill_manage"],
        skill_names=["release_followup_lookup"],
    )

    assert guidance == ""


def test_skills_list_query_ranks_lifecycle_metadata(monkeypatch, tmp_path):
    skill_dir = tmp_path / "vendor-escalation-workflow"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        """---
name: vendor-escalation-workflow
description: Reusable workflow for vendor escalation notes.
version: 2
scope: vendor escalations
supersedes: old-vendor-process
applies_when: vendor exception approval needs escalation
does_not_apply_when: one-off historical recall only
tags: [vendor, escalation]
---

# Vendor Escalation Workflow

Use the approved workflow for vendor escalations.
""",
        encoding="utf-8",
    )
    other_dir = tmp_path / "release-summary"
    other_dir.mkdir()
    (other_dir / "SKILL.md").write_text(
        """---
name: release-summary
description: Summarize release notes.
---

# Release Summary

Summarize release notes.
""",
        encoding="utf-8",
    )

    monkeypatch.setattr(skills_tool, "SKILLS_DIR", tmp_path)
    payload = json.loads(skills_tool.skills_list(query="vendor escalation approval workflow"))

    assert payload["success"] is True
    assert payload["count"] == 1
    skill = payload["skills"][0]
    assert skill["name"] == "vendor-escalation-workflow"
    assert skill["version"] == 2
    assert skill["scope"] == "vendor escalations"
    assert skill["match_score"] > 0
