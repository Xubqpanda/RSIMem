"""EP01 notes tasks must declare endpoints for every notes_* tool they advertise."""

from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
EP01_ROOT = (
    REPO_ROOT
    / "self-evolve-tasks-v2"
    / "memory_ability"
    / "EP01_prior_case_recall"
)


def test_ep01_notes_tools_have_matching_tool_endpoints() -> None:
    for task_yaml in sorted(EP01_ROOT.glob("*/task.yaml")):
        data = yaml.safe_load(task_yaml.read_text())
        tool_names = [tool["name"] for tool in data.get("tools", []) if tool["name"].startswith("notes_")]
        endpoint_names = [ep["tool_name"] for ep in data.get("tool_endpoints", [])]

        assert tool_names, f"{task_yaml} should advertise notes_* tools"
        assert set(tool_names) == set(endpoint_names), (
            f"{task_yaml} has mismatched notes tools/endpoints: "
            f"tools={tool_names}, endpoints={endpoint_names}"
        )
