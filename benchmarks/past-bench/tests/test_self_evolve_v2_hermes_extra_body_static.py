"""Static guards for Hermes self-evolve runtime config wiring."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SELF_EVOLVE_RUNNER = REPO_ROOT / "src" / "past_bench" / "runner" / "self_evolve.py"


def test_build_hermes_extra_body_carries_session_search_flag() -> None:
    source = SELF_EVOLVE_RUNNER.read_text()

    assert '"session_search_enabled": session_search_enabled' in source
    assert 'if session_search_enabled:' in source
    assert 'enabled_toolsets.append("session_search")' in source
    assert '"evidence_path": str(artifacts_dir / "rsimem_memory_events.jsonl")' in source
