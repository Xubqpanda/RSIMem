from pathlib import Path

from past_bench.runtime.manager import RuntimeSessionManager
from past_bench.runtime.protocol import BootstrapRequest


def test_runtime_manager_bootstrap_marks_cache(tmp_path: Path):
    registry_path = tmp_path / "agents.yaml"
    cache_dir = tmp_path / "cache"
    registry_path.write_text(
        """
agents:
  demo:
    adapter: openai_compat_chat
    install_policy: bootstrap_once
    bootstrap_commands:
      - "printf ready"
""".strip(),
        encoding="utf-8",
    )

    manager = RuntimeSessionManager(registry_path=registry_path, cache_dir=cache_dir)
    first = manager.bootstrap(BootstrapRequest(agent_name="demo"))
    second = manager.bootstrap(BootstrapRequest(agent_name="demo"))

    assert first.installed is True
    assert first.already_present is False
    assert first.commands_run == ["printf ready"]
    assert second.already_present is True
    assert (cache_dir / "demo.ready").exists()
