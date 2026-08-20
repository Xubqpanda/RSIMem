from pathlib import Path


def test_cli_minimax_judge_fallback_uses_anthropic_endpoint() -> None:
    source = Path("src/past_bench/cli.py").read_text(encoding="utf-8")
    assert '("MINIMAX_API_KEY",    "https://api.minimaxi.com/anthropic",     "MiniMax-M2.7")' in source
