import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_self_evolve_kb_fixtures_match_kb_search_schema() -> None:
    required = {"article_id", "title", "category", "content", "tags", "last_updated", "views"}
    fixture_root = ROOT / "self-evolve-tasks-v2" / "_shared" / "fixtures"

    for path in fixture_root.glob("*_kb/*.json"):
        articles = json.loads(path.read_text())
        assert isinstance(articles, list), path
        for article in articles:
            assert required.issubset(article), path
