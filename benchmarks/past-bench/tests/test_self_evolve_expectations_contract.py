import json
from pathlib import Path

from past_bench.models.self_evolve import SelfEvolveSequenceDefinition


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "self_evolve_v2"
MANIFESTS = sorted(CONFIG_ROOT.glob("hermes_self_evolve_v2_*_only.yaml"))
RETRIEVAL_THRESHOLD_FIELDS = {
    "min_memory_reads",
    "min_memory_injections",
    "min_skill_reads",
    "min_session_search_calls",
    "min_retrieval_signals",
}


def test_v2_expectations_align_with_manifest_metadata():
    for manifest in MANIFESTS:
        sequence = SelfEvolveSequenceDefinition.from_yaml(manifest)

        for episode in sequence.episodes:
            task_dir = sequence.resolve_task_yaml(episode.task).parent
            expectations = json.loads((task_dir / "expectations.json").read_text(encoding="utf-8"))
            artifact_contract = expectations["artifact_contract"]
            retrieval_contract = expectations["retrieval_contract"]

            assert expectations["latent_rule_id"] == episode.latent_rule_id
            assert expectations["expected_mechanism"] == episode.expected_persistence_signal
            assert expectations["bucket_role"] == episode.bucket

            assert artifact_contract["type"] in {"memory", "skill", "session_search", "mixed"}
            rule_keywords = artifact_contract.get("require_rule_keywords", [])
            assert isinstance(rule_keywords, list)
            if artifact_contract["min_count_delta"] > 0:
                assert rule_keywords
            assert artifact_contract["min_count_delta"] >= 0

            assert isinstance(retrieval_contract["evaluation_only"], bool)
            if episode.bucket == "evaluation":
                assert retrieval_contract["evaluation_only"] is True
            if episode.evaluation_requires_retrieval:
                assert any(
                    expectations.get(field, 0) > 0 or retrieval_contract.get(field, 0) > 0
                    for field in RETRIEVAL_THRESHOLD_FIELDS
                ), episode.task
