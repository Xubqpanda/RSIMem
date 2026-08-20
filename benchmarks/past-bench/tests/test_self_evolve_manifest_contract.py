from collections import Counter, defaultdict
from pathlib import Path

from past_bench.models.self_evolve import SelfEvolveSequenceDefinition


ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = sorted((ROOT / "configs" / "self_evolve_v2").glob("hermes_self_evolve_v2_*_only.yaml"))


def test_v2_family_manifests_have_ordered_family_shape():
    assert MANIFESTS

    for manifest in MANIFESTS:
        sequence = SelfEvolveSequenceDefinition.from_yaml(manifest)
        families = defaultdict(list)
        for episode in sequence.episodes:
            families[episode.family_id].append(episode)

        assert len(families) == 1, manifest

        family_id, episodes = next(iter(families.items()))
        buckets = Counter(episode.bucket for episode in episodes)
        assert buckets["baseline"] >= 1, family_id
        assert buckets["evaluation"] >= 1, family_id
        mechanisms = {episode.mechanism for episode in episodes}
        assert mechanisms.issubset({"memory", "skill", "session_search", "mixed"}), family_id
        assert len(mechanisms - {"mixed"}) == 1, family_id
        assert all(episode.latent_rule_id for episode in episodes), family_id

        evaluation_episodes = [episode for episode in episodes if episode.bucket == "evaluation"]
        assert all(episode.requires_fresh_session for episode in evaluation_episodes), family_id
        assert all(episode.evaluation_requires_retrieval for episode in evaluation_episodes), family_id
        assert all(episode.expected_persistence_signal for episode in episodes), family_id

        first_eval_index = next(i for i, episode in enumerate(episodes) if episode.bucket == "evaluation")
        first_control_index = next(
            (i for i, episode in enumerate(episodes) if episode.bucket == "control"),
            None,
        )
        if first_control_index is not None:
            assert first_eval_index < first_control_index, family_id

        for episode in evaluation_episodes:
            if episode.persistence_allowed:
                assert episode.history_mode == "from_anchor", episode.label
                assert episode.history_load_anchor, episode.label
