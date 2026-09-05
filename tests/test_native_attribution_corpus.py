from __future__ import annotations

import json

import pytest

from rsimem.native_attribution import attribute_native_observation
from rsimem.native_attribution_corpus import NativeAttributionCorpus, NativeAttributionCorpusStore
from rsimem.native_observation import extract_native_observations
from test_native_execution_audit import _fixture


def _corpus(tmp_path):
    run = _fixture(tmp_path)
    observations = extract_native_observations(run=run, output_root=tmp_path)
    candidates = tuple(attribute_native_observation(value, None) for value in observations)
    return NativeAttributionCorpus.create(
        protocol_id="native-attribution-repair-v1.fixture",
        accepted_run_ids=(run.run_id,),
        observations=observations,
        candidates=candidates,
        excluded_runs=({"run_id": "native-run.excluded", "reason": "usage_incomplete"},),
    )


def test_corpus_is_deterministic_content_free_and_append_once(tmp_path) -> None:
    corpus = _corpus(tmp_path)
    assert corpus.actionable_count == 0
    assert corpus.unresolved_count == len(corpus.candidates)
    serialized = json.dumps(corpus.payload())
    assert "final_response_text" not in serialized
    store = NativeAttributionCorpusStore(tmp_path / "corpus.json")
    assert store.put(corpus) is True
    assert store.put(corpus) is False


def test_corpus_rejects_forbidden_evaluation_fields(tmp_path) -> None:
    corpus = _corpus(tmp_path)
    with pytest.raises(ValueError, match="forbidden field"):
        NativeAttributionCorpus.create(
            protocol_id=corpus.protocol_id,
            accepted_run_ids=corpus.accepted_run_ids,
            observations=corpus.observations,
            candidates=corpus.candidates,
            excluded_runs=({"run_id": "native-run.bad", "task_score": 0.1},),
        )


def test_corpus_rejects_observation_candidate_mismatch(tmp_path) -> None:
    corpus = _corpus(tmp_path)
    with pytest.raises(ValueError, match="coverage mismatch"):
        NativeAttributionCorpus(
            corpus_id=corpus.corpus_id,
            protocol_id=corpus.protocol_id,
            accepted_run_ids=corpus.accepted_run_ids,
            observations=corpus.observations,
            candidates=corpus.candidates[:-1],
            excluded_runs=corpus.excluded_runs,
        )
