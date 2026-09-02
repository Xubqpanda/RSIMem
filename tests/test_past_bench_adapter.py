from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from rsimem.adapter_contracts import BenchmarkSplit
from rsimem.memory.family_matrix import PastFamilyMatrix
from rsimem.past_bench_adapter import PastBenchAdapter


def _fixture_root(tmp_path):
    matrix = PastFamilyMatrix.create_default()
    for family_id, task_name in (
        ("SM01_preference_adoption", "learn"),
        ("EP01_prior_case_recall", "episode"),
        ("PC01_sop_bootstrap_01", "bootstrap"),
    ):
        root = tmp_path / matrix.spec_for(family_id).task_root / task_name
        root.mkdir(parents=True)
        (root / "task.yaml").write_text(
            f"task_id: {family_id}-{task_name}\n"
            "prompt: private task text\n"
            "grader: hidden grader\n",
            encoding="utf-8",
        )
    return matrix


def test_past_adapter_enumerates_only_frozen_split_families(tmp_path) -> None:
    matrix = _fixture_root(tmp_path)
    adapter = PastBenchAdapter(
        tmp_path,
        matrix,
        split_family_ids={
            BenchmarkSplit.TRAIN: ("SM01_preference_adoption",),
            BenchmarkSplit.VALIDATION: ("EP01_prior_case_recall",),
            BenchmarkSplit.FINAL: ("PC01_sop_bootstrap_01",),
        },
    )
    train = adapter.enumerate_cases(BenchmarkSplit.TRAIN)
    validation = adapter.enumerate_cases(BenchmarkSplit.VALIDATION)
    final = adapter.enumerate_cases(BenchmarkSplit.FINAL)
    assert len(train) == len(validation) == len(final) == 1
    assert train[0].split is BenchmarkSplit.TRAIN
    assert train[0].case_id != validation[0].case_id
    assert train[0].task_template_id != validation[0].task_template_id

    result = adapter.reset(train[0])
    assert result.status.value == "supported"
    step_result, event = adapter.step(train[0])
    assert step_result.status.value == "supported"
    assert event.attributes == {}
    assert "grader" not in str(event)
    with pytest.raises(ValueError, match="not registered"):
        adapter.reset(replace(train[0], seed="past-seed.tampered"))


def test_past_adapter_requires_explicit_final_score_callback(tmp_path) -> None:
    matrix = _fixture_root(tmp_path)
    kwargs = {
        "split_family_ids": {
            BenchmarkSplit.TRAIN: ("SM01_preference_adoption",),
            BenchmarkSplit.VALIDATION: ("EP01_prior_case_recall",),
            BenchmarkSplit.FINAL: ("PC01_sop_bootstrap_01",),
        },
    }
    adapter = PastBenchAdapter(tmp_path, matrix, **kwargs)
    request = adapter.enumerate_cases(BenchmarkSplit.FINAL)[0]
    with pytest.raises(ValueError, match="final score provider"):
        adapter.evaluate_final(request)

    digest = hashlib.sha256(b"score").hexdigest()
    scored = PastBenchAdapter(
        tmp_path,
        matrix,
        final_score_digest_provider=lambda _: digest,
        **kwargs,
    )
    record = scored.evaluate_final(request)
    assert record.score_digest == digest
    assert record.evidence_plane.value == "final_evaluation"


def test_past_adapter_rejects_incomplete_split_or_unknown_family(tmp_path) -> None:
    matrix = _fixture_root(tmp_path)
    with pytest.raises(ValueError, match="cover train"):
        PastBenchAdapter(
            tmp_path,
            matrix,
            split_family_ids={BenchmarkSplit.TRAIN: ("SM01_preference_adoption",)},
        )
    with pytest.raises(ValueError, match="frozen PAST"):
        PastBenchAdapter(
            tmp_path,
            matrix,
            split_family_ids={
                BenchmarkSplit.TRAIN: ("SM99_unknown",),
                BenchmarkSplit.VALIDATION: ("EP01_prior_case_recall",),
                BenchmarkSplit.FINAL: ("PC01_sop_bootstrap_01",),
            },
        )
