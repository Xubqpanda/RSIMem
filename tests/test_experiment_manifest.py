from __future__ import annotations

import json
from pathlib import Path

import pytest

from rsimem.experiment_manifest import (
    execution_order,
    initialize_batch_manifest,
    record_attempt,
)


def test_execution_order_rotates_modes_across_replicates() -> None:
    assert execution_order(1) == (
        "native",
        "native+ledger",
        "native+adapter+ledger",
    )
    assert execution_order(2) == (
        "native+ledger",
        "native+adapter+ledger",
        "native",
    )
    assert execution_order(3) == (
        "native+adapter+ledger",
        "native",
        "native+ledger",
    )
    assert execution_order(4) == execution_order(1)


def test_manifest_records_revisions_schedule_and_actual_attempts(tmp_path: Path) -> None:
    path = tmp_path / "batch_manifest.json"
    initialize_batch_manifest(
        path,
        replicates=2,
        rsimem_commit="rsimem-head",
        past_bench_commit="past-last-change",
        past_bench_tree="past-tree",
        working_tree_dirty=False,
    )
    record_attempt(
        path,
        replicate=2,
        ordinal=1,
        mode="native+ledger",
        run_name="r02_native_ledger",
        status="running",
    )
    record_attempt(
        path,
        replicate=2,
        ordinal=1,
        mode="native+ledger",
        run_name="r02_native_ledger",
        status="completed",
    )

    value = json.loads(path.read_text(encoding="utf-8"))
    assert value["rsimemCommit"] == "rsimem-head"
    assert value["pastBenchCommit"] == "past-last-change"
    assert value["pastBenchTree"] == "past-tree"
    assert value["executionOrderByReplicate"]["2"][0] == "native+ledger"
    assert value["attempts"] == [{
        "failureStage": None,
        "mode": "native+ledger",
        "ordinal": 1,
        "replicate": 2,
        "runName": "r02_native_ledger",
        "status": "completed",
    }]

    with pytest.raises(ValueError, match="scheduled order"):
        record_attempt(
            path,
            replicate=2,
            ordinal=2,
            mode="native",
            run_name="wrong-order",
            status="running",
        )
