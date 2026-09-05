from __future__ import annotations

from pathlib import Path

import pytest

from rsimem.native_anchor import NativeAnchorStore, state_tree_digest


def _state(path: Path) -> Path:
    (path / "memories").mkdir(parents=True)
    (path / "memories" / "MEMORY.md").write_text("settled state", encoding="utf-8")
    (path / "state.db").write_bytes(b"sqlite-fixture")
    return path


def test_anchor_is_immutable_snapshot_not_alias_of_source(tmp_path) -> None:
    source = _state(tmp_path / "source")
    store = NativeAnchorStore(tmp_path / "anchors")
    receipt = store.freeze(
        source_state=source, source_run_id="native-run.fixture",
        family_id="SM01_preference_adoption", replicate=1,
        boundary_id="post-learn.fixture",
    )
    anchor = store.verify(receipt)
    (source / "memories" / "MEMORY.md").write_text("later mutation", encoding="utf-8")
    assert state_tree_digest(anchor) == receipt.anchor_tree_digest
    assert state_tree_digest(source) != receipt.anchor_tree_digest
    assert not (anchor / "memories" / "MEMORY.md").stat().st_mode & 0o200


def test_anchor_tampering_fails_closed(tmp_path) -> None:
    store = NativeAnchorStore(tmp_path / "anchors")
    receipt = store.freeze(
        source_state=_state(tmp_path / "source"), source_run_id="native-run.fixture",
        family_id="SM01_preference_adoption", replicate=1,
        boundary_id="post-learn.fixture",
    )
    anchor_file = store.root / receipt.anchor_directory / "state.db"
    anchor_file.chmod(0o600)
    anchor_file.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="content digest mismatch"):
        store.verify(receipt)


def test_repair_branches_share_base_but_do_not_share_mutations(tmp_path) -> None:
    store = NativeAnchorStore(tmp_path / "anchors")
    receipt = store.freeze(
        source_state=_state(tmp_path / "source"), source_run_id="native-run.fixture",
        family_id="SM01_preference_adoption", replicate=1,
        boundary_id="post-learn.fixture",
    )
    formation_root = tmp_path / "branches" / "formation"
    retrieval_root = tmp_path / "branches" / "retrieval"
    formation = store.materialize_branch(
        receipt=receipt, repair_axis="formation", branch_root=formation_root
    )
    retrieval = store.materialize_branch(
        receipt=receipt, repair_axis="retrieval", branch_root=retrieval_root
    )
    assert formation.base_anchor_digest == retrieval.base_anchor_digest == receipt.anchor_tree_digest
    assert formation.initial_branch_digest == retrieval.initial_branch_digest
    (formation_root / "memories" / "MEMORY.md").write_text("formation repair", encoding="utf-8")
    assert (retrieval_root / "memories" / "MEMORY.md").read_text(encoding="utf-8") == "settled state"
    assert state_tree_digest(store.verify(receipt)) == receipt.anchor_tree_digest


def test_unknown_repair_axis_is_rejected(tmp_path) -> None:
    store = NativeAnchorStore(tmp_path / "anchors")
    receipt = store.freeze(
        source_state=_state(tmp_path / "source"), source_run_id="native-run.fixture",
        family_id="SM01_preference_adoption", replicate=1,
        boundary_id="post-learn.fixture",
    )
    with pytest.raises(ValueError, match="invalid repair branch"):
        store.materialize_branch(
            receipt=receipt, repair_axis="all_axes", branch_root=tmp_path / "bad"
        )


def test_repeated_freeze_returns_same_verified_receipt(tmp_path) -> None:
    source = _state(tmp_path / "source")
    store = NativeAnchorStore(tmp_path / "anchors")
    kwargs = {
        "source_state": source,
        "source_run_id": "native-run.fixture",
        "family_id": "SM01_preference_adoption",
        "replicate": 1,
        "boundary_id": "post-learn.fixture",
    }
    first = store.freeze(**kwargs)
    second = store.freeze(**kwargs)
    assert first == second
    assert store.verify(second).name == second.anchor_directory
