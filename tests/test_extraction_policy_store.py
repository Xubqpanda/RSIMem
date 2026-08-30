from __future__ import annotations

import json
from dataclasses import replace

import pytest

from rsimem.lifecycle import RawResourceUsage
from rsimem.memory.contracts import MemoryKind
from rsimem.memory.extraction_policy_artifact import (
    ExtractionGenerationProvenance,
    ExtractionPolicyRule,
    ExtractionPolicySpec,
    ExtractionPromptPolicyArtifact,
    ExtractionRuleEdit,
    ExtractionRuleEditAction,
)
from rsimem.memory.extraction_policy_store import (
    ExtractionPolicyState,
    JsonExtractionPolicyStore,
)
from rsimem.memory.prompt_components import PromptPolicyStage, PromptSlotDescriptor


def _slot(*, wrapper: str = "3" * 64) -> PromptSlotDescriptor:
    return PromptSlotDescriptor(
        "fixture.semantic.extraction",
        MemoryKind.SEMANTIC,
        PromptPolicyStage.EXTRACTION,
        "1" * 64,
        "2" * 64,
        wrapper,
        "fixture-model-v1",
        "fixture-adapter-v1",
        ("policy_body", "source_messages"),
    )


def _root(slot: PromptSlotDescriptor | None = None) -> ExtractionPromptPolicyArtifact:
    return ExtractionPromptPolicyArtifact.create_root(
        slot=slot or _slot(),
        policy_version="root-v1",
        spec=ExtractionPolicySpec((
            ExtractionPolicyRule("scope", "Extract durable facts."),
            ExtractionPolicyRule("safety", "Exclude unsafe claims.", protected=True),
        )),
        max_body_chars=1_000,
        source_provenance="fixture-root-v1",
    )


def _child(
    parent: ExtractionPromptPolicyArtifact,
    *,
    version: str = "candidate-v2",
    text: str = "Extract durable facts and preferences.",
) -> ExtractionPromptPolicyArtifact:
    return ExtractionPromptPolicyArtifact.create_child(
        parent=parent,
        policy_version=version,
        edits=(ExtractionRuleEdit(
            f"edit.{version}",
            ExtractionRuleEditAction.REPLACE,
            "scope",
            ExtractionPolicyRule("scope", text),
        ),),
        generation_provenance=ExtractionGenerationProvenance(
            "optimizer-v1",
            "4" * 64,
            "corpus-v1",
            "cutoff-v1",
            "5" * 64,
            "6" * 64,
            RawResourceUsage(input_tokens=10, output_tokens=2, model_requests=1),
        ),
    )


def test_store_persists_root_child_active_and_restart_exactly(tmp_path) -> None:
    root = _root()
    path = tmp_path / "extraction-policies.json"
    store = JsonExtractionPolicyStore(path, trusted_root=root, slot=_slot())
    initial = store.initialize()
    assert initial.root == root
    assert initial.active is None
    child = _child(root)
    record, created = store.register(child)
    assert created is True
    assert record.state == ExtractionPolicyState.PROPOSAL
    assert store.register(child)[1] is False
    active, changed = store.transition(
        child.artifact_id,
        to_state=ExtractionPolicyState.ACTIVE,
        transition_id="transition.activate-v2",
        reason_code="validation_passed",
    )
    assert changed is True
    assert active.state == ExtractionPolicyState.ACTIVE

    restarted = JsonExtractionPolicyStore(path, trusted_root=root, slot=_slot())
    assert restarted.snapshot().active == child
    assert restarted.active_or_root().compiled_body == child.compiled_body
    assert restarted.transition(
        child.artifact_id,
        to_state=ExtractionPolicyState.ACTIVE,
        transition_id="transition.activate-v2",
        reason_code="validation_passed",
    )[1] is False


def test_store_rejects_unknown_inactive_parent_and_multiple_active(tmp_path) -> None:
    root = _root()
    store = JsonExtractionPolicyStore(
        tmp_path / "store.json",
        trusted_root=root,
        slot=_slot(),
    )
    first = _child(root, version="candidate-a", text="Extract durable A facts.")
    second = _child(root, version="candidate-b", text="Extract durable B facts.")
    store.register(first)
    store.register(second)
    store.transition(
        first.artifact_id,
        to_state=ExtractionPolicyState.ACTIVE,
        transition_id="transition.activate-a",
        reason_code="validation_passed",
    )
    with pytest.raises(ValueError, match="already active"):
        store.transition(
            second.artifact_id,
            to_state=ExtractionPolicyState.ACTIVE,
            transition_id="transition.activate-b",
            reason_code="validation_passed",
        )

    foreign_root = _root(replace(_slot(), slot_id="fixture.semantic.foreign"))
    foreign_child = _child(foreign_root, version="foreign-v2")
    with pytest.raises(ValueError, match="unknown or inactive"):
        store.register(foreign_child)


def test_corruption_cycle_wrapper_and_numeric_store_fail_closed_to_root(tmp_path) -> None:
    root = _root()
    path = tmp_path / "store.json"
    store = JsonExtractionPolicyStore(path, trusted_root=root, slot=_slot())
    child = _child(root)
    store.register(child)
    store.transition(
        child.artifact_id,
        to_state=ExtractionPolicyState.ACTIVE,
        transition_id="transition.activate",
        reason_code="validation_passed",
    )
    valid = json.loads(path.read_text(encoding="utf-8"))

    variants = []
    cycle = json.loads(json.dumps(valid))
    cycle["artifacts"][child.artifact_id]["parent_artifact_id"] = child.artifact_id
    variants.append(cycle)
    multiple = json.loads(json.dumps(valid))
    multiple["records"][root.artifact_id]["state"] = "active"
    variants.append(multiple)
    wrong_wrapper = json.loads(json.dumps(valid))
    wrong_wrapper["frozen_wrapper_digest"] = "9" * 64
    variants.append(wrong_wrapper)
    numeric = {"schema_version": 2, "artifacts": {}}
    variants.append(numeric)
    tampered = json.loads(json.dumps(valid))
    tampered["artifacts"][child.artifact_id]["compiled_body"] += " changed"
    variants.append(tampered)

    for payload in variants:
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError):
            store.snapshot()
        assert store.active_or_root() == root


def test_store_rejects_wrong_runtime_slot_before_writing(tmp_path) -> None:
    root = _root()
    with pytest.raises(ValueError, match="runtime slot"):
        JsonExtractionPolicyStore(
            tmp_path / "store.json",
            trusted_root=root,
            slot=_slot(wrapper="9" * 64),
        )
    assert not (tmp_path / "store.json").exists()


def test_store_rejects_symlinked_policy_path_and_lock(tmp_path) -> None:
    root = _root()
    target = tmp_path / "target"
    target.write_text("", encoding="utf-8")
    path = tmp_path / "store.json"
    path.symlink_to(target)
    store = JsonExtractionPolicyStore(path, trusted_root=root, slot=_slot())
    with pytest.raises(ValueError, match="symlink"):
        store.snapshot()

    path.unlink()
    lock = path.with_suffix(path.suffix + ".lock")
    lock_target = tmp_path / "lock-target"
    lock_target.write_text("", encoding="utf-8")
    lock.symlink_to(lock_target)
    with pytest.raises(ValueError, match="lock.*symlink"):
        store.initialize()
