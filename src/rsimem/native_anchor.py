"""Immutable native post-learn anchors and isolated repair branches."""

from __future__ import annotations

import hashlib
import fcntl
import json
import os
import re
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path


ANCHOR_SCHEMA = "rsimem-native-anchor-v1"
BRANCH_SCHEMA = "rsimem-repair-branch-v1"
_REPAIR_AXES = {"formation", "persistence", "maintenance", "retrieval", "application"}
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:+/-]{0,255}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _validate_identifier(value: object, name: str) -> None:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{name} must be a stable identifier")


def _validate_digest(value: object, name: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a sha256 digest")


def state_tree_digest(root: Path) -> str:
    root = Path(root).expanduser().resolve()
    if root.is_symlink() or not root.is_dir():
        raise ValueError("state tree is missing or symlinked")
    entries: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("state tree cannot contain symlinks")
        if path.is_file():
            data = path.read_bytes()
            entries.append({
                "path": path.relative_to(root).as_posix(),
                "digest": hashlib.sha256(data).hexdigest(),
                "size": len(data),
            })
    return _digest(entries)


def _set_tree_read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_file():
            path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        elif path.is_dir():
            path.chmod(stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
    root.chmod(stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)


def _set_tree_owner_writable(root: Path) -> None:
    root.chmod(stat.S_IRWXU)
    for path in root.rglob("*"):
        if path.is_dir():
            path.chmod(stat.S_IRWXU)
        elif path.is_file():
            path.chmod(stat.S_IRUSR | stat.S_IWUSR)


@dataclass(frozen=True, slots=True)
class NativeAnchorReceipt:
    anchor_id: str
    source_run_id: str
    family_id: str
    replicate: int
    boundary_id: str
    source_state_digest: str
    anchor_tree_digest: str
    anchor_directory: str
    schema: str = ANCHOR_SCHEMA

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema, "source_run_id": self.source_run_id,
            "family_id": self.family_id, "replicate": self.replicate,
            "boundary_id": self.boundary_id,
            "source_state_digest": self.source_state_digest,
            "anchor_tree_digest": self.anchor_tree_digest,
            "anchor_directory": self.anchor_directory,
        }

    def __post_init__(self) -> None:
        if self.schema != ANCHOR_SCHEMA or self.replicate < 1:
            raise ValueError("invalid native anchor receipt")
        for value, name in (
            (self.source_run_id, "source run ID"), (self.family_id, "family ID"),
            (self.boundary_id, "boundary ID"),
        ):
            _validate_identifier(value, name)
        _validate_digest(self.source_state_digest, "source state digest")
        _validate_digest(self.anchor_tree_digest, "anchor tree digest")
        if (
            not self.anchor_directory
            or Path(self.anchor_directory).is_absolute()
            or len(Path(self.anchor_directory).parts) != 1
        ):
            raise ValueError("anchor directory must be one relative path component")
        expected = "native-anchor." + _digest(self.identity_payload())[:40]
        if self.anchor_id != expected:
            raise ValueError("native anchor receipt identity mismatch")

    def payload(self) -> dict[str, object]:
        return {"anchor_id": self.anchor_id, **self.identity_payload()}


@dataclass(frozen=True, slots=True)
class RepairBranchReceipt:
    branch_id: str
    anchor_id: str
    base_anchor_digest: str
    repair_axis: str
    branch_directory: str
    initial_branch_digest: str
    schema: str = BRANCH_SCHEMA

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema, "anchor_id": self.anchor_id,
            "base_anchor_digest": self.base_anchor_digest,
            "repair_axis": self.repair_axis,
            "branch_directory": self.branch_directory,
            "initial_branch_digest": self.initial_branch_digest,
        }

    def __post_init__(self) -> None:
        if self.schema != BRANCH_SCHEMA or self.repair_axis not in _REPAIR_AXES:
            raise ValueError("invalid repair branch receipt")
        _validate_identifier(self.anchor_id, "anchor ID")
        _validate_digest(self.base_anchor_digest, "base anchor digest")
        _validate_digest(self.initial_branch_digest, "initial branch digest")
        if (
            not self.branch_directory
            or Path(self.branch_directory).is_absolute()
            or len(Path(self.branch_directory).parts) != 1
        ):
            raise ValueError("branch directory must be one relative path component")
        if self.branch_id != "repair-branch." + _digest(self.identity_payload())[:40]:
            raise ValueError("repair branch receipt identity mismatch")

    def payload(self) -> dict[str, object]:
        return {"branch_id": self.branch_id, **self.identity_payload()}


class NativeAnchorStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root).expanduser().resolve()

    def freeze(
        self, *, source_state: Path, source_run_id: str, family_id: str,
        replicate: int, boundary_id: str,
    ) -> NativeAnchorReceipt:
        source = Path(source_state).expanduser().resolve()
        source_digest = state_tree_digest(source)
        identity = {
            "source_run_id": source_run_id, "family_id": family_id,
            "replicate": replicate, "boundary_id": boundary_id,
            "source_state_digest": source_digest,
        }
        directory_name = "anchor." + _digest(identity)[:40]
        target = self.root / directory_name
        receipt_path = self.root / f"{directory_name}.json"
        self.root.mkdir(parents=True, exist_ok=True)
        lock_path = self.root / f"{directory_name}.lock"
        with lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if target.exists() or receipt_path.exists():
                receipt = self.load(receipt_path)
                self.verify(receipt)
                if (
                    receipt.source_run_id != source_run_id or receipt.family_id != family_id
                    or receipt.replicate != replicate or receipt.boundary_id != boundary_id
                    or receipt.source_state_digest != source_digest
                ):
                    raise ValueError("native anchor identity conflict")
                return receipt
            temporary = self.root / f".{directory_name}.tmp-{os.getpid()}"
            shutil.copytree(source, temporary, symlinks=False)
            anchor_digest = state_tree_digest(temporary)
            receipt_values = {
                "source_run_id": source_run_id, "family_id": family_id,
                "replicate": replicate, "boundary_id": boundary_id,
                "source_state_digest": source_digest, "anchor_tree_digest": anchor_digest,
                "anchor_directory": directory_name,
            }
            receipt = NativeAnchorReceipt(
                anchor_id="native-anchor." + _digest({"schema": ANCHOR_SCHEMA, **receipt_values})[:40],
                **receipt_values,
            )
            _set_tree_read_only(temporary)
            temporary.replace(target)
            temporary_receipt = receipt_path.with_name(f".{receipt_path.name}.tmp-{os.getpid()}")
            temporary_receipt.write_text(_canonical(receipt.payload()) + "\n", encoding="utf-8")
            temporary_receipt.replace(receipt_path)
            receipt_path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
            return receipt

    def load(self, receipt_path: Path) -> NativeAnchorReceipt:
        try:
            value = json.loads(Path(receipt_path).read_text(encoding="utf-8"))
            return NativeAnchorReceipt(
                anchor_id=value["anchor_id"], source_run_id=value["source_run_id"],
                family_id=value["family_id"], replicate=value["replicate"],
                boundary_id=value["boundary_id"], source_state_digest=value["source_state_digest"],
                anchor_tree_digest=value["anchor_tree_digest"],
                anchor_directory=value["anchor_directory"], schema=value["schema"],
            )
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError("malformed native anchor receipt") from exc

    def verify(self, receipt: NativeAnchorReceipt) -> Path:
        target = self.root / receipt.anchor_directory
        if not target.resolve().is_relative_to(self.root) or state_tree_digest(target) != receipt.anchor_tree_digest:
            raise ValueError("native anchor content digest mismatch")
        return target

    def materialize_branch(
        self, *, receipt: NativeAnchorReceipt, repair_axis: str, branch_root: Path,
    ) -> RepairBranchReceipt:
        source = self.verify(receipt)
        target = Path(branch_root).expanduser().resolve()
        if target.exists():
            raise ValueError("repair branch directory already exists")
        shutil.copytree(source, target, symlinks=False)
        _set_tree_owner_writable(target)
        initial_digest = state_tree_digest(target)
        values = {
            "anchor_id": receipt.anchor_id,
            "base_anchor_digest": receipt.anchor_tree_digest,
            "repair_axis": repair_axis,
            "branch_directory": target.name,
            "initial_branch_digest": initial_digest,
        }
        branch = RepairBranchReceipt(
            branch_id="repair-branch." + _digest({"schema": BRANCH_SCHEMA, **values})[:40],
            **values,
        )
        (target.parent / f"{target.name}.receipt.json").write_text(
            _canonical(branch.payload()) + "\n", encoding="utf-8"
        )
        return branch


__all__ = [
    "NativeAnchorReceipt", "NativeAnchorStore", "RepairBranchReceipt",
    "state_tree_digest",
]
