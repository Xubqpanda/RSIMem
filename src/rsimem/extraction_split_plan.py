"""Immutable train/validation/final split assignments for extraction trials."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Mapping


SPLIT_PLAN_SCHEMA_VERSION = 1
SPLIT_PLAN_SCHEMA = "extraction-split-plan-v1"
_DIGEST_LENGTH = 64


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _id(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or any(
        char.isspace() for char in value
    ):
        raise ValueError(f"{name} must be a stable identifier")
    return value


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != _DIGEST_LENGTH or any(
        char not in "0123456789abcdef" for char in value
    ):
        raise ValueError(f"{name} must be sha256")
    return value


class ExtractionSplitRole(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    FINAL_TEST = "final_test"


@dataclass(frozen=True, slots=True)
class ExtractionSplitAssignment:
    role: ExtractionSplitRole
    family_id: str
    task_template_group_id: str
    task_manifest_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", ExtractionSplitRole(self.role))
        _id(self.family_id, "split family ID")
        _id(self.task_template_group_id, "split task-template group ID")
        _sha(self.task_manifest_digest, "split task manifest digest")

    def payload(self) -> dict[str, str]:
        return {
            "role": self.role.value,
            "family_id": self.family_id,
            "task_template_group_id": self.task_template_group_id,
            "task_manifest_digest": self.task_manifest_digest,
        }


@dataclass(frozen=True, slots=True)
class ExtractionSplitPlan:
    plan_id: str
    assignments: tuple[ExtractionSplitAssignment, ...]
    schema: str = SPLIT_PLAN_SCHEMA
    schema_version: int = SPLIT_PLAN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SPLIT_PLAN_SCHEMA_VERSION or self.schema != SPLIT_PLAN_SCHEMA:
            raise ValueError("unsupported extraction split plan")
        _id(self.plan_id, "split plan ID")
        if not self.assignments:
            raise ValueError("extraction split plan requires assignments")
        if {value.role for value in self.assignments} != set(ExtractionSplitRole):
            raise ValueError("split plan must contain train, validation, and final-test roles")
        groups: set[tuple[str, str]] = set()
        manifests: dict[str, ExtractionSplitRole] = {}
        for assignment in self.assignments:
            key = (assignment.family_id, assignment.task_template_group_id)
            if key in groups:
                raise ValueError("split plan contains a duplicate family/template group")
            groups.add(key)
            previous = manifests.get(assignment.task_manifest_digest)
            if previous is not None and previous != assignment.role:
                raise ValueError("task manifest digest cannot cross split roles")
            manifests[assignment.task_manifest_digest] = assignment.role
        expected = f"split-plan.{_digest(self.identity_payload())[:40]}"
        if self.plan_id != expected:
            raise ValueError("extraction split plan ID mismatch")

    @classmethod
    def create(cls, assignments: tuple[ExtractionSplitAssignment, ...]) -> "ExtractionSplitPlan":
        values = {
            "assignments": tuple(sorted(assignments, key=lambda value: value.role.value)),
            "schema": SPLIT_PLAN_SCHEMA,
            "schema_version": SPLIT_PLAN_SCHEMA_VERSION,
        }
        identity = {
            "schema_version": values["schema_version"],
            "schema": values["schema"],
            "assignments": [value.payload() for value in values["assignments"]],
        }
        return cls(plan_id=f"split-plan.{_digest(identity)[:40]}", **values)

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "schema": self.schema,
            "assignments": [value.payload() for value in self.assignments],
        }

    def payload(self) -> dict[str, object]:
        return {"plan_id": self.plan_id, **self.identity_payload()}

    def assignment_for(
        self,
        *,
        role: ExtractionSplitRole,
        family_id: str,
        task_template_group_id: str,
        task_manifest_digest: str,
    ) -> ExtractionSplitAssignment:
        expected = ExtractionSplitAssignment(
            role,
            family_id,
            task_template_group_id,
            task_manifest_digest,
        )
        matches = tuple(value for value in self.assignments if value.role == expected.role)
        if matches != (expected,):
            raise ValueError("current extraction batch does not match split plan")
        return expected

    @classmethod
    def from_payload(cls, value: object) -> "ExtractionSplitPlan":
        if not isinstance(value, Mapping) or set(value) != {
            "plan_id", "schema_version", "schema", "assignments",
        } or not isinstance(value["assignments"], list):
            raise ValueError("malformed extraction split plan")
        try:
            assignments = tuple(
                ExtractionSplitAssignment(
                    item["role"],
                    item["family_id"],
                    item["task_template_group_id"],
                    item["task_manifest_digest"],
                )
                for item in value["assignments"]
            )
            return cls(
                plan_id=value["plan_id"],
                assignments=assignments,
                schema=value["schema"],
                schema_version=value["schema_version"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("malformed extraction split plan") from exc


def load_extraction_split_plan(path: Path) -> ExtractionSplitPlan:
    try:
        value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("extraction split plan cannot be read") from exc
    return ExtractionSplitPlan.from_payload(value)


def write_extraction_split_plan(path: Path, plan: ExtractionSplitPlan) -> bool:
    serialized = _canonical(plan.payload()) + "\n"
    target = path.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.read_text(encoding="utf-8") != serialized:
            raise ValueError("extraction split plan conflicts with existing content")
        return False
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        return True
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


__all__ = [
    "ExtractionSplitAssignment",
    "ExtractionSplitPlan",
    "ExtractionSplitRole",
    "SPLIT_PLAN_SCHEMA",
    "SPLIT_PLAN_SCHEMA_VERSION",
    "load_extraction_split_plan",
    "write_extraction_split_plan",
]
