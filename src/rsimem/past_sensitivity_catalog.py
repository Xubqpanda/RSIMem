"""Audit-only availability catalog for PAST Stage 3 sensitivity conditions.

This module reads only family-level control and episode identities.  It never
opens task prompts, expectations, graders, or oracle content.  The resulting
catalog is launcher planning evidence and must not be supplied to a method.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml

from .research_protocol import SensitivityCondition
from .sensitivity import SensitivityCase, SensitivityMatrix


PAST_SENSITIVITY_CATALOG_SCHEMA = "rsimem-past-sensitivity-catalog-v1"
PAST_SENSITIVITY_CATALOG_SCHEMA_VERSION = 1
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:+/-]{0,255}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _id(value: object, name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{name} must be a stable identifier")
    return value


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{name} must be a sha256 digest")
    return value


def _control_episode(
    episodes: tuple[str, ...],
    control: str,
    *,
    allow_semantic_legacy_fallback: bool = False,
) -> str | None:
    matches = tuple(value for value in episodes if control in value.lower())
    if len(matches) == 1:
        return matches[0]
    if control != "wrong_mechanism" or not allow_semantic_legacy_fallback:
        return None
    # Semantic v2 families declare the wrong-mechanism slot in control_set but
    # retain historical names such as `control_*_control`.  It is safe to use
    # that identity only when it is the sole control left after the explicit
    # no-persistence and shortcut slots are removed.
    fallback = tuple(
        value for value in episodes
        if "control" in value.lower()
        and "no_persistence" not in value.lower()
        and "shortcut" not in value.lower()
    )
    return fallback[0] if len(fallback) == 1 else None


@dataclass(frozen=True, slots=True)
class PastSensitivityAvailability:
    """One case's source-level ability to deploy its declared condition."""

    availability_id: str
    case_id: str
    family_id: str
    condition: SensitivityCondition
    panel: str
    target_kind: str
    episode_selector: str | None
    available: bool
    reason: str
    family_source_digest: str
    schema: str = PAST_SENSITIVITY_CATALOG_SCHEMA
    schema_version: int = PAST_SENSITIVITY_CATALOG_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema != PAST_SENSITIVITY_CATALOG_SCHEMA or self.schema_version != PAST_SENSITIVITY_CATALOG_SCHEMA_VERSION:
            raise ValueError("unsupported PAST sensitivity availability schema")
        for value, name in (
            (self.availability_id, "sensitivity availability ID"),
            (self.case_id, "sensitivity case ID"),
            (self.family_id, "sensitivity family ID"),
            (self.panel, "sensitivity panel"),
            (self.target_kind, "sensitivity target kind"),
            (self.reason, "sensitivity availability reason"),
        ):
            _id(value, name)
        object.__setattr__(self, "condition", SensitivityCondition(self.condition))
        if self.episode_selector is not None:
            _id(self.episode_selector, "sensitivity episode selector")
        if type(self.available) is not bool:
            raise ValueError("sensitivity availability must be bool")
        if self.available != (self.reason == "available"):
            raise ValueError("sensitivity availability reason does not match status")
        _sha(self.family_source_digest, "family source digest")
        expected = "past-sensitivity-availability." + _digest(self.identity_payload())[:40]
        if self.availability_id != expected:
            raise ValueError("sensitivity availability ID mismatch")

    def identity_payload(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "family_id": self.family_id,
            "condition": self.condition.value,
            "panel": self.panel,
            "target_kind": self.target_kind,
            "episode_selector": self.episode_selector,
            "available": self.available,
            "reason": self.reason,
            "family_source_digest": self.family_source_digest,
        }

    def payload(self) -> dict[str, object]:
        return {"availability_id": self.availability_id, **self.identity_payload()}


@dataclass(frozen=True, slots=True)
class PastSensitivityCatalog:
    """Immutable, audit-only readiness evidence for one sensitivity matrix."""

    catalog_id: str
    matrix_id: str
    matrix_digest: str
    past_root_digest: str
    entries: tuple[PastSensitivityAvailability, ...]
    schema: str = PAST_SENSITIVITY_CATALOG_SCHEMA
    schema_version: int = PAST_SENSITIVITY_CATALOG_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema != PAST_SENSITIVITY_CATALOG_SCHEMA or self.schema_version != PAST_SENSITIVITY_CATALOG_SCHEMA_VERSION:
            raise ValueError("unsupported PAST sensitivity catalog schema")
        _id(self.matrix_id, "sensitivity matrix ID")
        _sha(self.matrix_digest, "sensitivity matrix digest")
        _sha(self.past_root_digest, "PAST root digest")
        entries = tuple(self.entries)
        if not entries or len({(entry.case_id, entry.condition) for entry in entries}) != len(entries):
            raise ValueError("PAST sensitivity catalog entries must be unique")
        object.__setattr__(self, "entries", entries)
        expected = "past-sensitivity-catalog." + _digest(self.identity_payload())[:40]
        if self.catalog_id != expected:
            raise ValueError("PAST sensitivity catalog ID mismatch")

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "matrix_id": self.matrix_id,
            "matrix_digest": self.matrix_digest,
            "past_root_digest": self.past_root_digest,
            "entries": [entry.payload() for entry in self.entries],
        }

    def payload(self) -> dict[str, object]:
        return {"catalog_id": self.catalog_id, **self.identity_payload()}

    @property
    def execution_ready(self) -> bool:
        return all(entry.available for entry in self.entries)

    def unavailable(self) -> tuple[PastSensitivityAvailability, ...]:
        return tuple(entry for entry in self.entries if not entry.available)


def _family_document(path: Path, expected_family_id: str) -> tuple[tuple[str, ...], tuple[str, ...], str]:
    try:
        raw = path.read_bytes()
        value = yaml.safe_load(raw) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError("PAST family document cannot be read") from exc
    if not isinstance(value, Mapping) or value.get("family_id") != expected_family_id:
        raise ValueError("PAST family document identity mismatch")
    episodes = value.get("episode_order")
    controls = value.get("control_set")
    if (
        not isinstance(episodes, list)
        or not episodes
        or not all(isinstance(item, str) and _IDENTIFIER.fullmatch(item) for item in episodes)
        or len(episodes) != len(set(episodes))
        or not isinstance(controls, list)
        or not all(isinstance(item, str) and _IDENTIFIER.fullmatch(item) for item in controls)
    ):
        raise ValueError("PAST family control identity is malformed")
    return tuple(episodes), tuple(controls), hashlib.sha256(raw).hexdigest()


def _availability(case: SensitivityCase, episodes: tuple[str, ...], controls: tuple[str, ...], source_digest: str) -> PastSensitivityAvailability:
    condition = case.condition
    selector: str | None = None
    reason = "available"
    if condition is SensitivityCondition.NATIVE_STATIC:
        # Stage labels vary across semantic, episodic, and procedural slices.
        # A non-cold/non-control/non-evaluation episode is the frozen
        # family-level indication of a state-forming path; no task content is
        # inspected to make this determination.
        has_learning = any(
            not any(token in item.lower() for token in ("cold", "control", "eval"))
            for item in episodes
        )
        has_evaluation = any("eval" in item.lower() for item in episodes)
        selector = "family.learn_eval"
        if not (has_learning and has_evaluation):
            selector = None
            reason = "native_static_episode_path_missing"
    elif condition is SensitivityCondition.NO_PERSISTENCE:
        selector = _control_episode(episodes, "no_persistence")
        if "no_persistence" not in controls or selector is None:
            reason = "no_persistence_control_missing"
    elif condition is SensitivityCondition.SHORTCUT_CURRENT_INPUT:
        selector = _control_episode(episodes, "shortcut")
        if "shortcut" not in controls or selector is None:
            reason = "shortcut_control_missing"
    elif condition is SensitivityCondition.WRONG_MECHANISM:
        selector = _control_episode(
            episodes,
            "wrong_mechanism",
            allow_semantic_legacy_fallback=case.panel.value == "semantic",
        )
        if "wrong_mechanism" not in controls or selector is None:
            reason = "wrong_mechanism_control_missing"
    else:
        # Correct-type oracle artifacts must be registered separately.  A
        # learn/eval trajectory is not an oracle and cannot stand in for one.
        reason = "type_matched_oracle_seed_missing"
    values = {
        "case_id": case.case_id,
        "family_id": case.family_id,
        "condition": condition.value,
        "panel": case.panel.value,
        "target_kind": case.target_kind.value,
        "episode_selector": selector,
        "available": reason == "available",
        "reason": reason,
        "family_source_digest": source_digest,
    }
    return PastSensitivityAvailability(
        availability_id="past-sensitivity-availability." + _digest(values)[:40],
        **values,
    )


def build_past_sensitivity_catalog(
    *,
    matrix: SensitivityMatrix,
    past_bench_root: Path,
) -> PastSensitivityCatalog:
    """Audit all declared controls without reading task-level private content."""

    root = Path(past_bench_root).expanduser().resolve()
    if not (root / "pyproject.toml").is_file():
        raise ValueError("PAST-Bench root is invalid")
    from .memory.family_matrix import PastFamilyMatrix

    family_matrix = PastFamilyMatrix.create_default()
    if family_matrix.matrix_digest != matrix.family_matrix_digest:
        raise ValueError("sensitivity matrix family identity is not available to catalog")
    entries: list[PastSensitivityAvailability] = []
    family_digests: dict[str, str] = {}
    family_controls: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {}
    for case in matrix.cases:
        if case.family_id not in family_controls:
            # The family matrix keeps roots audit-only; it is allowed here but
            # never crosses the launcher metadata allowlist.
            spec = family_matrix.spec_for(case.family_id)
            episodes, controls, digest = _family_document(
                root / spec.task_root / "family.yaml", case.family_id
            )
            family_controls[case.family_id] = (episodes, controls)
            family_digests[case.family_id] = digest
        episodes, controls = family_controls[case.family_id]
        entries.append(_availability(case, episodes, controls, family_digests[case.family_id]))
    root_digest = _digest({"family_digests": family_digests})
    values = {
        "schema": PAST_SENSITIVITY_CATALOG_SCHEMA,
        "schema_version": PAST_SENSITIVITY_CATALOG_SCHEMA_VERSION,
        "matrix_id": matrix.matrix_id,
        "matrix_digest": matrix.matrix_digest,
        "past_root_digest": root_digest,
        "entries": [entry.payload() for entry in entries],
    }
    return PastSensitivityCatalog(
        catalog_id="past-sensitivity-catalog." + _digest(values)[:40],
        matrix_id=matrix.matrix_id,
        matrix_digest=matrix.matrix_digest,
        past_root_digest=root_digest,
        entries=tuple(entries),
    )


__all__ = [
    "PAST_SENSITIVITY_CATALOG_SCHEMA",
    "PAST_SENSITIVITY_CATALOG_SCHEMA_VERSION",
    "PastSensitivityAvailability",
    "PastSensitivityCatalog",
    "build_past_sensitivity_catalog",
]
