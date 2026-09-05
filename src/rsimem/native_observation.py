"""Content-free native-static lifecycle observations for offline attribution."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Mapping

from .memory.contracts import MemoryKind
from .native_attribution_run import NativeAttributionRunSpec
from .native_execution_audit import audit_native_execution


OBSERVATION_SCHEMA = "rsimem-native-lifecycle-observation-v1"


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _read_json(path: Path) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"native observation artifact is unreadable: {path.name}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"native observation artifact is malformed: {path.name}")
    return value


def _read_jsonl(path: Path) -> tuple[Mapping[str, object], ...]:
    if not path.is_file():
        return ()
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"native observation event is malformed: {path.name}") from exc
        if not isinstance(value, Mapping):
            raise ValueError(f"native observation event is malformed: {path.name}")
        values.append(value)
    return tuple(values)


class NativeLifecycleSurface(StrEnum):
    TRIGGER_SOURCE = "trigger_source"
    FORMATION = "formation"
    PERSISTENCE = "persistence"
    MAINTENANCE = "maintenance"
    RETRIEVAL = "retrieval"
    EXPOSURE_APPLICATION = "exposure_application"
    DOWNSTREAM = "downstream"


class ObservationStatus(StrEnum):
    OBSERVED = "observed"
    NOT_OBSERVED = "not_observed"


class NativeLifecycleEventType(StrEnum):
    SOURCE = "source"
    CANDIDATE = "candidate"
    FORMATION = "formation"
    ADMISSION = "admission"
    COMMIT = "commit"
    MAINTENANCE = "maintenance"
    RETRIEVAL = "retrieval"
    EXPOSURE = "exposure"
    APPLICATION = "application"
    TOOL = "tool"
    OUTCOME = "outcome"


_EVENT_SURFACE = {
    NativeLifecycleEventType.SOURCE: NativeLifecycleSurface.TRIGGER_SOURCE,
    NativeLifecycleEventType.CANDIDATE: NativeLifecycleSurface.FORMATION,
    NativeLifecycleEventType.FORMATION: NativeLifecycleSurface.FORMATION,
    NativeLifecycleEventType.ADMISSION: NativeLifecycleSurface.PERSISTENCE,
    NativeLifecycleEventType.COMMIT: NativeLifecycleSurface.PERSISTENCE,
    NativeLifecycleEventType.MAINTENANCE: NativeLifecycleSurface.MAINTENANCE,
    NativeLifecycleEventType.RETRIEVAL: NativeLifecycleSurface.RETRIEVAL,
    NativeLifecycleEventType.EXPOSURE: NativeLifecycleSurface.EXPOSURE_APPLICATION,
    NativeLifecycleEventType.APPLICATION: NativeLifecycleSurface.EXPOSURE_APPLICATION,
    NativeLifecycleEventType.TOOL: NativeLifecycleSurface.DOWNSTREAM,
    NativeLifecycleEventType.OUTCOME: NativeLifecycleSurface.DOWNSTREAM,
}


@dataclass(frozen=True, slots=True)
class NativeSurfaceObservation:
    event_id: str
    event_type: NativeLifecycleEventType
    surface: NativeLifecycleSurface
    status: ObservationStatus
    evidence_refs: tuple[str, ...]
    input_artifact_ids: tuple[str, ...]
    output_artifact_ids: tuple[str, ...]
    state_before_digest: str
    state_after_digest: str

    @classmethod
    def create(
        cls,
        *,
        event_type: NativeLifecycleEventType,
        surface: NativeLifecycleSurface,
        status: ObservationStatus,
        evidence_refs: tuple[str, ...],
        input_artifact_ids: tuple[str, ...],
        output_artifact_ids: tuple[str, ...],
        state_before_digest: str,
        state_after_digest: str,
    ) -> "NativeSurfaceObservation":
        values = {
            "event_type": NativeLifecycleEventType(event_type).value,
            "surface": NativeLifecycleSurface(surface).value,
            "status": ObservationStatus(status).value,
            "evidence_refs": sorted(set(evidence_refs)),
            "input_artifact_ids": sorted(set(input_artifact_ids)),
            "output_artifact_ids": sorted(set(output_artifact_ids)),
            "state_before_digest": state_before_digest,
            "state_after_digest": state_after_digest,
        }
        return cls(
            event_id="native-lifecycle-event." + _digest(values)[:40],
            event_type=NativeLifecycleEventType(event_type),
            surface=NativeLifecycleSurface(surface),
            status=ObservationStatus(status),
            evidence_refs=tuple(values["evidence_refs"]),
            input_artifact_ids=tuple(values["input_artifact_ids"]),
            output_artifact_ids=tuple(values["output_artifact_ids"]),
            state_before_digest=state_before_digest,
            state_after_digest=state_after_digest,
        )

    def __post_init__(self) -> None:
        if self.surface is not _EVENT_SURFACE[self.event_type]:
            raise ValueError("native lifecycle event type and surface do not match")
        for value in (self.state_before_digest, self.state_after_digest):
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError("native lifecycle state digest is invalid")
        if self.status is ObservationStatus.OBSERVED and not self.evidence_refs:
            raise ValueError("observed native lifecycle surface requires evidence")
        if self.status is ObservationStatus.NOT_OBSERVED and self.evidence_refs:
            raise ValueError("unobserved native lifecycle surface cannot cite evidence")
        expected = "native-lifecycle-event." + _digest(self.identity_payload())[:40]
        if self.event_id != expected:
            raise ValueError("native lifecycle event ID mismatch")

    def identity_payload(self) -> dict[str, object]:
        return {
            "event_type": self.event_type.value,
            "surface": self.surface.value,
            "status": self.status.value,
            "evidence_refs": list(self.evidence_refs),
            "input_artifact_ids": list(self.input_artifact_ids),
            "output_artifact_ids": list(self.output_artifact_ids),
            "state_before_digest": self.state_before_digest,
            "state_after_digest": self.state_after_digest,
        }

    def payload(self) -> dict[str, object]:
        return {"event_id": self.event_id, **self.identity_payload()}


@dataclass(frozen=True, slots=True)
class NativeEpisodeObservation:
    observation_id: str
    run_id: str
    trace_id: str
    task_id: str
    family_id: str
    memory_kind: MemoryKind | None
    events: tuple[NativeSurfaceObservation, ...]
    usage_digest: str
    evidence_plane: str = "benchmark_audit"

    def __post_init__(self) -> None:
        if tuple(value.event_type for value in self.events) != tuple(NativeLifecycleEventType):
            raise ValueError("native observation must contain every lifecycle event in order")
        if self.evidence_plane != "benchmark_audit":
            raise ValueError("native attribution observations are audit-only")
        if self.memory_kind is not None:
            object.__setattr__(self, "memory_kind", MemoryKind(self.memory_kind))
        if not isinstance(self.usage_digest, str) or len(self.usage_digest) != 64:
            raise ValueError("native observation usage digest is invalid")
        if self.observation_id != "native-observation." + _digest(self.identity_payload())[:40]:
            raise ValueError("native observation ID mismatch")

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema": OBSERVATION_SCHEMA,
            "run_id": self.run_id,
            "trace_id": self.trace_id,
            "task_id": self.task_id,
            "family_id": self.family_id,
            "memory_kind": self.memory_kind.value if self.memory_kind else None,
            "events": [value.payload() for value in self.events],
            "usage_digest": self.usage_digest,
            "evidence_plane": self.evidence_plane,
        }

    def payload(self) -> dict[str, object]:
        return {"observation_id": self.observation_id, **self.identity_payload()}


def extract_native_observations(
    *, run: NativeAttributionRunSpec, output_root: Path
) -> tuple[NativeEpisodeObservation, ...]:
    audit = audit_native_execution(run=run, output_root=output_root)
    trace_root = Path(output_root).expanduser().resolve() / run.trace_directory
    result = _read_json(trace_root / "sequence_results.json")
    episodes = result.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError("native sequence episodes are malformed")
    observations = []
    for episode in episodes:
        if not isinstance(episode, Mapping) or episode.get("episode_kind") == "reflection":
            continue
        trace_id = episode.get("trace_id")
        task_id = episode.get("task_id")
        trace_path = Path(str(episode.get("trace", ""))).resolve()
        if not isinstance(trace_id, str) or not isinstance(task_id, str):
            raise ValueError("native episode identity is malformed")
        sidecar = _read_json(trace_path.parent / "native_episode_identity.json")
        if sidecar.get("trace_id") != trace_id or sidecar.get("task_id") != task_id:
            raise ValueError("native episode sidecar identity mismatch")
        before = sidecar.get("artifact_before")
        after = sidecar.get("artifact_after")
        if not isinstance(before, Mapping) or not isinstance(after, Mapping):
            raise ValueError("native artifact identity is malformed")
        before_ids = tuple(before.get("artifact_ids") or ())
        after_ids = tuple(after.get("artifact_ids") or ())
        state_before = sidecar.get("state_before_digest")
        state_after = sidecar.get("state_after_digest")
        if not isinstance(state_before, str) or not isinstance(state_after, str):
            raise ValueError("native state identity is malformed")
        artifacts_dir = trace_path.parent / "artifacts"
        memory_events = _read_jsonl(artifacts_dir / "rsimem_memory_events.jsonl")
        process_events = _read_jsonl(artifacts_dir / "pure_process_event_archive.jsonl")
        memory_refs = tuple(
            str(value["eventId"]) for value in memory_events
            if isinstance(value.get("eventId"), str)
        )
        process_refs = tuple(
            str(value["event_id"]) for value in process_events
            if isinstance(value.get("event_id"), str)
        )
        internal = episode.get("internal_tools")
        if not isinstance(internal, Mapping):
            internal = {}
        mutation_count = int(internal.get("memory_write_count") or 0)
        mutation_count += int(internal.get("skill_create_count") or 0)
        mutation_count += int(internal.get("skill_update_count") or 0)
        action_counts = internal.get("memory_action_counts")
        maintenance_count = 0
        if isinstance(action_counts, Mapping):
            maintenance_count = sum(
                int(action_counts.get(name) or 0)
                for name in ("update", "delete", "remove")
            )
        maintenance_count += int(internal.get("skill_update_count") or 0)
        kinds = {str(value.get("kind") or "") for value in memory_events}
        process_kinds = {str(value.get("kind") or "") for value in process_events}
        internal_calls = internal.get("calls")
        if not isinstance(internal_calls, list):
            internal_calls = []
        mutation_refs = tuple(
            "native-call." + _digest({
                "trace_id": trace_id,
                "message_index": value.get("message_index"),
                "name": value.get("name"),
                "args": value.get("args"),
            })[:40]
            for value in internal_calls
            if isinstance(value, Mapping) and value.get("name") in {"memory", "skill_manage"}
        )
        changed_artifacts = tuple(sorted(set(before_ids).symmetric_difference(after_ids)))
        refs_by_type = {
            NativeLifecycleEventType.SOURCE: (trace_id,),
            NativeLifecycleEventType.CANDIDATE: mutation_refs if mutation_count else (),
            NativeLifecycleEventType.FORMATION: mutation_refs if mutation_count else (),
            NativeLifecycleEventType.ADMISSION: (),
            NativeLifecycleEventType.COMMIT: changed_artifacts,
            NativeLifecycleEventType.MAINTENANCE: mutation_refs if maintenance_count else (),
            NativeLifecycleEventType.RETRIEVAL: memory_refs if kinds.intersection({"query", "retrieved"}) else (),
            NativeLifecycleEventType.EXPOSURE: memory_refs if "injected" in kinds else (),
            NativeLifecycleEventType.APPLICATION: (),
            NativeLifecycleEventType.TOOL: process_refs if process_kinds.intersection({"tool_call", "tool_result"}) else (),
            NativeLifecycleEventType.OUTCOME: process_refs if "task_outcome" in process_kinds else (trace_id,),
        }
        events = tuple(NativeSurfaceObservation.create(
            event_type=event_type,
            surface=_EVENT_SURFACE[event_type],
            status=(ObservationStatus.OBSERVED if refs_by_type[event_type] else ObservationStatus.NOT_OBSERVED),
            evidence_refs=tuple(refs_by_type[event_type]),
            input_artifact_ids=before_ids,
            output_artifact_ids=after_ids,
            state_before_digest=state_before,
            state_after_digest=state_after,
        ) for event_type in NativeLifecycleEventType)
        values = {
            "schema": OBSERVATION_SCHEMA,
            "run_id": run.run_id,
            "trace_id": trace_id,
            "task_id": task_id,
            "family_id": run.family_id,
            "memory_kind": run.memory_kind,
            "events": [value.payload() for value in events],
            "usage_digest": audit.usage_digest,
            "evidence_plane": "benchmark_audit",
        }
        observations.append(NativeEpisodeObservation(
            observation_id="native-observation." + _digest(values)[:40],
            run_id=run.run_id,
            trace_id=trace_id,
            task_id=task_id,
            family_id=run.family_id,
            memory_kind=MemoryKind(run.memory_kind) if run.memory_kind else None,
            events=events,
            usage_digest=audit.usage_digest,
        ))
    if tuple(value.task_id for value in observations) != run.native_task_ids:
        raise ValueError("native observations do not cover the registered task order")
    return tuple(observations)


__all__ = [
    "NativeEpisodeObservation", "NativeLifecycleEventType", "NativeLifecycleSurface",
    "NativeSurfaceObservation", "OBSERVATION_SCHEMA", "ObservationStatus",
    "extract_native_observations",
]
