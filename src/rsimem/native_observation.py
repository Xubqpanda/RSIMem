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
    producer: str
    owner: str
    memory_kind: MemoryKind | None
    evidence_refs: tuple[str, ...]
    input_artifact_ids: tuple[str, ...]
    output_artifact_ids: tuple[str, ...]
    state_before_digest: str
    state_after_digest: str
    revision: str
    parent_event_ids: tuple[str, ...]
    observation_cutoff: str
    evidence_plane: str = "benchmark_audit"
    evidence_source: str = "runtime_observation"

    @classmethod
    def create(
        cls,
        *,
        event_type: NativeLifecycleEventType,
        surface: NativeLifecycleSurface,
        status: ObservationStatus,
        producer: str,
        owner: str,
        memory_kind: MemoryKind | None,
        evidence_refs: tuple[str, ...],
        input_artifact_ids: tuple[str, ...],
        output_artifact_ids: tuple[str, ...],
        state_before_digest: str,
        state_after_digest: str,
        revision: str,
        parent_event_ids: tuple[str, ...],
        observation_cutoff: str,
    ) -> "NativeSurfaceObservation":
        values = {
            "event_type": NativeLifecycleEventType(event_type).value,
            "surface": NativeLifecycleSurface(surface).value,
            "status": ObservationStatus(status).value,
            "producer": producer,
            "owner": owner,
            "memory_kind": MemoryKind(memory_kind).value if memory_kind else None,
            "evidence_refs": sorted(set(evidence_refs)),
            "input_artifact_ids": sorted(set(input_artifact_ids)),
            "output_artifact_ids": sorted(set(output_artifact_ids)),
            "state_before_digest": state_before_digest,
            "state_after_digest": state_after_digest,
            "revision": revision,
            "parent_event_ids": list(parent_event_ids),
            "observation_cutoff": observation_cutoff,
            "evidence_plane": "benchmark_audit",
            "evidence_source": "runtime_observation",
        }
        return cls(
            event_id="native-lifecycle-event." + _digest(values)[:40],
            event_type=NativeLifecycleEventType(event_type),
            surface=NativeLifecycleSurface(surface),
            status=ObservationStatus(status),
            producer=producer,
            owner=owner,
            memory_kind=MemoryKind(memory_kind) if memory_kind else None,
            evidence_refs=tuple(values["evidence_refs"]),
            input_artifact_ids=tuple(values["input_artifact_ids"]),
            output_artifact_ids=tuple(values["output_artifact_ids"]),
            state_before_digest=state_before_digest,
            state_after_digest=state_after_digest,
            revision=revision,
            parent_event_ids=tuple(parent_event_ids),
            observation_cutoff=observation_cutoff,
        )

    def __post_init__(self) -> None:
        if self.surface is not _EVENT_SURFACE[self.event_type]:
            raise ValueError("native lifecycle event type and surface do not match")
        if not self.producer or not self.owner or not self.observation_cutoff:
            raise ValueError("native lifecycle provenance is incomplete")
        if self.memory_kind is not None:
            object.__setattr__(self, "memory_kind", MemoryKind(self.memory_kind))
        if self.evidence_plane != "benchmark_audit" or self.evidence_source != "runtime_observation":
            raise ValueError("native lifecycle evidence boundary is invalid")
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
            "producer": self.producer,
            "owner": self.owner,
            "memory_kind": self.memory_kind.value if self.memory_kind else None,
            "evidence_refs": list(self.evidence_refs),
            "input_artifact_ids": list(self.input_artifact_ids),
            "output_artifact_ids": list(self.output_artifact_ids),
            "state_before_digest": self.state_before_digest,
            "state_after_digest": self.state_after_digest,
            "revision": self.revision,
            "parent_event_ids": list(self.parent_event_ids),
            "observation_cutoff": self.observation_cutoff,
            "evidence_plane": self.evidence_plane,
            "evidence_source": self.evidence_source,
        }

    def payload(self) -> dict[str, object]:
        return {"event_id": self.event_id, **self.identity_payload()}

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "NativeSurfaceObservation":
        if not isinstance(payload, Mapping):
            raise ValueError("native lifecycle event payload is malformed")
        expected = {
            "event_id", "event_type", "surface", "status", "producer", "owner",
            "memory_kind", "evidence_refs", "input_artifact_ids", "output_artifact_ids",
            "state_before_digest", "state_after_digest", "revision", "parent_event_ids",
            "observation_cutoff", "evidence_plane", "evidence_source",
        }
        if set(payload) != expected:
            raise ValueError("native lifecycle event payload fields are invalid")
        scalar_fields = (
            "event_id", "producer", "owner", "state_before_digest", "state_after_digest",
            "revision", "observation_cutoff", "evidence_plane", "evidence_source",
        )
        if any(not isinstance(payload[field], str) for field in scalar_fields):
            raise ValueError("native lifecycle event payload scalar types are invalid")
        collections = (
            "evidence_refs", "input_artifact_ids", "output_artifact_ids", "parent_event_ids"
        )
        if any(
            not isinstance(payload[field], list)
            or any(not isinstance(value, str) for value in payload[field])
            for field in collections
        ):
            raise ValueError("native lifecycle event payload collections are invalid")
        return cls(
            event_id=payload["event_id"],
            event_type=NativeLifecycleEventType(payload["event_type"]),
            surface=NativeLifecycleSurface(payload["surface"]),
            status=ObservationStatus(payload["status"]),
            producer=payload["producer"],
            owner=payload["owner"],
            memory_kind=MemoryKind(payload["memory_kind"]) if payload["memory_kind"] else None,
            evidence_refs=tuple(payload["evidence_refs"]),
            input_artifact_ids=tuple(payload["input_artifact_ids"]),
            output_artifact_ids=tuple(payload["output_artifact_ids"]),
            state_before_digest=payload["state_before_digest"],
            state_after_digest=payload["state_after_digest"],
            revision=payload["revision"],
            parent_event_ids=tuple(payload["parent_event_ids"]),
            observation_cutoff=payload["observation_cutoff"],
            evidence_plane=payload["evidence_plane"],
            evidence_source=payload["evidence_source"],
        )


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
    final_output_digest: str | None = None
    replicate_id: int | None = None
    evidence_plane: str = "benchmark_audit"

    def __post_init__(self) -> None:
        if tuple(value.event_type for value in self.events) != tuple(NativeLifecycleEventType):
            raise ValueError("native observation must contain every lifecycle event in order")
        if self.evidence_plane != "benchmark_audit":
            raise ValueError("native attribution observations are audit-only")
        if self.memory_kind is not None:
            object.__setattr__(self, "memory_kind", MemoryKind(self.memory_kind))
        if (
            not isinstance(self.usage_digest, str)
            or len(self.usage_digest) != 64
            or any(character not in "0123456789abcdef" for character in self.usage_digest)
        ):
            raise ValueError("native observation usage digest is invalid")
        if self.final_output_digest is not None and (
            not isinstance(self.final_output_digest, str)
            or len(self.final_output_digest) != 64
            or any(character not in "0123456789abcdef" for character in self.final_output_digest)
        ):
            raise ValueError("native observation final output digest is invalid")
        if self.replicate_id is not None and (type(self.replicate_id) is not int or self.replicate_id < 1):
            raise ValueError("native observation replicate ID is invalid")
        if self.observation_id != "native-observation." + _digest(self.identity_payload())[:40]:
            raise ValueError("native observation ID mismatch")

    def identity_payload(self) -> dict[str, object]:
        values = {
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
        if self.final_output_digest is not None:
            values["final_output_digest"] = self.final_output_digest
        if self.replicate_id is not None:
            values["replicate_id"] = self.replicate_id
        return values

    def payload(self) -> dict[str, object]:
        return {"observation_id": self.observation_id, **self.identity_payload()}

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "NativeEpisodeObservation":
        if not isinstance(payload, Mapping):
            raise ValueError("native observation payload is malformed")
        expected = {
            "observation_id", "schema", "run_id", "trace_id", "task_id", "family_id",
            "memory_kind", "events", "usage_digest", "evidence_plane",
        }
        allowed = (expected, expected | {"final_output_digest"}, expected | {"replicate_id"}, expected | {"final_output_digest", "replicate_id"})
        if set(payload) not in allowed or payload.get("schema") != OBSERVATION_SCHEMA:
            raise ValueError("native observation payload fields are invalid")
        scalar_fields = (
            "observation_id", "run_id", "trace_id", "task_id", "family_id", "usage_digest",
            "evidence_plane",
        )
        if any(not isinstance(payload[field], str) for field in scalar_fields):
            raise ValueError("native observation payload scalar types are invalid")
        final_output_digest = payload.get("final_output_digest")
        if final_output_digest is not None and not isinstance(
            final_output_digest, str
        ):
            raise ValueError("native observation final output digest type is invalid")
        replicate_id = payload.get("replicate_id")
        if replicate_id is not None and type(replicate_id) is not int:
            raise ValueError("native observation replicate ID type is invalid")
        raw_events = payload.get("events")
        if not isinstance(raw_events, list):
            raise ValueError("native observation events are malformed")
        return cls(
            observation_id=payload["observation_id"],
            run_id=payload["run_id"],
            trace_id=payload["trace_id"],
            task_id=payload["task_id"],
            family_id=payload["family_id"],
            memory_kind=MemoryKind(payload["memory_kind"]) if payload["memory_kind"] else None,
            events=tuple(NativeSurfaceObservation.from_payload(value) for value in raw_events),
            usage_digest=payload["usage_digest"],
            final_output_digest=final_output_digest,
            replicate_id=replicate_id,
            evidence_plane=payload["evidence_plane"],
        )


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
        if (
            run.memory_kind == "episodic"
            and sidecar.get("schema") != "past-bench-native-episode-identity-v2"
        ):
            raise ValueError("episodic native observation requires v2 session identity")
        before = sidecar.get("artifact_before")
        after = sidecar.get("artifact_after")
        if not isinstance(before, Mapping) or not isinstance(after, Mapping):
            raise ValueError("native artifact identity is malformed")
        before_ids_all = tuple(before.get("artifact_ids") or ())
        after_ids_all = tuple(after.get("artifact_ids") or ())
        prefixes = {
            "semantic": ("hermes-semantic:", "hermes-profile:"),
            "episodic": ("hermes-episodic:",),
            "procedural": ("hermes-procedural:",),
            None: ("hermes-",),
        }[run.memory_kind]
        before_ids = tuple(value for value in before_ids_all if value.startswith(prefixes))
        after_ids = tuple(value for value in after_ids_all if value.startswith(prefixes))
        state_before = sidecar.get("state_before_digest")
        state_after = sidecar.get("state_after_digest")
        if not isinstance(state_before, str) or not isinstance(state_after, str):
            raise ValueError("native state identity is malformed")
        artifacts_dir = trace_path.parent / "artifacts"
        memory_events = _read_jsonl(artifacts_dir / "rsimem_memory_events.jsonl")
        process_events = _read_jsonl(artifacts_dir / "pure_process_event_archive.jsonl")
        final_output = episode.get("final_response_text")
        final_output_digest = (
            hashlib.sha256(final_output.encode("utf-8")).hexdigest()
            if isinstance(final_output, str)
            else None
        )
        replicate_id = run.replicate
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
        events_list: list[NativeSurfaceObservation] = []
        for event_type in NativeLifecycleEventType:
            event = NativeSurfaceObservation.create(
                event_type=event_type,
                surface=_EVENT_SURFACE[event_type],
                status=(ObservationStatus.OBSERVED if refs_by_type[event_type] else ObservationStatus.NOT_OBSERVED),
                producer="hermes-past-runtime",
                owner="hermes-native",
                memory_kind=MemoryKind(run.memory_kind) if run.memory_kind else None,
                evidence_refs=tuple(refs_by_type[event_type]),
                input_artifact_ids=before_ids,
                output_artifact_ids=after_ids,
                state_before_digest=state_before,
                state_after_digest=state_after,
                revision=state_after,
                parent_event_ids=(events_list[-1].event_id,) if events_list else (),
                observation_cutoff=trace_id,
            )
            events_list.append(event)
        events = tuple(events_list)
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
        if final_output_digest is not None:
            values["final_output_digest"] = final_output_digest
        values["replicate_id"] = replicate_id
        observations.append(NativeEpisodeObservation(
            observation_id="native-observation." + _digest(values)[:40],
            run_id=run.run_id,
            trace_id=trace_id,
            task_id=task_id,
            family_id=run.family_id,
            memory_kind=MemoryKind(run.memory_kind) if run.memory_kind else None,
            events=events,
            usage_digest=audit.usage_digest,
            final_output_digest=final_output_digest,
            replicate_id=replicate_id,
        ))
    if tuple(value.task_id for value in observations) != run.native_task_ids:
        raise ValueError("native observations do not cover the registered task order")
    return tuple(observations)


__all__ = [
    "NativeEpisodeObservation", "NativeLifecycleEventType", "NativeLifecycleSurface",
    "NativeSurfaceObservation", "OBSERVATION_SCHEMA", "ObservationStatus",
    "extract_native_observations",
]
