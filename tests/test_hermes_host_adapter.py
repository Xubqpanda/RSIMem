from __future__ import annotations

from types import SimpleNamespace

from rsimem.adapter_contracts import AdapterStatus, CanonicalHostEvent, HostEventKind, MethodRunIdentity
from rsimem.hermes_host_adapter import HermesHostAdapter


class _Bridge:
    _run_id = "run.hermes.v1"
    _session_id = "session.hermes.v1"
    _task_id = "task.hermes.v1"
    config = SimpleNamespace(adapter_failure_policy=SimpleNamespace(value="bypass_native"))

    def __init__(self) -> None:
        self.wrapped = False

    def _wrap_skill_handlers(self) -> None:
        self.wrapped = True

    def _collect_completed_snapshot(self):
        return SimpleNamespace(
            snapshot_id="snapshot.hermes.v1",
            context_revision="revision.hermes.v1",
            segments=(SimpleNamespace(segment_id="segment.1"),),
            active_segment_ids=("segment.1",),
            current_turn_id="turn.1",
        )


def test_hermes_host_adapter_attaches_and_declares_capabilities() -> None:
    bridge = _Bridge()
    host = HermesHostAdapter(bridge)
    agent = SimpleNamespace(_memory_store=None, _session_db=None)
    host.attach(agent)
    assert bridge.wrapped is True
    assert host.capabilities.native_bypass is True
    assert host.capabilities.context_snapshot is True

    run = MethodRunIdentity(bridge._run_id, bridge._session_id, bridge._task_id, "revision.hermes.v1")
    assert host.prepare_session(run).status is AdapterStatus.SUPPORTED
    event = CanonicalHostEvent(
        event_id="host-event.hermes.v1",
        session_id=run.session_id,
        task_id=run.task_id,
        kind=HostEventKind.TURN_COMPLETED,
        revision=run.state_revision,
    )
    assert host.observe_event(event).status is AdapterStatus.SUPPORTED
    assert host.observe_event(event).reason_code == "duplicate_event"
    snapshot = host.snapshot_state()
    assert snapshot.state_schema == "hermes.context.snapshot.v1"
    assert snapshot.state_digest != "0" * 64
    assert host.restart(run).reason_code == "restart_requires_new_bridge"


def test_hermes_host_adapter_rejects_identity_mismatch() -> None:
    host = HermesHostAdapter(_Bridge())
    run = MethodRunIdentity("run.other.v1", "session.hermes.v1", "task.hermes.v1", "revision.hermes.v1")
    assert host.prepare_session(run).reason_code == "identity_mismatch"
