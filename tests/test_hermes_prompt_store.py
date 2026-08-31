from __future__ import annotations

from rsimem.hermes_past_bridge import _PromptMemoryStore
from rsimem.memory.contracts import MemoryArtifact, MemoryHit, MemoryKind


class _NativeStore:
    def format_for_system_prompt(self, target: str) -> str:
        return f"native:{target}"


class _FailingRuntime:
    def query(self, _query):
        raise RuntimeError("adapter unavailable")


class _Runtime:
    def __init__(self, hits) -> None:
        self.hits = tuple(hits)
        self.injected: list[object] = []

    def query(self, _query):
        return self.hits

    def mark_injected(self, hits, *, surface):
        self.injected.append((tuple(hits), surface))


class _Bridge:
    uses_adapter = True

    def __init__(self) -> None:
        self.runtime = _FailingRuntime()
        self._last_adapter_route = None
        self.recorded_prompts: list[object] = []

    def _record_process_observation(self, **_kwargs) -> None:
        return None

    def _exposure_context_revision(self) -> str:
        return "revision.fixture"

    def adapter_call(self, _operation, _adapter_call, native_call):
        self._last_adapter_route = "native_bypass"
        return native_call()

    def verify_projection(self, *_args) -> None:
        return None

    def record_semantic_prompt(self, *args, **kwargs) -> None:
        self.recorded_prompts.append((args, kwargs))


def test_native_bypass_does_not_create_semantic_future_trace() -> None:
    bridge = _Bridge()
    store = _PromptMemoryStore(bridge, _NativeStore())

    assert store.format_for_system_prompt("user") == "native:user"
    assert bridge.recorded_prompts == []


def test_adapter_prompt_records_the_same_retrieved_hits_once() -> None:
    artifact = MemoryArtifact(
        artifact_id="semantic.fixture",
        kind=MemoryKind.SEMANTIC,
        content="Use TSV for durable reports.",
        namespace="user",
        revision="revision.fixture",
    )
    hits = (MemoryHit(artifact=artifact, rank=1, backend="fixture-semantic"),)

    class AdapterBridge(_Bridge):
        def __init__(self) -> None:
            self.runtime = _Runtime(hits)
            self._last_adapter_route = None
            self._last_host_event_id = None
            self._last_host_source_revision = None
            self.recorded_prompts = []

        def adapter_call(self, _operation, adapter_call, _native_call):
            self._last_adapter_route = "adapter"
            return adapter_call()

    class NativeWithRender(_NativeStore):
        def _render_block(self, target, contents):
            return f"{target}:" + "\n".join(contents)

    bridge = AdapterBridge()
    store = _PromptMemoryStore(bridge, NativeWithRender())

    assert store.format_for_system_prompt("user") == (
        "user:Use TSV for durable reports."
    )
    assert len(bridge.recorded_prompts) == 1
    _args, kwargs = bridge.recorded_prompts[0]
    assert kwargs["artifact_ids"] == ("semantic.fixture",)
    assert kwargs["retrieved_hits"] == hits


def test_adapter_render_failure_does_not_mark_hits_as_injected() -> None:
    artifact = MemoryArtifact(
        artifact_id="semantic.render-failure",
        kind=MemoryKind.SEMANTIC,
        content="Use TSV for durable reports.",
        namespace="user",
        revision="revision.fixture",
    )
    hits = (MemoryHit(artifact=artifact, rank=1, backend="fixture-semantic"),)

    class FailingRenderBridge(_Bridge):
        def __init__(self) -> None:
            self.runtime = _Runtime(hits)
            self._last_adapter_route = None
            self._last_host_event_id = None
            self._last_host_source_revision = None
            self.recorded_prompts = []

        def adapter_call(self, _operation, adapter_call, native_call):
            try:
                self._last_adapter_route = "adapter"
                return adapter_call()
            except RuntimeError:
                self._last_adapter_route = "native_bypass"
                return native_call()

    class FailingRenderStore(_NativeStore):
        def _render_block(self, _target, _contents):
            raise RuntimeError("renderer unavailable")

    bridge = FailingRenderBridge()
    store = _PromptMemoryStore(bridge, FailingRenderStore())
    assert store.format_for_system_prompt("user") == "native:user"
    assert bridge.runtime.injected == []
    assert bridge.recorded_prompts == []
