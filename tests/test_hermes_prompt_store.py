from __future__ import annotations

from rsimem.hermes_past_bridge import _PromptMemoryStore


class _NativeStore:
    def format_for_system_prompt(self, target: str) -> str:
        return f"native:{target}"


class _FailingRuntime:
    def query(self, _query):
        raise RuntimeError("adapter unavailable")


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
