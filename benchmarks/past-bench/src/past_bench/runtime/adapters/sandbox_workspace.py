"""Minimal sandbox workspace shim for adapters that expect prepare/close hooks.

The Hermes adapter can run with or without a mounted sandbox workspace. The
current evaluation path only needs a stable hook surface, so this mirror is a
safe no-op unless future adapter code adds explicit file mirroring behavior.
"""

from __future__ import annotations


class SandboxWorkspaceMirror:
    def __init__(self, request) -> None:
        self.request = request

    def prepare(self) -> None:
        return

    def close(self) -> None:
        return
