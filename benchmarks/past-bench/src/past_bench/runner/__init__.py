"""Runner package with lazy exports to avoid heavy import side effects."""

from __future__ import annotations

from typing import Any

__all__ = ["ServiceManager", "run_task"]


def __getattr__(name: str) -> Any:
    if name == "run_task":
        from .loop import run_task

        return run_task
    if name == "ServiceManager":
        from .services import ServiceManager

        return ServiceManager
    raise AttributeError(name)
