"""Reusable Hermes host integration helpers."""

from .runtime import FROZEN_AUXILIARY_SESSION_SEARCH_MODEL, build_past_environment
from .hermes_integration import (
    HermesAdapterExecutionError,
    HermesAdapterFailurePolicy,
    HermesExecutionMode,
    HermesExperimentConfig,
    build_configured_hermes_runtime,
)
from .hermes_host_adapter import HermesHostAdapter
from .hermes_past_bridge import HermesPastBenchBridge

__all__ = [
    "FROZEN_AUXILIARY_SESSION_SEARCH_MODEL",
    "build_past_environment",
    "HermesAdapterExecutionError",
    "HermesAdapterFailurePolicy",
    "HermesExecutionMode",
    "HermesExperimentConfig",
    "HermesHostAdapter",
    "HermesPastBenchBridge",
    "build_configured_hermes_runtime",
]
