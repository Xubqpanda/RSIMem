"""Environment and runtime bindings shared by isolated Hermes launches."""

from __future__ import annotations

import os


FROZEN_AUXILIARY_SESSION_SEARCH_MODEL = "gpt-5.6-luna"


def build_past_environment(
    *,
    base_url: str,
    api_key: str | None,
    auxiliary_model: str = FROZEN_AUXILIARY_SESSION_SEARCH_MODEL,
) -> dict[str, str]:
    """Route Hermes auxiliary model calls through the run's provider."""

    environment = os.environ.copy()
    if not environment.get("ANTHROPIC_API_KEY"):
        compatibility_key = api_key or environment.get("GPT_LUNA_API_KEY")
        if compatibility_key:
            environment["ANTHROPIC_API_KEY"] = compatibility_key
    environment["AUXILIARY_SESSION_SEARCH_BASE_URL"] = base_url
    environment["AUXILIARY_SESSION_SEARCH_MODEL"] = auxiliary_model
    if api_key:
        environment["AUXILIARY_SESSION_SEARCH_API_KEY"] = api_key
    return environment


__all__ = [
    "FROZEN_AUXILIARY_SESSION_SEARCH_MODEL",
    "build_past_environment",
]
