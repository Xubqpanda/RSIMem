"""Top-level method identity for extraction-prompt adaptation."""

from __future__ import annotations

from enum import StrEnum
from typing import Mapping


EXTRACTION_ADAPTATION_OBJECTIVE = "delayed-extraction-utility-v1"


class AdaptiveArtifactKind(StrEnum):
    EXTRACTION_PROMPT = "extraction_prompt"
    LEGACY_THRESHOLD_EXPERIMENT = "legacy_threshold_experiment"


def require_extraction_prompt_artifact(value: object) -> None:
    """Reject non-prompt artifacts before an extraction runtime binds them."""

    kind = (
        value.get("artifact_kind")
        if isinstance(value, Mapping)
        else getattr(value, "artifact_kind", None)
    )
    if kind != AdaptiveArtifactKind.EXTRACTION_PROMPT:
        raise ValueError("extraction runtime requires an extraction prompt artifact")
