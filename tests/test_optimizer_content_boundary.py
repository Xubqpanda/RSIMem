from __future__ import annotations

import pytest

from rsimem.memory.optimizer_content_boundary import (
    OptimizerSecretBoundary,
    OptimizerTextRedaction,
)
from rsimem.memory.prompt_components import text_digest


def test_optimizer_boundary_redacts_only_its_copy_and_rejects_eval_evidence() -> None:
    boundary = OptimizerSecretBoundary()
    original = (
        "Authorization: Bearer secret-token inspect /mnt/private/run.json "
        "with sk-abcdefghijklmnopqrstuvwxyz012345"
    )
    projected = boundary.project(original)

    assert original.startswith("Authorization: Bearer")
    assert "secret-token" not in projected.text
    assert "/mnt/private" not in projected.text
    assert "sk-abcdefghijklmnopqrstuvwxyz" not in projected.text
    assert projected.redactions == (
        OptimizerTextRedaction.AUTHORIZATION,
        OptimizerTextRedaction.CREDENTIAL,
        OptimizerTextRedaction.MACHINE_PATH,
    )
    assert projected.source_digest == text_digest(original)
    assert projected.projected_digest == text_digest(projected.text)
    with pytest.raises(ValueError, match="forbidden evaluation evidence"):
        boundary.project("The official grader answer key says option B.")
    with pytest.raises(ValueError, match="forbidden evaluation evidence"):
        boundary.project("ordinary source", forbidden_values=("ordinary",))
