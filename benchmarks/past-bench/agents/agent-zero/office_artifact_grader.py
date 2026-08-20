"""Shared grader for file-oriented Office artifact tasks."""

from __future__ import annotations

import base64
import tempfile
from pathlib import Path
from typing import Any

from past_bench.graders.base import AbstractGrader
from past_bench.graders.office_artifact_checks import run_check
from past_bench.models.task import ArtifactCheck, TaskDefinition
from past_bench.models.trace import DimensionScores, MediaLoad, ToolDispatch, TraceMessage


class OfficeArtifactGrader(AbstractGrader):
    """Grade tasks by comparing sandbox output artifacts against local gold files."""

    def _materialize_result(
        self,
        check: ArtifactCheck,
        *,
        env_snapshot: dict | None,
        temp_dir: Path,
        cache: dict[str, Path],
    ) -> Path | None:
        cached = cache.get(check.result)
        if cached is not None:
            return cached

        snapshot_key = f"file:{check.result}"
        snapshot = (env_snapshot or {}).get(snapshot_key)
        if not isinstance(snapshot, dict) or "error" in snapshot:
            return None

        output_path = temp_dir / Path(check.result).name
        if snapshot.get("encoding") == "base64":
            output_path.write_bytes(base64.b64decode(snapshot.get("content", "")))
        else:
            output_path.write_text(snapshot.get("content", ""), encoding="utf-8")

        cache[check.result] = output_path
        return output_path

    def grade(
        self,
        messages: list[TraceMessage],
        dispatches: list[ToolDispatch],
        task: TaskDefinition,
        audit_data: dict[str, dict] | None = None,
        judge: Any | None = None,
        media_events: list[MediaLoad] | None = None,
        env_snapshot: dict | None = None,
    ) -> DimensionScores:
        scores = DimensionScores()
        scores.safety = 1.0

        task_root = Path(task.task_file).parent if task.task_file else Path.cwd()
        total_weight = sum(check.weight for check in task.artifact_checks) or 1.0
        completion = 0.0

        with tempfile.TemporaryDirectory(prefix="past_bench_office_grade_") as temp_dir_raw:
            temp_dir = Path(temp_dir_raw)
            cache: dict[str, Path] = {}
            for check in task.artifact_checks:
                options = dict(check.options)
                source_path = options.get("source_path")
                if isinstance(source_path, str) and not Path(source_path).is_absolute():
                    options["source_path"] = str(task_root / source_path)
                result_path = self._materialize_result(
                    check,
                    env_snapshot=env_snapshot,
                    temp_dir=temp_dir,
                    cache=cache,
                )
                if result_path is None:
                    continue
                expected_path = task_root / check.expected
                completion += check.weight * run_check(
                    check.func,
                    str(expected_path),
                    str(result_path),
                    options,
                )

        scores.completion = round(completion / total_weight, 4)
        scores.robustness = self.compute_robustness(dispatches)
        scores.efficiency_turns = len([message for message in messages if message.message.role == "assistant"])
        return scores
