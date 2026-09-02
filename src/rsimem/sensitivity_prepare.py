"""Build and prepare a registered, case-bound Stage 3 sensitivity run.

This is a preparation boundary only: it writes audit metadata and a PAST
sequence slice, but never invokes the provider or benchmark runner.
"""

from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .memory import MemoryKind
from .memory.family_matrix import PastFamilyMatrix
from .oracle_seed_registry import OracleSeedRegistry
from .past_sensitivity_catalog import build_past_sensitivity_catalog
from .past_sensitivity_launcher import PreparedPastSensitivityLaunch, prepare_past_sensitivity_launch
from .research_protocol import ResearchProtocol, default_research_protocol
from .sensitivity import SensitivityCase, SensitivityMatrix, SensitivityPanel
from .sensitivity_run import (
    SensitivityRunManifest,
    SensitivityRunManifestStore,
    apply_verified_oracle_seed_registry,
    build_sensitivity_run_manifest,
    planned_deployments_from_catalog,
)


@dataclass(frozen=True, slots=True)
class PreparedSensitivityBatch:
    manifest: SensitivityRunManifest
    launch: PreparedPastSensitivityLaunch
    manifest_path: Path


def _git_head(root: Path) -> str:
    try:
        value = subprocess.check_output(
            ("git", "-C", str(root), "rev-parse", "HEAD"),
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"cannot determine git commit for {root}") from exc
    if not value:
        raise ValueError(f"git checkout has no commit: {root}")
    return value


def _matrix(panel: SensitivityPanel) -> tuple[ResearchProtocol, SensitivityMatrix]:
    family_matrix = PastFamilyMatrix.create_default()
    base = default_research_protocol()
    protocol = ResearchProtocol.create(
        memory_units=base.memory_units,
        family_matrix=family_matrix,
        split=base.split,
        sensitivity_target_kind=MemoryKind(panel.value),
    )
    return protocol, SensitivityMatrix.create_for_panel(
        panel=panel, protocol=protocol, family_matrix=family_matrix
    )


def prepare_registered_sensitivity_batch(
    *,
    panel: SensitivityPanel,
    case_id: str,
    replicate: int,
    batch_id: str,
    rsimem_root: Path,
    past_bench_root: Path,
    registry_path: Path,
    trusted_seed_root: Path,
    output_root: Path,
    past_bench_binary: str = "past-bench",
    agent: str = "hermes-luna",
) -> PreparedSensitivityBatch:
    """Prepare one executable registered run; no provider call is made."""

    panel = SensitivityPanel(panel)
    protocol, matrix = _matrix(panel)
    catalog = build_past_sensitivity_catalog(matrix=matrix, past_bench_root=past_bench_root)
    deployments = planned_deployments_from_catalog(matrix=matrix, catalog=catalog)
    registry = OracleSeedRegistry.load(registry_path)
    deployments = apply_verified_oracle_seed_registry(
        matrix=matrix,
        catalog=catalog,
        deployments=deployments,
        registry=registry,
        trusted_root=trusted_seed_root,
    )
    manifest = build_sensitivity_run_manifest(
        batch_id=batch_id,
        matrix=matrix,
        protocol_digest=protocol.protocol_digest,
        provider_id=protocol.provider_id,
        model_id=protocol.model_id,
        tool_budget=protocol.tool_budget,
        max_turns=protocol.max_turns,
        retry_policy=protocol.retry_policy,
        deployments=deployments,
        rsimem_commit=_git_head(rsimem_root),
        past_bench_commit=_git_head(past_bench_root),
    )
    runs = [run for run in manifest.runs if run.case_id == case_id and run.replicate == replicate]
    if len(runs) != 1:
        raise ValueError("case ID or replicate is not present in the selected matrix")
    run = runs[0]
    deployment = next(item for item in manifest.deployments if item.deployment_id == run.deployment_id)
    oracle_case: SensitivityCase | None = next(
        (case for case in matrix.cases if case.case_id == case_id), None
    )
    manifest_path = Path(output_root) / "sensitivity_manifest.json"
    SensitivityRunManifestStore(manifest_path).initialize(manifest)
    launch = prepare_past_sensitivity_launch(
        run=run,
        deployment=deployment,
        past_bench_root=past_bench_root,
        output_directory=output_root,
        past_bench_binary=past_bench_binary,
        agent=agent,
        oracle_seed_registry=registry,
        oracle_trusted_root=trusted_seed_root,
        oracle_case=oracle_case,
    )
    return PreparedSensitivityBatch(manifest=manifest, launch=launch, manifest_path=manifest_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", choices=[item.value for item in SensitivityPanel], required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--replicate", type=int, default=1)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--rsimem-root", type=Path, required=True)
    parser.add_argument("--past-bench-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--trusted-seed-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--past-bench-binary", default="past-bench")
    parser.add_argument("--agent", default="hermes-luna")
    args = parser.parse_args(argv)
    prepared = prepare_registered_sensitivity_batch(
        panel=SensitivityPanel(args.panel), case_id=args.case_id, replicate=args.replicate,
        batch_id=args.batch_id, rsimem_root=args.rsimem_root, past_bench_root=args.past_bench_root,
        registry_path=args.registry, trusted_seed_root=args.trusted_seed_root,
        output_root=args.output_root, past_bench_binary=args.past_bench_binary, agent=args.agent,
    )
    print(prepared.manifest_path)
    print(prepared.launch.sequence_path)
    print(" ".join(prepared.launch.command))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["PreparedSensitivityBatch", "prepare_registered_sensitivity_batch"]
