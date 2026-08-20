"""Self-Evolve-Tasks-V2 Layer 1 metrics (§14.1) + 4-level reporting (§16).

This module consumes the `sequence_comparison.json` emitted by the self-evolve
runner and produces:

- **Task-level** records (one per episode, mostly pass-through)
- **Family-level** Layer 1 metrics and `FamilyEvolveScore`
- **Ability-level** aggregated scores across same-ability families
- **Benchmark-level** 7-dimensional capability vector plus headline summary

Layer 1 metrics from §14.1 of the design doc:

- `BaselineHeadroom`     - `OutcomeDelta`
- `AblationLift`
- `AutonomousReuseRate`
- `TransferRobustness`
- `ShortcutResistance`
- `MechanismConfidence`
- `FamilyEvolveScore = max(0, AblationLift) * MechanismConfidence
                       * AutonomousReuseRate * TransferRobustness`

Missing or ungathered inputs degrade to neutral defaults (1.0 for multiplicative
factors, 0.0 for additive deltas) and are recorded in a per-metric `basis` tag
so downstream reports can flag estimates versus measured values.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, Field

from past_bench.metrics.v2_layer2 import V2Layer2Metrics, compute_layer2
from past_bench.metrics.v2_layer3 import V2Layer3Metrics, compute_layer3
from past_bench.metrics.v2_trace_parse import (
    first_use_turn,
    repeat_violation_count,
    stale_leak_events,
)
from past_bench.models.self_evolve import (
    V2_ABILITIES,
    V2FamilyMetadata,
    load_v2_families,
)


# ── utility ──────────────────────────────────────────────────────────────

EPS = 1e-9


def _clip01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _safe_mean(xs: Iterable[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


# ── models ───────────────────────────────────────────────────────────────


class V2FamilyMetrics(BaseModel):
    """Layer 1 metrics for a single family (§14.1)."""

    family_id: str
    primary_ability: str
    expected_substrate: str
    primary_trigger: str

    baseline_headroom: float | None = None
    outcome_delta: float | None = None
    ablation_lift: float | None = None
    autonomous_reuse_rate: float | None = None
    transfer_robustness: float | None = None
    shortcut_resistance: float | None = None
    mechanism_confidence: float | None = None

    family_evolve_score: float | None = None

    # provenance tags so reports can distinguish measured vs inferred
    basis: dict[str, str] = Field(default_factory=dict)


class V2AbilityReport(BaseModel):
    """Aggregated score across same-ability families (§16)."""

    ability: str
    family_count: int
    families_with_positive_lift: int
    avg_outcome_delta: float | None = None
    avg_ablation_lift: float | None = None
    avg_mechanism_confidence: float | None = None
    avg_family_evolve_score: float | None = None


class V2BenchmarkReport(BaseModel):
    """Benchmark-level 7-dim capability vector and global summary (§16).

    The four named summary fields answer the §16 closing questions directly:
    which ability actually evolved, under which trigger, through which
    substrate, and how strong/robust the reuse was.
    """

    total_families: int
    families_with_metrics: int
    capability_vector: dict[str, float | None]  # ability -> avg_family_evolve_score
    overall_avg_outcome_delta: float | None = None
    overall_avg_ablation_lift: float | None = None
    overall_avg_family_evolve_score: float | None = None
    ability_reports: list[V2AbilityReport] = Field(default_factory=list)

    # §16 four-question summary
    which_ability_evolved: str | None = None
    under_which_trigger: str | None = None
    through_which_substrate: str | None = None
    reuse_strength_and_robustness: dict[str, float | None] = Field(default_factory=dict)
    out_of_scope_abilities: list[str] = Field(default_factory=list)


# ── family-level metric computation ──────────────────────────────────────


def _bucket_avg(bucket_summary: dict[str, Any], bucket: str, key: str) -> float | None:
    block = bucket_summary.get(bucket) or {}
    v = block.get(key)
    return float(v) if v is not None else None


def compute_family_metrics(
    family_id: str,
    family_meta: V2FamilyMetadata | None,
    with_family: dict[str, Any],
    without_family: dict[str, Any] | None,
    *,
    shortcut_eval_score: float | None = None,
) -> V2FamilyMetrics:
    """Compute Layer 1 metrics for one family from runner output.

    Parameters
    ----------
    family_id:
        Family identifier.
    family_meta:
        Optional loaded family.yaml. If provided, ability/trigger/substrate
        fall back to this metadata when the runner output does not expose them.
    with_family:
        The `family_summary[<family_id>]` dict from
        `sequence_comparison.json['with_persistence']`.
    without_family:
        Same block from `['without_persistence']`, or None when no control ran.
    shortcut_eval_score:
        Optional shortcut-control evaluation score (0–1). If None, shortcut
        resistance defaults to 1.0 and is marked as estimated.
    """

    basis: dict[str, str] = {}

    ability = (family_meta.primary_ability if family_meta else "")
    substrate = (family_meta.expected_substrate if family_meta else "")
    trigger = (family_meta.primary_trigger if family_meta else "")

    wp_bs = with_family.get("bucket_summary") or {}
    baseline_score = _bucket_avg(wp_bs, "baseline", "avg_task_score")
    eval_score = _bucket_avg(wp_bs, "evaluation", "avg_task_score")
    eval_near_score = _bucket_avg(wp_bs, "eval_near", "avg_task_score")
    eval_far_score = _bucket_avg(wp_bs, "eval_far", "avg_task_score")
    retrieval_rate_eval = _bucket_avg(wp_bs, "evaluation", "retrieval_before_success_rate")

    # BaselineHeadroom — §10. Reject saturated baselines.
    if baseline_score is not None:
        baseline_headroom = _clip01(1.0 - baseline_score)
        basis["baseline_headroom"] = "measured"
    else:
        baseline_headroom = None

    # OutcomeDelta within with_persistence
    if baseline_score is not None and eval_score is not None:
        outcome_delta = eval_score - baseline_score
        basis["outcome_delta"] = "measured"
    else:
        outcome_delta = None

    # AblationLift = with_persistence OutcomeDelta - without_persistence OutcomeDelta
    ablation_lift: float | None = None
    if without_family is not None:
        wp2_bs = without_family.get("bucket_summary") or {}
        b2 = _bucket_avg(wp2_bs, "baseline", "avg_task_score")
        e2 = _bucket_avg(wp2_bs, "evaluation", "avg_task_score")
        if baseline_score is not None and eval_score is not None and b2 is not None and e2 is not None:
            ablation_lift = (eval_score - baseline_score) - (e2 - b2)
            basis["ablation_lift"] = "measured"
    if ablation_lift is None:
        basis["ablation_lift"] = "unmeasured"

    # AutonomousReuseRate — proxy: retrieval_before_success_rate in evaluation bucket.
    # Until Phase C wires explicit reuse tagging this is the best signal we have.
    if retrieval_rate_eval is not None:
        autonomous_reuse_rate = _clip01(retrieval_rate_eval)
        basis["autonomous_reuse_rate"] = "proxy_retrieval_before_success"
    else:
        autonomous_reuse_rate = 1.0
        basis["autonomous_reuse_rate"] = "default_1.0_no_signal"

    # TransferRobustness — eval_far / max(eval_near, eps), clipped.
    # If buckets aren't split (legacy F01..F07 use single 'evaluation' bucket),
    # fall back to neutral 1.0.
    if eval_near_score is not None and eval_far_score is not None:
        if eval_near_score > EPS:
            transfer_robustness = _clip01(eval_far_score / eval_near_score)
        else:
            transfer_robustness = 0.0
        basis["transfer_robustness"] = "measured"
    else:
        transfer_robustness = 1.0
        basis["transfer_robustness"] = "default_1.0_no_near_far_split"

    # ShortcutResistance — 1 - shortcut_score / eval_score, clipped to [0,1].
    if shortcut_eval_score is not None and eval_score is not None and eval_score > EPS:
        shortcut_resistance = _clip01(1.0 - (shortcut_eval_score / eval_score))
        basis["shortcut_resistance"] = "measured"
    else:
        shortcut_resistance = 1.0
        basis["shortcut_resistance"] = "default_1.0_no_shortcut_control"

    # MechanismConfidence — pass-through from runner if present, else infer from
    # artifact / retrieval signals for this family.
    metrics_block = with_family.get("metrics") or {}
    mc = metrics_block.get("mechanism_score")
    if mc is None:
        mc = (with_family.get("evolve") or {}).get("mechanism_score")
    if mc is not None:
        mechanism_confidence = _clip01(float(mc))
        basis["mechanism_confidence"] = "runner_emitted"
    else:
        # Fallback: any artifact write or retrieval → 0.5, both → 1.0, else 0.0
        has_write = False
        has_retrieval = False
        for bk in ("learn", "update_or_stabilize", "reflection"):
            block = wp_bs.get(bk) or {}
            if (block.get("skill_create_count") or 0) + (block.get("memory_write_count") or 0) > 0:
                has_write = True
        for bk in ("evaluation", "eval_near", "eval_far"):
            block = wp_bs.get(bk) or {}
            if (block.get("retrieval_before_success_rate") or 0) > 0:
                has_retrieval = True
        if has_write and has_retrieval:
            mechanism_confidence = 1.0
        elif has_write or has_retrieval:
            mechanism_confidence = 0.5
        else:
            mechanism_confidence = 0.0
        basis["mechanism_confidence"] = "inferred_from_bucket_signals"

    # FamilyEvolveScore §14.1
    lift_clipped = max(0.0, ablation_lift) if ablation_lift is not None else None
    if None not in (lift_clipped, mechanism_confidence, autonomous_reuse_rate, transfer_robustness):
        family_evolve_score = (
            lift_clipped
            * mechanism_confidence
            * autonomous_reuse_rate
            * transfer_robustness
        )
    else:
        family_evolve_score = None

    return V2FamilyMetrics(
        family_id=family_id,
        primary_ability=ability,
        expected_substrate=substrate,
        primary_trigger=trigger,
        baseline_headroom=baseline_headroom,
        outcome_delta=outcome_delta,
        ablation_lift=ablation_lift,
        autonomous_reuse_rate=autonomous_reuse_rate,
        transfer_robustness=transfer_robustness,
        shortcut_resistance=shortcut_resistance,
        mechanism_confidence=mechanism_confidence,
        family_evolve_score=family_evolve_score,
        basis=basis,
    )


# ── ability + benchmark aggregation ──────────────────────────────────────


def aggregate_by_ability(
    family_metrics: list[V2FamilyMetrics],
) -> list[V2AbilityReport]:
    """Group family metrics by `primary_ability` and aggregate (§16)."""
    by_ab: dict[str, list[V2FamilyMetrics]] = {a: [] for a in sorted(V2_ABILITIES)}
    for fm in family_metrics:
        if fm.primary_ability in by_ab:
            by_ab[fm.primary_ability].append(fm)

    reports: list[V2AbilityReport] = []
    for ab, fams in by_ab.items():
        pos = sum(
            1 for f in fams if f.ablation_lift is not None and f.ablation_lift > 0
        )
        reports.append(
            V2AbilityReport(
                ability=ab,
                family_count=len(fams),
                families_with_positive_lift=pos,
                avg_outcome_delta=_safe_mean(f.outcome_delta for f in fams),
                avg_ablation_lift=_safe_mean(f.ablation_lift for f in fams),
                avg_mechanism_confidence=_safe_mean(f.mechanism_confidence for f in fams),
                avg_family_evolve_score=_safe_mean(f.family_evolve_score for f in fams),
            )
        )
    return reports


def build_benchmark_report(
    family_metrics: list[V2FamilyMetrics],
    ability_reports: list[V2AbilityReport],
    *,
    total_families: int,
    out_of_scope_abilities: set[str] | None = None,
) -> V2BenchmarkReport:
    in_scope = (
        {a for a in V2_ABILITIES if a not in (out_of_scope_abilities or set())}
    )
    vector: dict[str, float | None] = {
        r.ability: r.avg_family_evolve_score
        for r in ability_reports
        if r.ability in in_scope
    }

    # §16 four-question summary: pick the family with the highest score,
    # and expose the ability/trigger/substrate that drove it.
    scored = [f for f in family_metrics if f.family_evolve_score is not None]
    winner = max(scored, key=lambda f: f.family_evolve_score, default=None) if scored else None
    which_ability = winner.primary_ability if winner else None
    which_trigger = winner.primary_trigger if winner else None
    which_substrate = winner.expected_substrate if winner else None
    reuse_block: dict[str, float | None] = {
        "winning_family_evolve_score": winner.family_evolve_score if winner else None,
        "winning_ablation_lift": winner.ablation_lift if winner else None,
        "winning_transfer_robustness": winner.transfer_robustness if winner else None,
        "winning_autonomous_reuse_rate": winner.autonomous_reuse_rate if winner else None,
        "winning_mechanism_confidence": winner.mechanism_confidence if winner else None,
    }

    return V2BenchmarkReport(
        total_families=total_families,
        families_with_metrics=len(family_metrics),
        capability_vector=vector,
        overall_avg_outcome_delta=_safe_mean(f.outcome_delta for f in family_metrics),
        overall_avg_ablation_lift=_safe_mean(f.ablation_lift for f in family_metrics),
        overall_avg_family_evolve_score=_safe_mean(
            f.family_evolve_score for f in family_metrics
        ),
        ability_reports=ability_reports,
        which_ability_evolved=which_ability,
        under_which_trigger=which_trigger,
        through_which_substrate=which_substrate,
        reuse_strength_and_robustness=reuse_block,
        out_of_scope_abilities=sorted(out_of_scope_abilities or []),
    )


def _out_of_scope_abilities(v2_root: Path | None) -> set[str]:
    if v2_root is None or not v2_root.exists():
        return set()
    fams = load_v2_families(v2_root)
    by_ab: dict[str, list[str]] = {}
    for f in fams:
        by_ab.setdefault(f.primary_ability, []).append(f.status)
    return {
        ab for ab, statuses in by_ab.items()
        if statuses and all(s == "out_of_scope" for s in statuses)
    }


# ── runner-output ingestion + top-level driver ───────────────────────────


def _family_lookup(v2_root: Path | None) -> dict[str, V2FamilyMetadata]:
    if v2_root is None or not v2_root.exists():
        return {}
    return {f.family_id: f for f in load_v2_families(v2_root)}


def ingest_sequence_comparison(
    comparison_path: str | Path,
    *,
    v2_root: str | Path | None = "self-evolve-tasks-v2",
    shortcut_scores: dict[str, float] | None = None,
) -> list[V2FamilyMetrics]:
    """Load one `sequence_comparison.json` and emit family-level metrics."""
    data = json.loads(Path(comparison_path).read_text())
    with_block = data.get("with_persistence") or {}
    without_block = data.get("without_persistence") or None
    wp_families = with_block.get("family_summary") or {}
    wo_families = (without_block or {}).get("family_summary") or {}

    lookup = _family_lookup(Path(v2_root) if v2_root else None)
    shortcut_scores = shortcut_scores or {}

    out: list[V2FamilyMetrics] = []
    for fam_id, wp_fam in wp_families.items():
        # Match on loose family_id prefix. Runner uses names like
        # 'F01_skill_bootstrap' while v2 metadata uses 'PC01_sop_bootstrap'.
        meta = lookup.get(fam_id)
        if meta is None:
            # try primary mapping: scan for legacy_episodes matching the runner label
            for candidate in lookup.values():
                if fam_id in {candidate.family_id, *(candidate.legacy_episodes or [])}:
                    meta = candidate
                    break
        out.append(
            compute_family_metrics(
                family_id=fam_id,
                family_meta=meta,
                with_family=wp_fam,
                without_family=wo_families.get(fam_id),
                shortcut_eval_score=shortcut_scores.get(fam_id),
            )
        )
    return out


def _auto_load_shortcut_scores(trace_dir: Path) -> dict[str, float]:
    """Pick up `shortcut_scores.json` from the trace dir if present.

    Format: `{ "<family_id>": <eval_score_float>, ... }`. When Phase C/D
    adds shortcut-control task variants, writing that file next to
    `sequence_comparison.json` is enough for the metrics pipeline to pick
    them up — no other glue needed.
    """
    candidate = trace_dir / "shortcut_scores.json"
    if not candidate.exists():
        return {}
    try:
        data = json.loads(candidate.read_text())
        return {k: float(v) for k, v in data.items() if v is not None}
    except (ValueError, TypeError):
        return {}


def ingest_full_reports(
    comparison_path: str | Path,
    *,
    v2_root: str | Path | None = "self-evolve-tasks-v2",
    shortcut_scores: dict[str, float] | None = None,
) -> tuple[
    list[V2FamilyMetrics],
    list[V2Layer2Metrics],
    list[V2Layer3Metrics],
    list[dict[str, Any]],
]:
    """Ingest one `sequence_comparison.json` and emit Layers 1/2/3 + task-level rows."""
    data = json.loads(Path(comparison_path).read_text())
    with_block = data.get("with_persistence") or {}
    without_block = data.get("without_persistence") or None
    wp_families = with_block.get("family_summary") or {}
    wo_families = (without_block or {}).get("family_summary") or {}

    lookup = _family_lookup(Path(v2_root) if v2_root else None)
    shortcut_scores = shortcut_scores or _auto_load_shortcut_scores(Path(comparison_path).parent)

    layer1: list[V2FamilyMetrics] = []
    layer2: list[V2Layer2Metrics] = []
    layer3: list[V2Layer3Metrics] = []
    task_rows: list[dict[str, Any]] = []

    for fam_id, wp_fam in wp_families.items():
        meta = lookup.get(fam_id)
        if meta is None:
            # Resolve legacy family_ids like "F01_skill_bootstrap" by scanning
            # v2 families whose legacy_episodes start with the same F##_ prefix.
            legacy_prefix = fam_id.split("_")[0] + "_" if "_" in fam_id else ""
            for candidate in lookup.values():
                eps = candidate.legacy_episodes or []
                if legacy_prefix and any(e.startswith(legacy_prefix) for e in eps):
                    meta = candidate
                    break
                if fam_id in {candidate.family_id, *eps}:
                    meta = candidate
                    break

        layer1.append(
            compute_family_metrics(
                family_id=fam_id,
                family_meta=meta,
                with_family=wp_fam,
                without_family=wo_families.get(fam_id),
                shortcut_eval_score=shortcut_scores.get(fam_id),
            )
        )
        primary_ability = meta.primary_ability if meta else ""
        primary_trigger = meta.primary_trigger if meta else ""
        layer2.append(compute_layer2(fam_id, primary_ability, wp_fam, with_block))
        l3 = compute_layer3(fam_id, primary_trigger, wp_fam, with_block)

        # Wire trace-parser helpers into Layer 3 if the family declared
        # trigger/banned/stale phrases and the episode traces exist on disk.
        if meta is not None and (
            meta.trigger_phrases or meta.banned_phrases or meta.stale_phrases
        ):
            eps = wp_fam.get("episodes") or []
            first_turns: list[int] = []
            repeat_hits = 0
            leak_hits = 0
            trace_dir_base = Path(comparison_path).parent
            for ep in eps:
                trace_rel = ep.get("trace")
                if not trace_rel:
                    continue
                trace_path = (trace_dir_base.parent / trace_rel).resolve()
                if not trace_path.exists():
                    trace_path = Path(trace_rel)
                if not trace_path.exists():
                    continue
                if meta.trigger_phrases:
                    t = first_use_turn(trace_path, meta.trigger_phrases)
                    if t is not None:
                        first_turns.append(t)
                if meta.banned_phrases:
                    repeat_hits += repeat_violation_count(trace_path, meta.banned_phrases)
                if meta.stale_phrases:
                    leak_hits += stale_leak_events(trace_path, meta.stale_phrases)

            if first_turns:
                l3.adoption_latency = float(sum(first_turns) / len(first_turns))
                l3.basis["adoption_latency"] = "trace_parsed_first_use_turn_avg"
            if meta.banned_phrases:
                # Normalize against episode count so values are comparable
                denom = max(len(eps), 1)
                l3.repeat_violation_rate = repeat_hits / denom
                l3.basis["repeat_violation_rate"] = "trace_parsed_repeat_violation_count"
            if meta.stale_phrases:
                denom = max(len(eps), 1)
                l3.stale_leak_penalty = leak_hits / denom
                l3.basis["stale_leak_penalty"] = "trace_parsed_stale_leak_events"

        layer3.append(l3)

        # Task-level rows: one per episode in the with_persistence family summary
        for ep in wp_fam.get("episodes") or []:
            task_rows.append(
                {
                    "family_id": fam_id,
                    "primary_ability": primary_ability,
                    "primary_trigger": primary_trigger,
                    "task_id": ep.get("task_id"),
                    "task_name": ep.get("task_name"),
                    "task_score": ep.get("task_score"),
                    "passed": ep.get("passed"),
                    "total_turns": ep.get("total_turns"),
                    "tool_dispatch_count": ep.get("tool_dispatch_count"),
                    "artifacts": {
                        "skill_count": (ep.get("artifacts") or {}).get("skill_count", 0),
                        "memory_entries": len(
                            (ep.get("artifacts") or {}).get("memory_entries", [])
                        ),
                    },
                    "mechanism_fired": (ep.get("scores") or {}).get("mechanism_fired"),
                    "attribution": (ep.get("scores") or {}).get("attribution"),
                }
            )

    return layer1, layer2, layer3, task_rows


def emit_reports(
    family_metrics: list[V2FamilyMetrics],
    out_dir: str | Path,
    *,
    v2_root: str | Path | None = "self-evolve-tasks-v2",
    layer2: list[V2Layer2Metrics] | None = None,
    layer3: list[V2Layer3Metrics] | None = None,
    task_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Path]:
    """Write task/family/ability/benchmark JSON reports to `out_dir`.

    All 4 levels from §16:
    - `v2_task_report.json`      — per-episode rollup (if `task_rows` given)
    - `v2_family_report.json`    — Layer 1 metrics (+ optional Layer 2/3)
    - `v2_ability_report.json`   — ability aggregation
    - `v2_benchmark_report.json` — 7-dim capability vector
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    total = 0
    if v2_root is not None:
        total = len(load_v2_families(Path(v2_root))) if Path(v2_root).exists() else 0

    ability = aggregate_by_ability(family_metrics)
    oos = _out_of_scope_abilities(Path(v2_root)) if v2_root else set()
    benchmark = build_benchmark_report(
        family_metrics, ability,
        total_families=total or len(family_metrics),
        out_of_scope_abilities=oos,
    )

    paths: dict[str, Path] = {
        "family_report": out / "v2_family_report.json",
        "ability_report": out / "v2_ability_report.json",
        "benchmark_report": out / "v2_benchmark_report.json",
    }

    family_payload: list[dict[str, Any]] = []
    l2_by_id = {m.family_id: m for m in (layer2 or [])}
    l3_by_id = {m.family_id: m for m in (layer3 or [])}
    for fm in family_metrics:
        entry = fm.model_dump()
        if fm.family_id in l2_by_id:
            entry["layer2"] = l2_by_id[fm.family_id].model_dump()
        if fm.family_id in l3_by_id:
            entry["layer3"] = l3_by_id[fm.family_id].model_dump()
        family_payload.append(entry)

    paths["family_report"].write_text(json.dumps(family_payload, indent=2))
    paths["ability_report"].write_text(
        json.dumps([a.model_dump() for a in ability], indent=2)
    )
    paths["benchmark_report"].write_text(json.dumps(benchmark.model_dump(), indent=2))

    if task_rows is not None:
        paths["task_report"] = out / "v2_task_report.json"
        paths["task_report"].write_text(json.dumps(task_rows, indent=2))

    # §16 human-readable markdown summary alongside the JSON
    paths["markdown_report"] = out / "v2_benchmark_report.md"
    paths["markdown_report"].write_text(_render_markdown_report(benchmark, ability, family_metrics))

    return paths


def _render_markdown_report(
    benchmark: V2BenchmarkReport,
    ability_reports: list[V2AbilityReport],
    family_metrics: list[V2FamilyMetrics],
) -> str:
    def _fmt(x: float | None, digits: int = 3) -> str:
        return "—" if x is None else f"{x:.{digits}f}"

    lines: list[str] = [
        "# V2 Self-Evolve Benchmark Report",
        "",
        "## §16 Four-Question Summary",
        "",
        f"- **Which ability evolved:** `{benchmark.which_ability_evolved or 'no positive FamilyEvolveScore'}`",
        f"- **Under which trigger:** `{benchmark.under_which_trigger or '—'}`",
        f"- **Through which substrate:** `{benchmark.through_which_substrate or '—'}`",
        f"- **Winning FamilyEvolveScore:** "
        f"{_fmt(benchmark.reuse_strength_and_robustness.get('winning_family_evolve_score'))}"
        f"  (lift={_fmt(benchmark.reuse_strength_and_robustness.get('winning_ablation_lift'))},"
        f" reuse={_fmt(benchmark.reuse_strength_and_robustness.get('winning_autonomous_reuse_rate'))},"
        f" transfer={_fmt(benchmark.reuse_strength_and_robustness.get('winning_transfer_robustness'))},"
        f" mc={_fmt(benchmark.reuse_strength_and_robustness.get('winning_mechanism_confidence'))})",
        "",
        "## Overall Averages",
        "",
        f"- Total families in tree: **{benchmark.total_families}**",
        f"- Families with metrics this run: **{benchmark.families_with_metrics}**",
        f"- Avg outcome delta: {_fmt(benchmark.overall_avg_outcome_delta)}",
        f"- Avg ablation lift: {_fmt(benchmark.overall_avg_ablation_lift)}",
        f"- Avg FamilyEvolveScore: {_fmt(benchmark.overall_avg_family_evolve_score)}",
        f"- Out-of-scope abilities: {', '.join(benchmark.out_of_scope_abilities) or 'none'}",
        "",
        "## Ability Capability Vector (§16)",
        "",
        "| Ability | Families | Positive lift | Avg outcome Δ | Avg lift | Avg mechanism | Avg FamilyEvolveScore |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for ar in ability_reports:
        if ar.ability in benchmark.out_of_scope_abilities:
            continue
        lines.append(
            f"| `{ar.ability}` | {ar.family_count} | {ar.families_with_positive_lift} | "
            f"{_fmt(ar.avg_outcome_delta)} | {_fmt(ar.avg_ablation_lift)} | "
            f"{_fmt(ar.avg_mechanism_confidence)} | {_fmt(ar.avg_family_evolve_score)} |"
        )

    lines += [
        "",
        "## Per-Family Layer 1 Metrics",
        "",
        "| Family | Ability | Trigger | Substrate | Baseline Δ→ | Eval Δ | Ablation Lift | Reuse | Transfer | Shortcut | MC | Evolve |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for fm in family_metrics:
        lines.append(
            f"| `{fm.family_id}` | {fm.primary_ability} | {fm.primary_trigger} | {fm.expected_substrate} | "
            f"{_fmt(fm.baseline_headroom)} | {_fmt(fm.outcome_delta)} | {_fmt(fm.ablation_lift)} | "
            f"{_fmt(fm.autonomous_reuse_rate)} | {_fmt(fm.transfer_robustness)} | "
            f"{_fmt(fm.shortcut_resistance)} | {_fmt(fm.mechanism_confidence)} | "
            f"**{_fmt(fm.family_evolve_score)}** |"
        )
    lines += [
        "",
        "## Metric Basis",
        "",
        "For each metric per family, the `basis` tag distinguishes **measured** values "
        "(from real run data) from **defaults** (neutral fallback when the required signal "
        "was not collected). See `v2_family_report.json → basis` for details.",
        "",
    ]
    return "\n".join(lines) + "\n"
