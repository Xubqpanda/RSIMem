#!/usr/bin/env python3
"""
extract_appendix_data.py — Parse all PAST-Bench trace data and produce JSON
outputs for the paper appendix.

Outputs (written to scripts/appendix_data/):
  1. per_family_scores.json
  2. variance_analysis.json
  3. trajectory_episodes.json
  4. computational_cost.json
  5. failure_modes.json
  6. ablation_per_family.json
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

# ── paths ──────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
TRACE_ROOT = Path(os.environ.get("PAST_BENCH_TRACE_ROOT", REPO_ROOT / "self-evolve-traces-final"))
OUT_DIR = Path(os.environ.get("PAST_BENCH_APPENDIX_OUT_DIR", REPO_ROOT / "scripts" / "appendix_data"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── family → ability mapping ───────────────────────────────────────────────
FAMILY_ABILITY = {
    # Memory
    "SM01_preference_adoption": "Memory",
    "SM02_constraint_retention": "Memory",
    "SM05_weak_trigger_preference_adoption": "Memory",
    "EP01_prior_case_recall": "Memory",
    "EP02_exception_list_recall": "Memory",
    # Procedural
    "PC01_sop_bootstrap_01": "Procedural",
    "PC01_sop_bootstrap_02": "Procedural",
    "PC01_sop_bootstrap_03": "Procedural",
    "PC01_sop_bootstrap_04": "Procedural",
    "PC01_sop_bootstrap_05": "Procedural",
    "PC01_sop_bootstrap_06": "Procedural",
    "PC03_latent_rule_induction_01": "Procedural",
    "PC04_failure_to_rule_01": "Procedural",
    # Info (Proactive Information Gathering)
    "PG01_release_decision_followup": "Info",
    "PG02_ops_exception_desk": "Info",
    "PG03_oncall_handoff_lookup": "Info",
    "PG04_temporary_waiver_audit": "Info",
    "PG05_change_freeze_followup": "Info",
    "PG06_kappa_integration_review": "Info",
    # Update
    "SM03_fact_correction": "Update",
    "SM04_rule_migration": "Update",
    "SM06_temporary_exception_pollution": "Update",
    "SM07_scoped_rule_migration": "Update",
    "EP03_recall_then_modify": "Update",
    "PC02_sop_patch_01": "Update",
    "PC02_sop_patch_02": "Update",
}

# Also map ability directory names used in ablation
ABILITY_DIR_MAP = {
    "memory_ability": "Memory",
    "procedural_ability": "Procedural",
    "proactive_information_gathering": "Info",
    "update_ability": "Update",
}


def _safe_mean(vals):
    """Return mean or None for empty lists."""
    return round(mean(vals), 4) if vals else None


def _safe_stdev(vals):
    """Return sample stdev or None if <2 values."""
    return round(stdev(vals), 4) if len(vals) >= 2 else None


# ── Step 1: Discover and load every sequence_results.json ──────────────────

def find_sequence_files(root: Path):
    """Yield every sequence_results.json under *root*."""
    for dirpath, _, filenames in os.walk(root):
        if "sequence_results.json" in filenames:
            yield Path(dirpath) / "sequence_results.json"


def load_seq(path: Path):
    """Load a sequence_results.json, return dict with 'variant' and 'episodes'."""
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        # Some files are lists of dicts — take first
        data = data[0] if data else {"variant": None, "episodes": []}
    return data


def classify_path(path: Path):
    """
    Given an absolute path to a sequence_results.json, return a dict:
      framework, model, run, family (if determinable), variant_hint
    """
    rel = path.relative_to(TRACE_ROOT)
    parts = list(rel.parts)

    info = {
        "framework": None,
        "model": None,
        "run": None,
        "is_ablation": False,
        "ablation_setting": None,
    }

    if parts[0] == "_ablation":
        info["is_ablation"] = True
        info["ablation_setting"] = parts[1]  # e.g. base_hermes, planning, ...
        info["framework"] = "_ablation/" + parts[1]
        info["model"] = "minimax_m27"
        info["run"] = "run1"
        return info

    # Normal: framework/model/runN/...
    info["framework"] = parts[0]
    if len(parts) > 1:
        info["model"] = parts[1]
    if len(parts) > 2:
        info["run"] = parts[2]

    return info


def determine_variant(data, path: Path):
    """
    Determine whether a sequence_results.json is 'with_persistence' or
    'without_persistence'.  Uses the 'variant' field first, then falls
    back to the directory name.
    """
    v = data.get("variant")
    if v in ("with_persistence", "without_persistence"):
        return v
    # Check path segments
    for part in path.parts:
        if part == "with_persistence":
            return "with_persistence"
        if part == "without_persistence":
            return "without_persistence"
    # Default: treat as with_persistence (hermes single-file format)
    return "with_persistence"


# ── Master data structure ──────────────────────────────────────────────────
# episodes_db[framework][model][run][family_id][variant] = [episode, ...]
episodes_db = defaultdict(
    lambda: defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(list)
            )
        )
    )
)

# Ablation: ablation_db[setting][family_id] = [episodes...]
ablation_db = defaultdict(lambda: defaultdict(list))

print("=== Scanning trace files ===")
all_files = sorted(find_sequence_files(TRACE_ROOT))
print(f"Found {len(all_files)} sequence_results.json files")

loaded = 0
for fpath in all_files:
    info = classify_path(fpath)
    data = load_seq(fpath)
    variant = determine_variant(data, fpath)

    episodes = data.get("episodes", [])
    if not episodes:
        continue
    loaded += 1

    for ep in episodes:
        fam = ep.get("family_id") or "UNKNOWN"
        # Ensure family_id is set
        if fam == "UNKNOWN":
            # Try to infer from label
            label = ep.get("label", "")
            for known_fam in FAMILY_ABILITY:
                if known_fam.lower() in label.lower():
                    fam = known_fam
                    break

        if info["is_ablation"]:
            ablation_db[info["ablation_setting"]][fam].append(ep)
        else:
            fw = info["framework"]
            model = info["model"]
            run = info["run"]
            # For files in hermes format: single file has both w/ and w/o
            # episodes distinguished by persistence_allowed
            if variant == "with_persistence":
                if ep.get("persistence_allowed") is False:
                    episodes_db[fw][model][run][fam]["without_persistence"].append(ep)
                else:
                    episodes_db[fw][model][run][fam]["with_persistence"].append(ep)
            else:
                episodes_db[fw][model][run][fam][variant].append(ep)

print(f"Loaded episodes from {loaded} files")


# ── Helper: compute eval score for a list of episodes ─────────────────────

def eval_score(episodes, variant="with_persistence"):
    """
    Mean task_score of evaluation-bucket episodes.
    For hermes-style data: eval bucket or 'evaluation'.
    """
    eval_eps = [
        ep for ep in episodes
        if ep.get("bucket") in ("evaluation", "eval")
    ]
    if not eval_eps:
        return None
    return round(mean(e["task_score"] for e in eval_eps), 4)


def all_eval_scores_by_family(fw, model, run_id, fam_id, var):
    """Get list of eval episode scores for a specific config."""
    eps = episodes_db.get(fw, {}).get(model, {}).get(run_id, {}).get(fam_id, {}).get(var, [])
    return [
        ep["task_score"]
        for ep in eps
        if ep.get("bucket") in ("evaluation", "eval")
    ]


# ── Output 1: per_family_scores.json ─────────────────────────────────────

print("\n=== Building per_family_scores.json ===")

per_family_scores = {}

# Iterate over all framework/model combos
for fw in sorted(episodes_db):
    for model in sorted(episodes_db[fw]):
        runs = sorted(episodes_db[fw][model].keys())
        all_families = set()
        for r in runs:
            all_families.update(episodes_db[fw][model][r].keys())

        for fam in sorted(all_families):
            ability = FAMILY_ABILITY.get(fam, "Unknown")
            key = f"{fw}/{model}/{fam}"

            w_scores_by_run = []
            wo_scores_by_run = []

            for r in runs:
                w_vals = all_eval_scores_by_family(fw, model, r, fam, "with_persistence")
                wo_vals = all_eval_scores_by_family(fw, model, r, fam, "without_persistence")
                if w_vals:
                    w_scores_by_run.append(_safe_mean(w_vals))
                if wo_vals:
                    wo_scores_by_run.append(_safe_mean(wo_vals))

            w_mean = _safe_mean(w_scores_by_run) if w_scores_by_run else None
            wo_mean = _safe_mean(wo_scores_by_run) if wo_scores_by_run else None
            delta = round(w_mean - wo_mean, 4) if (w_mean is not None and wo_mean is not None) else None

            entry = {
                "framework": fw,
                "model": model,
                "family_id": fam,
                "ability": ability,
                "n_runs": len(runs),
                "runs_with_data": len(w_scores_by_run),
                "w_mean": w_mean,
                "wo_mean": wo_mean,
                "delta": delta,
            }
            if len(w_scores_by_run) >= 2:
                entry["w_std"] = _safe_stdev(w_scores_by_run)
            if len(wo_scores_by_run) >= 2:
                entry["wo_std"] = _safe_stdev(wo_scores_by_run)

            per_family_scores[key] = entry

# Group by ability for convenience
per_family_grouped = defaultdict(list)
for key, entry in per_family_scores.items():
    per_family_grouped[entry["ability"]].append(entry)

output1 = {
    "description": "Per-family eval scores: mean w/ and w/o persistence across runs, with delta",
    "by_ability": {
        ability: sorted(entries, key=lambda x: (x["framework"], x["model"], x["family_id"]))
        for ability, entries in sorted(per_family_grouped.items())
    },
    "all_entries": sorted(per_family_scores.values(),
                          key=lambda x: (x["framework"], x["model"], x["family_id"])),
}

with open(OUT_DIR / "per_family_scores.json", "w") as f:
    json.dump(output1, f, indent=2)
print(f"  Wrote {len(per_family_scores)} entries to per_family_scores.json")


# ── Output 2: variance_analysis.json ──────────────────────────────────────

print("\n=== Building variance_analysis.json ===")

variance_configs = [
    ("hermes", "minimax_m27"),
    ("hermes_plus", "minimax_m27"),
]

variance_analysis = {}

for fw, model in variance_configs:
    runs = sorted(episodes_db.get(fw, {}).get(model, {}).keys())
    if len(runs) < 2:
        continue

    config_key = f"{fw}/{model}"
    families_data = {}
    ability_agg = defaultdict(lambda: {"w_scores": [], "wo_scores": []})

    all_families = set()
    for r in runs:
        all_families.update(episodes_db[fw][model][r].keys())

    for fam in sorted(all_families):
        ability = FAMILY_ABILITY.get(fam, "Unknown")
        w_per_run = []
        wo_per_run = []

        for r in runs:
            w_vals = all_eval_scores_by_family(fw, model, r, fam, "with_persistence")
            wo_vals = all_eval_scores_by_family(fw, model, r, fam, "without_persistence")
            if w_vals:
                w_per_run.append(_safe_mean(w_vals))
            if wo_vals:
                wo_per_run.append(_safe_mean(wo_vals))

        families_data[fam] = {
            "ability": ability,
            "n_runs_w": len(w_per_run),
            "n_runs_wo": len(wo_per_run),
            "w_mean": _safe_mean(w_per_run),
            "w_std": _safe_stdev(w_per_run),
            "wo_mean": _safe_mean(wo_per_run),
            "wo_std": _safe_stdev(wo_per_run),
            "w_per_run": w_per_run,
            "wo_per_run": wo_per_run,
        }

        if w_per_run:
            ability_agg[ability]["w_scores"].extend(w_per_run)
        if wo_per_run:
            ability_agg[ability]["wo_scores"].extend(wo_per_run)

    ability_summary = {}
    for ability, agg in sorted(ability_agg.items()):
        ability_summary[ability] = {
            "w_mean": _safe_mean(agg["w_scores"]),
            "w_std": _safe_stdev(agg["w_scores"]),
            "wo_mean": _safe_mean(agg["wo_scores"]),
            "wo_std": _safe_stdev(agg["wo_scores"]),
            "n_w": len(agg["w_scores"]),
            "n_wo": len(agg["wo_scores"]),
        }

    variance_analysis[config_key] = {
        "runs": runs,
        "per_family": families_data,
        "per_ability": ability_summary,
    }

output2 = {
    "description": "Variance analysis for multi-run configs: per-family and per-ability mean +/- std",
    "configs": variance_analysis,
}

with open(OUT_DIR / "variance_analysis.json", "w") as f:
    json.dump(output2, f, indent=2)
print(f"  Wrote {len(variance_analysis)} configs to variance_analysis.json")


# ── Output 3: trajectory_episodes.json ────────────────────────────────────

print("\n=== Building trajectory_episodes.json ===")

# Representative families
TRAJECTORY_FAMILIES = ["SM01_preference_adoption", "PC01_sop_bootstrap_03",
                       "PG01_release_decision_followup", "SM03_fact_correction"]

# Fallbacks
TRAJECTORY_FALLBACKS = {
    "PC01_sop_bootstrap_03": ["PC01_sop_bootstrap_01", "PC01_sop_bootstrap_05"],
}

trajectory_data = {}
fw, model, run = "hermes", "minimax_m27", "run1"

for fam in TRAJECTORY_FAMILIES:
    candidates = [fam] + TRAJECTORY_FALLBACKS.get(fam, [])
    chosen = None
    for c in candidates:
        w_eps = episodes_db.get(fw, {}).get(model, {}).get(run, {}).get(c, {}).get("with_persistence", [])
        if w_eps:
            chosen = c
            break

    if chosen is None:
        # Try run2 or run3
        for alt_run in ["run2", "run3"]:
            for c in candidates:
                w_eps = episodes_db.get(fw, {}).get(model, {}).get(alt_run, {}).get(c, {}).get("with_persistence", [])
                if w_eps:
                    chosen = c
                    run_used = alt_run
                    break
            if chosen:
                break
        if chosen is None:
            trajectory_data[fam] = {"note": "no data found", "with_persistence": [], "without_persistence": []}
            continue
    else:
        run_used = run

    for var in ["with_persistence", "without_persistence"]:
        eps = episodes_db.get(fw, {}).get(model, {}).get(run_used, {}).get(chosen, {}).get(var, [])
        traj = []
        for i, ep in enumerate(eps):
            traj.append({
                "index": i,
                "label": ep.get("label", ""),
                "phase": ep.get("phase", ""),
                "bucket": ep.get("bucket", ""),
                "task_score": ep.get("task_score"),
                "persistence_allowed": ep.get("persistence_allowed"),
            })

        key = fam if fam == chosen else f"{fam} (using {chosen})"
        if key not in trajectory_data:
            trajectory_data[key] = {"source": f"{fw}/{model}/{run_used}/{chosen}"}
        trajectory_data[key][var] = traj

output3 = {
    "description": "Episode-by-episode trajectory for representative families from Hermes/MiniMax",
    "families": trajectory_data,
}

with open(OUT_DIR / "trajectory_episodes.json", "w") as f:
    json.dump(output3, f, indent=2)
print(f"  Wrote {len(trajectory_data)} families to trajectory_episodes.json")


# ── Output 4: computational_cost.json ─────────────────────────────────────

print("\n=== Building computational_cost.json ===")

cost_data = {}

for fw in sorted(episodes_db):
    for model in sorted(episodes_db[fw]):
        config_key = f"{fw}/{model}"
        all_tokens = []
        all_wall_times = []
        total_tokens = 0
        total_wall = 0.0
        episode_count = 0

        for run_id in sorted(episodes_db[fw][model]):
            for fam in episodes_db[fw][model][run_id]:
                for var in episodes_db[fw][model][run_id][fam]:
                    for ep in episodes_db[fw][model][run_id][fam][var]:
                        tu = ep.get("token_usage") or {}
                        ti = ep.get("timing") or {}

                        tok = tu.get("total_tokens")
                        wall = ti.get("wall_time_s")

                        if tok is not None:
                            all_tokens.append(tok)
                            total_tokens += tok
                        if wall is not None:
                            all_wall_times.append(wall)
                            total_wall += wall
                        episode_count += 1

        cost_data[config_key] = {
            "episode_count": episode_count,
            "avg_tokens_per_episode": round(mean(all_tokens), 1) if all_tokens else None,
            "avg_wall_time_per_episode_s": round(mean(all_wall_times), 2) if all_wall_times else None,
            "total_tokens": total_tokens,
            "total_wall_time_s": round(total_wall, 2),
            "total_wall_time_h": round(total_wall / 3600, 2) if total_wall else 0,
        }

output4 = {
    "description": "Computational cost per (framework, model): avg tokens/episode, avg wall time/episode, totals",
    "configs": cost_data,
}

with open(OUT_DIR / "computational_cost.json", "w") as f:
    json.dump(output4, f, indent=2)
print(f"  Wrote {len(cost_data)} configs to computational_cost.json")


# ── Output 5: failure_modes.json ──────────────────────────────────────────

print("\n=== Building failure_modes.json ===")

failure_data = {"hermes/minimax_m27": {"runs": {}, "summary": {}}}

fw, model = "hermes", "minimax_m27"
runs = sorted(episodes_db.get(fw, {}).get(model, {}).keys())

global_failure_cats = defaultdict(int)
global_failure_families = defaultdict(list)

for run_id in runs:
    run_failures = []
    for fam in sorted(episodes_db[fw][model][run_id]):
        ability = FAMILY_ABILITY.get(fam, "Unknown")
        w_eps = episodes_db[fw][model][run_id][fam].get("with_persistence", [])

        for ep in w_eps:
            if ep.get("bucket") not in ("evaluation", "eval"):
                continue
            score = ep.get("task_score", 1.0)
            if score >= 0.5:
                continue

            # Failure — categorize
            arts = ep.get("artifacts") or {}
            ret = ep.get("retrieval_signals") or {}

            mem_entries = arts.get("memory_entries", [])
            mem_count = len(mem_entries) if isinstance(mem_entries, list) else (mem_entries or 0)
            skill_count = arts.get("skill_count", 0)

            mem_read = ret.get("memory_read_count", 0) or 0
            mem_inj = ret.get("memory_injection_count", 0) or 0
            skill_read = ret.get("skill_read_count", 0) or 0
            retrieval_count = ret.get("retrieval_signal_count", 0) or 0
            used_expected = ret.get("used_expected_signal", False)

            category = "unknown"

            if ability == "Memory":
                if mem_count == 0 and mem_read == 0:
                    category = "no_write_no_read"
                elif mem_count > 0 and (mem_read == 0 and mem_inj == 0):
                    category = "wrote_but_no_retrieval"
                elif mem_count > 0 and (mem_read > 0 or mem_inj > 0):
                    category = "retrieved_but_wrong_answer"
                else:
                    category = "no_write_but_read_attempted"

            elif ability == "Procedural":
                if skill_count == 0:
                    category = "no_skill_created"
                elif skill_count > 0 and skill_read == 0:
                    category = "skill_created_not_used"
                elif skill_count > 0 and skill_read > 0:
                    category = "skill_used_but_wrong"
                else:
                    category = "other_procedural"

            elif ability == "Info":
                if retrieval_count == 0:
                    category = "no_retrieval_attempted"
                elif not used_expected:
                    category = "retrieval_missed_target"
                else:
                    category = "retrieval_ok_but_wrong_answer"

            elif ability == "Update":
                if mem_count == 0 and skill_count == 0:
                    category = "no_update_written"
                elif not used_expected:
                    category = "stale_or_wrong_retrieval"
                else:
                    category = "update_written_but_wrong"

            failure_entry = {
                "family_id": fam,
                "ability": ability,
                "label": ep.get("label", ""),
                "task_score": score,
                "category": category,
                "memory_entries_count": mem_count,
                "skill_count": skill_count,
                "retrieval_signal_count": retrieval_count,
                "used_expected_signal": used_expected,
            }
            run_failures.append(failure_entry)
            global_failure_cats[category] += 1
            global_failure_families[fam].append(category)

    failure_data["hermes/minimax_m27"]["runs"][run_id] = {
        "failure_count": len(run_failures),
        "failures": run_failures,
    }

# Build summary
cat_summary = dict(sorted(global_failure_cats.items(), key=lambda x: -x[1]))
fam_summary = {}
for fam, cats in sorted(global_failure_families.items()):
    cat_counts = defaultdict(int)
    for c in cats:
        cat_counts[c] += 1
    fam_summary[fam] = {
        "total_failures": len(cats),
        "categories": dict(sorted(cat_counts.items(), key=lambda x: -x[1])),
    }

failure_data["hermes/minimax_m27"]["summary"] = {
    "total_failures_all_runs": sum(global_failure_cats.values()),
    "by_category": cat_summary,
    "by_family": fam_summary,
}

output5 = {
    "description": "Failure mode analysis for eval episodes scoring < 0.5 under w/ persistence (Hermes/MiniMax, all runs)",
    "data": failure_data,
}

with open(OUT_DIR / "failure_modes.json", "w") as f:
    json.dump(output5, f, indent=2)
total_failures = sum(global_failure_cats.values())
print(f"  Wrote failure_modes.json ({total_failures} total failures across {len(runs)} runs)")


# ── Output 6: ablation_per_family.json ────────────────────────────────────

print("\n=== Building ablation_per_family.json ===")

ablation_output = {}

for setting in sorted(ablation_db):
    setting_data = {}
    for fam in sorted(ablation_db[setting]):
        ability = FAMILY_ABILITY.get(fam, "Unknown")
        eps = ablation_db[setting][fam]

        # All ablation files are with_persistence variant
        eval_eps = [ep for ep in eps if ep.get("bucket") in ("evaluation", "eval")]
        all_eps_scores = [ep["task_score"] for ep in eps if ep.get("task_score") is not None]
        eval_scores = [ep["task_score"] for ep in eval_eps if ep.get("task_score") is not None]

        setting_data[fam] = {
            "ability": ability,
            "total_episodes": len(eps),
            "eval_episodes": len(eval_eps),
            "eval_mean": _safe_mean(eval_scores),
            "all_episodes_mean": _safe_mean(all_eps_scores),
        }

    ablation_output[setting] = setting_data

output6 = {
    "description": "Ablation per-family w/ persistence eval scores for each ablation setting",
    "settings": ablation_output,
}

with open(OUT_DIR / "ablation_per_family.json", "w") as f:
    json.dump(output6, f, indent=2)
print(f"  Wrote {len(ablation_output)} ablation settings to ablation_per_family.json")


# ── Summary ────────────────────────────────────────────────────────────────

print("\n=== Summary ===")
print(f"Output directory: {OUT_DIR}")
for fname in sorted(os.listdir(OUT_DIR)):
    fpath = OUT_DIR / fname
    size_kb = fpath.stat().st_size / 1024
    print(f"  {fname:40s} {size_kb:8.1f} KB")

# Print some high-level stats
print("\n--- Frameworks/Models found ---")
for fw in sorted(episodes_db):
    for model in sorted(episodes_db[fw]):
        runs = sorted(episodes_db[fw][model].keys())
        n_families = set()
        for r in runs:
            n_families.update(episodes_db[fw][model][r].keys())
        print(f"  {fw}/{model}: {len(runs)} runs, {len(n_families)} families")

print("\n--- Ablation settings ---")
for setting in sorted(ablation_db):
    fams = sorted(ablation_db[setting].keys())
    print(f"  {setting}: {len(fams)} families")

print("\nDone.")
