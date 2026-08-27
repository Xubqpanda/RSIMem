#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RSIMEM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PAST_BENCH_ROOT="${PAST_BENCH_ROOT:-${RSIMEM_ROOT}/benchmarks/past-bench}"
PAST_BENCH_BIN="${RSIMEM_ROOT}/.venv/bin/past-bench"
PYTHON_BIN="${RSIMEM_ROOT}/.venv/bin/python"
REPLICATES="${RSIMEM_REPLICATES:-3}"
REPLICATE_START="${RSIMEM_REPLICATE_START:-1}"
REPLICATE_END="${RSIMEM_REPLICATE_END:-${REPLICATES}}"
TASK_FAMILY="memory_ability/SM01_preference_adoption"
ADAPTIVE_CONFIG="${RSIMEM_ADAPTIVE_CONFIG:-}"

if [[ -z "${GPT_LUNA_API_KEY:-}" ]]; then
  echo "GPT_LUNA_API_KEY is required." >&2
  exit 2
fi
if [[ -z "${ADAPTIVE_CONFIG}" || ! -f "${ADAPTIVE_CONFIG}" ]]; then
  echo "RSIMEM_ADAPTIVE_CONFIG must name a prepared adaptive JSON config." >&2
  exit 2
fi
ADAPTIVE_CONFIG="$(realpath "${ADAPTIVE_CONFIG}")"
if [[ ! -x "${PAST_BENCH_BIN}" || ! -f "${PAST_BENCH_ROOT}/pyproject.toml" ]]; then
  echo "RSIMem or PAST-Bench environment is incomplete." >&2
  exit 2
fi
if [[ ! "${REPLICATES}" =~ ^[1-9][0-9]*$ ]] || (( REPLICATES < 3 )); then
  echo "RSIMEM_REPLICATES must be at least 3." >&2
  exit 2
fi
if (
  [[ ! "${REPLICATE_START}" =~ ^[1-9][0-9]*$ ]] ||
  [[ ! "${REPLICATE_END}" =~ ^[1-9][0-9]*$ ]] ||
  (( REPLICATE_START > REPLICATE_END )) ||
  (( REPLICATE_END > REPLICATES ))
); then
  echo "RSIMEM_REPLICATE_START/END must select the configured replicate range." >&2
  exit 2
fi

batch_id="${RSIMEM_BATCH_ID:-$(date +%Y%m%d_%H%M%S)}"
if [[ ! "${batch_id}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "RSIMEM_BATCH_ID contains unsupported characters." >&2
  exit 2
fi
if [[ -n "$(git -C "${RSIMEM_ROOT}" status --porcelain)" ]]; then
  echo "Formal adaptive batches require a clean RSIMem working tree." >&2
  exit 2
fi
if [[ -n "$(git -C "${RSIMEM_ROOT}" status --porcelain -- benchmarks/past-bench)" ]]; then
  echo "Formal adaptive batches require a clean PAST-Bench tree." >&2
  exit 2
fi

batch_root="${RSIMEM_ROOT}/outputs/adaptive_sm01/hermes_luna/${batch_id}"
manifest_path="${batch_root}/batch_manifest.json"
rsimem_commit="$(git -C "${RSIMEM_ROOT}" rev-parse HEAD)"
past_bench_commit="$(git -C "${RSIMEM_ROOT}" log -1 --format=%H -- benchmarks/past-bench)"
past_bench_tree="$(git -C "${RSIMEM_ROOT}" rev-parse HEAD:benchmarks/past-bench)"
proxy_args=()
[[ -z "${PAST_BENCH_PROXY:-}" ]] || proxy_args=(--proxy "${PAST_BENCH_PROXY}")
mkdir -p "${batch_root}"

PYTHONPATH="${RSIMEM_ROOT}/src" "${PYTHON_BIN}" -m rsimem.preflight \
  --state-dir "${batch_root}/preflight_state" \
  --past-bench-root "${PAST_BENCH_ROOT}" \
  --registry "${RSIMEM_ROOT}/configs/agents.yaml" \
  --agent hermes-luna \
  --require-provider

PYTHONPATH="${RSIMEM_ROOT}/src" "${PYTHON_BIN}" -c '
import sys
from pathlib import Path
from rsimem.experiment_manifest import (
    ADAPTIVE_METHOD_VARIANTS,
    initialize_batch_manifest,
    resolved_adaptive_policy_profile,
    resolved_environment_profile,
    resolved_family_budget,
    resolved_model_profile,
    resolved_run_profile,
)
run = resolved_run_profile(Path(sys.argv[4]))
initialize_batch_manifest(
    Path(sys.argv[1]),
    replicates=int(sys.argv[2]),
    task_family=sys.argv[3],
    agent="hermes-luna",
    runtime=run["runtime"],
    model=resolved_model_profile(Path(sys.argv[5]), "hermes-luna", temperature=run["temperature"]),
    judge=run["judge"],
    budget=resolved_family_budget(Path(sys.argv[6])),
    environment=resolved_environment_profile(),
    persistence_isolation={"strategy": "per_attempt_trace_directory", "compareNoPersistence": True},
    adapter_projection_verification=True,
    rsimem_commit=sys.argv[7],
    rsimem_working_tree_dirty=False,
    past_bench_commit=sys.argv[8],
    past_bench_tree=sys.argv[9],
    past_bench_dirty=False,
    execution_modes=ADAPTIVE_METHOD_VARIANTS,
    adaptive_policy=resolved_adaptive_policy_profile(Path(sys.argv[10])),
    semantic_feedback_contract="sm01_tsv_v1",
)
' "${manifest_path}" "${REPLICATES}" "${TASK_FAMILY}" \
  "${RSIMEM_ROOT}/configs/past_bench_luna_smoke.yaml" \
  "${RSIMEM_ROOT}/configs/agents.yaml" \
  "${PAST_BENCH_ROOT}/self-evolve-tasks-v2/${TASK_FAMILY}" \
  "${rsimem_commit}" "${past_bench_commit}" "${past_bench_tree}" \
  "${ADAPTIVE_CONFIG}"

manifest_call() {
  local operation="$1"
  shift
  PYTHONPATH="${RSIMEM_ROOT}/src" "${PYTHON_BIN}" -c '
import sys
from pathlib import Path
from rsimem.experiment_manifest import next_attempt_name, record_attempt
path = Path(sys.argv[2])
if sys.argv[1] == "next":
    value = next_attempt_name(path, replicate=int(sys.argv[3]), ordinal=int(sys.argv[4]), mode=sys.argv[5], base_run_name=sys.argv[6])
    print(value if value is not None else "__SKIP__")
else:
    record_attempt(path, replicate=int(sys.argv[3]), ordinal=int(sys.argv[4]), mode=sys.argv[5], run_name=sys.argv[6], status=sys.argv[7], failure_stage=sys.argv[8] or None)
' "${operation}" "${manifest_path}" "$@"
}

echo "Adaptive SM01 batch: ${batch_root}"
echo "Replicates: ${REPLICATE_START}-${REPLICATE_END} of ${REPLICATES}"
echo "RSIMem commit: ${rsimem_commit}"
echo "PAST-Bench commit/tree: ${past_bench_commit} / ${past_bench_tree}"

cd "${PAST_BENCH_ROOT}"
for replicate in $(seq "${REPLICATE_START}" "${REPLICATE_END}"); do
  mapfile -t methods < <(
    PYTHONPATH="${RSIMEM_ROOT}/src" "${PYTHON_BIN}" -c '
import sys
from rsimem.experiment_manifest import ADAPTIVE_METHOD_VARIANTS, execution_order
print("\n".join(execution_order(int(sys.argv[1]), ADAPTIVE_METHOD_VARIANTS)))
' "${replicate}"
  )
  echo "Replicate ${replicate} order: ${methods[*]}"
  ordinal=0
  for method in "${methods[@]}"; do
    ordinal=$((ordinal + 1))
    IFS=$'\t' read -r persistence_variant rsimem_mode lifecycle_mode semantic_mode feedback_contract adaptive_required < <(
      PYTHONPATH="${RSIMEM_ROOT}/src" "${PYTHON_BIN}" -c '
import sys
from rsimem.experiment_manifest import adaptive_method_execution_profile
p = adaptive_method_execution_profile(sys.argv[1])
print("\t".join((p["persistenceVariant"], p["rsimemMode"], p["lifecycleEvaluatorMode"], p["semanticWritebackMode"], p["semanticFeedbackContract"], str(p["adaptiveConfigRequired"]).lower())))
' "${method}"
    )
    base_name="${batch_id}_r$(printf '%02d' "${replicate}")_${method//-/_}"
    run_name="$(manifest_call next "${replicate}" "${ordinal}" "${method}" "${base_name}")"
    [[ "${run_name}" != "__SKIP__" ]] || continue
    trace_dir="${batch_root}/${run_name}"
    adaptive_args=()
    [[ "${adaptive_required}" != "true" ]] || adaptive_args=(--rsimem-adaptive-config "${ADAPTIVE_CONFIG}")
    manifest_call record "${replicate}" "${ordinal}" "${method}" "${run_name}" running ""
    echo "=== replicate=${replicate} method=${method} ==="
    if ! "${PAST_BENCH_BIN}" evolve \
      --family "${TASK_FAMILY}" \
      --agent hermes-luna \
      --runtime local \
      --sandbox \
      --sandbox-tools \
      --persistence-variant "${persistence_variant}" \
      --no-judge \
      --config "${RSIMEM_ROOT}/configs/past_bench_luna_smoke.yaml" \
      --registry "${RSIMEM_ROOT}/configs/agents.yaml" \
      --trace-dir "${trace_dir}" \
      --rsimem-mode "${rsimem_mode}" \
      --rsimem-adapter-failure-policy fail_closed \
      --rsimem-verify-native-projection \
      --rsimem-lifecycle-evaluator-mode "${lifecycle_mode}" \
      --rsimem-lifecycle-policy-version adaptive-sm01-lifecycle-v1 \
      --rsimem-lifecycle-compiler-version uncompiled-v0 \
      --rsimem-semantic-writeback-mode "${semantic_mode}" \
      --rsimem-semantic-feedback-contract "${feedback_contract}" \
      "${adaptive_args[@]}" \
      "${proxy_args[@]}"; then
      manifest_call record "${replicate}" "${ordinal}" "${method}" "${run_name}" failed past_bench
      exit 1
    fi
    if ! "${PYTHON_BIN}" -c '
import json
import sys
from pathlib import Path
run_dir = Path(sys.argv[1])
results = json.loads((run_dir / "sequence_results.json").read_text(encoding="utf-8"))
variant = results.get("variant")
if variant not in {"with_persistence", "without_persistence"}:
    raise ValueError("single-variant results have an unknown persistence identity")
(run_dir / "sequence_comparison.json").write_text(
    json.dumps({variant: results}, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
' "${trace_dir}"; then
      manifest_call record "${replicate}" "${ordinal}" "${method}" "${run_name}" failed normalize
      exit 1
    fi
    if ! PYTHONPATH="${RSIMEM_ROOT}/src" "${PYTHON_BIN}" -m rsimem.ledger \
      "${trace_dir}/sequence_comparison.json" \
      --output "${trace_dir}/ledger.jsonl" \
      --judge-disabled; then
      manifest_call record "${replicate}" "${ordinal}" "${method}" "${run_name}" failed ledger
      exit 1
    fi
    if ! PYTHONPATH="${RSIMEM_ROOT}/src" "${PYTHON_BIN}" -m rsimem.audit \
      "${trace_dir}" --output "${trace_dir}/audit.json"; then
      manifest_call record "${replicate}" "${ordinal}" "${method}" "${run_name}" failed audit
      exit 1
    fi
    if ! PYTHONPATH="${RSIMEM_ROOT}/src" "${PYTHON_BIN}" -c '
import json
import sys
from pathlib import Path
method, run_dir, manifest_path = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
report = json.loads((run_dir / "audit.json").read_text(encoding="utf-8"))
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
utility = report.get("staticUtility") or {}
events = utility.get("events", 0)
if method in {"static-rsimem", "adaptive-rsimem"}:
    if events < 1 or utility.get("uniqueExecutions", 0) < 1:
        raise ValueError("RSIMem method produced no utility evidence")
else:
    if events != 0:
        raise ValueError("control method unexpectedly emitted utility evidence")
if method == "adaptive-rsimem":
    active = manifest["configuration"]["adaptivePolicy"]["activePolicyVersion"]
    if utility.get("policyVersions") != [active]:
        raise ValueError("adaptive utility evidence does not use manifest ACTIVE policy")
' "${method}" "${trace_dir}" "${manifest_path}"; then
      manifest_call record "${replicate}" "${ordinal}" "${method}" "${run_name}" failed method_audit
      exit 1
    fi
    manifest_call record "${replicate}" "${ordinal}" "${method}" "${run_name}" completed ""
  done
done

echo "Adaptive SM01 batch complete: ${batch_root}"
