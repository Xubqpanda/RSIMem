#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RSIMEM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PAST_BENCH_ROOT="${PAST_BENCH_ROOT:-${RSIMEM_ROOT}/benchmarks/past-bench}"
PAST_BENCH_BIN="${RSIMEM_ROOT}/.venv/bin/past-bench"
PYTHON_BIN="${RSIMEM_ROOT}/.venv/bin/python"
REPLICATES="${RSIMEM_REPLICATES:-1}"
TASK_FAMILY="memory_ability/SM01_preference_adoption"

if [[ -z "${GPT_LUNA_API_KEY:-}" ]]; then
  echo "GPT_LUNA_API_KEY is required." >&2
  exit 2
fi
if [[ ! -x "${PAST_BENCH_BIN}" ]]; then
  echo "Missing ${PAST_BENCH_BIN}; create the RSIMem environment first." >&2
  exit 2
fi
if [[ ! -f "${PAST_BENCH_ROOT}/pyproject.toml" ]]; then
  echo "PAST_BENCH_ROOT does not point to a PAST-Bench checkout: ${PAST_BENCH_ROOT}" >&2
  exit 2
fi
if [[ ! "${REPLICATES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "RSIMEM_REPLICATES must be a positive integer." >&2
  exit 2
fi

batch_id="${RSIMEM_BATCH_ID:-$(date +%Y%m%d_%H%M%S)}"
if [[ ! "${batch_id}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "RSIMEM_BATCH_ID contains unsupported characters." >&2
  exit 2
fi
batch_root="${RSIMEM_ROOT}/outputs/matched/hermes_luna_sm01/${batch_id}"
manifest_path="${batch_root}/batch_manifest.json"
rsimem_commit="$(git -C "${RSIMEM_ROOT}" rev-parse HEAD)"
past_bench_commit="$(git -C "${RSIMEM_ROOT}" log -1 --format=%H -- benchmarks/past-bench)"
past_bench_tree="$(git -C "${RSIMEM_ROOT}" rev-parse HEAD:benchmarks/past-bench)"
rsimem_working_tree_dirty=false
if [[ -n "$(git -C "${RSIMEM_ROOT}" status --porcelain)" ]]; then
  rsimem_working_tree_dirty=true
fi
past_bench_dirty=false
if [[ -n "$(git -C "${RSIMEM_ROOT}" status --porcelain -- benchmarks/past-bench)" ]]; then
  past_bench_dirty=true
fi
proxy_args=()
if [[ -n "${PAST_BENCH_PROXY:-}" ]]; then
  proxy_args=(--proxy "${PAST_BENCH_PROXY}")
fi
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
    initialize_batch_manifest,
    resolved_environment_profile,
    resolved_family_budget,
    resolved_model_profile,
    resolved_run_profile,
)

run_profile = resolved_run_profile(Path(sys.argv[4]))
initialize_batch_manifest(
    Path(sys.argv[1]),
    replicates=int(sys.argv[2]),
    task_family=sys.argv[3],
    agent="hermes-luna",
    runtime=run_profile["runtime"],
    model=resolved_model_profile(
        Path(sys.argv[5]),
        "hermes-luna",
        temperature=run_profile["temperature"],
    ),
    judge=run_profile["judge"],
    budget=resolved_family_budget(Path(sys.argv[6])),
    environment=resolved_environment_profile(),
    persistence_isolation={
        "strategy": "per_attempt_trace_directory",
        "compareNoPersistence": True,
    },
    adapter_projection_verification=True,
    rsimem_commit=sys.argv[7],
    rsimem_working_tree_dirty=sys.argv[8] == "true",
    past_bench_commit=sys.argv[9],
    past_bench_tree=sys.argv[10],
    past_bench_dirty=sys.argv[11] == "true",
)
' "${manifest_path}" "${REPLICATES}" "${TASK_FAMILY}" \
  "${RSIMEM_ROOT}/configs/past_bench_luna_smoke.yaml" \
  "${RSIMEM_ROOT}/configs/agents.yaml" \
  "${PAST_BENCH_ROOT}/self-evolve-tasks-v2/${TASK_FAMILY}" \
  "${rsimem_commit}" "${rsimem_working_tree_dirty}" \
  "${past_bench_commit}" "${past_bench_tree}" "${past_bench_dirty}"

record_attempt() {
  PYTHONPATH="${RSIMEM_ROOT}/src" "${PYTHON_BIN}" -c '
import sys
from pathlib import Path
from rsimem.experiment_manifest import record_attempt

record_attempt(
    Path(sys.argv[1]),
    replicate=int(sys.argv[2]),
    ordinal=int(sys.argv[3]),
    mode=sys.argv[4],
    run_name=sys.argv[5],
    status=sys.argv[6],
    failure_stage=sys.argv[7] or None,
)
' "${manifest_path}" "$1" "$2" "$3" "$4" "$5" "${6:-}"
}

next_attempt_name() {
  PYTHONPATH="${RSIMEM_ROOT}/src" "${PYTHON_BIN}" -c '
import sys
from pathlib import Path
from rsimem.experiment_manifest import next_attempt_name

name = next_attempt_name(
    Path(sys.argv[1]),
    replicate=int(sys.argv[2]),
    ordinal=int(sys.argv[3]),
    mode=sys.argv[4],
    base_run_name=sys.argv[5],
)
print(name if name is not None else "__SKIP__")
' "${manifest_path}" "$1" "$2" "$3" "$4"
}

echo "PAST-Bench: ${PAST_BENCH_ROOT}"
echo "Batch root: ${batch_root}"
echo "Replicates: ${REPLICATES}"
echo "RSIMem:     ${rsimem_commit}"
echo "PAST-Bench: ${past_bench_commit} (tree ${past_bench_tree})"

cd "${PAST_BENCH_ROOT}"
for replicate in $(seq 1 "${REPLICATES}"); do
  mapfile -t modes < <(
    PYTHONPATH="${RSIMEM_ROOT}/src" "${PYTHON_BIN}" -c '
import sys
from rsimem.experiment_manifest import execution_order
print("\n".join(execution_order(int(sys.argv[1]))))
' "${replicate}"
  )
  echo "Replicate ${replicate} order: ${modes[*]}"
  ordinal=0
  for mode in "${modes[@]}"; do
    ordinal=$((ordinal + 1))
    mode_slug="${mode//+/_}"
    base_run_name="${batch_id}_r$(printf '%02d' "${replicate}")_${mode_slug}"
    run_name="$(next_attempt_name "${replicate}" "${ordinal}" "${mode}" "${base_run_name}")"
    if [[ "${run_name}" == "__SKIP__" ]]; then
      echo "Skipping completed replicate=${replicate} mode=${mode}"
      continue
    fi
    trace_dir="${batch_root}/${run_name}"
    echo
    echo "=== replicate=${replicate} mode=${mode} ==="
    record_attempt "${replicate}" "${ordinal}" "${mode}" "${run_name}" running

    if ! "${PAST_BENCH_BIN}" evolve \
      --family "${TASK_FAMILY}" \
      --agent hermes-luna \
      --runtime local \
      --sandbox \
      --sandbox-tools \
      --compare-no-persistence \
      --no-judge \
      --config "${RSIMEM_ROOT}/configs/past_bench_luna_smoke.yaml" \
      --registry "${RSIMEM_ROOT}/configs/agents.yaml" \
      --trace-dir "${trace_dir}" \
      --rsimem-mode "${mode}" \
      --rsimem-adapter-failure-policy fail_closed \
      --rsimem-verify-native-projection \
      "${proxy_args[@]}"; then
      record_attempt "${replicate}" "${ordinal}" "${mode}" "${run_name}" failed past_bench
      exit 1
    fi

    if ! PYTHONPATH="${RSIMEM_ROOT}/src" "${PYTHON_BIN}" -m rsimem.ledger \
      "${trace_dir}/sequence_comparison.json" \
      --output "${trace_dir}/ledger.jsonl" \
      --judge-disabled; then
      record_attempt "${replicate}" "${ordinal}" "${mode}" "${run_name}" failed ledger
      exit 1
    fi

    if ! PYTHONPATH="${RSIMEM_ROOT}/src" "${PYTHON_BIN}" -m rsimem.audit \
      "${trace_dir}" \
      --output "${trace_dir}/audit.json"; then
      record_attempt "${replicate}" "${ordinal}" "${mode}" "${run_name}" failed audit
      exit 1
    fi
    record_attempt "${replicate}" "${ordinal}" "${mode}" "${run_name}" completed
  done
done

echo
echo "Matched batch complete: ${batch_root}"
