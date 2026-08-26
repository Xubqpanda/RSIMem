#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RSIMEM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PAST_BENCH_ROOT="${PAST_BENCH_ROOT:-${RSIMEM_ROOT}/benchmarks/past-bench}"
PAST_BENCH_BIN="${RSIMEM_ROOT}/.venv/bin/past-bench"
PYTHON_BIN="${RSIMEM_ROOT}/.venv/bin/python"
REPLICATES="${RSIMEM_REPLICATES:-1}"

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

batch_id="$(date +%Y%m%d_%H%M%S)"
batch_root="${RSIMEM_ROOT}/outputs/matched/hermes_luna_sm01/${batch_id}"
manifest_path="${batch_root}/batch_manifest.json"
rsimem_commit="$(git -C "${RSIMEM_ROOT}" rev-parse HEAD)"
past_bench_commit="$(git -C "${RSIMEM_ROOT}" log -1 --format=%H -- benchmarks/past-bench)"
past_bench_tree="$(git -C "${RSIMEM_ROOT}" rev-parse HEAD:benchmarks/past-bench)"
working_tree_dirty=false
if [[ -n "$(git -C "${RSIMEM_ROOT}" status --porcelain)" ]]; then
  working_tree_dirty=true
fi
proxy_args=()
if [[ -n "${PAST_BENCH_PROXY:-}" ]]; then
  proxy_args=(--proxy "${PAST_BENCH_PROXY}")
fi
mkdir -p "${batch_root}"

PYTHONPATH="${RSIMEM_ROOT}/src" "${PYTHON_BIN}" -c '
import sys
from pathlib import Path
from rsimem.experiment_manifest import initialize_batch_manifest

initialize_batch_manifest(
    Path(sys.argv[1]),
    replicates=int(sys.argv[2]),
    rsimem_commit=sys.argv[3],
    past_bench_commit=sys.argv[4],
    past_bench_tree=sys.argv[5],
    working_tree_dirty=sys.argv[6] == "true",
)
' "${manifest_path}" "${REPLICATES}" "${rsimem_commit}" "${past_bench_commit}" "${past_bench_tree}" "${working_tree_dirty}"

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
    run_name="${batch_id}_r$(printf '%02d' "${replicate}")_${mode_slug}"
    trace_dir="${batch_root}/${run_name}"
    echo
    echo "=== replicate=${replicate} mode=${mode} ==="
    record_attempt "${replicate}" "${ordinal}" "${mode}" "${run_name}" running

    if ! "${PAST_BENCH_BIN}" evolve \
      --family memory_ability/SM01_preference_adoption \
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
