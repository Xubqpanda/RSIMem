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
modes=("native" "native+ledger" "native+adapter+ledger")
proxy_args=()
if [[ -n "${PAST_BENCH_PROXY:-}" ]]; then
  proxy_args=(--proxy "${PAST_BENCH_PROXY}")
fi
mkdir -p "${batch_root}"

"${PYTHON_BIN}" -c '
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
path.write_text(json.dumps({
    "schemaVersion": 1,
    "family": "memory_ability/SM01_preference_adoption",
    "agent": "hermes-luna",
    "model": "gpt-5.6-luna",
    "runtime": "local",
    "temperature": 0.0,
    "judgeEnabled": False,
    "compareNoPersistence": True,
    "adapterFailurePolicy": "fail_closed",
    "executionModes": ["native", "native+ledger", "native+adapter+ledger"],
    "replicates": int(sys.argv[2]),
    "seedControl": "not_exposed_by_current_runtime",
    "resourceEvidence": "Each run writes raw ledger.jsonl and audit.json evidence.",
}, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
' "${batch_root}/batch_manifest.json" "${REPLICATES}"

echo "PAST-Bench: ${PAST_BENCH_ROOT}"
echo "Batch root: ${batch_root}"
echo "Replicates: ${REPLICATES}"
echo "Modes:      ${modes[*]}"

cd "${PAST_BENCH_ROOT}"
for replicate in $(seq 1 "${REPLICATES}"); do
  for mode in "${modes[@]}"; do
    mode_slug="${mode//+/_}"
    run_name="${batch_id}_r$(printf '%02d' "${replicate}")_${mode_slug}"
    trace_dir="${batch_root}/${run_name}"
    echo
    echo "=== replicate=${replicate} mode=${mode} ==="

    "${PAST_BENCH_BIN}" evolve \
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
      "${proxy_args[@]}"

    PYTHONPATH="${RSIMEM_ROOT}/src" "${PYTHON_BIN}" -m rsimem.ledger \
      "${trace_dir}/sequence_comparison.json" \
      --output "${trace_dir}/ledger.jsonl" \
      --judge-disabled

    PYTHONPATH="${RSIMEM_ROOT}/src" "${PYTHON_BIN}" -m rsimem.audit \
      "${trace_dir}" \
      --output "${trace_dir}/audit.json"
  done
done

echo
echo "Matched batch complete: ${batch_root}"
