#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RSIMEM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PAST_BENCH_ROOT="${PAST_BENCH_ROOT:-${RSIMEM_ROOT}/benchmarks/past-bench}"
PAST_BENCH_BIN="${RSIMEM_ROOT}/.venv/bin/past-bench"

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

run_id="$(date +%Y%m%d_%H%M%S)"
trace_dir="${RSIMEM_ROOT}/outputs/smoke/hermes_luna_sm01/${run_id}"
proxy_args=()
if [[ -n "${PAST_BENCH_PROXY:-}" ]]; then
  proxy_args=(--proxy "${PAST_BENCH_PROXY}")
fi

echo "PAST-Bench: ${PAST_BENCH_ROOT}"
echo "Trace dir:  ${trace_dir}"
echo "Agent:      hermes-luna (gpt-5.6-luna, Responses API, reasoning=medium)"

cd "${PAST_BENCH_ROOT}"
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
  "${proxy_args[@]}" \
  "$@"

PYTHONPATH="${RSIMEM_ROOT}/src" "${RSIMEM_ROOT}/.venv/bin/python" -m rsimem.ledger \
  "${trace_dir}/sequence_comparison.json" \
  --output "${trace_dir}/ledger.jsonl" \
  --judge-disabled

PYTHONPATH="${RSIMEM_ROOT}/src" "${RSIMEM_ROOT}/.venv/bin/python" -m rsimem.audit \
  "${trace_dir}" \
  --output "${trace_dir}/audit.json"
