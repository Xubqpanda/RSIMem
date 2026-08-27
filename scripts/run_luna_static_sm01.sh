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

if [[ -z "${GPT_LUNA_API_KEY:-}" ]]; then
  echo "GPT_LUNA_API_KEY is required." >&2
  exit 2
fi
if [[ ! -x "${PAST_BENCH_BIN}" || ! -f "${PAST_BENCH_ROOT}/pyproject.toml" ]]; then
  echo "RSIMem or PAST-Bench environment is incomplete." >&2
  exit 2
fi
if [[ ! "${REPLICATES}" =~ ^[3-9][0-9]*$ ]]; then
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
batch_root="${RSIMEM_ROOT}/outputs/static_sm01/hermes_luna/${batch_id}"
manifest_path="${batch_root}/batch_manifest.json"
rsimem_commit="$(git -C "${RSIMEM_ROOT}" rev-parse HEAD)"
past_bench_commit="$(git -C "${RSIMEM_ROOT}" log -1 --format=%H -- benchmarks/past-bench)"
past_bench_tree="$(git -C "${RSIMEM_ROOT}" rev-parse HEAD:benchmarks/past-bench)"
rsimem_dirty=false
[[ -z "$(git -C "${RSIMEM_ROOT}" status --porcelain)" ]] || rsimem_dirty=true
past_bench_dirty=false
[[ -z "$(git -C "${RSIMEM_ROOT}" status --porcelain -- benchmarks/past-bench)" ]] || past_bench_dirty=true
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
    STATIC_METHOD_VARIANTS,
    initialize_batch_manifest,
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
    persistence_isolation={"strategy": "per_attempt_trace_directory", "compareNoPersistence": False},
    adapter_projection_verification=True,
    rsimem_commit=sys.argv[7],
    rsimem_working_tree_dirty=sys.argv[8] == "true",
    past_bench_commit=sys.argv[9],
    past_bench_tree=sys.argv[10],
    past_bench_dirty=sys.argv[11] == "true",
    execution_modes=STATIC_METHOD_VARIANTS,
)
' "${manifest_path}" "${REPLICATES}" "${TASK_FAMILY}" \
  "${RSIMEM_ROOT}/configs/past_bench_luna_smoke.yaml" \
  "${RSIMEM_ROOT}/configs/agents.yaml" \
  "${PAST_BENCH_ROOT}/self-evolve-tasks-v2/${TASK_FAMILY}" \
  "${rsimem_commit}" "${rsimem_dirty}" "${past_bench_commit}" \
  "${past_bench_tree}" "${past_bench_dirty}"

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

echo "Static SM01 batch: ${batch_root}"
echo "Replicates: ${REPLICATE_START}-${REPLICATE_END} of ${REPLICATES}"
echo "RSIMem commit: ${rsimem_commit}"
echo "PAST-Bench commit/tree: ${past_bench_commit} / ${past_bench_tree}"

cd "${PAST_BENCH_ROOT}"
for replicate in $(seq "${REPLICATE_START}" "${REPLICATE_END}"); do
  mapfile -t methods < <(
    PYTHONPATH="${RSIMEM_ROOT}/src" "${PYTHON_BIN}" -c '
import sys
from rsimem.experiment_manifest import STATIC_METHOD_VARIANTS, execution_order
print("\n".join(execution_order(int(sys.argv[1]), STATIC_METHOD_VARIANTS)))
' "${replicate}"
  )
  echo "Replicate ${replicate} order: ${methods[*]}"
  ordinal=0
  for method in "${methods[@]}"; do
    ordinal=$((ordinal + 1))
    persistence_variant="with_persistence"
    semantic_mode="disabled"
    if [[ "${method}" == "no-persistence" ]]; then
      persistence_variant="without_persistence"
    elif [[ "${method}" == "static-rsimem" ]]; then
      semantic_mode="static"
    fi
    base_name="${batch_id}_r$(printf '%02d' "${replicate}")_${method//-/_}"
    run_name="$(manifest_call next "${replicate}" "${ordinal}" "${method}" "${base_name}")"
    if [[ "${run_name}" == "__SKIP__" ]]; then
      continue
    fi
    trace_dir="${batch_root}/${run_name}"
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
      --rsimem-mode native+ledger \
      --rsimem-adapter-failure-policy fail_closed \
      --rsimem-lifecycle-evaluator-mode deterministic \
      --rsimem-lifecycle-policy-version static-sm01-v1 \
      --rsimem-lifecycle-compiler-version uncompiled-v0 \
      --rsimem-semantic-writeback-mode "${semantic_mode}" \
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
comparison = {variant: results}
(run_dir / "sequence_comparison.json").write_text(
    json.dumps(comparison, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
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
    manifest_call record "${replicate}" "${ordinal}" "${method}" "${run_name}" completed ""
  done
done

echo "Static SM01 batch complete: ${batch_root}"
