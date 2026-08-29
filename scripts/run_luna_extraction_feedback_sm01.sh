#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RSIMEM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PAST_BENCH_ROOT="${PAST_BENCH_ROOT:-${RSIMEM_ROOT}/benchmarks/past-bench}"
PAST_BENCH_BIN="${RSIMEM_ROOT}/.venv/bin/past-bench"
PYTHON_BIN="${RSIMEM_ROOT}/.venv/bin/python"
TASK_FAMILY="${RSIMEM_EXTRACTION_TASK_FAMILY:-memory_ability/SM01_preference_adoption}"
FAMILY_ROOT="${PAST_BENCH_ROOT}/self-evolve-tasks-v2/${TASK_FAMILY}"
EXPERIMENT_CONFIG="${RSIMEM_EXTRACTION_EXPERIMENT_CONFIG:-${RSIMEM_ROOT}/configs/extraction_feedback_sm01.json}"
AGENT_REGISTRY="${RSIMEM_AGENT_REGISTRY:-${RSIMEM_ROOT}/configs/agents.yaml}"
RUN_CONFIG="${RSIMEM_ROOT}/configs/past_bench_luna_smoke.yaml"
METHOD="static-extraction-rsimem"
FEEDBACK_CONTRACT="${RSIMEM_EXTRACTION_FEEDBACK_CONTRACT:-}"
# The default SM01 invocation resolves to --rsimem-semantic-feedback-contract sm01_tsv_v1;
# other registered families may be selected explicitly through the environment.

if [[ -z "${GPT_LUNA_API_KEY:-}" ]]; then
  echo "GPT_LUNA_API_KEY is required." >&2
  exit 2
fi
if [[ -z "${RSIMEM_BATCH_ID:-}" ]]; then
  echo "RSIMEM_BATCH_ID is required for a formal feedback batch." >&2
  exit 2
fi
if [[ ! "${RSIMEM_BATCH_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._:-]*$ ]]; then
  echo "RSIMEM_BATCH_ID is not a stable identifier." >&2
  exit 2
fi
if [[ ! -x "${PAST_BENCH_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
  echo "The RSIMem virtual environment is incomplete." >&2
  exit 2
fi
if [[ ! -f "${FAMILY_ROOT}/family.yaml" ]]; then
  echo "The requested vendored family is incomplete: ${TASK_FAMILY}" >&2
  exit 2
fi

if [[ -z "${FEEDBACK_CONTRACT}" ]]; then
  FEEDBACK_CONTRACT="$(PYTHONPATH="${RSIMEM_ROOT}/src" "${PYTHON_BIN}" - "${TASK_FAMILY}" <<'PY'
import sys
from rsimem.memory.future_trace import SemanticFeedbackContract, _SEMANTIC_FEEDBACK_FAMILIES
family = sys.argv[1].split("/", 1)[-1]
matches = [contract.value for contract, value in _SEMANTIC_FEEDBACK_FAMILIES.items() if value == family]
if len(matches) != 1:
    raise SystemExit("family has no unique registered semantic feedback contract")
print(matches[0])
PY
  )"
fi

batch_root="${RSIMEM_ROOT}/outputs/extraction_feedback/hermes_luna/${RSIMEM_BATCH_ID}"
manifest_path="${batch_root}/batch_manifest.json"
batch_registry="${RSIMEM_ROOT}/outputs/extraction_formal/batch_registry.json"
mkdir -p "${batch_root}"

# Manifest registration performs both clean-tree checks before any provider call.
PYTHONPATH="${RSIMEM_ROOT}/src" "${PYTHON_BIN}" \
  -m rsimem.extraction_experiment_preflight \
  --manifest "${manifest_path}" \
  --batch-registry "${batch_registry}" \
  --batch-id "${RSIMEM_BATCH_ID}" \
  --rsimem-root "${RSIMEM_ROOT}" \
  --past-bench-root "${PAST_BENCH_ROOT}" \
  --family-root "${FAMILY_ROOT}" \
  --agent-registry "${AGENT_REGISTRY}" \
  --run-config "${RUN_CONFIG}" \
  --experiment-config "${EXPERIMENT_CONFIG}"

PYTHONPATH="${RSIMEM_ROOT}/src" "${PYTHON_BIN}" -m rsimem.preflight \
  --state-dir "${batch_root}/provider_preflight" \
  --past-bench-root "${PAST_BENCH_ROOT}" \
  --registry "${AGENT_REGISTRY}" \
  --agent hermes-luna \
  --require-provider

manifest_call() {
  local operation="$1"
  shift
  PYTHONPATH="${RSIMEM_ROOT}/src" "${PYTHON_BIN}" -c '
import sys
from pathlib import Path
from rsimem.extraction_experiment_manifest import (
    next_extraction_attempt_name,
    record_extraction_attempt,
)
operation, manifest, replicate, ordinal, method, run_name = sys.argv[1:7]
if operation == "next":
    value = next_extraction_attempt_name(
        Path(manifest),
        replicate=int(replicate),
        ordinal=int(ordinal),
        method=method,
        base_run_name=run_name,
    )
    print(value if value is not None else "__SKIP__")
else:
    record_extraction_attempt(
        Path(manifest),
        replicate=int(replicate),
        ordinal=int(ordinal),
        method=method,
        run_name=run_name,
        status=sys.argv[7],
        failure_stage=sys.argv[8] or None,
    )
' "${operation}" "${manifest_path}" "$@"
}

replicates="$(PYTHONPATH="${RSIMEM_ROOT}/src" "${PYTHON_BIN}" -c '
import sys
from pathlib import Path
from rsimem.extraction_experiment_manifest import load_extraction_manifest
print(load_extraction_manifest(Path(sys.argv[1]))["replicates"])
' "${manifest_path}")"
proxy_args=()
[[ -z "${PAST_BENCH_PROXY:-}" ]] || proxy_args=(--proxy "${PAST_BENCH_PROXY}")

echo "Formal extraction feedback batch: ${batch_root}"
echo "Method: ${METHOD}"
echo "Replicates: ${replicates}"

for replicate in $(seq 1 "${replicates}"); do
  ordinal=1
  base_name="${RSIMEM_BATCH_ID}_r$(printf '%02d' "${replicate}")_static_extraction_rsimem"
  run_name="$(manifest_call next "${replicate}" "${ordinal}" "${METHOD}" "${base_name}")"
  [[ "${run_name}" != "__SKIP__" ]] || continue
  trace_dir="${batch_root}/${run_name}"
  manifest_call record "${replicate}" "${ordinal}" "${METHOD}" "${run_name}" running ""

  if ! (
    cd "${PAST_BENCH_ROOT}"
    "${PAST_BENCH_BIN}" evolve \
      --family "${TASK_FAMILY}" \
      --agent hermes-luna \
      --runtime local \
      --sandbox \
      --sandbox-tools \
      --persistence-variant with_persistence \
      --no-judge \
      --config "${RUN_CONFIG}" \
      --registry "${AGENT_REGISTRY}" \
      --trace-dir "${trace_dir}" \
      --background-review-wait-s 0 \
      --rsimem-mode native+ledger \
      --rsimem-adapter-failure-policy fail_closed \
      --rsimem-lifecycle-evaluator-mode disabled \
      --rsimem-semantic-writeback-mode static \
      --rsimem-semantic-writeback-timeout-seconds 30 \
      --rsimem-semantic-writeback-max-output-tokens 4096 \
      --rsimem-semantic-feedback-contract "${FEEDBACK_CONTRACT}" \
      "${proxy_args[@]}"
  ); then
    manifest_call record "${replicate}" "${ordinal}" "${METHOD}" "${run_name}" failed past_bench
    exit 1
  fi

  if ! "${PYTHON_BIN}" -c '
import json
import sys
from pathlib import Path
run_dir = Path(sys.argv[1])
results = json.loads((run_dir / "sequence_results.json").read_text(encoding="utf-8"))
if results.get("variant") != "with_persistence":
    raise ValueError("formal extraction run has the wrong persistence variant")
(run_dir / "sequence_comparison.json").write_text(
    json.dumps({"with_persistence": results}, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
' "${trace_dir}"; then
    manifest_call record "${replicate}" "${ordinal}" "${METHOD}" "${run_name}" failed normalize
    exit 1
  fi

  if ! PYTHONPATH="${RSIMEM_ROOT}/src" "${PYTHON_BIN}" -m rsimem.ledger \
    "${trace_dir}/sequence_comparison.json" \
    --output "${trace_dir}/ledger.jsonl" \
    --judge-disabled; then
    manifest_call record "${replicate}" "${ordinal}" "${METHOD}" "${run_name}" failed ledger
    exit 1
  fi
  if ! PYTHONPATH="${RSIMEM_ROOT}/src" "${PYTHON_BIN}" -m rsimem.audit \
    "${trace_dir}" --output "${trace_dir}/audit.json"; then
    failure_stage="$(PYTHONPATH="${RSIMEM_ROOT}/src" "${PYTHON_BIN}" -c '
import json
import sys
from pathlib import Path
from rsimem.extraction_experiment_analysis import classify_extraction_audit_failure
audit = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(classify_extraction_audit_failure(audit))
' "${trace_dir}/audit.json")"
    manifest_call record "${replicate}" "${ordinal}" "${METHOD}" "${run_name}" failed "${failure_stage}"
    exit 1
  fi
  if ! PYTHONPATH="${RSIMEM_ROOT}/src" "${PYTHON_BIN}" -c '
import json
import sys
from pathlib import Path
from rsimem.memory.extraction_projection import JsonExtractionSourceRecordStore
from rsimem.memory.process_corpus import JsonProcessCorpusStore, ProcessCorpus
from rsimem.memory.process_feedback import JsonProcessFeedbackLedger
run_dir = Path(sys.argv[1])
audit = json.loads((run_dir / "audit.json").read_text(encoding="utf-8"))
if audit.get("ok") is not True or audit.get("issues") != []:
    raise ValueError("formal extraction audit failed")
utility = audit.get("staticUtility") or {}
if utility.get("events") != 0:
    raise ValueError("plain extraction parent emitted utility decisions")
paths = tuple(run_dir.rglob("extraction_sources.jsonl"))
if not paths or not any(JsonExtractionSourceRecordStore(path).records() for path in paths):
    raise ValueError("formal extraction run emitted no source evidence")
process_paths = tuple(run_dir.rglob("rsimem_process_feedback.jsonl"))
events = tuple(
    event
    for path in process_paths
    for event in JsonProcessFeedbackLedger(path).events
)
if not events:
    raise ValueError("formal extraction run emitted no process feedback corpus")
manifest = json.loads((run_dir.parent / "batch_manifest.json").read_text(encoding="utf-8"))
split = manifest["split"]
corpus = ProcessCorpus.create(
    events,
    split_role=split["role"],
    family_id=split["familyId"],
    task_template_group_id=split["taskTemplateGroupId"],
    task_manifest_digest=split["taskManifestDigest"],
)
JsonProcessCorpusStore(run_dir / "process_corpus.json").put(corpus)
' "${trace_dir}"; then
    manifest_call record "${replicate}" "${ordinal}" "${METHOD}" "${run_name}" failed process_corpus
    exit 1
  fi
  manifest_call record "${replicate}" "${ordinal}" "${METHOD}" "${run_name}" completed ""
done

PYTHONPATH="${RSIMEM_ROOT}/src" "${PYTHON_BIN}" \
  -m rsimem.extraction_experiment_analysis \
  "${batch_root}" \
  --output "${batch_root}/extraction_analysis.json"

echo "Formal extraction feedback batch complete: ${batch_root}"
