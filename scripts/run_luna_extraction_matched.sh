#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RSIMEM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PAST_BENCH_ROOT="${PAST_BENCH_ROOT:-${RSIMEM_ROOT}/benchmarks/past-bench}"
PAST_BENCH_BIN="${RSIMEM_ROOT}/.venv/bin/past-bench"
PYTHON_BIN="${RSIMEM_ROOT}/.venv/bin/python"
AGENT_REGISTRY="${RSIMEM_AGENT_REGISTRY:-${RSIMEM_ROOT}/configs/agents.yaml}"
PAST_AGENT_REGISTRY="${RSIMEM_PAST_AGENT_REGISTRY:-${PAST_BENCH_ROOT}/configs/agents.yaml}"
PAST_AGENT="${RSIMEM_PAST_AGENT:-hermes}"
PAST_AGENT_PROFILE="${RSIMEM_PAST_AGENT_PROFILE:-openai}"
PAST_MODEL="${RSIMEM_PAST_MODEL:-gpt-5.6-luna}"
PAST_BASE_URL="${RSIMEM_PAST_BASE_URL:-https://coding.tu-zi.com/v1}"
# RSIMem preflight consumes --agent-registry "${AGENT_REGISTRY}"; the PAST
# runtime consumes its own --registry "${PAST_AGENT_REGISTRY}" schema.
# (The historical invocation spelling --registry "${AGENT_REGISTRY}" is kept
# here as a compatibility marker; it must not be used for the PAST schema.)
TRIAL_CONFIG="${RSIMEM_EXTRACTION_TRIAL_CONFIG:-}"
EXPERIMENT_CONFIG="${RSIMEM_EXTRACTION_EXPERIMENT_CONFIG:-}"
# Formal matched validation always uses the checked-in family/template split.
# Callers may provide an immutable replacement plan for a separately authored
# experiment, but omitting it must not silently disable the split gate.
SPLIT_PLAN="${RSIMEM_EXTRACTION_SPLIT_PLAN:-${RSIMEM_ROOT}/configs/extraction_split_plan_sm02_sm03_sm04.json}"
BATCH_ID="${RSIMEM_BATCH_ID:-}"
TASK_FAMILY="${RSIMEM_EXTRACTION_TASK_FAMILY:-}"

[[ -n "${GPT_LUNA_API_KEY:-}" ]] || { echo "GPT_LUNA_API_KEY is required." >&2; exit 2; }
export OPENAI_API_KEY="${OPENAI_API_KEY:-${GPT_LUNA_API_KEY}}"
[[ -n "${TRIAL_CONFIG}" && -f "${TRIAL_CONFIG}" ]] || { echo "RSIMEM_EXTRACTION_TRIAL_CONFIG is required." >&2; exit 2; }
[[ -n "${EXPERIMENT_CONFIG}" && -f "${EXPERIMENT_CONFIG}" ]] || { echo "RSIMEM_EXTRACTION_EXPERIMENT_CONFIG is required." >&2; exit 2; }
[[ -n "${BATCH_ID}" && "${BATCH_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._:-]*$ ]] || { echo "RSIMEM_BATCH_ID is invalid." >&2; exit 2; }
[[ -n "${TASK_FAMILY}" ]] || { echo "RSIMEM_EXTRACTION_TASK_FAMILY is required." >&2; exit 2; }
[[ -x "${PAST_BENCH_BIN}" && -x "${PYTHON_BIN}" ]] || { echo "RSIMem virtual environment is incomplete." >&2; exit 2; }
[[ -z "$(git -C "${RSIMEM_ROOT}" status --porcelain)" ]] || { echo "Formal matched validation requires a clean RSIMem tree." >&2; exit 2; }
[[ -z "$(git -C "${PAST_BENCH_ROOT}" status --porcelain)" ]] || { echo "Formal matched validation requires a clean PAST-Bench tree." >&2; exit 2; }
[[ -f "${SPLIT_PLAN}" ]] || { echo "RSIMEM_EXTRACTION_SPLIT_PLAN does not exist." >&2; exit 2; }
split_plan_args=(--split-plan "${SPLIT_PLAN}")

family_root="${PAST_BENCH_ROOT}/self-evolve-tasks-v2/${TASK_FAMILY}"
[[ -f "${family_root}/family.yaml" ]] || { echo "Requested PAST family is incomplete." >&2; exit 2; }
batch_root="${RSIMEM_ROOT}/outputs/extraction_matched/${BATCH_ID}"
manifest_path="${batch_root}/batch_manifest.json"
registry_path="${RSIMEM_ROOT}/outputs/extraction_formal/batch_registry.json"
mkdir -p "${batch_root}"

PYTHONPATH="${RSIMEM_ROOT}/src" "${PYTHON_BIN}" -m rsimem.extraction_matched_preflight \
  --manifest "${manifest_path}" --batch-registry "${registry_path}" --batch-id "${BATCH_ID}" \
  --rsimem-root "${RSIMEM_ROOT}" --past-bench-root "${PAST_BENCH_ROOT}" \
  --family-root "${family_root}" --agent-registry "${AGENT_REGISTRY}" \
  --run-config "${RSIMEM_ROOT}/configs/past_bench_luna_smoke.yaml" \
  --experiment-config "${EXPERIMENT_CONFIG}" --trial-config "${TRIAL_CONFIG}" \
  "${split_plan_args[@]}"

# Freeze the result-independent process-signal contract before any task run.
PYTHONPATH="${RSIMEM_ROOT}/src" "${PYTHON_BIN}" - "${manifest_path}" "${batch_root}" <<'PY'
import sys
from pathlib import Path
from rsimem.extraction_experiment_manifest import load_extraction_manifest
from rsimem.memory.signal_protocol import (
    PROCESS_SIGNAL_PROTOCOL_FILENAME,
    JsonProcessSignalAnalysisProtocolStore,
    protocol_for_extraction_manifest,
)
manifest_path, batch_root = map(Path, sys.argv[1:])
manifest = load_extraction_manifest(manifest_path)
JsonProcessSignalAnalysisProtocolStore(
    batch_root / PROCESS_SIGNAL_PROTOCOL_FILENAME
).freeze(protocol_for_extraction_manifest(manifest))
PY

# Keep provider connectivity outside the matched task/accounting surface and
# fail before any parent/candidate task starts when completion content is not
# available.  Only the content-free probe result is persisted.
provider_probe_path="${batch_root}/provider_probe.json"
if ! PYTHONPATH="${RSIMEM_ROOT}/src" "${PYTHON_BIN}" -m rsimem.provider_probe \
  --base-url "${PAST_BASE_URL}" \
  --model "${PAST_MODEL}" \
  >"${provider_probe_path}"; then
  echo "Provider completion probe failed; see ${provider_probe_path}." >&2
  exit 1
fi

replicates="$(PYTHONPATH="${RSIMEM_ROOT}/src" "${PYTHON_BIN}" -c 'from pathlib import Path; from rsimem.extraction_experiment_manifest import load_extraction_manifest; import sys; print(load_extraction_manifest(Path(sys.argv[1]))["replicates"])' "${manifest_path}")"
feedback_contract="$(PYTHONPATH="${RSIMEM_ROOT}/src" "${PYTHON_BIN}" - "${EXPERIMENT_CONFIG}" <<'PY'
import sys
from pathlib import Path
from rsimem.extraction_experiment_preflight import load_extraction_preflight_config
from rsimem.memory.future_trace import SemanticFeedbackContract, _SEMANTIC_FEEDBACK_FAMILIES
family = load_extraction_preflight_config(Path(sys.argv[1]))["familyId"]
matches = [contract.value for contract, value in _SEMANTIC_FEEDBACK_FAMILIES.items() if value == family]
if len(matches) != 1:
    raise ValueError("family has no unique semantic feedback contract")
print(matches[0])
PY
)"
proxy_args=()
[[ -z "${PAST_BENCH_PROXY:-}" ]] || proxy_args=(--proxy "${PAST_BENCH_PROXY}")

manifest_call() {
  PYTHONPATH="${RSIMEM_ROOT}/src" "${PYTHON_BIN}" - "$@" <<'PY'
import sys
from pathlib import Path
from rsimem.extraction_experiment_manifest import next_extraction_attempt_name, record_extraction_attempt
operation, manifest, replicate, ordinal, method, run_name = sys.argv[1:7]
if operation == "next":
    value = next_extraction_attempt_name(Path(manifest), replicate=int(replicate), ordinal=int(ordinal), method=method, base_run_name=run_name)
    print(value if value is not None else "__SKIP__")
else:
    record_extraction_attempt(Path(manifest), replicate=int(replicate), ordinal=int(ordinal), method=method, run_name=run_name, status=sys.argv[7], failure_stage=sys.argv[8] or None)
PY
}

for replicate in $(seq 1 "${replicates}"); do
  mapfile -t methods < <(PYTHONPATH="${RSIMEM_ROOT}/src" "${PYTHON_BIN}" -c 'from rsimem.extraction_experiment_manifest import extraction_execution_order; import sys; print("\n".join(extraction_execution_order(int(sys.argv[1]))))' "${replicate}")
  ordinal=0
  for method in "${methods[@]}"; do
    ordinal=$((ordinal + 1))
    base_name="${BATCH_ID}_r$(printf '%02d' "${replicate}")_${method//-/_}"
    run_name="$(manifest_call next "${manifest_path}" "${replicate}" "${ordinal}" "${method}" "${base_name}")"
    [[ "${run_name}" != "__SKIP__" ]] || continue
    trace_dir="${batch_root}/${run_name}"
    trial_args=()
    [[ "${method}" != "adaptive-extraction-rsimem" ]] || trial_args=(--rsimem-extraction-trial-config "${TRIAL_CONFIG}")
    manifest_call record "${manifest_path}" "${replicate}" "${ordinal}" "${method}" "${run_name}" running ""
    if ! (
      cd "${PAST_BENCH_ROOT}"
      "${PAST_BENCH_BIN}" evolve --family "${TASK_FAMILY}" --agent "${PAST_AGENT}" \
        --agent-profile "${PAST_AGENT_PROFILE}" --model "${PAST_MODEL}" --base-url "${PAST_BASE_URL}" --runtime local \
        --sandbox --sandbox-tools --persistence-variant with_persistence --no-judge \
        --config "${RSIMEM_ROOT}/configs/past_bench_luna_smoke.yaml" --registry "${PAST_AGENT_REGISTRY}" \
        --trace-dir "${trace_dir}" --background-review-wait-s 0 \
        --rsimem-mode native+ledger --rsimem-adapter-failure-policy fail_closed \
        --rsimem-lifecycle-evaluator-mode disabled --rsimem-semantic-writeback-mode static \
        --rsimem-semantic-feedback-contract "${feedback_contract}" "${trial_args[@]}" "${proxy_args[@]}"
    ); then
      manifest_call record "${manifest_path}" "${replicate}" "${ordinal}" "${method}" "${run_name}" failed past_bench
      exit 1
    fi
    if ! "${PYTHON_BIN}" - "${trace_dir}" <<'PY'
import json, sys
from pathlib import Path
run = Path(sys.argv[1])
result = json.loads((run / "sequence_results.json").read_text())
if result.get("variant") != "with_persistence":
    raise ValueError("wrong persistence variant")
(run / "sequence_comparison.json").write_text(json.dumps({"with_persistence": result}, ensure_ascii=True, sort_keys=True) + "\n")
PY
    then
      manifest_call record "${manifest_path}" "${replicate}" "${ordinal}" "${method}" "${run_name}" failed normalize
      exit 1
    fi
    if ! PYTHONPATH="${RSIMEM_ROOT}/src" "${PYTHON_BIN}" -m rsimem.ledger "${trace_dir}/sequence_comparison.json" --output "${trace_dir}/ledger.jsonl" --judge-disabled; then
      manifest_call record "${manifest_path}" "${replicate}" "${ordinal}" "${method}" "${run_name}" failed ledger
      exit 1
    fi
    if ! PYTHONPATH="${RSIMEM_ROOT}/src" "${PYTHON_BIN}" -m rsimem.audit "${trace_dir}" --output "${trace_dir}/audit.json"; then
      manifest_call record "${manifest_path}" "${replicate}" "${ordinal}" "${method}" "${run_name}" failed audit
      exit 1
    fi
    if ! PYTHONPATH="${RSIMEM_ROOT}/src" "${PYTHON_BIN}" - "${trace_dir}" "${manifest_path}" <<'PY'
import json, sys
from pathlib import Path
from rsimem.memory.process_corpus import JsonProcessCorpusStore, ProcessCorpus
from rsimem.memory.pure_process import JsonPureProcessCorpusStore, PureProcessCorpus
from rsimem.memory.process_feedback import JsonProcessFeedbackLedger, audit_process_events
from rsimem.memory.process_signal import (
    JsonProcessSignalCaseStore,
    build_process_signal_cases,
)
from rsimem.memory.signal_protocol import (
    PROCESS_SIGNAL_OBSERVATION_WINDOW,
    PROCESS_SIGNAL_PROTOCOL_FILENAME,
    JsonProcessSignalAnalysisProtocolStore,
    validate_protocol_for_extraction_manifest,
)
run_dir, manifest_path = map(Path, sys.argv[1:])
events = tuple(
    event
    for path in sorted(run_dir.rglob("rsimem_process_feedback.jsonl"))
    for event in JsonProcessFeedbackLedger(path).events
)
if not events:
    raise ValueError("formal matched run emitted no process feedback corpus")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
protocol = JsonProcessSignalAnalysisProtocolStore(
    manifest_path.parent / PROCESS_SIGNAL_PROTOCOL_FILENAME
).get()
if protocol is None:
    raise ValueError("formal matched run has no frozen process signal protocol")
protocol = validate_protocol_for_extraction_manifest(protocol, manifest)
split = manifest["split"]
corpus = ProcessCorpus.create(
    events,
    split_role="validation",
    family_id=split["familyId"],
    task_template_group_id=split["taskTemplateGroupId"],
    task_manifest_digest=split["taskManifestDigest"],
)
pure_corpus = PureProcessCorpus.create(events)
# Shared-cold traces may be present under both the nested shared directory and
# the attempt directory.  Collapse exact logical duplicates before auditing;
# Both corpus forms reject conflicting payloads for one event ID.
process_errors = audit_process_events(corpus.events)
if process_errors:
    raise ValueError("formal process feedback audit failed: " + "; ".join(process_errors))
JsonProcessCorpusStore(run_dir / "process_corpus.json").put(corpus)
JsonPureProcessCorpusStore(run_dir / "pure_process_corpus.json").put(pure_corpus)
attempt = next(
    item for item in manifest["attemptHistory"]
    if Path(item["outputDirectory"]).resolve() == run_dir.resolve()
    or Path(item["outputDirectory"]).name == run_dir.name
)
method = attempt["method"]
policy_digest = manifest["semanticPolicy"]["activeArtifactByMethod"][method]["artifactDigest"]
cases = build_process_signal_cases(
    pure_corpus.events,
    frozen_policy_digest=policy_digest,
    source_task_template_id="source." + split["taskTemplateGroupId"],
    future_task_template_id="future." + split["taskTemplateGroupId"],
    observation_window=PROCESS_SIGNAL_OBSERVATION_WINDOW,
    replicate_id="replicate." + str(attempt["replicate"]),
    analysis_protocol_id=protocol.protocol_id,
)
if not cases:
    raise ValueError("formal matched run emitted no process signal cases")
case_store = JsonProcessSignalCaseStore(run_dir / "process_signal_cases.jsonl")
for case in cases:
    case_store.append(case)
PY
    then
      manifest_call record "${manifest_path}" "${replicate}" "${ordinal}" "${method}" "${run_name}" failed process_corpus
      exit 1
    fi
    manifest_call record "${manifest_path}" "${replicate}" "${ordinal}" "${method}" "${run_name}" completed ""
  done
done

PYTHONPATH="${RSIMEM_ROOT}/src" "${PYTHON_BIN}" - "${batch_root}" "${manifest_path}" "${TRIAL_CONFIG}" <<'PY'
import sys
from pathlib import Path
from rsimem.extraction_validation_evidence import assemble_extraction_matched_evidence_batch
from rsimem.extraction_experiment_manifest import load_extraction_manifest
from rsimem.extraction_validation_runtime import load_extraction_matched_trial_profile
from rsimem.memory.extraction_prompt_validation import (
    ExtractionPromptValidationSplit, ExtractionSplitAssignment,
    ExtractionValidationSplitRole,
)
root, manifest, trial = map(Path, sys.argv[1:])
profile = load_extraction_matched_trial_profile(trial)
registered = load_extraction_manifest(manifest)
split = ExtractionPromptValidationSplit(
    "live-validation." + registered["experimentId"][:24],
    (ExtractionSplitAssignment(
        ExtractionValidationSplitRole.VALIDATION,
        registered["split"]["familyId"],
        registered["split"]["taskTemplateGroupId"],
        registered["split"]["taskManifestDigest"],
    ),),
)
assemble_extraction_matched_evidence_batch(
    root,
    parent=profile.parent,
    candidate=profile.candidate,
    offline_decision=profile.offline_decision,
    split=split,
    output_path=root / "matched_evidence.json",
)
PY
