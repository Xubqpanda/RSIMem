#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"

AGENT_PROFILE="minimax"
RUNTIME="local"
CONFIG=""
JUDGE_MODEL="MiniMax-M2.7"
TRACE_ROOT="traces/hermes_plus_mechanism_addition_$(date +%Y%m%d_%H%M%S)"
DRY_RUN=0
NO_JUDGE=0
ONLY_MECHANISM=""
ONLY_FAMILIES=""
EXTRA_ARGS=()

MECHANISMS=(planning memory skill gate closeout)
WAVES=(
  "procedural_ability/PC01_sop_bootstrap_02 memory_ability/SM01_preference_adoption memory_ability/EP01_prior_case_recall procedural_ability/PC01_sop_bootstrap_01 memory_ability/EP02_exception_list_recall"
  "procedural_ability/PC01_sop_bootstrap_03 memory_ability/SM02_constraint_retention proactive_information_gathering/PG01_release_decision_followup procedural_ability/PC03_latent_rule_induction_01"
  "procedural_ability/PC01_sop_bootstrap_04 memory_ability/SM05_weak_trigger_preference_adoption proactive_information_gathering/PG04_temporary_waiver_audit procedural_ability/PC04_failure_to_rule_01"
  "update_ability/PC02_sop_patch_02 update_ability/SM03_fact_correction proactive_information_gathering/PG05_change_freeze_followup update_ability/PC02_sop_patch_01"
  "update_ability/SM04_rule_migration procedural_ability/PC01_sop_bootstrap_05 proactive_information_gathering/PG02_ops_exception_desk proactive_information_gathering/PG06_kappa_integration_review"
  "update_ability/SM06_temporary_exception_pollution procedural_ability/PC01_sop_bootstrap_06 proactive_information_gathering/PG03_oncall_handoff_lookup update_ability/EP03_recall_then_modify"
  "update_ability/SM07_scoped_rule_migration"
)

usage() {
  cat <<'EOF'
Usage:
  scripts/run_hermes_plus_mechanisms_full.sh [options] [-- extra past-bench args]

Runs Hermes+ single-mechanism additions over every self-evolve v2 family:
  planning, memory, skill, gate, closeout x all family.yaml files.

Options:
  --trace-root <dir>      Top-level trace output directory.
  --agent-profile <name>  Agent profile. Default: minimax
  --runtime <name>        Runtime. Default: local
  --config <path>         Optional local config.yaml override. Default: none.
  --judge-model <name>    Judge model. Default: MiniMax-M2.7
  --mechanism <name>      Run one mechanism only. Also accepts all or none.
  --families <list>       Comma-separated family list or ability/family paths.
  --no-judge              Disable judge.
  --dry-run               Print commands without executing them.
  -h, --help              Show this help message.

Example:
  scripts/run_hermes_plus_mechanisms_full.sh \
    --trace-root traces/table5_mechanism_addition_m27

  scripts/run_hermes_plus_mechanisms_full.sh --dry-run --mechanism memory
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --trace-root)
      TRACE_ROOT="$2"
      shift 2
      ;;
    --agent-profile)
      AGENT_PROFILE="$2"
      shift 2
      ;;
    --runtime)
      RUNTIME="$2"
      shift 2
      ;;
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --judge-model)
      JUDGE_MODEL="$2"
      shift 2
      ;;
    --mechanism)
      ONLY_MECHANISM="$2"
      shift 2
      ;;
    --families)
      ONLY_FAMILIES="$2"
      shift 2
      ;;
    --no-judge)
      NO_JUDGE=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS=("$@")
      break
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -n "$ONLY_MECHANISM" ]]; then
  case "$ONLY_MECHANISM" in
    planning|memory|skill|gate|closeout|all|none)
      MECHANISMS=("$ONLY_MECHANISM")
      ;;
    *)
      echo "Unknown mechanism: $ONLY_MECHANISM" >&2
      exit 1
      ;;
  esac
fi

if [[ -n "$ONLY_FAMILIES" ]]; then
  IFS=',' read -r -a FILTERED_FAMILIES <<< "$ONLY_FAMILIES"
  SELECTED_WAVES=()
  for wave in "${WAVES[@]}"; do
    selected_wave=()
    for family in $wave; do
      for requested in "${FILTERED_FAMILIES[@]}"; do
        if [[ "$family" == "$requested" || "${family##*/}" == "$requested" ]]; then
          selected_wave+=("$family")
          break
        fi
      done
    done
    if [[ ${#selected_wave[@]} -gt 0 ]]; then
      SELECTED_WAVES+=("${selected_wave[*]}")
    fi
  done
  if [[ ${#SELECTED_WAVES[@]} -eq 0 ]]; then
    echo "No families matched --families: $ONLY_FAMILIES" >&2
    exit 1
  fi
  WAVES=("${SELECTED_WAVES[@]}")
fi

count_families() {
  local total=0
  local wave
  local family

  for wave in "${WAVES[@]}"; do
    for family in $wave; do
      total=$(( total + 1 ))
    done
  done
  printf '%d' "$total"
}

trace_path_for() {
  local trace_dir="$1"
  if [[ "$trace_dir" = /* ]]; then
    printf '%s' "$trace_dir"
  else
    printf '%s/%s' "$ROOT_DIR" "$trace_dir"
  fi
}

run_one() {
  local mechanism="$1"
  local family="$2"
  local index="$3"
  local total="$4"
  local ability
  local family_id
  local trace_dir
  local trace_path
  local log_path
  local cmd

  ability="${family%%/*}"
  family_id="${family##*/}"
  trace_dir="$TRACE_ROOT/$mechanism/$ability/$family_id"
  trace_path="$(trace_path_for "$trace_dir")"
  log_path="$trace_path/run.log"

  cmd=(
    uv run past-bench evolve
    --family "$family"
    --agent hermes-plus
    --agent-profile "$AGENT_PROFILE"
    --runtime "$RUNTIME"
    --trace-dir "$trace_dir"
    --sandbox
    --sandbox-tools
    --compare-no-persistence
  )
  if [[ -n "$CONFIG" ]]; then
    cmd+=(--config "$CONFIG")
  fi

  if [[ "$NO_JUDGE" -eq 0 ]]; then
    cmd+=(--judge-model "$JUDGE_MODEL")
  else
    cmd+=(--no-judge)
  fi

  if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    cmd+=("${EXTRA_ARGS[@]}")
  fi

  # Skip if healthy trace already exists
  if [[ -f "$trace_path/sequence_summary.json" ]]; then
    local broken
    broken=$(python3 -c "
import json,sys
s=json.load(open('$trace_path/sequence_summary.json'))
eps=s.get('episodes',[])
n=sum(1 for e in eps if e.get('token_usage',{}).get('input_tokens',0)==0 and e.get('token_usage',{}).get('output_tokens',0)==0)
print(n)
" 2>/dev/null || echo "1")
    if [[ "$broken" -eq 0 ]]; then
      printf '\n[%03d/%03d] mechanism=%s family=%s - SKIP (healthy trace exists)\n' "$index" "$total" "$mechanism" "$family"
      return 0
    else
      printf '\n[%03d/%03d] mechanism=%s family=%s - re-running (broken trace, %s failed episodes)\n' "$index" "$total" "$mechanism" "$family" "$broken"
      rm -rf "$trace_path"
    fi
  fi

  printf 'trace: %s\n' "$trace_dir"
  printf 'log: %s\n' "$log_path"
  printf 'cmd: HERMES_PLUS_MECHANISMS=%q' "$mechanism"
  printf ' %q' "${cmd[@]}"
  printf '\n'

  if [[ "$DRY_RUN" -eq 0 ]]; then
    mkdir -p "$trace_path"
    (
      cd "$ROOT_DIR"
      HERMES_PLUS_MECHANISMS="$mechanism" "${cmd[@]}"
    ) > "$log_path" 2>&1
  fi
}

run_wave() {
  local mechanism="$1"
  local wave_index="$2"
  local wave="$3"
  local total="$4"
  local pids=()
  local labels=()
  local family
  local pid
  local failed=0

  printf '\n=== mechanism=%s wave=%s ===\n' "$mechanism" "$wave_index"
  for family in $wave; do
    INDEX=$(( INDEX + 1 ))
    if [[ "$DRY_RUN" -eq 1 ]]; then
      run_one "$mechanism" "$family" "$INDEX" "$total"
    else
      run_one "$mechanism" "$family" "$INDEX" "$total" &
      pids+=("$!")
      labels+=("$mechanism/$family")
    fi
  done

  if [[ "$DRY_RUN" -eq 0 ]]; then
    for i in "${!pids[@]}"; do
      pid="${pids[$i]}"
      if ! wait "$pid"; then
        printf 'FAILED: %s\n' "${labels[$i]}" >&2
        failed=1
      fi
    done
    if [[ "$failed" -ne 0 ]]; then
      exit 1
    fi
  fi
}

FAMILY_COUNT="$(count_families)"
TOTAL=$(( ${#MECHANISMS[@]} * FAMILY_COUNT ))
INDEX=0

printf 'Hermes+ mechanism-addition full run\n'
printf 'trace_root: %s\n' "$TRACE_ROOT"
printf 'mechanisms: %s\n' "${MECHANISMS[*]}"
printf 'judge_model: %s\n' "$JUDGE_MODEL"
printf 'families: %d\n' "$FAMILY_COUNT"
printf 'waves per mechanism: %d\n' "${#WAVES[@]}"
printf 'total runs: %d\n' "$TOTAL"

for mechanism in "${MECHANISMS[@]}"; do
  for i in "${!WAVES[@]}"; do
    run_wave "$mechanism" "$(( i + 1 ))" "${WAVES[$i]}" "$TOTAL"
  done
done

printf '\nAll runs complete. Trace root: %s\n' "$TRACE_ROOT"
