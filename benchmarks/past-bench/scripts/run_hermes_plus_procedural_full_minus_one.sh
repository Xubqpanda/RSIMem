#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
TRACE_ROOT="${1:-traces/ablation/procedural_full_minus_one_minimax_m27}"

CONFIGS=(
  "minus_planning:memory,skill,gate,closeout"
  "minus_memory:planning,skill,gate,closeout"
  "minus_skill:planning,memory,gate,closeout"
  "minus_gate:planning,memory,skill,closeout"
  "minus_closeout:planning,memory,skill,gate"
  "full_highspeed:planning,memory,skill,gate,closeout"
  "minus_all:none"
)

WAVES=(
  "PC01_sop_bootstrap_01 PC01_sop_bootstrap_02 PC01_sop_bootstrap_03 PC03_latent_rule_induction_01 PC01_sop_bootstrap_04 PC04_failure_to_rule_01"
)

MINIMAX_BASE_URL="${MINIMAX_BASE_URL:-https://api.minimaxi.com/v1}"
MINIMAX_MODEL="${MINIMAX_MODEL:-MiniMax-M2.7}"
HERMES_STREAM_READ_TIMEOUT="${HERMES_STREAM_READ_TIMEOUT:-30}"
HERMES_API_TIMEOUT="${HERMES_API_TIMEOUT:-300}"
HERMES_DISABLE_STREAMING="${HERMES_DISABLE_STREAMING:-1}"

if [[ -z "${MINIMAX_API_KEY:-}" ]]; then
  echo "MINIMAX_API_KEY is not set" >&2
  exit 2
fi

trace_path_for() {
  if [[ "$1" = /* ]]; then
    printf '%s' "$1"
  else
    printf '%s/%s' "$ROOT_DIR" "$1"
  fi
}

trace_is_valid() {
  python3 - \
    "$1/with_persistence/sequence_summary.json" \
    "$1/without_persistence/sequence_summary.json" \
    "$1/sequence_comparison.json" <<'PY'
import json
import sys

with open(sys.argv[3]) as f:
    comparison = json.load(f)
serialized = json.dumps(comparison)
if "HTTP 429" in serialized or "API call failed after 3 retries" in serialized:
    raise SystemExit(1)
for path in sys.argv[1:3]:
    with open(path) as f:
        summary = json.load(f)
    episodes = summary.get("episodes", [])
    if not episodes:
        raise SystemExit(1)
    for episode in episodes:
        usage = episode.get("token_usage", {})
        if usage.get("input_tokens", 0) == 0 and usage.get("output_tokens", 0) == 0:
            raise SystemExit(1)
PY
}

run_family() {
  local label="$1"
  local mechanisms="$2"
  local family="$3"
  local port_offset="$4"
  local trace_dir="$TRACE_ROOT/$label/procedural_ability/$family"
  local trace_path
  trace_path="$(trace_path_for "$trace_dir")"

  if [[ -f "$trace_path/sequence_comparison.json" ]]; then
    if trace_is_valid "$trace_path"; then
      printf 'SKIP %s/%s\n' "$label" "$family"
      return
    fi
    printf 'INVALID existing trace: %s\n' "$trace_path" >&2
    return 1
  fi

  mkdir -p "$trace_path"
  (
    cd "$ROOT_DIR"
    HERMES_PLUS_MECHANISMS="$mechanisms" \
      HERMES_STREAM_READ_TIMEOUT="$HERMES_STREAM_READ_TIMEOUT" \
      HERMES_API_TIMEOUT="$HERMES_API_TIMEOUT" \
      HERMES_DISABLE_STREAMING="$HERMES_DISABLE_STREAMING" \
      uv run past-bench evolve \
        --family "procedural_ability/$family" \
        --agent hermes-plus \
        --agent-profile minimax \
        --model "$MINIMAX_MODEL" \
        --base-url "$MINIMAX_BASE_URL" \
        --runtime local \
        --config config.yaml \
        --temperature 0 \
        --port-offset "$port_offset" \
        --trace-dir "$trace_dir" \
        --sandbox \
        --sandbox-tools \
        --compare-no-persistence \
        --judge-model "$MINIMAX_MODEL"
  ) > "$trace_path/run.log" 2>&1

  if ! trace_is_valid "$trace_path"; then
    printf 'INVALID new trace: %s\n' "$trace_path" >&2
    return 1
  fi
}

for entry in "${CONFIGS[@]}"; do
  label="${entry%%:*}"
  mechanisms="${entry#*:}"
  printf '\n=== %s (%s) ===\n' "$label" "$mechanisms"

  for wave in "${WAVES[@]}"; do
    pids=()
    index=0
    for family in $wave; do
      run_family "$label" "$mechanisms" "$family" "$((index * 500))" &
      pids+=("$!")
      index=$((index + 1))
    done

    for pid in "${pids[@]}"; do
      wait "$pid"
    done
  done
done

printf '\nComplete: %s\n' "$TRACE_ROOT"
