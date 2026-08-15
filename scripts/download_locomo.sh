#!/usr/bin/env bash
set -euo pipefail

# LoCoMo is small enough for a smoke test, but is kept out of Git.
repo_root="$(cd "$(dirname "$0")/.." && pwd)"
target_dir="${repo_root}/data/raw/locomo"
target_file="${target_dir}/locomo10.json"

mkdir -p "$target_dir"

if [[ -f "$target_file" ]]; then
  echo "LoCoMo already exists: $target_file"
  exit 0
fi

curl -L --fail --retry 3 \
  https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json \
  -o "$target_file"

python - "$target_file" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text(encoding="utf-8"))
if not isinstance(data, list) or not data:
    raise SystemExit("Downloaded LoCoMo file is empty or has an unexpected format")
print(f"Downloaded {len(data)} LoCoMo trajectories to {path}")
PY
