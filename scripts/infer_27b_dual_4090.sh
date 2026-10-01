#!/usr/bin/env bash
set -euo pipefail
if (( $# < 3 )); then
  echo "Usage: $0 MODEL INPUT.jsonl OUTPUT.jsonl [extra inference arguments]" >&2
  exit 2
fi
model="$1"; input="$2"; output="$3"; shift 3
python -m innerjev.infer --model "$model" --input "$input" --output "$output" \
  --gpus 2 --precision int8 --gpu-memory 20GiB --batch-size 1 "$@"
