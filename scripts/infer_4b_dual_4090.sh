#!/usr/bin/env bash
set -euo pipefail
if (( $# < 3 )); then
  echo "Usage: $0 MODEL INPUT.jsonl OUTPUT.jsonl [extra inference arguments]" >&2
  exit 2
fi
model="$1"; input="$2"; output="$3"; shift 3
if [[ -e "$output" ]]; then
  echo "Output already exists: $output" >&2
  exit 2
fi
parts="$(mktemp -d)"
trap 'rm -rf "$parts"' EXIT
torchrun --standalone --nnodes 1 --nproc_per_node 2 -m innerjev.infer \
  --model "$model" --input "$input" --output "$parts" --replicas 2 \
  --gpus 1 --precision bf16 --batch-size 4 "$@"
python -m innerjev.merge_predictions --input "$input" --parts "$parts" --output "$output"
