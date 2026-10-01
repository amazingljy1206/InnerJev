#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
torchrun --standalone --nnodes 1 --nproc_per_node 4 -m innerjev.train \
  --config "$root/configs/4B/full.json" "$@"
