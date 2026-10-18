#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# NPROC_PER_NODE=2 bash train.sh ... for two GPUs.
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-1}" train.py "$@"
