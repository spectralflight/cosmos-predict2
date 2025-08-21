#!/usr/bin/env -S bash -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <num_gpus>"
    exit 1
fi
NUM_GPUS="${1}"
shift

uv run torchrun --nproc_per_node=${NUM_GPUS} examples/video2world.py --num_gpus ${NUM_GPUS} "$@"
