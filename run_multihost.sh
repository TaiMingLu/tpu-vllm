#!/bin/bash
set -euo pipefail

# Simple launcher for multi-host TPU execution
# Usage: ./run_multihost.sh <tpu-name> <tpu-type-or-chip-count>
# Example:
#   ./run_multihost.sh terry-v6e-32 v6e-32
#   ./run_multihost.sh terry-v6e-32 32

if [ $# -lt 2 ]; then
    echo "Usage: $0 <tpu-name> <tpu-type-or-chip-count>"
    echo "Examples:"
    echo "  $0 terry-v6e-32 v6e-32    # TPU type (extracts '32')"
    echo "  $0 terry-v6e-32 32        # Direct chip count"
    echo "  $0 my-v5e-16 v5e-16"
    echo "  $0 my-v5e-16 16"
    exit 1
fi

TPU_NAME=$1
TPU_TYPE_OR_SIZE=$2

# Extract chip count if TPU type provided (e.g., "v6e-32" -> "32")
if [[ "$TPU_TYPE_OR_SIZE" =~ ^v[0-9]+[a-z]*-([0-9]+)$ ]]; then
    TENSOR_PARALLEL_SIZE="${BASH_REMATCH[1]}"
    echo "Detected TPU type: ${TPU_TYPE_OR_SIZE} -> ${TENSOR_PARALLEL_SIZE} chips"
else
    # Assume it's already a chip count
    TENSOR_PARALLEL_SIZE=$TPU_TYPE_OR_SIZE
fi

echo "Running on TPU: ${TPU_NAME}"
echo "Tensor parallel size: ${TENSOR_PARALLEL_SIZE}"
echo

python3 multihost_runner.py \
    --tpu-name="${TPU_NAME}" \
    --command="export TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE} && bash full_loop_vllm_v6e_multi.sh" \
    --script-dir=.
