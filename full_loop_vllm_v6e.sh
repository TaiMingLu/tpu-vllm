#!/bin/bash
set -euo pipefail

# Get the directory where this script lives
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Copy Python script to work-dir to avoid import conflicts with local tpu_inference/
cp "${SCRIPT_DIR}/sequence_kd_parquet_vllm.py" ~/work-dir/

# Change to work-dir
cd ~/work-dir

# Environment variables
HF_ACCESS_TOKEN=${HF_ACCESS_TOKEN:-""}
BUCKET_NAME=${BUCKET_NAME:-taiming_europe_west4_a}

# Configuration (matching your MaxText setup exactly)
RUN_NAME="sequence-kd-vllm-v6e"
DATASET_PATH="/home/terry/gcs-bucket/HF_HOME/finewebedu/sample/100BT"
DATA_SPLIT="train"
TEXT_COLUMN="text"
TEACHER_MODEL_NAME="llama3.1-1b"
MODEL_PATH="/home/terry/gcs-bucket/ckpts/pretrain_param_only_hf/llama3.1-1b-finewebedu-vanilla-s42"
TOKENIZER_PATH="/home/terry/gcs-bucket/HF_HOME/Llama-3.1-8B"
MAX_PREFILL_LENGTH=1024
MAX_TARGET_LENGTH=2048
GEN_BATCH_SIZE=1024
TENSOR_PARALLEL_SIZE=1
TEMPERATURE=1.2                # Lower = more focused, less random (was 1.0)
TOP_P=0.9                      # Slightly more restrictive (was 0.95)
REPETITION_PENALTY=1.5         # Very strong penalty (was 1.2, try 1.3-1.5)
FREQUENCY_PENALTY=1.5          # Strong frequency penalty (was 1.0, try 1.5-2.0)
PRESENCE_PENALTY=1.0           # Strong presence penalty (was 0.5, try 1.0)
MAX_EXAMPLES=20000000

OUTPUT_DIR="/tmp/sequence-kd-vllm/output"
GCS_BUCKET_PATH="/home/terry/gcs-bucket/sequence_kd_data/finewebedu/sample-100BT/vllm-T50BS42_new"

printf '\n=== Sequence KD Config (vLLM) ===\n'
printf 'Run name: %s\n' "$RUN_NAME"
printf 'Bucket: %s\n' "$BUCKET_NAME"
printf 'Dataset: %s (%s)\n' "$DATASET_PATH" "$DATA_SPLIT"
printf 'Teacher model: %s\n' "$TEACHER_MODEL_NAME"
printf 'Model path: %s\n' "$MODEL_PATH"
printf 'Tokenizer: %s\n' "$TOKENIZER_PATH"
printf 'Output dir: %s\n' "$OUTPUT_DIR"
printf 'GCS bucket path: %s\n' "$GCS_BUCKET_PATH"
printf '==================================\n\n'

# Activate vLLM environment
source ~/work-dir/vllm_env/bin/activate

# Create directories
mkdir -p /tmp/sequence-kd-vllm
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${GCS_BUCKET_PATH}"

# Run generation (using local copy in work-dir)
python3 -u sequence_kd_parquet_vllm.py \
  --input-dir "${DATASET_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --model-path "${MODEL_PATH}" \
  --tokenizer-path "${TOKENIZER_PATH}" \
  --hf-access-token "${HF_ACCESS_TOKEN}" \
  --text-column "${TEXT_COLUMN}" \
  --batch-size ${GEN_BATCH_SIZE} \
  --max-prefill-length ${MAX_PREFILL_LENGTH} \
  --max-target-length ${MAX_TARGET_LENGTH} \
  --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
  --temperature ${TEMPERATURE} \
  --top-p ${TOP_P} \
  --repetition-penalty ${REPETITION_PENALTY} \
  --frequency-penalty ${FREQUENCY_PENALTY} \
  --presence-penalty ${PRESENCE_PENALTY} \
  --gcs-bucket-path "${GCS_BUCKET_PATH}" \
  --save-every-n-batches 4

echo
echo "================================================================================"
echo "✓ Sequence KD generation completed successfully!"
echo "Output saved to: ${GCS_BUCKET_PATH}"
echo "================================================================================"
