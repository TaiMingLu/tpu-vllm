#!/bin/bash
set -euo pipefail

# TPU Configuration
TPU_NAME="taiming-v6e-8_000103"
ZONE="europe-west4-a"
PROJECT_ID="vision-mix"

# Environment variables (replicate MaxText setup)
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
MAX_TARGET_LENGTH=4096
GEN_BATCH_SIZE=128
TENSOR_PARALLEL_SIZE=1
TEMPERATURE=0.8
TOP_P=0.95
MAX_EXAMPLES=20000000

OUTPUT_DIR="/tmp/sequence-kd-vllm/output"
GCS_BUCKET_PATH="/home/terry/gcs-bucket/sequence_kd_data/finewebedu/sample-100BT/vllm-T50BS42"

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

chmod 600 ~/.ssh/id_rsa
chmod 600 ~/.ssh/google_rsa

# Copy the Python script to TPU
echo "Copying generation script to TPU..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
gcloud compute tpus tpu-vm scp \
  --project=${PROJECT_ID} \
  --zone=${ZONE} \
  --worker=0 \
  "${SCRIPT_DIR}/sequence_kd_parquet_vllm.py" \
  terry@${TPU_NAME}:~/work-dir/sequence_kd_parquet_vllm.py

if [ $? -ne 0 ]; then
    echo "Error: Failed to copy script to TPU"
    exit 1
fi

echo "✓ Script copied successfully"
echo

# Run on TPU
echo "Running sequence KD generation on TPU..."
echo "================================================================================"
gcloud compute tpus tpu-vm ssh terry@${TPU_NAME} \
  --project=${PROJECT_ID} --zone=${ZONE} \
  --worker=0 \
  --ssh-key-file="~/.ssh/id_rsa" \
  --command="
    set -euo pipefail

    # Activate vLLM environment
    source ~/work-dir/vllm_env/bin/activate
    cd ~/work-dir

    # Create directories
    mkdir -p /tmp/sequence-kd
    mkdir -p '${OUTPUT_DIR}'
    mkdir -p '${GCS_BUCKET_PATH}'

    # Run generation
    python3 -u sequence_kd_parquet_vllm.py \\
      --input-dir '${DATASET_PATH}' \\
      --output-dir '${OUTPUT_DIR}' \\
      --model-path '${MODEL_PATH}' \\
      --tokenizer-path '${TOKENIZER_PATH}' \\
      --hf-access-token '${HF_ACCESS_TOKEN}' \\
      --text-column '${TEXT_COLUMN}' \\
      --batch-size ${GEN_BATCH_SIZE} \\
      --max-prefill-length ${MAX_PREFILL_LENGTH} \\
      --max-target-length ${MAX_TARGET_LENGTH} \\
      --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \\
      --temperature ${TEMPERATURE} \\
      --top-p ${TOP_P} \\
      --gcs-bucket-path '${GCS_BUCKET_PATH}' \\
      --save-every-n-batches 4
  "

if [ $? -eq 0 ]; then
    echo
    echo "================================================================================"
    echo "✓ Sequence KD generation completed successfully!"
    echo "Output saved to: ${GCS_BUCKET_PATH}"
    echo "================================================================================"
else
    echo
    echo "================================================================================"
    echo "✗ Generation failed. Check the output above for errors."
    echo "================================================================================"
    exit 1
fi
