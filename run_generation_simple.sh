#!/bin/bash

# Configuration
TPU_NAME="taiming-v6e-8_000103"
ZONE="europe-west4-a"
PROJECT_ID="vision-mix"

# Model configuration
MODEL_PATH="/home/terry/gcs-bucket/ckpts/pretrain_param_only_hf/llama3.1-1b-finewebedu-vanilla-s42"
TOKENIZER_PATH="/home/terry/gcs-bucket/HF_HOME/Llama-3.1-8B"

# Input/Output paths - CHANGE THESE to your actual data paths
INPUT_PARQUET="/home/terry/gcs-bucket/data/prompts.parquet"
OUTPUT_PARQUET="/home/terry/gcs-bucket/data/completions.parquet"

# Generation parameters
TEMPERATURE=0.8
TOP_P=0.95
TOP_K=-1
MAX_TOKENS=512
MAX_MODEL_LEN=2048
REPETITION_PENALTY=1.0

# Column names in your parquet file
PROMPT_COLUMN="prompt"
COMPLETION_COLUMN="completion"

# Optional: limit number of rows for testing
# MAX_ROWS="--max-rows 100"
MAX_ROWS=""

# Set SSH permissions
chmod 600 ~/.ssh/id_rsa
chmod 600 ~/.ssh/google_rsa

echo "================================================================================"
echo "vLLM Parquet Generation"
echo "================================================================================"
echo "TPU: ${TPU_NAME}"
echo "Model: ${MODEL_PATH}"
echo "Input: ${INPUT_PARQUET}"
echo "Output: ${OUTPUT_PARQUET}"
echo "================================================================================"
echo

# Step 1: Copy the Python script to TPU
echo "Step 1: Copying generation script to TPU..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
gcloud compute tpus tpu-vm scp \
  --project=${PROJECT_ID} \
  --zone=${ZONE} \
  --worker=0 \
  "${SCRIPT_DIR}/generate_from_parquet.py" \
  terry@${TPU_NAME}:~/work-dir/generate_from_parquet.py

if [ $? -ne 0 ]; then
    echo "Error: Failed to copy script to TPU"
    exit 1
fi

echo "✓ Script copied successfully"
echo

# Step 2: Run generation on TPU
echo "Step 2: Running generation on TPU..."
echo "================================================================================"
gcloud compute tpus tpu-vm ssh terry@${TPU_NAME} \
  --project=${PROJECT_ID} --zone=${ZONE} \
  --worker=0 \
  --ssh-key-file="~/.ssh/id_rsa" \
  --command="
    set -e

    # Activate vLLM environment
    source ~/work-dir/vllm_env/bin/activate

    # Change to work directory
    cd ~/work-dir

    # Run generation
    python3 generate_from_parquet.py \\
      --input '${INPUT_PARQUET}' \\
      --output '${OUTPUT_PARQUET}' \\
      --model '${MODEL_PATH}' \\
      --tokenizer '${TOKENIZER_PATH}' \\
      --prompt-column '${PROMPT_COLUMN}' \\
      --completion-column '${COMPLETION_COLUMN}' \\
      --temperature ${TEMPERATURE} \\
      --top-p ${TOP_P} \\
      --top-k ${TOP_K} \\
      --max-tokens ${MAX_TOKENS} \\
      --max-model-len ${MAX_MODEL_LEN} \\
      --repetition-penalty ${REPETITION_PENALTY} \\
      --tensor-parallel-size 1 \\
      --compression snappy \\
      --skip-special-tokens \\
      ${MAX_ROWS}
  "

if [ $? -eq 0 ]; then
    echo
    echo "================================================================================"
    echo "✓ Generation completed successfully!"
    echo "Output saved to: ${OUTPUT_PARQUET}"
    echo "================================================================================"
else
    echo
    echo "================================================================================"
    echo "✗ Generation failed. Check the output above for errors."
    echo "================================================================================"
    exit 1
fi
