#!/bin/bash
set -euo pipefail

# Get the directory where this script lives
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Copy Python script to work-dir to avoid import conflicts with local tpu_inference/
cp "${SCRIPT_DIR}/analyze_dataset_tokens.py" ~/work-dir/

# Change to work-dir
cd ~/work-dir

# Configuration (matching your dataset)
DATASET_PATH="/home/terry/gcs-bucket/HF_HOME/finewebedu/sample/100BT"
TOKENIZER_PATH="/home/terry/gcs-bucket/HF_HOME/Llama-3.1-8B"
TEXT_COLUMN="text"
NUM_SAMPLES=10000  # Analyze 10k random samples (adjust as needed)
OUTPUT_PATH="/tmp/token_stats.json"

printf '\n=== Dataset Token Analysis ===\n'
printf 'Dataset: %s\n' "$DATASET_PATH"
printf 'Tokenizer: %s\n' "$TOKENIZER_PATH"
printf 'Text column: %s\n' "$TEXT_COLUMN"
printf 'Number of samples: %d\n' "$NUM_SAMPLES"
printf 'Output path: %s\n' "$OUTPUT_PATH"
printf '==============================\n\n'

# Activate vLLM environment (has transformers)
source ~/work-dir/vllm_env/bin/activate

# Create output directory
mkdir -p /tmp

# Run analysis (using local copy in work-dir)
python3 -u analyze_dataset_tokens.py \
  --input-dir "${DATASET_PATH}" \
  --tokenizer-path "${TOKENIZER_PATH}" \
  --text-column "${TEXT_COLUMN}" \
  --num-samples ${NUM_SAMPLES} \
  --output "${OUTPUT_PATH}" \
  --random-seed 42

echo
echo "================================================================================"
echo "✓ Analysis completed successfully!"
echo "Results saved to: ${OUTPUT_PATH}"
echo "Raw lengths saved to: /tmp/token_stats.lengths.npy"
echo "================================================================================"
