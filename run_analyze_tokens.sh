#!/bin/bash

# Configuration (matching your dataset)
TPU_NAME="taiming-v6e-8_000103"
ZONE="europe-west4-a"
PROJECT_ID="vision-mix"

DATASET_PATH="/home/terry/gcs-bucket/HF_HOME/finewebedu/sample/100BT"
TOKENIZER_PATH="/home/terry/gcs-bucket/HF_HOME/Llama-3.1-8B"
TEXT_COLUMN="text"
NUM_SAMPLES=10000  # Analyze 10k random samples (adjust as needed)

chmod 600 ~/.ssh/id_rsa

echo "================================================================================"
echo "Analyzing Token Lengths in Dataset"
echo "================================================================================"
echo "Dataset: ${DATASET_PATH}"
echo "Tokenizer: ${TOKENIZER_PATH}"
echo "Samples: ${NUM_SAMPLES}"
echo "================================================================================"
echo

# Copy script to TPU
echo "Copying analysis script to TPU..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
gcloud compute tpus tpu-vm scp \
  --project=${PROJECT_ID} \
  --zone=${ZONE} \
  --worker=0 \
  "${SCRIPT_DIR}/analyze_dataset_tokens.py" \
  terry@${TPU_NAME}:~/work-dir/analyze_dataset_tokens.py

if [ $? -ne 0 ]; then
    echo "Error: Failed to copy script to TPU"
    exit 1
fi

echo "✓ Script copied"
echo

# Run analysis on TPU
echo "Running analysis on TPU..."
echo "================================================================================"
gcloud compute tpus tpu-vm ssh terry@${TPU_NAME} \
  --project=${PROJECT_ID} --zone=${ZONE} \
  --worker=0 \
  --ssh-key-file="~/.ssh/id_rsa" \
  --command="
    set -e

    # Activate vLLM environment (has transformers)
    source ~/work-dir/vllm_env/bin/activate

    cd ~/work-dir

    # Run analysis
    python3 analyze_dataset_tokens.py \\
      --input-dir '${DATASET_PATH}' \\
      --tokenizer-path '${TOKENIZER_PATH}' \\
      --text-column '${TEXT_COLUMN}' \\
      --num-samples ${NUM_SAMPLES} \\
      --output /tmp/token_stats.json \\
      --random-seed 42

    # Show the results file
    echo
    echo 'Results saved to /tmp/token_stats.json'
    echo 'Raw lengths saved to /tmp/token_stats.lengths.npy'
  "

if [ $? -eq 0 ]; then
    echo
    echo "================================================================================"
    echo "✓ Analysis completed successfully!"
    echo "================================================================================"
else
    echo
    echo "================================================================================"
    echo "✗ Analysis failed. Check the output above for errors."
    echo "================================================================================"
    exit 1
fi
