#!/bin/bash
set -euo pipefail

# Combined setup + run script for multi-host TPU
# This runs on each worker and does setup if needed, then runs generation
#
# Environment variables expected:
#   TENSOR_PARALLEL_SIZE - total chips across all hosts

echo "=== Worker $(hostname) starting ==="

# Wait for any running apt processes to finish (unattended-upgrades)
echo "Waiting for apt lock..."
while sudo fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1 ; do
    echo "  Waiting for other apt process to finish..."
    sleep 5
done

# Check if venv exists, if not do setup
if [ ! -d ~/work-dir/vllm_env ]; then
    echo "=== Setting up vLLM environment ==="

    # Install Python 3.12
    echo "Installing Python 3.12..."
    sudo apt-get update && sudo apt-get install -y python3.12 python3.12-venv

    # Create work directory
    echo "Setting up work directory..."
    mkdir -p ~/work-dir
    cd ~/work-dir

    # Create Python virtual environment
    echo "Creating Python 3.12 virtual environment..."
    python3.12 -m venv vllm_env --symlinks

    # Activate and install vLLM
    source vllm_env/bin/activate
    echo "Installing vllm-tpu..."
    pip install vllm-tpu

    echo "=== Setup complete ==="
else
    echo "vLLM environment already exists, skipping setup"
fi

# Now run the generation script
echo "=== Starting generation ==="
bash full_loop_vllm_v6e_multi.sh
