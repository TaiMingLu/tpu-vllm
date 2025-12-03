#!/bin/bash
set -euo pipefail

# Combined setup + run script for multi-host TPU
# This runs on each worker and does setup if needed, then runs generation
#
# Environment variables expected:
#   TENSOR_PARALLEL_SIZE - total chips across all hosts

echo "=== Worker $(hostname) starting ==="

# Kill any existing processes using the TPU (from previous failed runs)
echo "Killing any existing TPU processes..."
device_name="vfio/"  # v5e/v6e use vfio
pids=$(sudo lsof -t /dev/${device_name}* 2>/dev/null | sort -u || true)
if [[ -n "${pids}" ]]; then
    echo "Found existing processes: ${pids}"
    for pid in ${pids}; do
        echo "Killing process ${pid}..."
        sudo kill -9 "${pid}" 2>/dev/null || true
        # Wait for process to die
        tail --pid="${pid}" -f /dev/null 2>/dev/null || true
    done
    echo "Existing processes killed"
fi
sudo rm -f /tmp/libtpu_lockfile
echo "TPU is ready"

# Wait for any running apt processes to finish (unattended-upgrades)
echo "Waiting for apt lock..."
while sudo fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1 ; do
    echo "  Waiting for other apt process to finish..."
    sleep 5
done

# Check if vllm-tpu is actually installed (not just venv exists)
VLLM_INSTALLED=false
if [ -d ~/work-dir/vllm_env ]; then
    source ~/work-dir/vllm_env/bin/activate
    if pip show vllm-tpu >/dev/null 2>&1 || pip show vllm >/dev/null 2>&1; then
        VLLM_INSTALLED=true
        echo "vLLM already installed: $(pip show vllm-tpu 2>/dev/null | grep Version || pip show vllm 2>/dev/null | grep Version)"
    fi
    deactivate 2>/dev/null || true
fi

if [ "$VLLM_INSTALLED" = false ]; then
    echo "=== Setting up vLLM environment ==="

    # Install Python 3.12
    echo "Installing Python 3.12..."
    sudo apt-get update && sudo apt-get install -y python3.12 python3.12-venv

    # Create work directory
    echo "Setting up work directory..."
    rm -rf ~/work-dir/vllm_env  # Clean up any partial install
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
    echo "Skipping setup - vLLM already installed"
fi

# Now run the generation script
echo "=== Starting generation ==="
bash full_loop_vllm_v6e_multi.sh
