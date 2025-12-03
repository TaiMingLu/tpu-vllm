#!/bin/bash
set -euo pipefail

# Combined setup + run script for multi-host TPU with Ray
# This runs on each worker and does setup if needed, then runs generation
#
# Environment variables expected:
#   TENSOR_PARALLEL_SIZE - total chips across all hosts
#   RAY_HEAD_IP - IP address of Ray head node (worker 0)
#   WORKER_ID - which worker this is (0, 1, 2, ...)

echo "=== Worker $(hostname) starting ==="
echo "WORKER_ID: ${WORKER_ID:-not set}"
echo "RAY_HEAD_IP: ${RAY_HEAD_IP:-not set}"

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

# Stop any existing Ray processes
ray stop 2>/dev/null || true
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
    echo "Installing vllm-tpu and ray..."
    pip install vllm-tpu ray[default]

    echo "=== Setup complete ==="
else
    echo "Skipping setup - vLLM already installed"
    # Make sure ray is installed
    source ~/work-dir/vllm_env/bin/activate
    pip show ray >/dev/null 2>&1 || pip install ray[default]
fi

# Activate venv for Ray
source ~/work-dir/vllm_env/bin/activate

# Start Ray cluster
WORKER_ID=${WORKER_ID:-0}
RAY_PORT=6379

if [ "$WORKER_ID" == "0" ]; then
    echo "=== Starting Ray head node ==="
    ray start --head --port=$RAY_PORT --num-cpus=4 --resources='{"TPU": 4}'

    # Give head node time to start
    sleep 5

    # Now run the generation script (only on head node)
    echo "=== Starting generation ==="
    bash full_loop_vllm_v6e_multi.sh
else
    echo "=== Starting Ray worker node ==="
    # Connect to head node
    ray start --address="${RAY_HEAD_IP}:${RAY_PORT}" --num-cpus=4 --resources='{"TPU": 4}'

    # Keep alive while head node runs
    echo "Ray worker started, waiting for head node to finish..."
    while ray status >/dev/null 2>&1; do
        sleep 10
    done
    echo "Ray cluster shut down, exiting"
fi
