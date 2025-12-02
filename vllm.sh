TPU_NAME="taiming-v6e-8_000103"
ZONE="europe-west4-a"
PROJECT_ID="vision-mix"

chmod 600 ~/.ssh/id_rsa
chmod 600 ~/.ssh/google_rsa

gcloud compute tpus tpu-vm ssh terry@${TPU_NAME} \
  --project=${PROJECT_ID} --zone=${ZONE} \
  --worker=all \
  --ssh-key-file="~/.ssh/id_rsa" \
  --command='
    set -e  # Exit on error

    echo "=== Installing system dependencies ==="
    sudo apt-get update
    sudo apt-get install -y libopenblas-base libopenmpi-dev libomp-dev

    echo "=== Cloning repositories ==="
    # Clone vllm (use your fork if needed)
    if [ ! -d ~/vllm ]; then
        git clone https://github.com/vllm-project/vllm.git ~/vllm
    else
        echo "vllm directory already exists, skipping clone"
    fi

    # Clone tpu-inference (use your fork if available)
    if [ ! -d ~/tpu-inference ]; then
        git clone https://github.com/TaiMingLu/tpu-vllm.git ~/tpu-inference
    else
        echo "tpu-inference directory already exists, skipping clone"
    fi

    echo "=== Creating Python virtual environment ==="
    python3.12 -m venv ~/vllm_env --symlinks
    source ~/vllm_env/bin/activate

    # Upgrade pip
    pip install --upgrade pip setuptools wheel

    echo "=== Installing vllm from source ==="
    cd ~/vllm
    pip install -r requirements/tpu.txt
    VLLM_TARGET_DEVICE="tpu" pip install -e .

    echo "=== Installing tpu-inference and all dependencies from requirements_vllm.txt ==="
    cd ~/tpu-inference
    pip install -r requirements_vllm.txt
    pip install -e .

    echo "=== Installation complete! ==="
    echo "Virtual environment location: ~/vllm_env"
    echo "To activate: source ~/vllm_env/bin/activate"
    echo "vllm location: ~/vllm"
    echo "tpu-inference location: ~/tpu-inference"

    # Verify installation
    python3 -c "import vllm; print(f\"vLLM version: {vllm.__version__}\")"
    python3 -c "import tpu_inference; print(\"tpu-inference imported successfully\")"
    '