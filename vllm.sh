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
    rm -rf ~/vllm ~/tpu-inference
    git clone https://github.com/TaiMingLu/tpu-vllm.git ~/tpu-inference
    python3.11 -m venv ~/vllm_env --symlinks
    source ~/vllm_env/bin/activate
    pip install --upgrade pip setuptools wheel
    cd ~/tpu-inference
    pip install -r requirements_vllm.txt
    pip install vllm-tpu
    pip install -e .
    '

gcloud compute tpus tpu-vm ssh terry@${TPU_NAME} \
  --project=${PROJECT_ID} --zone=${ZONE} \
  --worker=0 \
  --ssh-key-file="~/.ssh/id_rsa" \
  --command='
    source ~/vllm_env/bin/activate
    python -c "import vllm; import tpu_inference; print(\"Ready!\")"
    '