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
    sudo apt-get update && sudo apt-get install -y python3.12 python3.12-venv
    rm -rf ~/work-dir
    mkdir ~/work-dir
    cd ~/work-dir
    python3.12 -m venv vllm_env --symlinks
    source vllm_env/bin/activate
    pip install vllm-tpu
    python -c "import vllm; import tpu_inference; print(\"vLLM ready!\")"
    '

gcloud compute tpus tpu-vm ssh terry@${TPU_NAME} \
  --project=${PROJECT_ID} --zone=${ZONE} \
  --worker=0 \
  --ssh-key-file="~/.ssh/id_rsa" \
  --command='
    source ~/vllm_env/bin/activate
    python -c "import vllm; import tpu_inference; print(\"Ready!\")"
    '