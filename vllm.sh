TPU_NAME="taiming-v6e-8_000103"
ZONE="europe-west4-a"
PROJECT_ID="vision-mix"

chmod 600 ~/.ssh/id_rsa
chmod 600 ~/.ssh/google_rsa

gcloud compute tpus tpu-vm ssh terry@${TPU_NAME} \
  --project=${PROJECT_ID} --zone=${ZONE} \
  --worker=all \
  --ssh-key-file="~/.ssh/id_rsa" \
  --command=''