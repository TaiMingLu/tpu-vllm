#!/bin/bash

TPU_NAME="taiming-v6e-8_000103"
ZONE="europe-west4-a"
PROJECT_ID="vision-mix"

chmod 600 ~/.ssh/id_rsa
chmod 600 ~/.ssh/google_rsa

gcloud compute tpus tpu-vm ssh terry@${TPU_NAME} \
  --project=${PROJECT_ID} --zone=${ZONE} \
  --worker=0 \
  --ssh-key-file="~/.ssh/id_rsa" \
  --command='
    echo "=== Looking for tokenizer files in bucket ==="
    find /home/terry/gcs-bucket -name "tokenizer.model" -o -name "tokenizer.json" 2>/dev/null | head -20
    '
