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
    echo "=== Checking Docker images ==="
    sudo docker images
    echo ""
    echo "=== Checking disk space ==="
    df -h /var/lib/docker
    echo ""
    echo "=== Checking Docker version ==="
    sudo docker --version
    '
