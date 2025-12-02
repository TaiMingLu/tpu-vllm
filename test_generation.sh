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
    source ~/work-dir/vllm_env/bin/activate
    python3 - <<EOF
from vllm import LLM, SamplingParams

# Your converted model
model_path = "/home/terry/gcs-bucket/ckpts/pretrain_param_only_hf/llama3.1-1b-finewebedu-vanilla-s42"
tokenizer_path = "/home/terry/gcs-bucket/HF_HOME/Llama-3.1-8B"

print(f"Loading model from: {model_path}")
print(f"Using tokenizer from: {tokenizer_path}")
llm = LLM(
    model=model_path,
    tokenizer=tokenizer_path,
    tensor_parallel_size=1,
    max_model_len=512,
)

prompts = [
    "The capital of France is",
    "Once upon a time",
    "In machine learning,",
]

sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=50,
    skip_special_tokens=True,
)

print("\nGenerating responses...\n")
outputs = llm.generate(prompts, sampling_params)

for prompt, output in zip(prompts, outputs):
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}")
    print("-" * 80)
EOF
    '
