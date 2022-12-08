#!/bin/bash

# list of tuples (exp, domains)
declare -a experiences=(
  "0 general"
  "1 general security"
  "2 general security android"
  "3 general security android web"
  "4 general security android web guava"
)

TASK="call"
STRATEGY="naive"

for exp in "${experiences[@]}"; do
  id="${exp[0]}"
  read -a domains <<< "${exp[@]:1}"

  for d in "${domains[@]}"; do
    echo "exp ${id} - ${d}"
    CUDA_VISIBLE_DEVICES=3 python run_inference.py \
      run=inference \
      hydra=output_inference \
      model.model_name_or_path="./run_outputs/code-gpt2-small/ft_${TASK}_${STRATEGY}/exp_${id}" \
      run.task=${TASK} \
      run.domain="${d}"
  done;
done;
