#!/bin/bash

# list of tuples (exp, domains)
declare -a experiences=(
  "0 general"
  "1 general security"
  "2 general security android"
  "3 general security android web"
  "4 general security android web guava"
)

TASK="usage"
STRATEGY="naive"

for exp in "${experiences[@]}"; do
  id="${exp[@]:0:1}"
  read -a domains <<< "${exp[@]:1}"

  for d in "${domains[@]}"; do
    echo "exp ${id} - ${d}"
    python evaluator.py \
      --ref "../../run_outputs/code-gpt2-small/ft_${TASK}_${STRATEGY}/exp_${id}/${TASK}_${d}/gt.txt" \
      --pre "../../run_outputs/code-gpt2-small/ft_${TASK}_${STRATEGY}/exp_${id}/${TASK}_${d}/predictions.txt"
  done;
done;
