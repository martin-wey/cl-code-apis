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
    python calc_code_bleu.py \
      --refs "../../../run_outputs/code-gpt2-small/ft_${TASK}_${STRATEGY}/exp_${id}/${TASK}_${d}/gt.txt" \
      --hyp "../../../run_outputs/code-gpt2-small/ft_${TASK}_${STRATEGY}/exp_${id}/${TASK}_${d}/predictions.txt" \
      --lang java
  done;
done;
