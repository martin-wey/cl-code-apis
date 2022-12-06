# Continual learning for code

### Pre-training a model from scratch

### Fine-tuning a model
Run fine-tuning on API call completion task using the in-distribution dataset.
```shell
python run_finetuning.py \
  run=finetuning \
  hydra=output_finetuning \
  run.dataset_name=gh-java-methods-small-id-ft-train \
  run.domain=id-all \
  run.task=api-call-completion \
  model.model_name_or_path=./run_outputs/code-gpt2-small/checkpoint \
  model.model_base_path=./run_outputs/code-gpt2-small
```
This will output log and model files in a new directory: 
`./run_outputs/code-gpt2-small/finetuning_api-call-completion_id-all/`

Run fine-tuning on API call completion task using the Guava out-of-distribution dataset.
```shell
python run_finetuning.py \
  run=finetuning \
  hydra=output_finetuning \
  run.dataset_name=gh-java-methods-small-ood-train \
  run.domain=ood-guava \
  run.task=api-call-completion \
  model.model_name_or_path=./run_outputs/code-gpt2-small/checkpoint \
  model.model_base_path=./run_outputs/code-gpt2-small
```
You can set `run.task=api_usage_completion` to fine-tune the model on API usage completion task.

### Inference
Run inference on API call completion 
```shell
python run_inference.py \
  run=inference \
  hydra=output_inference \
  run.dataset_name=gh-java-methods-small-id-ft-test \
  run.domain=all \
  run.experiment=naive \
  run.task=api-call-completion \
  model.model_name_or_path=./run_outputs/code-gpt2-small/checkpoint
```
This will output log and model files in a new directory: 
`./run_outputs/code-gpt2-small/checkpoint/naive/api-call-completion/gh-java-methods-small-id-ft-test_all`