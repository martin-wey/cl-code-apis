# Continual learning for code

### Pre-training a new model

### Evaluating a model

Get loss and perplexity on a dataset with a GPT-2 model.
```shell
python run_inference.py \
  run=code_generation \
  run.run_name=perplexity_ood \
  run.dataset_name=martiwey/gh-java-methods-small-ood \
  run.evaluate=perplexity \
  run.batch_size=16 \
  model.model_name_or_path=./run_outputs/clm_pretraining_code-gpt2-small/step_20000
```