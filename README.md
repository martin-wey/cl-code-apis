# On the Usage of Continual Learning for Out-of-Distribution Generalization in Pre-trained Languages Models of Code

This is the official replication package associated with the FSE 23' submission:
```On the Usage of Continual Learning for Out-of-Distribution Generalization" in Pre-trained Languages Models of Code```.

In this readme, we provide details on how to setup our codebase for experimenting with continual fine-tuning. We include links to download our datasets and models locally. Unfortunately, we cannot leverage HuggingFace's hub as it does not allow double-blind sharing of repositories.

## Installation
1. Clone this repository using ```git```.
2. Setup a ```Python 3``` virtual environment and install the requirements.
```shell
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```
Note that Pytorch's version may not be suitable for your environment. Make sure the installed version matches your ```CUDA``` version.

## Data acquisition
All the data used in our experiments can be downloaded here: https://zenodo.org/record/7600769#.Y9y_FXbML-h. The data where downloaded from GitHub using Google BigQuery. 
We do not share the 68M Java methods (full dataset extracted from GitHub), but intend to publish the data later.

Here is a description of each dataset:
* ```gh-java-methods-small-id-train```: in-distribution pre-training data (~9M java methods).
* ```gh-java-methods-small-id-valid```: in-distribution validation data.
* ```gh-java-methods-small-id-test```: in-distribution test data
* ```gh-java-methods-small-ood-train```: out-of-distribution fine-tuning data.
* ```gh-java-methods-small-ood-test```: out-of-distribution test data.

The datasets are compressed using parquet format. You do not need to download the training/validation data if you do not intend to pre-training or fine-tuning models.

We recommend storing the dataset folders inside a ```data``` folder in the folder of your local clone of this repository.

## Models checkpoints
We provide all models checkpoints used in our experiments. ```code-gpt-small``` refers to the decoder based on GPT2 architecture, and ```code-roberta-base``` refers to the encoder based on RoBERTa architecture.
The models can be downloaded from the same link as the data: https://zenodo.org/record/7600769#.Y9y_FXbML-h.

The hyperparameters and details about the models' architectures can be found in the following configuration files:
* ```./configuration/model/code-gpt2-small_config.json``` for the decoder.
* ```./configuration/model/code-roberta-base_config.json``` for the encoder.

Here is a description of each model:
* ```code-gpt-small```: checkpoint of the decoder after pre-training.
* ```code-roberta-base```: checkpoint of the encoder after pre-training.
* ```code-gpt-small_ft_$method$```: checkpoint of the decoder after fine-tuning using a specific method (*i.e.*, rwalk, ewc, naive)
* ```code-roberta-base_$method$```: checkpoint of the decoder after fine-tuning using a specific method (*i.e.*, rwalk, ewc, naive) 

Inside each fine-tuning checkpoint, you will find five sub-folders. Each sub-folder contains the checkpoint of the model after a specific fine-tuning step.
For the paper, we fine-tuned the models sequentially on the OOD datasets following the order: General &rarr; Security &rarr; Android &rarr; Web &rarr; Guava. Therefore, the folder ```exp_0``` contains the model's checkpoint after fine-tuning the pre-trained model on the ```General``` OOD dataset. The folder ```exp_1``` contains the model's checkpoint after fine-tuning the model from checkpoint ```exp_0``` on the ```Security``` OOD dataset, etc.

We recommend storing the dataset folders inside a ```models``` folder in the folder of your local clone of this repository.

## Replicating experiments

We use [Hydra](https://hydra.cc/) to run all our codes using configuration files. The configuration files can be found in the folder ```configuration```. You can choose to change the .yaml files manually or to change variables directly in the shell.

Additionally, for pre-training and fine-tuning we use [Wandb](https://wandb.ai/site). We left the usage of Wandb optional.

You do not need to pre-train or fine-tune any model as we provide all the checkpoints. Nonetheless, we provide explanation on how to run a pre-training and a fine-tuning for sake of completeness.

### Pre-training a model from scratch
We leverage [Accelerate](https://huggingface.co/docs/accelerate/index) and [deepspeed](https://huggingface.co/docs/accelerate/usage_guides/deepspeed) to pre-train our models. We provide the configuration we used in the file ```accelerator_deepspeed.yaml```.

To pre-train a GPT2-like or a RoBERTa-like model, run the following command:
```shell
accelerate launch --config_file accelerator_deepspeed.yaml run_pretraining.py \
  model=(code-roberta-base | code-gpt2-small) \
  run=pretraining
  use_wandb=(true | false) \
  hydra=output_pretraining
```
You can change hyperparameters of the model by editing variables from the ```./configuration/run/pretraining.yaml``` file.

For instance, to set the weight_decay to 0.01, set `run.weight_decay=0.01`.

### Fine-tuning a model
You can fine-tune a model using the following command:
```shell
python run_finetuning.py \
  run=finetuning \
  run.strategy=(naive | cumulative | ewc | rwalk | si | replay) \
  run.train_batch_size=8 \
  run.valid_batch_size=8 \
  run.learning_rate= 5e-5 \
  run.num_epochs_per_experience=10 \
  run.patience=2 \
  model.model_name_or_path=./models/code-gpt2-small \
  model.model_base_path=./models/code-gpt2-small \
  hydra=output_finetuning 
```
This will output log and model files in a new directory: 
`./models/code-gpt2-small/ft_$method$/` where $method$ refers to the fine-tuning approach used.

You can change hyperparameters of each fine-tuning strategy in the ```./configuration/run/finetuning.yaml``` configuration file.

### Inference
To run zero-shot inference on the in-distribution dataset, run the following command:
```shell
python run_inference.py \
  run=inference \
  run.dataset_name=./data/gh-java-methods-small-id-ft-test \
  run.domain=all \
  run.task=(call | usage) \
  model.model_name_or_path=./models/code-gpt2-small
  hydra=output_inference
```
This will test `code-gpt-small` in zero-shot on the ID data and all domains (by concatenating the test set). You do separate zero-shot testing by setting the variable `run.domain` to a specific domain (e.g., general, security, ...) as specified in the `./configuration/run/inference.yaml` configuration file.

To run inference on a fine-tuned model, run the following command:
```shell
python run_inference.py \
  run=inference \
  run.dataset_name=./data/gh-java-methods-small-ood-test \
  run.domain=(all | general | security | android | web | guava) \
  run.task=(call | usage) \
  model.model_name_or_path=./models/code-gpt2-small_ft_$method$/exp_$id$
  hydra=output_inference
```
Note that fine-tuning produces five checkpoints, *i.e.*, one after each fine-tuning step. Therefore, you need to specify which checkpoint you want to test, *e.g.*, `exp_0`, `exp_1`, etc.
