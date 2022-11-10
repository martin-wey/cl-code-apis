# Continual learning for code

## Generating the pre-training and fine-tuning datasets

### Data preprocessing
We used 50k-c java dataset as raw data. It consists of 50k compilable Java projects. 
From each project, we extract their java methods and the API usage sequences in every method.

Next step is to preprocess the dataset using the `utils/preprocessing.py` script as follows:
```shell
python utils/preprocessing.py \
  --dataset_dir ./data \
  --output_dir ./data/preprocessed \
  --samples_per_file 100000 \
  --line_max 250 \
  --num_proc 8 \
  --seed 42
```
It will deduplicate the dataset and filter out samples with too many lines (arg `line_max`). The preprocessed
data are saved and compressed under the specified output directory. 

### Data splitting

To split the preprocessed data into in-distribution and out-of-distribution data, use the `utils/data_splitting.py` script: