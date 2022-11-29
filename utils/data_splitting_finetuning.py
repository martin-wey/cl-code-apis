"""
Splitting a fine-tuning dataset into train and test.

For OOD dataset, we select samples according to the API.
    For each API, we randomly select 10% samples to be included in the OOD test data.
For ID dataset, we randomly select 10% of the samples as ID test data.

In both case, the remaining samples are in the train data used for fine-tuning.
We do not select validation samples as we select them on-the-fly prior to fine-tuning in the fine-tuning script.
"""
import argparse
import pickle

from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
from transformers import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='martiwey/gh-java-methods-small-id-ft', type=str,
                        help='Name of the dataset on the Huggingface hub.')
    parser.add_argument('--dataset_type', default='id', type=str,
                        help='Type of the dataset to process (id/ood)')
    parser.add_argument('--train_dataset_name', default='martiwey/gh-java-methods-small-id-ft-train', type=str,
                        help='Name of the training dataset on the Huggingface hub.')
    parser.add_argument('--test_dataset_name', default='martiwey/gh-java-methods-small-id-ft-test', type=str,
                        help='Name of the testing dataset on the Huggingface hub.')
    parser.add_argument('--ood_data_file', default='data/ood_data_statistics.pkl', type=str,
                        help='Pickle file containing the OOD sample ids.')
    parser.add_argument('--percentage_test_samples', default=.10,
                        help='Percentage of samples to include in the test data.')
    parser.add_argument('--num_proc', default=32, type=int,
                        help='Number of process to use.')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed for replication.')
    args = parser.parse_args()

    if args.seed > 0:
        set_seed(args.seed)

    ds = load_dataset(args.dataset_name, split='train', use_auth_token=True)

    if args.dataset_type == 'ood':
        train_datasets = []
        test_datasets = []

        with open(args.ood_data_file, 'rb') as fin:
            ood_data = pickle.load(fin)
        domains = ood_data.keys()

        for domain in domains:
            apis = ood_data[domain]['apis']
            for api in tqdm(apis):
                ds_api = ds.filter(lambda e: e['api'] == api)
                ds_api = ds_api.shuffle(seed=args.seed)
                n_test_samples = int(len(ds_api) * args.percentage_test_samples)
                ds_api_test = ds_api.select(list(range(n_test_samples)))
                ds_api_train = ds_api.select(list(range(n_test_samples, len(ds_api))))
                test_datasets.append(ds_api_test)
                train_datasets.append(ds_api_train)

        train_dataset = concatenate_datasets(train_datasets)
        test_dataset = concatenate_datasets(test_datasets)
    else:
        ds = ds.shuffle(seed=args.seed)
        n_test_samples = int(len(ds) * args.percentage_test_samples)
        test_dataset = ds.select(list(range(n_test_samples)))
        train_dataset = ds.select(list(range(n_test_samples, len(ds))))

    train_dataset.push_to_hub(args.train_dataset_name)
    test_dataset.push_to_hub(args.test_dataset_name)


if __name__ == '__main__':
    main()
