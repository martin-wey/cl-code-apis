"""
Splitting preprocessed data into in-distribution and out-of-distribution datasets.

OOD data: samples using at least one of the target APIs -> continual fine-tuning.
ID data: rest of the samples -> pre-training & fine-tuning.
"""
import argparse
import pickle

from datasets import load_dataset, concatenate_datasets
from transformers import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='martiwey/gh-java-methods-small', type=str,
                        help='Name of the dataset on the Huggingface hub.')
    parser.add_argument('--ood_data_file', default='data/ood_data_statistics.pkl', type=str,
                        help='Pickle file containing the OOD sample ids.')
    parser.add_argument('--id_train_dataset_name', default='martiwey/gh-java-methods-small-id-train', type=str,
                        help='Name of the in-distribution training dataset on the Huggingface hub.')
    parser.add_argument('--id_valid_dataset_name', default='martiwey/gh-java-methods-small-id-valid', type=str,
                        help='Name of the in-distribution valid dataset on the Huggingface hub.')
    parser.add_argument('--id_ft_dataset_name', default='martiwey/gh-java-methods-small-id-ft', type=str,
                        help='Name of the in-distribution fine-tuning dataset on the Huggingface hub.')
    parser.add_argument('--ood_dataset_name', default='martiwey/gh-java-methods-small-ood', type=str,
                        help='Name of the out-of-distribution dataset on the Huggingface hub.')
    parser.add_argument('--n_valid_samples', default=100000,
                        help='Number of validation samples to extract from the ID dataset.')
    parser.add_argument('--n_ft_samples', default=500000,
                        help='Number of samples to extract from the ID dataset for fine-tuning.')
    parser.add_argument('--num_proc', default=32, type=int,
                        help='Number of process to use.')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed for replication.')
    args = parser.parse_args()

    if args.seed > 0:
        set_seed(args.seed)

    with open(args.ood_data_file, 'rb') as fin:
        ood_data = pickle.load(fin)

    dataset = load_dataset(args.dataset_name, split='train', use_auth_token=True)

    all_ids = []
    for domain, domain_data in ood_data.items():
        print(f'Extracting dataset for domain `{domain}`')
        domain_ds = dataset.select(domain_data['samples'])
        all_ids += domain_data['samples']
        print(domain_ds)

        ids_per_api = {}
        for api, api_data in domain_data['sample_ids_per_api'].items():
            ids_per_api[api] = api_data['api_in_samples']

        def get_sample_api(idx):
            for api, ids in ids_per_api.items():
                if idx in ids:
                    return {'domain': domain, 'api': api}

        def preprocess(idx):
            """Chain all preprocessing steps into one function to not fill cache."""
            results = dict()
            results.update(get_sample_api(idx))
            return results

        domain_ds = domain_ds.map(lambda _, idx: preprocess(idx), with_indices=True, num_proc=args.num_proc)
        domain_data['dataset'] = domain_ds

    print('Concatenating domain datasets into OOD dataset.')
    ood_dataset = concatenate_datasets([d['dataset'] for k, d in ood_data.items()])
    print(ood_dataset)
    print('Pushing OOD dataset to Huggingface hub.')
    # ood_dataset.push_to_hub(args.ood_dataset_name)

    print('Extracting ID datasets.')
    id_dataset = dataset.filter(lambda e, idx: idx not in all_ids, with_indices=True, num_proc=args.num_proc)
    # we do not reshuffle `id_dataset` as it was already shuffled in `create_small_dataset.py`.
    print('Splitting ID dataset into train/validation/fine-tuning sets.')
    id_ft_dataset = id_dataset.select(list(range(args.n_ft_samples)))
    id_valid_dataset = id_dataset.select(list(range(args.n_ft_samples, (args.n_ft_samples + args.n_valid_samples))))
    id_train_dataset = id_dataset.select(list(range((args.n_ft_samples + args.n_valid_samples), len(id_dataset))))
    print(f'ID dataset: {id_dataset}')
    print(f'Fine-tuning dataset: {id_ft_dataset}')
    print(f'Train dataset: {id_train_dataset}')
    print(f'Valid dataset: {id_valid_dataset}')

    print('Pushing ID datasets to Huggingface hub.')
    id_ft_dataset.push_to_hub(args.id_ft_dataset_name)
    id_train_dataset.push_to_hub(args.id_train_dataset_name)
    id_valid_dataset.push_to_hub(args.id_valid_dataset_name)


if __name__ == '__main__':
    main()
