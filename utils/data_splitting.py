"""
Splitting preprocessed data into in-distribution and out-of-distribution datasets.

OOD data: samples using at least one of the target APIs -> fine-tuning.
ID data: rest of the samples -> pre-training

1. Load dataset from preprocessed folder.
2. Go through the samples, add field containing the list of APIs used, store sample ids of those using the target APIs.
3. Filter out samples using the stored id -> OOD data.
4. Rest of the samples -> ID data.
5. Export datasets in sub-folders /id and /ood
"""
import argparse
import gzip
import os
import shutil
from pathlib import Path

from datasets import load_dataset
from transformers import set_seed

APIS = ('InputStreamReader', 'Collections')


def filter_contains_api(example):
    """Add a column to the example determining whether it uses at least one target api."""
    api_usages = list(filter(lambda e: e != '', example['api_seq'].split('|')))
    # remove useless spaces
    api_usages = [u.split('.') for u in api_usages]
    # remove the generic types of the API class, keep the API class name
    api_usages = list(map(lambda e: e[0].strip().split()[0], api_usages))
    api_usages_set = set(api_usages)

    for api in APIS:
        if api in api_usages_set:
            return {'ood': True}
    else:
        return {'ood': False}


def compress_file(file_path):
    """Compress a file with g-zip."""
    with open(file_path, 'rb') as f_in:
        with gzip.open(str(file_path) + '.gz', 'wb', compresslevel=6) as f_out:
            shutil.copyfileobj(f_in, f_out)
    os.unlink(file_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', default='./data/preprocessed', type=str,
                        help='Directory where the preprocessed data files are stored.')
    parser.add_argument('--samples_per_file', default=100000, type=int,
                        help='Number of samples to store per preprocessed data file.')
    parser.add_argument('--num_proc', default=8, type=int,
                        help='Number of process to use.')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed for replication.')
    args = parser.parse_args()

    if args.seed > 0:
        set_seed(args.seed)

    ds = load_dataset(args.dataset_dir, split='train')
    ds = ds.map(filter_contains_api, num_proc=args.num_proc)
    ds_id = ds.filter(lambda example: not example['ood'], num_proc=args.num_proc)
    ds_ood = ds.filter(lambda example: example['ood'], num_proc=args.num_proc)

    # Save data in batches of samples_per_file
    output_dir = Path(args.dataset_dir)

    for (ds, f) in ((ds_id, 'in'), (ds_ood, 'ood')):
        for file_number, index in enumerate(range(0, len(ds), args.samples_per_file)):
            file_path = str(output_dir / f / f'file-{file_number + 1:03}.json')
            end_index = min(len(ds), index + args.samples_per_file)
            ds.select(list(range(index, end_index))).to_json(file_path)
            compress_file(file_path)


if __name__ == '__main__':
    main()