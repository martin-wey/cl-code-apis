"""
Dataset preprocessing and upload on Huggingface hub.
Adapted from: https://github.com/huggingface/transformers/blob/main/examples/research_projects/codeparrot/scripts/preprocessing.py
"""
import argparse
import gzip
import hashlib
import os
import re
import shutil
from pathlib import Path

from datasets import load_dataset
from transformers import set_seed

PATTERN = re.compile(r"\s+")


def get_hash(example):
    """Get hash of source_code field."""
    return {'hash': hashlib.md5(re.sub(PATTERN, '', example['source_code']).encode('utf-8')).hexdigest()}


def line_stats(example):
    """Calculates the number of line of the method."""
    return {'lines': len(example['source_code'].splitlines())}


def check_uniques(example, uniques):
    """Check if current hash is still in set of unique hashes and remove if true."""
    if example["hash"] in uniques:
        uniques.remove(example['hash'])
        return True
    else:
        return False


def preprocess(example):
    """Chain all preprocessing steps into one function to not fill cache."""
    results = dict()
    results.update(get_hash(example))
    results.update(line_stats(example))
    return results


def filter(example, uniques, args):
    """Filter dataset with heuristics. Config, test and has_no_keywords files are removed with a given probability."""
    if not check_uniques(example, uniques):
        return False
    elif example['lines'] > args.line_max:
        return False
    else:
        return True


def compress_file(file_path):
    """Compress a file with g-zip."""
    with open(file_path, "rb") as f_in:
        with gzip.open(str(file_path) + ".gz", "wb", compresslevel=6) as f_out:
            shutil.copyfileobj(f_in, f_out)
    os.unlink(file_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', default='./data', type=str,
                        help='Directory where the csv raw data file is stored.')
    parser.add_argument('--output_dir', default='./data_preprocessed', type=str,
                        help='Output directory.')
    parser.add_argument('--samples_per_file', default=100000, type=int,
                        help='Number of samples to store per preprocessed data file.')
    parser.add_argument('--line_max', default=250, type=int,
                        help='Maximum line length in the method, otherwise it is filtered.')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed for replication.')
    args = parser.parse_args()

    if args.seed > 0:
        set_seed(args.seed)

    ds = load_dataset(args.dataset_dir, data_files={'train': 'data_small.csv'}, split='train')
    ds = ds.map(preprocess, num_proc=8)

    uniques = set(ds.unique('hash'))
    frac = len(uniques) / len(ds)
    print(f"Fraction of duplicates: {1 - frac:.2%}")

    ds_filter = ds.filter(filter, fn_kwargs={'uniques': uniques, 'args': args})

    # Save data in batches of samples_per_file
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    for file_number, index in enumerate(range(0, len(ds_filter), args.samples_per_file)):
        file_path = str(output_dir / f"file-{file_number + 1:03}.json")
        end_index = min(len(ds_filter), index + args.samples_per_file)
        ds_filter.select(list(range(index, end_index))).to_json(file_path)
        compress_file(file_path)


if __name__ == '__main__':
    main()
