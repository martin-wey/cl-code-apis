import json
import os
import pathlib
import random
import re
import shutil

from tqdm import tqdm

from utils import download_url, unzip_file

CSN_DATASET_SPLIT_PATH = 'https://github.com/guoday/CodeBERT/raw/master/GraphCodeBERT/codesearch/dataset.zip'
CSN_DATASET_BASE_PATH = 'https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/'
CSN_LANGUAGES = ('java', 'python', 'php', 'go', 'javascript', 'ruby')


def download_codesearchnet_dataset(dataset_dir):
    """Download CodeSearchNet dataset and clean it using GraphCodeBERT cleaning splits (Guo et al's)

    Return:
        dataset_dir (str): the path containing the downloaded dataset.
    """
    zip_file_path = 'dataset.zip'

    if not os.path.exists(zip_file_path):
        print('Downloading CodeSearchNet dataset...')
        download_url(CSN_DATASET_SPLIT_PATH, zip_file_path)
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)

    unzip_file(zip_file_path, './')
    os.mkdir(dataset_dir)

    for lang in CSN_LANGUAGES:
        print(f'***** Creating {lang} dataset. *****')
        if not os.path.isfile(os.path.join(dataset_dir, f'{lang}.zip')):
            print(f'- Downloading CodeSearchNet {lang} dataset.')
            download_url(os.path.join(CSN_DATASET_BASE_PATH, f'{lang}.zip'), os.path.join(dataset_dir, f'{lang}.zip'))
        if not os.path.exists(os.path.join(dataset_dir, lang)):
            unzip_file(os.path.join(dataset_dir, f'{lang}.zip'), dataset_dir)

        print(f'- Filtering {lang} dataset.')
        data = {}
        # gzip all .gz files and add them to `data` with their url as key
        for file in tqdm(pathlib.Path(os.path.join(dataset_dir, lang)).rglob('*.gz')):
            unzip_file(str(file), '', str(file)[:-3])
            os.remove(file)
            with open(str(file)[:-3]) as f:
                for line in f:
                    js = json.loads(line)
                    data[js['url']] = js
        for split in ['train', 'valid', 'test', 'codebase']:
            with open(os.path.join(dataset_dir, lang, f'{split}.jsonl'), 'w') as f1, open(
                    os.path.join('dataset', lang, f'{split}.txt'), encoding='utf-8') as f2:
                for line in f2:
                    line = line.strip()
                    # we only keep code snippets that are clean (based on GraphCodeBERT cleaning)
                    #   by matching the url with a key in `data`.
                    if line in data:
                        f1.write(json.dumps(data[line]) + '\n')
        shutil.rmtree(os.path.join(dataset_dir, lang, 'final'))
    # clean folders
    for file in os.listdir(dataset_dir):
        if re.match('.*.(zip|pkl|py|sh)', file):
            os.remove(os.path.join(dataset_dir, file))


def create_splits(dataset_path, split):
    """Might be useful to create splits for other datasets."""
    dataset_dir = os.path.dirname(dataset_path)
    if (os.path.isfile(os.path.join(dataset_dir, 'train.jsonl'))
            and os.path.isfile(os.path.join(dataset_dir, 'test.jsonl'))
            and os.path.isfile(os.path.join(dataset_dir, 'valid.jsonl'))):
        print('Splits already created.')
        return
    with open(dataset_path, 'r') as f:
        data = list(f)
        # we already seeded random package in the main
        random.shuffle(data)
        total_count = len(data)
        train_count = int(split[0] * total_count)
        valid_count = int(split[1] * total_count)
        train_data = data[:train_count]
        valid_data = data[train_count:train_count + valid_count]
        test_data = data[train_count + valid_count:]
    with open(os.path.join(dataset_dir, 'train.jsonl'), 'w') as f1, \
            open(os.path.join(dataset_dir, 'valid.jsonl'), 'w') as f2, \
            open(os.path.join(dataset_dir, 'test.jsonl'), 'w') as f3:
        print('Creating training split.')
        for line in tqdm(train_data):
            js = json.loads(line)
            f1.write(json.dumps(js) + '\n')
        print('Creating validation split.')
        for line in tqdm(valid_data):
            js = json.loads(line)
            f2.write(json.dumps(js) + '\n')
        print('Creating test split.')
        for line in tqdm(test_data):
            js = json.loads(line)
            f3.write(json.dumps(js) + '\n')


if __name__ == '__main__':
    download_codesearchnet_dataset(dataset_dir='csn/')
