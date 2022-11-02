"""
BPE tokenizer training.
Adapted from: https://github.com/huggingface/transformers/blob/main/examples/research_projects/codeparrot/scripts/bpe_training.py
"""
import argparse

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, set_seed
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode


def batch_iterator(iter_dataset, args, batch_size=10):
    for _ in tqdm(range(0, args.n_examples, batch_size)):
        yield [next(iter_dataset)['source_code'] for _ in range(batch_size)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', default='./data_preprocessed', type=str,
                        help='Directory where the csv raw data file is stored.')
    parser.add_argument('--output_dir', default='./bpe_tokenizer', type=str,
                        help='Output directory.')
    parser.add_argument('--tokenizer_type', default='gpt2', type=str,
                        help='Type of the tokenizer to train.')
    parser.add_argument('--n_examples', default=200000, type=int,
                        help='Number of examples to train the tokenizer on.')
    parser.add_argument('--vocab_size', default=32768, type=int,
                        help='Vocabulary size.')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed for replication.')
    args = parser.parse_args()

    if args.seed > 0:
        set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_type)
    base_vocab = list(bytes_to_unicode().values())

    dataset = load_dataset(args.dataset_dir, split='train', streaming=True)
    iter_dataset = iter(dataset)

    new_tokenizer = tokenizer.train_new_from_iterator(
        batch_iterator(iter_dataset, args), vocab_size=args.vocab_size, initial_alphabet=base_vocab
    )
    new_tokenizer.save_pretrained(args.output_dir)


if __name__ == '__main__':
    main()
