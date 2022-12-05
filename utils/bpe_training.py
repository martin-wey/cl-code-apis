"""
BPE tokenizer training.
"""
import argparse

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, set_seed
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode as bytes_to_unicode_gpt2
from transformers.models.roberta.tokenization_roberta import bytes_to_unicode as bytes_to_unicode_roberta


def batch_iterator(dataset, args, batch_size=1000):
    for start_idx in tqdm(range(0, args.n_examples, batch_size)):
        samples = dataset[start_idx:start_idx + batch_size]
        yield samples['source_code']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='martiwey/gh-java-methods-small-id-train', type=str,
                        help='Dataset name on the Huggingface hub.')
    parser.add_argument('--output_dir', default='./data/bpe_tokenizer', type=str,
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
    if args.tokenizer_type == 'gpt2':
        base_vocab = list(bytes_to_unicode_gpt2().values())
    else:
        base_vocab = list(bytes_to_unicode_roberta().values())

    dataset = load_dataset(args.dataset_name, split='train', use_auth_token=True)

    new_tokenizer = tokenizer.train_new_from_iterator(
        batch_iterator(dataset, args), vocab_size=args.vocab_size, initial_alphabet=base_vocab
    )
    new_tokenizer.save_pretrained(args.output_dir)


if __name__ == '__main__':
    main()
