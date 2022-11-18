from argparse import Namespace

from datasets import load_dataset

# Settings
config = {
    'in_dataset_name': 'martiwey/github-java-methods',
    'out_dataset_name': 'martiwey/github-java-methods-small',
    'n_samples': 10000000,
    'seed': 42,
    'shard_size': 1000 << 20
}

args = Namespace(**config)

dataset = load_dataset(args.in_dataset_name, split='train', use_auth_token=True)
print(dataset)
print(f"Shuffling data...")
dataset = dataset.shuffle(seed=args.seed)
dataset = dataset.select(list(range(args.n_samples)))
print(dataset)

# Save dataset in HF hub repo
dataset.push_to_hub(args.out_dataset_name, max_shard_size=args.shard_size)
