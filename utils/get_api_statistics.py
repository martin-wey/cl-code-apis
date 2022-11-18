import argparse
from collections import Counter
import pickle
import os

from datasets import load_dataset
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='martiwey/github-java-methods', type=str,
                        help='Name of the dataset on the Huggingface hub.')
    parser.add_argument('--num_proc', default=8, type=int,
                        help='Number of process to use.')
    args = parser.parse_args()

    dataset = load_dataset(args.dataset_name, split='train', use_auth_token=True)
    print(dataset)

    inits_per_api = Counter()
    calls_per_api = Counter()
    calls_per_api_methods = Counter()
    samples_per_api = Counter()
    api_in_samples = {}

    for index, row in enumerate(tqdm(dataset, desc='Retrieving API statistics from data.')):
        # split api usages
        api_usages = list(filter(lambda e: e != '', row['api_seq'].split('|')))
        api_usages = [u.split('.') for u in api_usages]
        # remove java wildcards (e.g., ArrayList<String> -> ArrayList)
        #   and get a list of api usage in the form of [api_class, api_call, index_in_source_code, jdk|non-jdk]
        api_usages = list(map(lambda e: [e[0].strip().split(' ')[0],
                                         e[1].strip().split(' ')[0],
                                         e[1].strip().split(' ')[1]], api_usages))

        for api_usage in api_usages:
            # update counters
            if api_usage[1] == '<init>':
                inits_per_api.update([api_usage[0]])
            else:
                calls_per_api.update([api_usage[0]])
                calls_per_api_methods.update([f'{api_usage[0]}.{api_usage[1]}'])

            if api_usage[0] not in api_in_samples.keys():
                samples_per_api.update([api_usage[0]])
                api_in_samples[api_usage[0]] = [index]
            else:
                if index not in api_in_samples[api_usage[0]]:
                    samples_per_api.update([api_usage[0]])
                    api_in_samples[api_usage[0]].append(index)

    api_statistics = {
        'inits_per_api': inits_per_api,
        'calls_per_api': calls_per_api,
        'calls_per_api_methods': calls_per_api_methods,
        'samples_per_api': samples_per_api,
        'api_in_samples': api_in_samples
    }

    output_dir = os.path.join('data', args.dataset_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(f'{output_dir}/api_statistics.pkl', 'wb') as fout:
        pickle.dump(api_statistics, fout)

    print(f'API statistics saved: {output_dir}/api_statistics.pkl')


if __name__ == '__main__':
    main()
