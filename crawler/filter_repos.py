"""
Filter crawled repositories by library.
example usage:
    > python filter_repos.py torch torch>=1.7.0
"""

import sys
import json

import pkg_resources
from tqdm import tqdm

SEARCH_LIB_NAME = sys.argv[1]
SEARCH_LIB_SPECS = sys.argv[2]
LANGUAGE = 'python' if len(sys.argv) <= 3 else sys.argv[3]  # Default to Python, if none passed.


def main():
    repositories = []
    with open(f'TopLists/{LANGUAGE}-top-repos.jsonl', 'r') as f:
        for line in f:
            repositories.append(json.loads(line))

    print('Retrieving repositories dependencies specifications.')
    repositories_libs, bad_requirements_idx = globals()[f'get_libs_{LANGUAGE}'](repositories)

    # remove repositories with invalid requirements
    for idx in sorted(bad_requirements_idx, reverse=True):
        del repositories[idx]
    # merge repositories information with their dependencies
    repos_info = list(zip(repositories, repositories_libs))

    print('Extracting repositories having the searched library as dependency.')
    valid_repo_idx = []
    for i, (repository, libs) in enumerate(tqdm(repos_info)):
        lib_in_repo = globals()[f'is_lib_in_repo_{LANGUAGE}'](libs)
        if lib_in_repo:
            valid_repo_idx.append(i)

    valid_repositories = [repos_info[i][0] for i in valid_repo_idx]
    with open(f'TopLists/{LANGUAGE}-top-repos_{SEARCH_LIB_NAME}.txt', 'w') as f:
        for repository in valid_repositories:
            f.write(f'{repository["url"]}\t{repository["stars"]}\n')


def is_lib_in_repo_python(repo_libs):
    found = False
    """
    try:
        search_lib_req = pkg_resources.Requirement.parse(SEARCH_LIB_SPECS)
        search_lib_dist = pkg_resources.Distribution(project_name=search_lib_req)
    except pkg_resources.DistributionNotFound:
        raise ValueError(f'Library distribution not found: {SEARCH_LIB_NAME}\tv{SEARCH_LIB_VERSION}.')
    """
    search_lib_req = pkg_resources.Requirement.parse(SEARCH_LIB_SPECS)
    for lib in repo_libs:
        # https://setuptools.pypa.io/en/latest/pkg_resources.html#requirement-methods-and-attributes
        lib_dist = pkg_resources.find_distributions(SEARCH_LIB_SPECS)
        print(lib_dist)
        if search_lib_req.__contains__(lib_dist.version):
            found = True
            break
    return found


def get_libs_python(repositories):
    repositories_libs = []
    bad_requirements_idx = []
    for i, repository in enumerate(tqdm(repositories)):
        try:
            libs = repository['libs']
            install_requires = [requirement for requirement in pkg_resources.parse_requirements(libs)]
            repositories_libs.append(install_requires)
        except pkg_resources.extern.packaging.requirements.InvalidRequirement:
            bad_requirements_idx.append(i)
    return repositories_libs, bad_requirements_idx


if __name__ == '__main__':
    main()
