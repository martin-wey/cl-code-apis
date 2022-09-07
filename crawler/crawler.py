"""
GitHub repository crawling script.
Adapted from: https://github.com/VHellendoorn/Code-LMs/blob/main/Data/gh_crawler.py
"""

import requests
import sys
import time
import json

git_credentials = sys.argv[1]
headers = {'Authorization': f'token {git_credentials}'}

# Constants & language argument.
NUM_REPOS = 10
MIN_STARS = 50
LAST_ACTIVE = '2020-01-01'
LANGUAGE = 'python' if len(sys.argv) <= 2 else sys.argv[2]  # Default to Java, if none passed.
REQUIREMENTS_FILE = {
    'python': 'requirements.txt'
}


def main():
    repositories = set()  # Keep track of a set of repositories seen to avoid duplicate entries across pages.
    next_max_stars = 1_000_000_000
    with open(f'TopLists/{LANGUAGE}-top-repos.jsonl', 'w') as f:
        while len(repositories) < NUM_REPOS:
            results = run_query(next_max_stars)
            if not results:
                break
            new_repositories = [repository for repository, _ in results]
            next_max_stars = min([stars for _, stars in results])

            # If a query returns no new repositories, drop it.
            if len(repositories | set(new_repositories)) == len(repositories):
                break
            for (repository_url, repository_libs), stars in sorted(results, key=lambda e: e[1], reverse=True):
                if repository_url not in repositories:
                    repositories.add(repository_url)
                    item = {'url': repository_url, 'stars': stars, 'libs': repository_libs}
                    f.write(json.dumps(item) + '\n')
            f.flush()
            print(f'Collected {len(repositories):,} repositories so far; lowest number of stars: {next_max_stars:,}')


def run_query(max_stars):
    end_cursor = None  # Used to track pagination.
    repositories = set()

    while end_cursor != "":
        # Extracts non-fork, recently active repositories in the provided language, in groups of 100.
        # Leaves placeholders for maximum stars and page cursor. The former allows us to retrieve more than 1,000 repositories
        # by repeatedly lowering the bar.
        query = f"""
        {{
          search(query: "language:{LANGUAGE} fork:false pushed:>{LAST_ACTIVE} sort:stars stars:<{max_stars}", type: REPOSITORY, first: 100 {', after: "' + end_cursor + '"' if end_cursor else ''}) {{
            edges {{
              node {{
                ... on Repository {{
                  name
                  owner {{
                    login
                  }}
                  url
                  isPrivate
                  isDisabled
                  isLocked
                  stargazers {{
                    totalCount
                  }}
                  object(expression: "master:{REQUIREMENTS_FILE[LANGUAGE]}") {{
                    ... on Blob {{
                      text
                    }}
                  }}
                }}
              }}
            }}
            pageInfo {{
              hasNextPage
              endCursor
            }}
          }}
        }}
        """
        print(f'  Retrieving next page; {len(repositories)} repositories in this batch so far.')

        content, success = send_query(query)
        end_cursor = get_end_cursor(content)
        new_repositories, is_done = get_repositories(content)
        repositories.update(new_repositories)
        if len(repositories) > NUM_REPOS or is_done:
            break
    return repositories


def send_query(query):
    # Attempt a query up to three times, pausing when a query limit is hit.
    attempts = 0
    success = False
    while not success and attempts < 3:
        request = requests.post('https://api.github.com/graphql', json={'query': query}, headers=headers)
        content = request.json()
        if 'data' not in content or 'search' not in content['data']:
            # If this is simply a signal to pause querying, wait two minutes.
            if 'message' in content and 'wait' in content['message']:
                attempts += 1
                time.sleep(120)
            # Otherwise, assume we've hit the end of the stream.
            else:
                break
        else:
            success = True
    return content, success


def get_end_cursor(content):
    page_info = content['data']['search']['pageInfo']
    has_next_page = page_info['hasNextPage']
    if has_next_page:
        return page_info['endCursor']
    return ""


def get_repositories(content):
    edges = content['data']['search']['edges']
    repositories_with_stars = []
    for edge in edges:
        if edge['node']['isPrivate'] is False and edge['node']['isDisabled'] is False \
                and edge['node']['isLocked'] is False and edge['node']['object'] is not None:
            star_count = edge['node']['stargazers']['totalCount']
            if star_count < MIN_STARS:
                return repositories_with_stars, True
            repository_url = edge['node']['url']
            repository_libs = edge['node']['object']['text']
            repositories_with_stars.append(((repository_url, repository_libs), star_count))
    return repositories_with_stars, False


if __name__ == '__main__':
    main()
