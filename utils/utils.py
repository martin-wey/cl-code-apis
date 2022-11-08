import re
import tokenize
from io import StringIO
import numpy as np

from .jdk_apis import JDK_APIS


def get_apis_tokenized(tokenizer, selected_apis=None):
    #  api_cls_ood_tokenized, api_calls_ood_tokenized = get_apis_tokenized(tokenizer, ('HashMap', 'Set'))
    api_list = JDK_APIS.split()
    api_dict = {}

    for api in api_list:
        api_usage = api.split('.')
        if selected_apis is not None:
            if api.startswith(selected_apis):
                if api_usage[0] not in api_dict:
                    api_dict[api_usage[0]] = []
                api_dict[api_usage[0]].append(api_usage[1])
        else:
            if api_usage[0] not in api_dict:
                api_dict[api_usage[0]] = []
            api_dict[api_usage[0]].append(api_usage[1])

    print(api_dict)

    # api_cls_tokenized = tokenizer(list(selected_apis), return_attention_mask=False).input_ids
    # api_calls_tokenized = tokenizer(api_calls, return_attention_mask=False).input_ids

    return None, None


def find_patterns_in_batch_ids(batch_ids, patterns):
    batch_size = batch_ids.shape[0]
    block_size = batch_ids.shape[1]
    match_idx_array = [[] for _ in range(batch_size)]

    for i in range(batch_size):
        ids = batch_ids.cpu().numpy()[i, :]
        for pattern in patterns:
            search = np.asarray(pattern)
            search_len = len(search)
            sublists = []
            for j in range(block_size):
                if j + search_len < block_size:
                    sublists.append(ids[j:j + search_len])
            sublists = np.asarray(sublists)
            match_idx = np.flatnonzero((sublists == search).all(1))
            for match in match_idx:
                for id in range(match, match + search_len):
                    match_idx_array[i].append(id)

    return match_idx_array


def remove_comments_and_docstrings(source, lang):
    if lang in ['python']:
        """
        Returns 'source' minus comments and docstrings.
        """
        io_obj = StringIO(source)
        out = ""
        prev_toktype = tokenize.INDENT
        last_lineno = -1
        last_col = 0
        for tok in tokenize.generate_tokens(io_obj.readline):
            token_type = tok[0]
            token_string = tok[1]
            start_line, start_col = tok[2]
            end_line, end_col = tok[3]
            if start_line > last_lineno:
                last_col = 0
            if start_col > last_col:
                out += (" " * (start_col - last_col))
            # Remove comments:
            if token_type == tokenize.COMMENT:
                pass
            # This series of conditionals removes docstrings:
            elif token_type == tokenize.STRING:
                if prev_toktype != tokenize.INDENT:
                    # This is likely a docstring; double-check we're not inside an operator:
                    if prev_toktype != tokenize.NEWLINE:
                        if start_col > 0:
                            out += token_string
            else:
                out += token_string
            prev_toktype = token_type
            last_col = end_col
            last_lineno = end_line
        temp = []
        for x in out.split('\n'):
            if x.strip() != "":
                temp.append(x)
        return '\n'.join(temp)
    elif lang in ['ruby']:
        return source
    else:
        def replacer(match):
            s = match.group(0)
            if s.startswith('/'):
                return " "  # note: a space and not an empty string
            else:
                return s

        pattern = re.compile(
            r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
            re.DOTALL | re.MULTILINE
        )
        temp = []
        for x in re.sub(pattern, replacer, source).split('\n'):
            if x.strip() != "":
                temp.append(x)
        return '\n'.join(temp)
