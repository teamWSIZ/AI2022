from random import random

from helpers import log, to_str

"""
W tym file-u generujemy różne dane/przykłady poddawane potem analizie przez (proste) sieci neuronowe. 
"""


def get_radom_pattern(list_len: int, probability: float):
    """
    Generate a random list of length `list_len` with 0's or 1's; probability of each 1 is `probability`.
    """
    patt = [0] * list_len
    for i in range(list_len):
        if random() < probability: patt[i] = 1
    return patt


def get_patterns_single_1(list_len: int, probability: float, n_samples: int):
    """
    Generate `n_n_samples` lists of length `list_len` with 0's and 1's; p(1) = `probability`.
    """
    patterns = [get_radom_pattern(list_len, probability) for _ in range(n_samples)]
    outputs = [1 if sum(p) == 1 else 0 for p in patterns]
    log(f'Generated {n_samples} patterns, {int(100 * sum(outputs) / n_samples)}% positive')
    # input()
    return patterns, outputs


def get_patterns_for_mask(list_len: int, probability: float, n_samples: int, mask: str):
    samples = [get_radom_pattern(list_len, probability) for _ in range(n_samples)]
    outputs = [1 if mask in to_str(k) else 0 for k in samples]
    log(f'Generated {n_samples} patterns, {int(100 * sum(outputs) / n_samples)}% positive')
    return samples, outputs


def get_patterns_multimask(list_len: int, probability: float, n_samples: int, masks: list[str]):
    samples = [get_radom_pattern(list_len, probability) for _ in range(n_samples)]
    outputs = []
    len_output = len(masks) + 1
    for s in samples:
        o = [0] * len_output
        str_s = to_str(s)
        for i, m in enumerate(masks):
            if m in str_s: o[i] = 1
        if sum(o) == 0: o[-1] = 1
        outputs.append(o)
    stats = [0] * len_output
    for o in outputs:
        for i in range(len_output): stats[i] += o[i]

    log(f'Generated {n_samples} patterns, stats: {stats}')
    return samples, outputs


if __name__ == '__main__':
    ll, ou = get_patterns_single_1(4, 0.20, 50)
    # ll, ou = get_patterns_multimask(14, 0.25, 15, ['111', '000', '101'])
    for s, ou in zip(ll, ou):
        print(f'sample: {s} output/label:{ou}')
