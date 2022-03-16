from random import random

from helpers import log

"""
W tym file-u generujemy różne dane/przykłady poddawane potem analizie przez (proste) sieci neuronowe. 
"""


def get_1():
    samples = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    output = [1, 1, 1]
    return samples, output


def get_0():
    samples = [[0, 0, 0], [1, 1, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]]
    output = [0, 0, 0, 0, 0]
    return samples, output


def get_radom_pattern(list_len: int, probability: float):
    """
    Generate a random list of length `list_len` with 0's or 1's; probability of each 1 is `probability`.
    """
    patt = [0] * list_len
    for i in range(list_len):
        if random() < probability: patt[i] = 1
    return patt


def get_patterns(list_len: int, probability: float, n_samples: int):
    """
    Generate `n_n_samples` lists of length `list_len` with 0's and 1's; p(1) = `probability`.
    """
    patterns = [get_radom_pattern(list_len, probability) for _ in range(n_samples)]
    outputs = [1 if sum(p) == 1 else 0 for p in patterns]
    log(f'Generated {n_samples} patterns, {int(100 * sum(outputs) / n_samples)}% positive')
    # input()
    return patterns, outputs


if __name__ == '__main__':
    # print(get_radom_pattern(10, 0.2))
    ll, ou = get_patterns(10, 0.25, 5)
    print(ll)
    print(ou)
