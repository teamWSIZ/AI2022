from random import random

from ptorch.helpers import log


def get_1():
    samples = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    output = [1, 1, 1]
    return samples, output


def get_0():
    samples = [[0, 0, 0], [1, 1, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]]
    output = [0, 0, 0, 0, 0]
    return samples, output


def get_radom_pattern(len: int, probability: float):
    patt = [0] * len
    for i in range(len):
        if random() < probability: patt[i] = 1
    return patt


def get_patterns(len: int, probability: float, n_samples: int):
    patterns = [get_radom_pattern(len, probability) for _ in range(n_samples)]
    outputs = [1 if sum(p) == 1 else 0 for p in patterns]
    log(f'Generated {n_samples} patterns, {int(100 * sum(outputs) / n_samples)}% positive')
    # input()
    return patterns, outputs


if __name__ == '__main__':
    # print(get_radom_pattern(10, 0.2))
    get_patterns(10, 0.15, 500)
