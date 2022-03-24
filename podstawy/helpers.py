from datetime import datetime

from torch import Tensor


def format_list(l: list[float]) -> str:
    res = '['
    for x in l:
        res += f' {x:5.2f}'
    res += ']'
    return res

def format_t(t: Tensor, n_few = 4):
    s = ''
    for i in range(n_few):
        s += format_list(t[i].tolist()) + ','
    return s[:-1]

def to_str(w):
    """Convert a list of int's to one string"""
    return ''.join([str(k) for k in w])


def format_list_int(l: list[int]) -> str:
    return format_list([float(x) for x in l])


def clock():
    return datetime.now().strftime('%H:%M:%S.%f')[:]


def log(what):
    print(f'[{clock()}]: {what}')
