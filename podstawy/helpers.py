from datetime import datetime


def format_list(l: list[float]) -> str:
    res = '['
    for x in l:
        res += f' {x:5.3f}'
    res += ']'
    return res


def format_list_int(l: list[int]) -> str:
    return format_list([float(x) for x in l])

def clock():
    return datetime.now().strftime('%H:%M:%S.%f')[:]


def log(what):
    print(f'[{clock()}]: {what}')
