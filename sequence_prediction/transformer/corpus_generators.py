from random import randint


def get_journey(length=10000, max_distance=40) -> list[int]:
    # 0 = tree, 1 =semaphore, 2 = station Left, 3 = station right
    # periodic first
    journey = [0] * length
    at = 0
    station = 2
    while at + 1 < length:
        journey[at] = 1
        journey[at + 1] = station
        # station = 2 if station == 3 else 3
        # at += randint(3, max_distance)
        at += 10
    return journey


def get_small_samples(length=10000) -> list[int]:
    # alphabet = 0,1,2,3; if there ever is a "2" it is always followed by 3, if there ever is 3, it is always in "23"
    journey = [randint(0, 3) for _ in range(length)]
    for i in range(1, length):
        if journey[i] == 3: journey[i - 1] = 2
        if journey[i - 1] == 2: journey[i] = 3
    # now remove half of these "23" complexes (they appear too often)
    for i in range(1, length):
        if journey[i] == 3 and i % 2 == 0:
            journey[i] = 0
            journey[i - 1] = 0
    return journey


def get_periodic_samples(length=10000, dist=3) -> list[int]:
    # alphabet = 0,1,2,3; repeating pattern of 000213000213000213000213...
    journey = []
    pat = [0] * dist
    pat.extend([2, 1, 3])
    while len(journey) < length:
        journey.extend(pat)
    journey = journey[:length]
    return journey


def get_periodic(length=1000, alphabet=10) -> list[int]:
    data = [i % alphabet for i in range(length)]
    return data


if __name__ == '__main__':
    print(get_journey(500))
