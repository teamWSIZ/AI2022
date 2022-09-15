from random import randint

N = 15
LR = 0.01
DI = 0.02


def reward(position):
    # todo: globalna funkcja ustawiająca nagrody za osiągnięcie pewnych stanów
    if position == 0:
        return 2
    if position == N - 1:
        return 10
    return 0


def simulate(state: int, action: int) -> int:
    """
    "Engine" pozwalający na "symulację gry", czyli znalezienie następnego stanu na podstawie obecnego stanu
    i wybranej akcji.

    :param state: pozycja na moście (aktualna)
    :param action: liczba typu 1 lub -1
    :return: int -- nowa pozycja na moście
    """
    # state == pozycja na moście (obecna)
    # action = +1 (w prawo), 0 (w lewo)
    next_position = 0
    if action == 0:
        next_position = state - 1  # fixme: tu można zmienić działanie akcji "0" na takie jak w tekście tutoriala
    elif action == 1:
        next_position = state + 1
    next_position = max(0, next_position)
    next_position = min(N - 1, next_position)
    return next_position

################################################
# Definicje strategii -- czyli funkcji podających który kierunek wybrać w zależności
# od aktualnej wiedzy (tablica Q), oraz ewentualnie aktualnej epoki (ile kroków eksploracji wykonano)
#


def random_strategy(state, q, epoch) -> int:
    action = randint(0, 1)
    return action


def follow_best_reward_strategy(state, q, epoch) -> int:
    if q[state][0] > q[state][1]: return 0
    else: return 1


def mixed_strategy(state, q, epoch) -> int:
    if epoch < 100000:
        return random_strategy(state, q, epoch)
    elif randint(0, 10) < 7:
        return follow_best_reward_strategy(state, q, epoch)
    return random_strategy(state, q, epoch)


def explore(start_position: int, horizon_steps: int, initialQ: list[list], strategy, epoch):
    # Q[s][0] -- spodziewana nagroda po action=0 w stanie "s"
    # Q[s][1] -- spodziewana nagroda po action=1 w stanie "s"
    q = initialQ.copy()
    state = start_position
    # print('----------------')
    path = []
    for _ in range(horizon_steps):
        # strategia kompletnie losowa
        # action = randint(0, 1)
        action = strategy(state, q, epoch)
        new_state = simulate(state, action)
        reward_new_state = reward(new_state)

        expected_next_reward = max(q[new_state][0], q[new_state][1])
        delta = LR * (reward_new_state + DI * (expected_next_reward - q[state][action]))
        q[state][action] += delta
        # print(f'new state:{new_state}, exp_n_rew:{expected_next_reward}, delta={delta}')
        path.append(new_state)
        state = new_state
    return q


def print_Q(Q):
    print('---\nQ values:')
    rows = []
    for a in range(2):
        rows.append([Q[i][a] for i in range(len(Q))])
    signs = ['←', '→']
    for (sign, row) in zip(signs, rows):
        print(f'{sign}[' + ','.join([f'{x:7.3f}' for x in row]) + ']')
    print('---')


def get_normalized_q(q: list[list]):
    mx_reward = 0.001
    for s in range(N):
        for a in range(2):
            mx_reward = max(mx_reward, q[s][a])
    norm_Q = q.copy()
    for s in range(N):
        for a in range(2):
            norm_Q[s][a] /= mx_reward
    return norm_Q


if __name__ == '__main__':
    total_rewards = 0
    EXPLORATIONS = 100001
    Q = [[0] * 2 for _ in range(N)]  # global state,action → reward information --- to be _learned_
    for epoch in range(EXPLORATIONS):
        start = randint(0, N - 1)
        Q = explore(start_position=start, horizon_steps=30, initialQ=Q,
                    strategy=mixed_strategy,
                    epoch=epoch)
        if epoch % 1000 == 0:
            print_Q(get_normalized_q(Q))
