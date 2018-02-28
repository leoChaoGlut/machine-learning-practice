episode = 10
alpha = 0.1
gamma = 0.9


def init_q_table():
    pass


def init_state():
    pass


def choose_action():
    pass


def take_action():
    pass


if __name__ == '__main__':
    q_table = init_q_table()
    for eps in range(episode):
        state = init_state()
        while True:
            action = choose_action()
            next_state, reward, finished = take_action()

            # if next_state

            reality = reward + gamma * q_table.loc[next_state, :].max()
            predict = q_table.loc[state, action]
            q_table.loc[state, action] += alpha * (reality - predict)

            state = next_state
