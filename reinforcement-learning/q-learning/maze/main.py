from Maze import Maze, Point, Actions
from QLearning import QLearning

if __name__ == '__main__':
    epsilon = 0.9
    alpha = 0.1
    gamma = 0.9

    max_row = 4
    max_col = 4
    actions = [Actions.LEFT, Actions.RIGHT, Actions.UP, Actions.DOWN]
    worker = Point(0, 0)
    treasure = Point(2, 2)

    env = Maze(
        max_row=max_row,
        max_col=max_col,
        worker=worker,
        treasure=treasure,
        refresh_interval=0.3
    )
    rl = QLearning(
        epsilon=epsilon,
        alpha=alpha,
        gamma=gamma,
        max_row=max_row,
        max_col=max_col,
        actions=actions
    )

    for episode in range(10):

        cur_state = env.reset()
        step_counter = 0

        while True:
            step_counter += 1

            env.display()

            action = rl.choose_action(cur_state)

            next_state, reward = env.move(action)

            rl.learn(
                cur_state=cur_state,
                action=action,
                reward=reward,
                next_state=next_state
            )

            cur_state = next_state

            if reward != 0:
                print('steps:{}'.format(step_counter))
                print(rl.q_table)
                input()
                break
