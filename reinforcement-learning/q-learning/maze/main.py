from Maze import Maze, Point, Actions
from QLearning import QLearning

"""
使用强化学习走迷宫,由于刚开始的状态是随机的,所以当迷宫面积较大的时候,可能会导致训练时间太长,可以尝试利用 A* 算法原理,
帮助更快找到treasure
"""

if __name__ == '__main__':
    epsilon = 0.9
    alpha = 0.1
    gamma = 0.9

    max_row = 5
    max_col = 5
    actions = [Actions.LEFT, Actions.RIGHT, Actions.UP, Actions.DOWN]
    worker = Point(0, 0)
    treasure = Point(3, 4)

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
