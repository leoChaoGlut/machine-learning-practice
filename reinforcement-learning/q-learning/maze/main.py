from Maze import Maze, Point, Actions
from QLearning import QLearning

"""
使用强化学习走迷宫,由于刚开始的状态是随机的,所以当迷宫面积较大的时候,可能会导致训练时间太长,可以尝试利用 A* 算法原理,
帮助更快找到treasure

1: worker
4: obstacle
8: treasure

"""

if __name__ == '__main__':
    refresh_interval = 0.1  # 该参数用于定时显示迷宫情况

    episode = 50  # 训练多少回合

    epsilon = 0.9  # 使用历史经验的概率, 若值为0.9,则有 90% 的情况下,会根据历史经验选择 action, 10% 的情况下,随机选择 action
    learning_rate = 0.01  # 根据公式可知,该值越大,则旧训练数据被保留的就越少
    discount_factor = 0.9  #

    max_row = 4
    max_col = 4
    actions = [Actions.LEFT, Actions.RIGHT, Actions.UP, Actions.DOWN]
    worker = Point(0, 0)
    treasure = Point(2, 2)
    obstacles = [Point(1, 2), Point(2, 1)]

    env = Maze(
        max_row=max_row,
        max_col=max_col,
        worker=worker,
        treasure=treasure,
        obstacles=obstacles,
        refresh_interval=refresh_interval
    )
    rl = QLearning(
        epsilon=epsilon,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        max_row=max_row,
        max_col=max_col,
        actions=actions
    )
    step_counter_arr = []

    env.display()

    for eps in range(episode):

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
                break

        if reward > 0:
            step_counter_arr.append(step_counter)

        print('round:{}, reward:{}, steps:{}'.format(eps, reward, step_counter))
        # time.sleep(1) # 如果想看模型进步历程,可以将这一行的注释去掉

    print('steps record:{}'.format(step_counter_arr))
    print(rl.q_table)
