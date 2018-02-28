import os

import numpy as np
import pandas as pd

MAX_ROW = 3  # x 的边界
MAX_COL = 3  # y 的边界
N_STATES = MAX_ROW * MAX_COL  # 2维迷宫
ACTIONS = ['left', 'right', 'up', 'down']  # 探索者的可用动作
EPSILON = 0.9  # 贪婪度 greedy
ALPHA = 0.1  # 学习率
GAMMA = 0.9  # 奖励递减值
MAX_EPISODES = 13  # 最大回合数
FRESH_TIME = 0.1  # 移动间隔时间


class Point:
    def __init__(self):
        self.__init__(0, 0)

    def __init__(self, row, col):
        self.row = row
        self.col = col


class CoordinateSystem:
    def __init__(self, max_row, max_col, worker, treasure, obstacles=None):
        self.max_row = max_row
        self.max_col = max_col
        self.worker = worker
        self.treasure = treasure

        if obstacles:
            self.obstacles = obstacles
        else:
            self.obstacles = []

    def move(self, point, action):
        print(action)

        if action == 'left' and point.col > 0:
            point.col -= 1
        elif action == 'right' and point.col < self.max_col - 2:
            point.col += 1
        elif action == 'up' and point.row > 0:
            point.row -= 1
        elif action == 'down' and point.row < self.max_row - 2:
            point.row += 1
        else:
            raise Exception('Not supported action.')

        return self.feedback(point)

    def feedback(self):
        state = str(self.worker.row) + ',' + str(self.worker.col)
        if self.worker.row == self.treasure.row and self.worker.col == self.treasure.col:
            reward = 1
        for obstacle in self.obstacles:
            if self.worker.row == obstacle.row and self.worker.col == obstacle.col:
                reward = -1
        return state, reward

    def display(self):
        os.system('clear')
        arr = np.zeros((self.max_row, self.max_col))
        arr[self.treasure.row][self.treasure.col] = 8
        arr[self.worker.row][self.worker.col] = 1
        print(arr)


def build_index():
    index = []
    for x in range(MAX_ROW):
        for y in range(MAX_COL):
            index.append(str(x) + ',' + str(y))
    return index


def build_q_table(n_states, actions):
    table = pd.DataFrame(
        data=np.zeros((n_states, len(actions))),
        index=build_index(),
        columns=actions,  # columns 对应的是行为名称,
    )
    return table


def choose_action(state, q_table):
    state_actions = q_table.iloc[state, :]  # 选出这个 state 的所有 action 值
    if (np.random.uniform() > EPSILON) or (state_actions.all() == 0):  # 非贪婪 or 或者这个 state 还没有探索过
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_actions.argmax()  # 贪婪模式
    return action_name


def rl():
    worker = Point()
    treasure = Point(1, 1)
    coordinateSystem = CoordinateSystem(MAX_ROW, MAX_COL, worker, treasure)

    q_table = build_q_table(N_STATES, ACTIONS)  # 初始 q table
    for episode in range(MAX_EPISODES):  # 回合
        step_counter = 0
        state = 0  # 回合初始位置
        is_terminated = False  # 是否回合结束
        coordinateSystem.display()
        while not is_terminated:
            action = choose_action(state, q_table)  # 选行为
            state_new, reward = coordinateSystem.feedback(worker)  # 实施行为并得到环境的反馈
            q_predict = q_table.loc[state, action]  # 估算的(状态-行为)值
            if state_new != 'terminal':
                q_target = reward + GAMMA * q_table.iloc[state_new, :].max()  # 实际的(状态-行为)值 (回合没结束)
            else:
                q_target = reward  # 实际的(状态-行为)值 (回合结束)
                is_terminated = True  # terminate this episode

            q_table.loc[state, action] += ALPHA * (q_target - q_predict)  # q_table 更新
            state = state_new  # 探索者移动到下一个 state

            coordinateSystem.display()

            step_counter += 1

    return q_table


if __name__ == "__main__":
    coordinate = Point(0, 0, 3, 3)
    coordinate.move('left')
