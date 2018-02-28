import numpy as np
import pandas as pd


class QLearning:
    def __init__(self, epsilon, learning_rate, discount_factor, max_row, max_col, actions):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.actions = actions

        self.q_table = pd.DataFrame(
            data=np.zeros((max_row * max_col, len(actions))),
            index=self.build_index(max_row, max_col),
            columns=actions,  # columns 对应的是行为名称,
        )

    def choose_action(self, state):
        if np.random.uniform() < self.epsilon:
            action = self.q_table.loc[state, :]
            action = action.reindex(np.random.permutation(action.index))  # some actions have same value
            action = action.idxmax()
        else:
            action = np.random.choice(self.actions)
        return action

    def build_index(self, max_row, max_col):
        from Maze import Point
        index = []
        for row in range(max_row):
            for col in range(max_col):
                index.append(Point(row, col).toString())
        return index

    def learn(self, cur_state, action, reward, next_state):
        q_predict = self.q_table.loc[cur_state, action]
        if reward == 0:
            q_reality = reward + self.discount_factor * self.q_table.loc[next_state, :].max()
        else:  # 代表碰到 obstacle or tresure
            q_reality = reward
        self.q_table.loc[cur_state, action] += self.learning_rate * (q_reality - q_predict)
