import numpy as np
import pandas as pd


class QLearningAgent:
    def __init__(self, epsilon, learning_rate, discount_factor, actions):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.actions = actions
        self.q_table = pd.DataFrame(columns=actions)

    def choose_action(self, state):
        self.create_state_if_not_exists(state)

        if np.random.uniform() < self.epsilon:
            action = self.q_table.loc[state, :]
            action = action.reindex(np.random.permutation(action.index))
            action = action.idxmax()
        else:
            action = np.random.choice(self.actions)

        return action

    def build_index(self, max_row, max_col):
        from MazeEnv import Point
        index = []
        for row in range(max_row):
            for col in range(max_col):
                index.append(Point(row, col).toString())
        return index

    def learn(self, cur_state, action, reward, next_state):
        self.create_state_if_not_exists(next_state)
        q_predict = self.q_table.loc[cur_state, action]
        if self.not_finished(reward):
            q_reality = reward + self.discount_factor * self.q_table.loc[next_state, :].max()
        else:  # 代表碰到 obstacle or tresure
            q_reality = reward
        self.q_table.loc[cur_state, action] += self.learning_rate * (q_reality - q_predict)

    def create_state_if_not_exists(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series(
                    data=[0] * len(self.actions),
                    index=self.actions,
                    name=state
                )
            )

    def not_finished(self, reward):
        return reward == 0
