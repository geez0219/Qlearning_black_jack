"""
This part of code is the Q learning brain, which is a brain of the agent.
All decisions are made in here.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

import numpy as np
import pandas as pd

class QLearningTable:
    def __init__(self, actions, learning_rate=0.001, reward_decay=1, e_greedy=0.1):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = {}

    def choose_action(self, observation):
        self.check_state_exist(observation)
        # action selection
        state_action = self.q_table[str(observation)]

        if np.random.uniform() > self.epsilon:
            # choose best action
            action = self.actions[np.random.choice(np.where(state_action == np.max(state_action))[0])]
            # some actions may have the same value, randomly choose on in these actions
        else:
            # choose random action
            action = self.actions[np.random.choice(len(state_action))]
        return action

    def learn(self, s, a, r, s_, done):
        q_predict = self.q_table[str(s)][a]
        if done:
            q_target = r
        else:
            self.check_state_exist(s_)
            q_target = r + self.gamma * self.q_table[str(s_)].max()  # next state is not terminal

        self.q_table[str(s)][a] += self.lr * (q_target - q_predict)  # update

    def check_state_exist(self, state):
        stateName = str(state)
        canDouble, canSplit = state[2][1], state[2][2]

        if stateName not in self.q_table:
            if canSplit == 1:
                self.q_table[stateName] = np.zeros(4, dtype=np.float32)
            elif canDouble == 1:
                self.q_table[stateName] = np.zeros(3, dtype=np.float32)
            else:
                self.q_table[stateName] = np.zeros(2, dtype=np.float32)
