import gym
import numpy as np
import matplotlib.pyplot as plt
import time
from Policy_Iteration import abstractTrainer


class trainer(abstractTrainer):
    def __init__(self, gamma=1, eps=1e-10, evaluate_interval=100, maxInterval=4000):
        super(trainer, self).__init__()
        self.env = gym.make('FrozenLake8x8-v0')
        self.obs_dim = self.env.observation_space.n
        self.action_dim = self.env.action_space.n
        self.env.reset()  # 初始化环境
        self.env.seed(0)
        self.gamma = gamma
        self.eps = eps
        self.policy = np.random.randint(0, self.action_dim, self.obs_dim)  # 初始化一个随机策略
        self.state_value = np.zeros((self.obs_dim,))
        self.evaluate_interval = evaluate_interval
        self.maxInterval = 4000

    def valueFunctionUpdate(self):
        old_state_value = self.state_value.copy()
        for state in range(self.obs_dim):
            action = self.policy[state]
            value = 0
            for prob, next_state, reward, done in self.env.env.P[state][action]:
                value += prob * (reward + old_state_value[next_state] * self.gamma)
            self.state_value[state] = value
        return

    def policyUpdate(self):
        for state in range(self.obs_dim):
            state_action_value = [0] * self.action_dim
            for action in range(self.action_dim):
                value = 0
                for prob, next_state, reward, done in self.env.env.P[state][action]:
                    value += prob * (reward + self.state_value[next_state] * self.gamma)
                state_action_value[action] = value
            self.policy[state] = np.argmax(state_action_value)
        return

    def valueIteration(self):
        old_state_table = self.state_value.copy()
        for count in range(self.maxInterval):
            self.valueFunctionUpdate()
            if count % self.evaluate_interval == 0 and count:
                print("进行了第{}次迭代".format(count))
                self.policyUpdate()
                if np.sum(abs(old_state_table - self.state_value)) < self.eps:
                    print("在第{}次迭代，价值函数已经成功收敛".format(count))
                    break
                old_state_table = self.state_value.copy()


if __name__ == '__main__':
    trainer = trainer()
    trainer.valueIteration()
    trainer.printPolicy()
    trainer.printValue()
