import gym
import numpy as np
import matplotlib.pyplot as plt
import time


class abstractTrainer:
    def __init__(self):
        pass

    def printPolicy(self):
        for i in range(int(self.obs_dim ** (1 / 2))):
            for j in range(int(self.obs_dim ** (1 / 2))):
                state = i * 8 + j
                action = self.policy[state]
                if action == 0:
                    print("←", end="\t")
                elif action == 1:
                    print("↓", end="\t")
                elif action == 2:
                    print("→", end="\t")
                else:
                    print("↑", end="\t")
            print()

    def printValue(self):
        for i in range(int(self.obs_dim ** (1 / 2))):
            for j in range(int(self.obs_dim ** (1 / 2))):
                state = i * 8 + j
                print("{:.3f}".format(self.state_value[state]), end="\t")
            print()

    def render(self):
        rewards = 0
        cur_obs = 0
        self.env.reset()
        while True:
            action = self.policy[cur_obs]
            cur_obs, reward, done, _ = self.env.step(action)
            rewards += reward
            # self.env.render()
            # time.sleep(0.5)
            if done:
                return rewards

    def evaluate(self, maxsteps):
        count = 0
        sum = 0
        while count < maxsteps:
            count += 1
            sum += self.render()
        print(sum / count)


class trainer(abstractTrainer):
    def __init__(self, gamma=1, eps=1e-10):
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

    def policyUpdate(self):
        while True:  # 更新价值函数,直至当前policy下收敛
            old_state_value = self.state_value.copy()
            for state in range(self.obs_dim):
                action = self.policy[state]
                value = 0
                for prob, next_state, reward, done in self.env.env.P[state][action]:
                    value += prob * (reward + self.gamma * old_state_value[next_state])
                self.state_value[state] = value
            if np.sum(abs(old_state_value - self.state_value)) < self.eps:
                break

        for state in range(self.obs_dim):  # 选择一个当前状态的最佳策略更新策略
            state_action_value = np.zeros((self.action_dim,))
            for action in range(self.action_dim):
                value = 0
                for prob, next_state, reward, done in self.env.env.P[state][action]:
                    value += prob * (reward + self.gamma * self.state_value[next_state])
                state_action_value[action] = value
            self.policy[state] = np.argmax(state_action_value)
        return

    def policyIteration(self):
        count = 0
        while True:
            count += 1
            old_policy = self.policy.copy()
            self.policyUpdate()
            print("当前第{}次迭代".format(count))
            if (old_policy == self.policy).all():
                print("在第{}次迭代时，已经找到一个最佳的策略，最佳策略如下所示".format(count))
                self.printPolicy()
                break


if __name__ == '__main__':
    trainer = trainer()
    trainer.policyIteration()
    trainer.printValue()
    trainer.evaluate(1000)
