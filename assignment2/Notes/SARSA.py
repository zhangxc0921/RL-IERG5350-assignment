import time
import gym
import numpy as np
import matplotlib.pyplot as plt


class CliffWalking:
    def __init__(self, gamma=0.9, alpha=0.1, theta=0.03, maxIntervals=500):  #
        self.env = gym.make('CliffWalking-v0')
        self.obs_dim = self.env.observation_space.n
        self.act_dim = self.env.action_space.n
        self.Q_table = np.zeros((self.obs_dim, self.act_dim))
        self.env.reset()
        self.env.seed(0)
        self.reward_list = []

    def update_theta(self, episode_nums):  # 随episode的增加动态改变theta
        return 1 / (episode_nums + 1)

    def print_plot(self):
        episodes_list = list(range(len(self.reward_list)))
        plt.plot(episodes_list, self.reward_list)
        plt.xlabel("Episodes' number")
        plt.ylabel("Total rewards")
        plt.title('{} on {}'.format(self.algorithm, 'Cliff Walking'))
        plt.show()

    def policy(self, state):
        if np.random.random() < self.theta:  # 此时选择一个随机策略
            action = np.random.randint(self.act_dim)
        else:  # 此时选择贪心策略
            action = np.argmax(self.Q_table[state])
        return action


class Sarsa(CliffWalking):
    def __init__(self, gamma=0.9, alpha=0.1, theta=0.03, maxIntervals=1000):
        super(Sarsa, self).__init__()
        self.algorithm = "SARSA"
        self.gamma = gamma  # 折扣因子
        self.alpha = alpha  # 学习因子
        self.theta = theta  # 贪心因子
        self.maxIntervals = maxIntervals
        self.env.reset()
        self.env.seed(0)

    def updateInterval(self):
        for i in range(self.maxIntervals):
            cur_obs = self.env.reset()
            # self.theta = self.update_theta(i) # 是否选择动态改变贪心因子
            action = self.policy(cur_obs)
            rewards = 0
            while True:
                next_obs, reward, done, _ = self.env.step(action)  # next_obs对应于s'
                next_action = self.policy(next_obs)  # next_action对应于a'
                rewards += reward
                td_error = reward + self.gamma * self.Q_table[next_obs][next_action] - self.Q_table[cur_obs][action]
                self.Q_table[cur_obs][action] += self.alpha * td_error
                action = next_action
                cur_obs = next_obs
                if done or reward == -100:  # 这个环境有点问题，当掉入悬崖时，不会显示done=True
                    break
            self.reward_list.append(rewards)


if __name__ == '__main__':
    Sarsa_agent = Sarsa()
    Sarsa_agent.updateInterval()
    Sarsa_agent.print_plot()
