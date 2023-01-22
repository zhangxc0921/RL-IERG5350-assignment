from SARSA import CliffWalking
import numpy as np


class Q_Learing(CliffWalking):
    def __init__(self, gamma=0.9, alpha=0.1, theta=0.03, maxIntervals=10000):
        super(Q_Learing, self).__init__()
        self.algorithm = "Q_learning"
        self.gamma = gamma
        self.alpha = alpha
        self.theta = theta
        self.maxintervals = maxIntervals
        self.env.reset()
        self.env.seed(0)

    def updateInterval(self):
        for i in range(self.maxintervals):
            # self.theta = self.update_theta(i) # 是否选择动态改变贪心因子
            cur_obs = self.env.reset()
            rewards = 0
            while True:
                action = self.policy(cur_obs)
                next_obs, reward, done, _ = self.env.step(action)  # next_obs对应于s'
                rewards += reward
                td_error = reward + self.gamma * np.max(self.Q_table[next_obs]) - self.Q_table[cur_obs][action]
                self.Q_table[cur_obs][action] += td_error * self.alpha
                cur_obs = next_obs
                if done or reward == -100:  # 到达终点 或者 掉入悬崖
                    break
            self.reward_list.append(rewards)


if __name__ == '__main__':
    qlearning = Q_Learing()
    qlearning.updateInterval()
    qlearning.print_plot()
