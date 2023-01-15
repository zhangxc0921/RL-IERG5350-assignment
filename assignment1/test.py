# Run this cell without modification

# Import some packages that we need to use
from utils import *
import gym
import numpy as np
from collections import deque

# Solve the TODOs and remove `pass`

# [TODO] Just a reminder. Do you add your name and student
# ID in the table at top of the notebook?

# Create the environment
env = gym.make('FrozenLake8x8-v1')

# You need to reset the environment immediately after instantiating env.
env.reset()  # [TODO] uncomment this line
# Seed the environment
env.seed(0)  # [TODO] uncomment this line
env.reset()


def evaluate(policy, num_episodes, seed=0, env_name='FrozenLake8x8-v1', render=False):
    """[TODO] You need to implement this function by yourself. It
    evaluate the given policy and return the mean episode reward.
    We use `seed` argument for testing purpose.
    You should pass the tests in the next cell.

    :param policy: a function whose input is an interger (observation)
    :param num_episodes: number of episodes you wish to run
    :param seed: an interger, used for testing.
    :param env_name: the name of the environment
    :param render: a boolean flag. If true, please call _render_helper
    function.
    :return: the averaged episode reward of the given policy.
    """

    # Create environment (according to env_name, we will use env other than 'FrozenLake8x8-v0')
    env = gym.make(env_name)
    env.seed(seed)
    rewards = []
    for i in range(num_episodes):
        obs = env.reset()
        act = policy(obs)
        ep_reward = 0
        while True:
            obs, reward, done, info = env.step(act)
            act = policy(obs)
            ep_reward += reward
            if done:
                break

        rewards.append(ep_reward)

    return np.mean(rewards)


# Run this cell without modification

# Run this cell to test the correctness of your implementation of `evaluate`.
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3


def expert(obs):  # obs 是位置，当前地图是8*8的地图，obs按行列递增一维排列
    """Go down if agent at the right edge, otherwise go right."""
    return DOWN if (obs + 1) % 8 == 0 else RIGHT


def assert_equal(seed, value, env_name):
    ret = evaluate(expert, 1000, seed, env_name=env_name)
    assert ret == value, \
        "When evaluate on seed {}, 1000 episodes, in {} environment, the " \
        "averaged reward should be {}. But you get {}." \
        "".format(seed, env_name, value, ret)


assert_equal(0, 0.065, 'FrozenLake8x8-v1')
assert_equal(1, 0.059, 'FrozenLake8x8-v1')
assert_equal(2, 0.055, 'FrozenLake8x8-v1')

assert_equal(0, 0.026, 'FrozenLake-v1')
assert_equal(1, 0.034, 'FrozenLake-v1')
assert_equal(2, 0.028, 'FrozenLake-v1')

print("Test Passed!")
print("\nAs a baseline, the mean episode reward of a hand-craft "
      "agent is: ", evaluate(expert, 1000))