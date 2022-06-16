import random
from typing import List, Tuple

import gym
import numpy as np
import torch


def playGame(agent, seed: int) -> Tuple[List, float]:
    """[summary]

    Returns:
        Tuple[List, float]: episode_data & total_reward
    """
    # initialize seeds (torch, np, random)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    data_buffer = []
    env = gym.make("CartPole-v1")
    state = env.reset()
    total_reward, done = 0, False

    while not done:
        action = agent.selectAction(
            state, tuple(range(env.action_space.n)))
        next_state, reward, done, _ = env.step(action)
        total_reward += reward

        # collect data
        data_buffer.append(
            (state, reward, action, next_state, done))

        state = next_state
        # debug
        # env.render()

    env.close()
    return data_buffer, total_reward
