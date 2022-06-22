import logging

import torch
from icecream import ic

from agent.d3qn import D3QN
from env import SekiroEnv
from train_utils import ReplayBuffer, playGame


if __name__ == "__main__":
    ic.disable()
    logging.basicConfig(level=logging.DEBUG)

    agent = D3QN()
    agent.setDevice(torch.device("cuda:0"))
    agent.save(version="test")
    replay_buffer = ReplayBuffer()

    # stage 1
    env = SekiroEnv()
    episode_data, reward = playGame(env, agent, 19937)
    replay_buffer.add(episode_data)
    replay_buffer.save("test")

    # stage 2
    # replay_buffer.load("./dataset/data_test.pkl")
    # data_batch = replay_buffer.sample()
    # loss = agent.trainStep(data_batch)
