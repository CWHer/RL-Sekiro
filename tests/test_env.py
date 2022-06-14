import logging
import random

from icecream import ic

from env import SekiroEnv

if __name__ == "__main__":
    ic.disable()
    logging.basicConfig(level=logging.DEBUG)

    env = SekiroEnv()

    state = env.reset()
    for _ in range(20):
        action = random.choice(env.actionSpace())
        state, reward, done, _ = env.step(action)
