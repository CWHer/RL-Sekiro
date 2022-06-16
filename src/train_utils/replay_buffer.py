import logging
import pickle
from collections import deque, namedtuple

import numpy as np
import torch
from config import TRAIN_CONFIG


class ReplayBuffer():
    Data = namedtuple("data", "state reward action next_state done")

    def __init__(self) -> None:
        self.buffer = deque(maxlen=TRAIN_CONFIG.replay_size)

    def size(self) -> int:
        return len(self.buffer)

    def save(self, version):
        dataset_dir = TRAIN_CONFIG.dataset_dir

        import os
        if not os.path.exists(dataset_dir):
            os.mkdir(dataset_dir)

        logging.info(f"save replay buffer version({version})")
        with open(dataset_dir +
                  f"/data_{version}.pkl", "wb") as f:
            pickle.dump(self.buffer, f)

    def load(self, data_dir):
        logging.info(f"load replay buffer {data_dir}")
        with open(data_dir, "rb") as f:
            self.buffer = pickle.load(f)

    def enough(self):
        """[summary]
        whether data is enough to start training
        """
        return self.size() > TRAIN_CONFIG.thr

    def __enhanceData(self,
                      states, rewards, actions,
                      next_states, dones):
        # TODO:
        enhanced_data = (states, rewards, actions, next_states, dones)
        return enhanced_data

    def add(self, episode_data):
        enhanced_data = self.__enhanceData(*episode_data)
        self.buffer.extend(enhanced_data)

    def sample(self):
        indices = np.random.choice(
            len(self.buffer), TRAIN_CONFIG.batch_size)
        data_batch = map(
            lambda x: torch.as_tensor(np.stack(x)),
            zip(*[self.buffer[i] for i in indices]))
        return list(data_batch)
