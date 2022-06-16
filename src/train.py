import logging
import random
from functools import partial

from tensorboardX import SummaryWriter
from tqdm import tqdm

from agent.d3qn import D3QN
from config import DATA_CONFIG, TRAIN_CONFIG
from train_utils import ReplayBuffer, playGame
from utils import timeLog


class Trainer():
    def __init__(self, loc="cpu") -> None:
        self.agent = D3QN()
        self.agent.setDevice(loc)
        self.replay_buffer = ReplayBuffer()

        self.n_best = 0
        self.best_reward = 1e-5

        self.seed = partial(
            random.randint, a=0, b=20000905)
        self.writer = SummaryWriter(TRAIN_CONFIG.log_dir)

    @timeLog
    def __collectData(self, epoch) -> float:
        logging.info("collect data")

        mean_reward = 0
        n_game = TRAIN_CONFIG.n_game
        for _ in tqdm(range(n_game)):
            episode_data, reward = \
                playGame(self.agent, self.seed())
            self.replay_buffer.add(episode_data)
            mean_reward += reward
        mean_reward /= n_game
        self.writer.add_scalar("total_reward", mean_reward, epoch)

        self.agent.updateEpsilon()
        return mean_reward

    @timeLog
    def __train(self, epoch):
        logging.info("train model")

        mean_loss = 0
        train_epochs = min(
            TRAIN_CONFIG.train_epochs,
            self.replay_buffer.size() // DATA_CONFIG.batch_size)
        for _ in tqdm(range(train_epochs)):
            data_batch = self.replay_buffer.sample()
            loss = self.agent.trainStep(data_batch)
            mean_loss += loss
        mean_loss /= train_epochs
        self.writer.add_scalar("loss", mean_loss, epoch)

    def run(self):
        """[summary]
        NOTE: training pipeline contains:
             1. collect data    2. train model    3. update best model
        """

        for i in range(TRAIN_CONFIG.epochs):
            logging.info(f"\n=====Epoch {i}=====")

            # >>>>> collect data
            reward = self.__collectData(epoch=i)
            logging.info(f"buffer size {self.replay_buffer.size()}")
            # save data
            if (i + 1) % DATA_CONFIG.save_freq == 0:
                self.replay_buffer.save(version=f"epoch{i}")

            # >>>>> train model
            if self.replay_buffer.enough():
                self.__train(epoch=i)

            # >>>>> update model
            if reward / self.best_reward \
                    > TRAIN_CONFIG.update_thr:
                self.n_best += 1
                self.agent.save(self.n_best)
                self.best_reward = reward

            self.writer.flush()

        logging.info(f"best reward: {self.best_reward:<.2f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    trainer = Trainer(loc="cuda:0")
    trainer.run()
