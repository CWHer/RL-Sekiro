import random
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from config import AGENT_CONFIG
from utils import timeLog

from .d3qn_utils import Encoder, VANet


class D3QN():
    def __init__(self) -> None:
        self.device = torch.device("cpu")

        self.encoder = Encoder()
        self.q_net = VANet().to(self.device)
        # NOTE: target net is used for training
        self.target_net = VANet().to(self.device)

        self.optimizer = optim.Adam(
            self.q_net.parameters(),
            lr=AGENT_CONFIG.learning_rate,
            weight_decay=AGENT_CONFIG.l2_weight)

        self.epsilon = AGENT_CONFIG.init_epsilon

    def updateEpsilon(self):
        self.epsilon -= AGENT_CONFIG.delta_epsilon
        self.epsilon = max(self.epsilon, AGENT_CONFIG.min_epsilon)

    def setDevice(self, device: torch.device):
        self.device = device
        self.q_net.to(device)
        self.target_net.to(device)

    def save(self, version):
        checkpoint_dir = AGENT_CONFIG.checkpoint_dir

        import os
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)

        print(f"save network & optimizer / version({version})")
        torch.save(
            self.target_net.state_dict(),
            checkpoint_dir + f"/model_{version}")
        torch.save(
            self.optimizer.state_dict(),
            checkpoint_dir + f"/optimizer_{version}")

    def load(self, model_dir, optimizer_dir=None):
        print("load network {}".format(model_dir))

        self.q_net.load_state_dict(torch.load(
            model_dir, map_location=self.device))
        self.target_net.load_state_dict(self.q_net.state_dict())

        if not optimizer_dir is None:
            print("load optimizer {}".format(optimizer_dir))
            self.optimizer.load_state_dict(torch.load(optimizer_dir))

    def softUpdateTarget(self):
        for t_param, param in zip(
                self.target_net.parameters(),
                self.q_net.parameters()):
            t_param.data = (
                t_param.data * (1.0 - AGENT_CONFIG.tau) +
                param.data * AGENT_CONFIG.tau)

    def predict(self, features):
        if features.ndim < 4:
            features = np.expand_dims(features, 0)

        features = torch.as_tensor(
            features).float().to(self.device)
        with torch.no_grad():
            q_values = self.q_net(features)
        return q_values.detach().cpu().numpy()

    @timeLog
    def selectAction(self, states, actions):
        def epsilonGreedy(q_values, actions, epsilon):
            return random.choice(actions) \
                if np.random.rand() < epsilon \
                else actions[q_values.argmax(axis=1).item()]
        features = self.encoder.encode(states)
        q_values = self.predict(features)
        action = epsilonGreedy(q_values, actions, self.epsilon)
        return action

    def trainStep(self, data_batch: Tuple) -> float:
        """[summary]

        Returns:
            loss
        """
        states, rewards, actions, next_states, dones = data_batch
        states = torch.as_tensor(np.stack(
            [self.encoder.encode(state) for state in states])).to(self.device)
        rewards = torch.as_tensor(
            np.stack(rewards)).float().view(-1, 1).to(self.device)
        actions = torch.as_tensor(
            np.stack(actions)).long().view(-1, 1).to(self.device)
        next_states = torch.as_tensor(np.stack(
            [self.encoder.encode(state) for state in next_states])).to(self.device)
        dones = torch.as_tensor(
            np.stack(dones)).float().view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)

        with torch.no_grad():
            max_actions = self.q_net(next_states).argmax(dim=1)
            tq_values = self.target_net(
                next_states).gather(1, max_actions.view(-1, 1))

        q_targets = rewards + \
            (1 - dones) * AGENT_CONFIG.gamma * tq_values
        loss = F.mse_loss(q_values, q_targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.softUpdateTarget()

        return loss.item()
