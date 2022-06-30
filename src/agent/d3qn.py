import logging
import random
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from config import AGENT_CONFIG
from utils import timeLog

from .d3qn_utils import AuxiliaryNet, DuelNet, Encoder


class D3QN():
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.encoder = Encoder()

        self.q_net = DuelNet().to(self.device)
        # NOTE: target net has no gradients
        self.target_net = DuelNet().to(self.device)
        self.target_net.eval()

        self.auxiliary_net = AuxiliaryNet().to(self.device)
        self.auxiliary_net.train()

        self.optimizer = optim.Adam(
            [{"params": self.q_net.parameters(),
              "lr": AGENT_CONFIG.learning_rate},
             {"params": self.auxiliary_net.parameters(),
             "lr": AGENT_CONFIG.learning_rate * 10}],
            weight_decay=AGENT_CONFIG.l2_weight)

        self.epsilon = AGENT_CONFIG.init_epsilon

    def updateEpsilon(self):
        self.epsilon -= AGENT_CONFIG.delta_epsilon
        self.epsilon = max(self.epsilon, AGENT_CONFIG.min_epsilon)

    def setDevice(self, device: torch.device):
        self.device = device

        self.q_net.to(device)
        self.target_net.to(device)
        self.auxiliary_net.to(device)

    def save(self, version):
        checkpoint_dir = AGENT_CONFIG.checkpoint_dir

        import os
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)

        logging.info(f"save network & optimizer / version({version})")
        torch.save(
            {"q_net": self.q_net.state_dict(),
             "target_net": self.target_net.state_dict(),
             "auxiliary_net": self.auxiliary_net.state_dict(), },
            checkpoint_dir + f"/model_{version}")
        torch.save(
            self.optimizer.state_dict(),
            checkpoint_dir + f"/optimizer_{version}")

    def load(self, model_dir, optimizer_dir=None):
        logging.info(f"load network {model_dir}")

        checkpoints = torch.load(
            model_dir, map_location=self.device)
        self.q_net.load_state_dict(checkpoints["q_net"])
        self.target_net.load_state_dict(checkpoints["target_net"])
        self.auxiliary_net.load_state_dict(checkpoints["auxiliary_net"])

        if not optimizer_dir is None:
            logging.info(f"load optimizer {optimizer_dir}")
            self.optimizer.load_state_dict(torch.load(optimizer_dir))

    def resetState(self) -> None:
        self.encoder.reset()

    def encodeState(self, last_action, state):
        self.encoder.update(last_action, state)
        return self.encoder.state()

    def softUpdateTarget(self):
        for t_param, param in zip(
                self.target_net.parameters(),
                self.q_net.parameters()):
            t_param.data = (
                t_param.data * (1.0 - AGENT_CONFIG.tau) +
                param.data * AGENT_CONFIG.tau)

    def predict(self, frames, attributes):
        self.q_net.eval()

        frames = torch.as_tensor(
            np.expand_dims(frames, axis=0)).to(self.device)
        attributes = torch.as_tensor(
            np.expand_dims(attributes, axis=0)).to(self.device)
        with torch.no_grad():
            q_values, _ = self.q_net(frames, attributes)
        return q_values.detach().cpu().numpy()

    @timeLog
    def selectAction(self, state, actions):
        def epsilonGreedy(q_values, actions, epsilon):
            return random.choice(actions) \
                if np.random.rand() < epsilon \
                else actions[q_values.argmax(axis=1).item()]

        frames = state[0].astype(np.float32) / 255
        attributes = np.array(state[-4:], dtype=np.float32)
        q_value = self.predict(frames, attributes)
        action = epsilonGreedy(q_value, actions, self.epsilon)

        logging.info(q_value)
        return action

    def trainStep(self, data_batch: Tuple) -> Tuple[float, float]:
        """[summary]

        Returns:
            Tuple[float, float]: q_loss, auxiliary_loss
        """
        self.q_net.train()

        states, rewards, actions, next_states, dones = data_batch
        frames = torch.as_tensor(
            np.stack(states[..., 0]).astype(np.float32) / 255).to(self.device)
        attributes = torch.as_tensor(
            np.array(states[..., -4:], dtype=np.float32)).to(self.device)
        rewards = torch.as_tensor(rewards).float().view(-1, 1).to(self.device)
        actions = torch.as_tensor(actions).long().view(-1, 1).to(self.device)
        next_frames = torch.as_tensor(
            np.stack(next_states[..., 0]).astype(np.float32) / 255).to(self.device)
        next_attributes = torch.as_tensor(
            np.array(next_states[..., -4:], dtype=np.float32)).to(self.device)
        dones = torch.as_tensor(dones).float().view(-1, 1).to(self.device)

        q_values, current_latent = self.q_net(frames, attributes)
        q_values = q_values.gather(1, actions)

        with torch.no_grad():
            max_actions = self.q_net(
                next_frames, next_attributes)[0].argmax(dim=1)
            tq_values, next_latent = \
                self.target_net(next_frames, next_attributes)
            tq_values = tq_values.gather(1, max_actions.view(-1, 1))

        q_targets = rewards + \
            (1 - dones) * AGENT_CONFIG.gamma * tq_values
        q_loss = F.mse_loss(q_values, q_targets)

        # NOTE: auxiliary loss
        # fmt: off
        predicted_next_latent, predicted_actions, \
            predicted_frames = self.auxiliary_net(current_latent, next_latent, actions)
        auxiliary_loss = \
            F.mse_loss(predicted_next_latent, next_latent) + \
            F.cross_entropy(predicted_actions, actions.view(-1), ignore_index=-1) + \
            F.mse_loss(predicted_frames, torch.cat([frames, next_frames], dim=0))
        # fmt: on

        loss = q_loss + auxiliary_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.softUpdateTarget()

        return q_loss.item(), auxiliary_loss.item()
