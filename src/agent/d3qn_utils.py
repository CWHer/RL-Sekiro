from typing import Tuple

import numpy as np
import numpy.typing as npt
import torch.nn as nn
import torch.nn.functional as F
from config import AGENT_CONFIG


class Encoder():
    def encode(self, state: Tuple[npt.NDArray[np.uint8],
                                  float, float, float, float]) \
            -> npt.NDArray[np.float32]:
        """[summary]

        State:
            focus_area      npt.NDArray[np.uint8]
            agent_hp        float, [0, 1]
            agent_ep        float, [0, 1]
            boss_hp         float, [0, 1]
        """
        focus_area = state[0].astype(np.float32) / 255
        attributes = map(lambda x: np.full(
            focus_area.shape, x, dtype=np.float32) / 100, state[1:])
        features = np.stack(
            [focus_area] + list(attributes))
        return features


def conv3x3(in_channels, out_channels):
    return nn.Conv2d(
        in_channels, out_channels,
        kernel_size=3, padding=1, bias=False)


class ResBlock(nn.Module):
    def __init__(self, n_channels):
        super().__init__()

        self.net = nn.Sequential(
            conv3x3(n_channels, n_channels),
            # nn.BatchNorm2d(num_channels),
            nn.ReLU(),
            conv3x3(n_channels, n_channels),
            # nn.BatchNorm2d(num_channels)
        )

    def forward(self, x):
        y = self.net(x)
        return F.relu(x + y)


class VANet(nn.Module):
    def __init__(self):
        super().__init__()

        in_channels = AGENT_CONFIG.in_channels
        hidden_channels = AGENT_CONFIG.n_channels

        resnets = [
            ResBlock(hidden_channels)
            for _ in range(AGENT_CONFIG.n_res)]
        self.common_layers = nn.Sequential(
            conv3x3(in_channels, hidden_channels),
            nn.ReLU(), *resnets)

        # A head
        self.A_output = nn.Sequential(
            nn.Conv2d(hidden_channels, 4, kernel_size=1),
            nn.ReLU(), nn.Flatten(),
            nn.LazyLinear(AGENT_CONFIG.action_size),
        )

        # V head
        self.V_output = nn.Sequential(
            nn.Conv2d(hidden_channels, 2, kernel_size=1),
            nn.ReLU(), nn.Flatten(),
            nn.LazyLinear(256),
            nn.ReLU(), nn.Linear(256, 1),
        )

    def forward(self, x):
        x = self.common_layers(x)
        A = self.A_output(x)
        V = self.V_output(x)
        Q = V + A - A.mean(dim=1).view(-1, 1)
        return Q
