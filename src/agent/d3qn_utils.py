from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt
import torch.nn as nn
import torch.nn.functional as F
from config import AGENT_CONFIG


class Encoder():
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.last_focus: Optional[npt.NDArray[np.uint8]] = None
        self.last_action: Optional[int] = None
        self.current_state = None

    def update(self, last_action: int,
               state: Tuple[npt.NDArray[np.uint8],
                            float, float, float, float]) -> None:
        self.last_focus = state[0] \
            if self.current_state is None \
            else self.current_state[0]
        self.last_action = last_action
        self.current_state = state

    def state(self) -> Tuple[npt.NDArray[np.uint8],
                             npt.NDArray[np.uint8],
                             float, float, float, float, int]:
        """[summary]

        State:
            last_focus      npt.NDArray[np.uint8]
            focus_area      npt.NDArray[np.uint8]
            agent_hp        float, [0, 1]
            agent_ep        float, [0, 1]
            boss_hp         float, [0, 1]
            last_action     int, [-1, 4)
        """
        return (self.last_focus, *self.current_state, self.last_action)

    def encode(self, state: Tuple[npt.NDArray[np.uint8],
                                  npt.NDArray[np.uint8],
                                  float, float, float, float, int]) \
            -> npt.NDArray[np.float32]:
        focus_areas = map(
            lambda x: x.astype(np.float32) / 255, state[:2])
        attributes = map(lambda x: np.full(
            state[0].shape, x, dtype=np.float32), state[2:])
        features = np.stack(list(focus_areas) + list(attributes))
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
