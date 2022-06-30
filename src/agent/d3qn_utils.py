from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import AGENT_CONFIG


class Encoder():
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.last_action: Optional[int] = None
        self.current_state = None

    def update(self, last_action: int,
               state: Tuple[npt.NDArray[np.uint8],
                            float, float, float]) -> None:
        self.last_action = last_action
        self.current_state = state

    def state(self) -> Tuple[npt.NDArray[np.uint8],
                             float, float, float, float]:
        """[summary]

        State:
            focus_area      npt.NDArray[np.uint8], "RGB", C x H x W
            agent_hp        float, [0, 1]
            agent_ep        float, [0, 1]
            boss_hp         float, [0, 1]
            last_action     float, {-1, 0, ..., 6}
        """
        return (*self.current_state, float(self.last_action))


class ResBlock(nn.Module):
    def __init__(self, n_channels):
        super().__init__()

        self.conv_net = nn.Sequential(
            nn.Conv2d(n_channels, n_channels,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_channels, n_channels,
                      kernel_size=3, padding=1),
        )

    def forward(self, x):
        y = self.conv_net(x)
        return F.relu(x + y)


class AuxiliaryNet(nn.Module):
    def __init__(self):
        super().__init__()

        # NOTE: auxiliary tasks
        # 1. Forward Dynamics
        #   [current_latent, action] -> [next_latent]
        self.forward_dynamics_net = nn.Sequential(
            nn.LazyLinear(1024), nn.ReLU(),
            nn.LazyLinear(512), nn.ReLU(),
            nn.LazyLinear(AGENT_CONFIG.hidden_size),
        )

        # 2. Inverse Dynamics
        #   [current_latent, next_latent] -> [action]
        self.inverse_dynamics_net = nn.Sequential(
            nn.LazyLinear(1024), nn.ReLU(),
            nn.LazyLinear(512), nn.ReLU(),
            nn.LazyLinear(AGENT_CONFIG.n_action),
        )

        # 3. AutoEncoder
        n_channels = AGENT_CONFIG.n_channels // 8
        size = int((AGENT_CONFIG.hidden_size / n_channels) ** 0.5)
        self.decoder = nn.Sequential(
            nn.Unflatten(dim=-1, unflattened_size=(n_channels, size, size)),
            nn.LazyConvTranspose2d(n_channels,
                                   kernel_size=2, stride=2),
            *(ResBlock(n_channels)
              for _ in range(AGENT_CONFIG.n_res)),
            nn.LazyConvTranspose2d(n_channels * 2,
                                   kernel_size=2, stride=2),
            *(ResBlock(n_channels * 2)
              for _ in range(AGENT_CONFIG.n_res)),
            nn.LazyConvTranspose2d(n_channels * 4,
                                   kernel_size=2, stride=2),
            *(ResBlock(n_channels * 4)
              for _ in range(AGENT_CONFIG.n_res)),
            nn.LazyConvTranspose2d(n_channels * 8,
                                   kernel_size=2, stride=2),
            *(ResBlock(n_channels * 8)
              for _ in range(AGENT_CONFIG.n_res)),
            nn.LazyConv2d(out_channels=3, kernel_size=3, padding=1),
        )

    def forward(self,
                current_latent: torch.Tensor,
                next_latent: torch.Tensor,
                actions: torch.Tensor):
        """[summary]

        Args:
            current_latent (torch.Tensor): B x K
            next_latent (torch.Tensor): B x K
            actions (torch.Tensor): B x 1
        """
        x = torch.hstack([current_latent, actions])
        predicted_next_latent = self.forward_dynamics_net(x)

        x = torch.hstack([current_latent, next_latent])
        predicted_actions = self.inverse_dynamics_net(x)

        x = torch.cat([current_latent, next_latent], dim=0)
        predicted_frames = self.decoder(x)

        return predicted_next_latent, \
            predicted_actions, predicted_frames


class DuelNet(nn.Module):
    def __init__(self):
        super().__init__()
        n_channels = AGENT_CONFIG.n_channels

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=n_channels,
                      kernel_size=3, padding=1),
            *(ResBlock(n_channels)
              for _ in range(AGENT_CONFIG.n_res)),
            nn.MaxPool2d(kernel_size=2),  # 64 x 64
            nn.LazyConv2d(out_channels=n_channels // 2,
                          kernel_size=3, padding=1),
            *(ResBlock(n_channels // 2)
              for _ in range(AGENT_CONFIG.n_res)),
            nn.MaxPool2d(kernel_size=2),  # 32 x 32
            nn.LazyConv2d(out_channels=n_channels // 4,
                          kernel_size=3, padding=1),
            *(ResBlock(n_channels // 4)
              for _ in range(AGENT_CONFIG.n_res)),
            nn.MaxPool2d(kernel_size=2),  # 16 x 16
            nn.LazyConv2d(out_channels=n_channels // 8,
                          kernel_size=3, padding=1),
            *(ResBlock(n_channels // 8)
              for _ in range(AGENT_CONFIG.n_res)),
            nn.MaxPool2d(kernel_size=2),  # 8 x 8
            nn.Flatten()
        )

        # A head
        self.a_output = nn.Sequential(
            nn.LazyLinear(512), nn.ReLU(),
            nn.LazyLinear(256), nn.ReLU(),
            nn.LazyLinear(AGENT_CONFIG.n_action),
        )

        # V head
        self.v_output = nn.Sequential(
            nn.LazyLinear(512), nn.ReLU(),
            nn.LazyLinear(256), nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self,
                frames: torch.Tensor,
                attributes: torch.Tensor):
        """[summary]

        Args:
            frames (torch.Tensor): B x C X H X W
            attributes (torch.Tensor): B x N
        """
        latent_features = self.encoder(frames)
        features = torch.cat(
            [latent_features, attributes], dim=1)
        a_values = self.a_output(features)
        v_values = self.v_output(features)
        q_values = v_values + a_values - \
            a_values.mean(dim=1).view(-1, 1)
        return q_values, latent_features
