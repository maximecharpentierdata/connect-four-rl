import sys
from typing import Tuple, List
import numpy as np

import torch
from torch import nn
from agents.agent import Agent

from connect_four_env.connect_four_env import ConnectFourGymEnv


class ValueNetwork(nn.Module):
    def __init__(self, grid_size: Tuple[int], n_channels: int):
        super(ValueNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, n_channels, 4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(np.prod(grid_size), 1),
        )

    def forward(self, grid):
        value = self.layers(grid)
        return value


class DeepVAgent(Agent):
    def __init__(
        self,
        n_channels: int,
        player_number: int,
        epsilon: float = 0,
        board_shape: Tuple[int, int] = (6, 7),
        seed: int = 42,
    ):
        self.value_network = ValueNetwork(board_shape, n_channels)
        self.player_number = player_number
        self.random = np.random.default_rng(seed)
        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.value_network.parameters)
        self.epsilon = epsilon

    def get_move(self, state: np.ndarray) -> int:
        self.value_network.eval()

        actions_states = ConnectFourGymEnv.get_next_actions_states(
            state, self.player_number
        )
        q_values = dict()

        for action, state in actions_states:
            q_values[action] = self.value_network.predict(state)

        if self.random.random() < self.epsilon:
            action = self.random.choice(q_values.values(), 1)[0]
        else:
            sorted_q_values = sorted(
                q_values.keys(), key=lambda action: q_values[action], reverse=True
            )
            action = sorted_q_values[0]

        return action

    def learn_from_episode(self, states: List[np.ndarray], gains: List[float]):
        assert len(states) == len(gains), "not as many states as there are gains"
        self.value_network.train()
        self.optimizer.zero_grad()
        criterion = self.loss(self.value_network(states), gains)
        criterion.backward()
        self.optimizer.step()

        return criterion.item
