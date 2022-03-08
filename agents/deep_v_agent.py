from typing import Tuple, List
import numpy as np

import torch
from torch import nn
from agents.agent import Agent

from connect_four_env.connect_four_env import ConnectFourGymEnv


class ValueNetwork(nn.Module):
    def __init__(self, board_size: Tuple[int], n_channels: int, kernel_size: int = 4):
        super(ValueNetwork, self).__init__()
        conved_size = np.prod(board_size - (kernel_size - 1) * np.ones(2, np.int))
        self.layers = nn.Sequential(
            nn.Conv2d(1, n_channels, 4, dtype=torch.float64),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(conved_size * n_channels, 1, dtype=torch.float64),
        )

    def forward(self, boards: np.ndarray) -> float:
        if len(boards.shape) == 3:
            boards = torch.from_numpy(boards[:, np.newaxis, ...])
        elif len(boards.shape) == 2:
            boards = torch.from_numpy(boards[np.newaxis, np.newaxis, ...])

        value = self.layers(boards)
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
        self.optimizer = torch.optim.Adam(self.value_network.parameters())
        self.epsilon = epsilon

    def get_move(self, state: np.ndarray, explore: bool = True) -> int:
        self.value_network.eval()

        actions, next_states = ConnectFourGymEnv.get_next_actions_states(
            state, self.player_number
        )
        # This is necessary because the agent has to learn with him having one number
        # in order to be able to know which tokens are his and which are his opponent's
        next_states = self.player_number * np.array(next_states) 

        next_states_values = self.value_network(next_states)

        q_values = dict(zip(actions, next_states_values))

        if explore and (self.random.random() < self.epsilon):
            action = self.random.choice(list(q_values.keys()))
        else:
            # TODO maybe use argmax plutôt que de trier ?
            # TODO donner une action random si plusieurs sont à égalité
            sorted_q_values = sorted(
                q_values.keys(), key=lambda action: q_values[action], reverse=True
            )
            action = sorted_q_values[0]

        return action

    def learn_from_episode(self, states: List[np.ndarray], gains: np.ndarray):
        assert len(states) == len(gains), "not as many states as there are gains"

        # see comment in self.get_move()
        states = self.player_number * np.array(states) 

        self.value_network.train()
        self.optimizer.zero_grad()
        criterion = self.loss(self.value_network(states), torch.from_numpy(gains[:, np.newaxis]))
        criterion.backward()
        self.optimizer.step()

        return criterion.item()
