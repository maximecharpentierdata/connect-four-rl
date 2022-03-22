import os
from typing import List, Tuple, Union

import numpy as np
import torch
from torch import nn

from agents.agent import Agent
from connect_four_env.utils import get_next_actions_states
import constants


class ValueNetwork(nn.Module):
    def __init__(self, board_size: Tuple[int], n_channels: int, kernel_size: int = 4):
        super(ValueNetwork, self).__init__()
        conved_size = np.prod(board_size - (kernel_size - 1) * np.ones(2, int))
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=n_channels,
                kernel_size=4,
                dtype=torch.float64,
            ),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(conved_size * n_channels, 64, dtype=torch.float64),
            nn.LeakyReLU(),
            nn.Linear(64, 1, dtype=torch.float64),
        )

    def forward(self, boards: Union[np.ndarray, List[np.ndarray]]) -> float:
        boards = np.array(boards)
        if len(boards.shape) == 3:
            boards = torch.from_numpy(boards[:, np.newaxis, ...])
        elif len(boards.shape) == 2:
            boards = torch.from_numpy(boards[np.newaxis, np.newaxis, ...])

        value = self.layers(boards)
        return value


class DeepVAgent(Agent):
    def __init__(
        self,
        n_channels: int = 64,
        player_number: int = constants.PLAYER1,
        epsilon: float = 0,
        board_shape: Tuple[int, int] = (6, 7),
        seed: int = 42,
        stochastic: bool = False,
    ):
        super().__init__(player_number=player_number, board_shape=board_shape)
        self.value_network = ValueNetwork(board_shape, n_channels)
        self.random = np.random.default_rng(seed)
        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.value_network.parameters(), lr=0.001)
        self.epsilon = epsilon
        self.stochastic = stochastic

    def _translate_state(self, state: np.ndarray) -> np.ndarray:
        return self.player_number * state

    def get_move(
        self, state: np.ndarray, explore: bool = True, get_values=False
    ) -> Union[int, Tuple[int, Tuple[List[int], List[float]]]]:
        self.value_network.eval()

        actions, next_states = get_next_actions_states(state, self.player_number)
        next_states_translated = [self._translate_state(state) for state in next_states]
        next_states_values = self.value_network(next_states_translated)

        if explore and (self.random.random() < self.epsilon):
            action = self.random.choice(actions)
        else:
            if self.stochastic:
                exp_values = np.exp(next_states_values.detach().numpy().flatten())
                index_action = np.random.choice(
                    len(actions), p=exp_values / sum(exp_values)
                )
            else:
                index_action = np.random.choice(
                    np.flatnonzero(next_states_values == next_states_values.max())
                )
            action = actions[index_action]

        if get_values:
            return action, (actions, [value.item() for value in next_states_values])
        else:
            return action

    def learn_from_episode(self, states: List[np.ndarray], gains: np.ndarray):
        assert len(states) == len(gains), "not as many states as there are gains"

        states_translated = [self._translate_state(state) for state in states]

        self.value_network.train()
        self.optimizer.zero_grad()
        criterion = self.loss(
            self.value_network(states_translated),
            torch.from_numpy(gains[:, np.newaxis]),
        )
        criterion.backward()
        self.optimizer.step()

        return criterion.item()

    def save(self, name: str, path: str = "private/saved_agents") -> None:
        os.makedirs(path, exist_ok=True)
        name = name + ".pt"
        torch.save(self.value_network, os.path.join(path, name))

    def load(self, name: str, path: str = "private/saved_agents"):
        name = name + ".pt"
        with open(os.path.join(path, name), "rb") as file:
            self.value_network = torch.load(file)

    def duplicate(self) -> Agent:
        agent = DeepVAgent(1, self.player_number)
        agent.value_network = self.value_network
        return agent
