from typing import Tuple

import gym
import numpy as np
from gym import spaces

import constants
from connect_four_env.rendering import render_board
from connect_four_env.utils import modify_board


class ConnectFourGymEnv(gym.Env):
    def __init__(self, board_size: Tuple[int] = (6, 7)):
        super(ConnectFourGymEnv, self).__init__()

        self.board = np.zeros(board_size)

        self.reward_range = (0, 1)

        self.action_space = spaces.MultiDiscrete((2 * [board_size[1]]))

        self.observation_space = spaces.MultiDiscrete([3] * board_size[0] * board_size[1])

        self.result = None

        self.history = []

    def _take_action(self, player_value: int, column: int, keep_history: bool) -> int:
        row = modify_board(self.board, player_value, column)
        if keep_history:
            self.history.append(self.board.copy())
        return row

    def _check_row(self, row: int) -> bool:
        return row < 0 or row >= self.board.shape[0]

    def _check_column(self, column: int) -> bool:
        return column < 0 or column >= self.board.shape[1]

    def _check_direction(self, row: int, column: int, direction: Tuple[int, int]) -> bool:
        total = -1
        i = 0
        while (
            self.board[row + i * direction[0], column + i * direction[1]] == self.board[row, column]
        ):
            total += 1
            i += 1
            if self._check_row(row + i * direction[0]) or self._check_column(
                column + i * direction[1]
            ):
                break
        i = 0
        while (
            self.board[row + i * direction[0], column + i * direction[1]] == self.board[row, column]
        ):
            total += 1
            i -= 1
            if self._check_row(row + i * direction[0]) or self._check_column(
                column + i * direction[1]
            ):
                break
        return total >= 4

    def _game_over(self, row: int, column: int) -> bool:
        # Check if the game is over
        directions = [(1, 0), (1, 1), (0, 1), (1, -1)]
        for direction in directions:
            if self._check_direction(row, column, direction):
                return True
        return False

    def _is_tie(self) -> bool:
        return np.sum(self.board == 0) == 0

    def get_final_reward(self, player_number):
        if self.result == player_number:
            reward = constants.WINNER_REWARD
        elif self.result == 0:
            reward = constants.IDLE_REWARD
        else:
            reward = constants.LOSER_REWARD
        return reward

    def step(self, action: Tuple[int, int], keep_history=False):
        # Execute one time step within the environment
        reward = constants.IDLE_REWARD
        player = action[0]
        column = action[1]
        row = self._take_action(player, column, keep_history)

        done = False
        if self._game_over(row, column):
            done = True
            reward = constants.WINNER_REWARD
            self.result = player

        if self._is_tie():
            done = True
            reward = constants.IDLE_REWARD
            self.result = 0

        return self.board, reward, done, {}

    def reset(self) -> np.ndarray:
        # Reset the state of the environment to an initial state
        self.board = np.zeros_like(self.board)
        self.history = []
        return self.board

    def render(self, mode="human", figsize=(10.5, 9), slot_size=3000):
        """Render the environment to the screen"""
        render_board(self.board, figsize, slot_size)
