import gym
from gym import spaces
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt
from ipywidgets import interact, widgets


class ConnectFourGymEnv(gym.Env):
    PLAYER1 = -1
    PLAYER2 = 1

    LOSER_REWARD = -1
    WINNER_REWARD = 1
    IDLE_REWARD = 0

    def __init__(self, board_size: Tuple[int] = (6, 7)):
        super(ConnectFourGymEnv, self).__init__()

        self.board = np.zeros(board_size)

        self.reward_range = (0, 1)

        self.action_space = spaces.MultiDiscrete((2 * [board_size[1]]))

        self.observation_space = spaces.MultiDiscrete(
            [3] * board_size[0] * board_size[1]
        )

        self.result = None

        self.history = []

    @staticmethod
    def _get_fall_row(board, column: int) -> int:
        row = 0
        while board[row, column] != 0:
            row += 1
            if row >= board.shape[0]:
                raise ValueError(f"{column} is already full")
        return row

    @staticmethod
    def modify_board(board, player_value, column):
        row = ConnectFourGymEnv._get_fall_row(board, column)
        board[row, column] = player_value
        return row

    def _take_action(self, player_value: int, column: int, keep_history: bool) -> int:
        row = ConnectFourGymEnv.modify_board(self.board, player_value, column)
        if keep_history:
            self.history.append(self.board.copy())
        return row

    def _check_row(self, row: int) -> bool:
        return row < 0 or row >= self.board.shape[0]

    def _check_column(self, column: int) -> bool:
        return column < 0 or column >= self.board.shape[1]

    def _check_direction(
        self, row: int, column: int, direction: Tuple[int, int]
    ) -> bool:
        total = -1
        i = 0
        while (
            self.board[row + i * direction[0], column + i * direction[1]]
            == self.board[row, column]
        ):
            total += 1
            i += 1
            if self._check_row(row + i * direction[0]) or self._check_column(
                column + i * direction[1]
            ):
                break
        i = 0
        while (
            self.board[row + i * direction[0], column + i * direction[1]]
            == self.board[row, column]
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

    def _draw(self) -> bool:
        return np.sum(self.board == 0) == 0

    def get_final_reward(self, player_number):
        if self.result == player_number:
            reward = self.WINNER_REWARD
        elif self.result == 0:
            reward = self.IDLE_REWARD
        else:
            reward = self.LOSER_REWARD
        return reward

    def step(self, action: Tuple[int, int], keep_history=False):
        # Execute one time step within the environment
        reward = self.IDLE_REWARD
        player = action[0]
        column = action[1]
        row = self._take_action(player, column, keep_history)

        done = False
        if self._game_over(row, column):
            done = True
            reward = self.WINNER_REWARD
            self.result = player

        if self._draw():
            done = True
            reward = self.IDLE_REWARD
            self.result = 0

        return self.board, reward, done, {}

    @staticmethod
    def get_next_actions_states(
        state: np.ndarray, player_value: int
    ) -> List[Tuple[int, np.ndarray]]:
        next_states = []
        for column in range(state.shape[1]):
            try:
                row = ConnectFourGymEnv.modify_board(state, player_value, column)
                next_states.append((column, state.copy()))  # Maybe optimize later
                state[row, column] = 0
            except ValueError:
                pass
        return next_states

    def reset(self) -> np.ndarray:
        # Reset the state of the environment to an initial state
        self.board = np.zeros_like(self.board)
        self.history = []
        return self.board

    def render(self, mode="human", figsize=(10.5, 9), slot_size=3000):
        """Render the environment to the screen"""
        self._render_board(self.board, figsize, slot_size)

    def _render_board(self, board, figsize=(10.5, 9), slot_size=3000):
        plt.figure(figsize=figsize, facecolor="blue")

        row, col = np.indices(board.shape)
        for slot_value, color in [
            (self.PLAYER1, "yellow"),
            (self.PLAYER2, "red"),
            (0, "grey"),
        ]:
            x = col[board == slot_value].flatten()
            y = row[board == slot_value].flatten()
            plt.scatter(x, y, c=color, s=slot_size)

        plt.xlim(-0.5, board.shape[1] - 0.5)
        plt.ylim(-0.5, board.shape[0] - 0.5)
        plt.axis("off")
        plt.show()

    def render_history(self):
        """This is designed to be used in a notebook, be careful"""
        interact(
            lambda turn: self._render_board(self.history[turn]),
            turn=widgets.IntSlider(min=0, max=len(self.history) - 1, step=1, value=0),
        )
