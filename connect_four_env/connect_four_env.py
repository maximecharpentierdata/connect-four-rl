import gym
from gym import spaces
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt
from ipywidgets import interact, widgets

class ConnectFourGymEnv(gym.Env):
    PLAYER1 = -1
    PLAYER2 = 1


    def __init__(self, board_size:Tuple[int]=(6, 7)):
        super(ConnectFourGymEnv, self).__init__()
        
        self.board = np.zeros(board_size)

        self.reward_range = (0, 1)
        
        self.action_space = spaces.MultiDiscrete(board_size)
        
        self.observation_space = spaces.MultiDiscrete([3] *  7 * 6)

        self.history = []

    def _get_fall_row(self, column: int) -> int:
        row = 0
        while self.board[row, column] != 0:
            row += 1
            if row >= self.board.shape[1]:
                raise ValueError(f"{column} is already full")
        return row

    def _take_action(self, player_value: int, column: int, keep_history: bool) -> int:
        row = self._get_fall_row(column)
        self.board[row, column] = player_value
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
        while self.board[row + i * direction[0], column + i * direction[1]] == self.board[row, column]:
            total += 1
            i += 1
            if self._check_row(row + i * direction[0]) or self._check_column(column + i * direction[1]):
                break
        i = 0
        while self.board[row + i * direction[0], column + i * direction[1]] == self.board[row, column]:
            total += 1
            i -= 1
            if self._check_row(row + i * direction[0]) or self._check_column(column + i * direction[1]):
                break
        return total >= 4

    def _game_over(self, row: int, column: int) -> bool:
        # Check if the game is over
        directions = [(1, 0), (1, 1), (0, 1), (1, -1)]
        for direction in directions:
            if self._check_direction(row, column, direction):
                return True
        return False

    def step(self, action:Tuple[int, int], keep_history=False):
        # Execute one time step within the environment

        reward = (0, 0)

        column = action[0]
        row = self._take_action(self.PLAYER1, column, keep_history)
        done = False
        if self._game_over(row, column):
            done = True
            reward = (1, -1)

        if not done:
            column = action[1]
            row = self._take_action(self.PLAYER2, column, keep_history)
            done = False
            if self._game_over(row, column):
                done = True
                reward = (-1, 1)

        return self.board, reward, done, {}
            
        
    def reset(self):
        # Reset the state of the environment to an initial state
        self.board = np.zeros_like(self.board)
    
    def render(self, mode='human', close=False, figsize=(10.5, 9), slot_size=3000):
        """Render the environment to the screen"""
        self.render_board(self.board, figsize, slot_size)

    def render_board(self, board, figsize=(10.5, 9), slot_size=3000):
        plt.figure(figsize=figsize, facecolor='blue')

        row, col = np.indices(board.shape)
        for slot_value, color in [
            (self.PLAYER1, "yellow"), 
            (self.PLAYER2, "red"), 
            (0, "grey")
        ]:
            x = col[board == slot_value].flatten()
            y = row[board == slot_value].flatten()
            plt.scatter(x, y, c=color, s=slot_size)

        plt.xlim(-.5, board.shape[1]-.5)
        plt.ylim(-.5, board.shape[0]-.5)
        plt.axis("off")
        plt.show()
        
    def render_history(self):
        """This is designed to be used in a notebook, be careful"""
        interact(
            lambda turn: self.render_board(self.history[turn]), 
            turn=widgets.IntSlider(min=0, max=len(self.history)-1, step=1, value=0)
        )
        
        