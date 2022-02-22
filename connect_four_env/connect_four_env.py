import gym
from gym import spaces
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt

class ConnectFourGymEnv(gym.Env):
    PLAYER1 = 1
    PLAYER2 = -1


    def __init__(self, board_size:Tuple[int]=(6, 7)):
        super(ConnectFourGymEnv, self).__init__()
        
        self.board = np.zeros(board_size)

        self.reward_range = (0, 1)
        
        self.action_space = spaces.MultiDiscrete([7, 7])
        
        self.observation_space = spaces.MultiDiscrete([3] *  7 * 6)

    def _get_fall_row(self, column: int) -> int:
        row = 0
        while self.board[row, column] != 0:
            row += 1
            if row >= self.board.shape[1]:
                raise ValueError(f"{column} is already full")
        return row

    def _take_action(self, player_value: int, column: int) -> int:
        row = self._get_fall_row(column)
        self.board[row, column] = player_value
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

    def step(self, action:Tuple[int, int]):
        # Execute one time step within the environment

        reward = (0, 0)

        player_value = -1
        column = action[0]
        row = self._take_action(player_value, column)
        done = False
        if self._game_over(row, column):
            done = True
            reward = (1, -1)

        if not done:
            player_value = 1
            column = action[1]
            row = self._take_action(player_value, column)
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
        plt.figure(figsize=figsize, facecolor='blue')

        row, col = np.indices(self.board.shape)
        for slot_value, color in [
            (self.PLAYER1, "yellow"), 
            (self.PLAYER2, "red"), 
            (0, "grey")
        ]:
            x = col[self.board == slot_value].flatten()
            y = row[self.board == slot_value].flatten()
            plt.scatter(x, y, c=color, s=slot_size)

        plt.xlim(-.5, self.board.shape[1]-.5)
        plt.ylim(-.5, self.board.shape[0]-.5)
        plt.axis("off")

        plt.show()
        