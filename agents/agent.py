from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import numpy as np


class Agent(ABC):
    def __init__(self, player_number: int, board_shape: Tuple[int, int] = (6, 7)):
        self.player_number = player_number
        self.board_shape = board_shape

    @abstractmethod
    def get_move(
        self, state: np.ndarray, explore: bool = True, get_values: bool = False
    ) -> Union[int, Tuple[int, Tuple[List[int], List[float]]]]:
        pass

    @abstractmethod
    def learn_from_episode(self, states: List[np.ndarray], gains: List[float]):
        pass
