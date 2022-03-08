from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple


class Agent(ABC):
    @abstractmethod
    def __init__(self, player_number: int, board_shape: Tuple[int, int] = (6, 7)):
        pass

    @abstractmethod
    def get_move(self, state: np.ndarray, explore: bool = True) -> int:
        pass

    @abstractmethod
    def learn_from_episode(self, states: List[np.ndarray], gains: List[float]):
        pass
