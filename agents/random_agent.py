from agents.agent import Agent
from typing import Tuple, List
import numpy as np
from connect_four_env.connect_four_env import ConnectFourGymEnv


class RandomAgent(Agent):
    """
    A random agent that chooses a random action at each step.
    """

    def __init__(self, player_number: int, board_shape: Tuple[int, int]):
        self.player_number = player_number
        self.board_shape = board_shape

    def get_move(self, state: np.ndarray) -> int:

        actions_states = ConnectFourGymEnv.get_next_actions_states(
            state, self.player_number
        )

        random_action = np.random.randint(0, len(actions_states))
        return actions_states[random_action][0]

    def learn_from_episode(self, states: List[np.ndarray], gains: List[float]):
        pass
