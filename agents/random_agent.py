from typing import List, Tuple, Union

import numpy as np

import constants
from agents.agent import Agent
from connect_four_env.utils import get_next_actions_states


class RandomAgent(Agent):
    """
    A random agent that chooses a random action at each step.
    """

    def __init__(
        self,
        player_number: int = constants.PLAYER1,
        board_shape: Tuple[int, int] = (6, 7),
        stochastic: bool = False,
    ):
        super().__init__(
            player_number=player_number, board_shape=board_shape, stochastic=stochastic
        )

    def get_move(
        self, state: np.ndarray, explore: bool = True, get_values: bool = False
    ) -> Union[int, Tuple[int, Tuple[List[int], List[float]]]]:

        actions, _ = get_next_actions_states(state, self.player_number)

        random_action = int(np.random.randint(0, len(actions)))

        if get_values:
            return actions[random_action], (
                actions,
                [1.0 / len(actions)] * len(actions),
            )
        else:
            return actions[random_action]

    def learn_from_episode(self, states: List[np.ndarray], gains: List[float]):
        pass
