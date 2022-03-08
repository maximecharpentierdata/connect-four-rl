import time
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from ipywidgets import widgets

from agents.agent import Agent
from connect_four_env.connect_four_env import ConnectFourGymEnv


class HumanAgent(Agent):
    """
    Allows a human to play the game.
    """

    def __init__(self, player_number: int, board_shape: Tuple[int, int]):
        super().__init__(player_number, board_shape)
        self.player_number = player_number
        self.board_shape = board_shape
        self.clicked = False
        self.action_selected = None
        self.figsize = None

    def render_board(self, board, figsize=(10.5, 9), slot_size=3000):
        fig = plt.figure(figsize=figsize, facecolor="blue")

        row, col = np.indices(board.shape)
        for slot_value, color in [
            (-1, "yellow"),
            (1, "red"),
            (0, "grey"),
        ]:
            x = col[board == slot_value].flatten()
            y = row[board == slot_value].flatten()
            plt.scatter(x, y, c=color, s=slot_size)

        plt.xlim(-0.5, board.shape[1] - 0.5)
        plt.ylim(-0.5, board.shape[0] - 0.5)
        plt.axis("off")
        plt.show()

        return fig

    def get_move(self, state: np.ndarray) -> int:

        output = widgets.Output()

        with output:
            self.render_board(state)

        actions, _ = ConnectFourGymEnv.get_next_actions_states(state, self.player_number)

        buttons = []

        for action in actions:
            buttons.append(widgets.Button(description=str(action)))

        clicked = False

        def on_button_clicked(b):
            with output:
                print("clicked")
            with open("test.txt", "w") as f:
                f.write(b.description)
            nonlocal clicked
            self.action_selected = int(b.description)
            clicked = True

        for button in buttons:
            button.on_click(on_button_clicked)

        display(output)
        display(widgets.HBox(buttons, layout=widgets.Layout(width="100%")))

        while not clicked:
            with output:
                print(clicked)
            time.sleep(0.5)

        self.clicked = False
        return self.action_selected

    def learn_from_episode(self, states: List[np.ndarray], gains: List[float]):
        pass
