import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from ipywidgets import widgets

import constants
from connect_four_env.utils import get_fall_row
from typing import Optional, Tuple, List


def render_board(
    board,
    figsize: Tuple[float, float] = (10.5, 9),
    slot_size: int = 3000,
    agent_values: Optional[Tuple[List[int], List[float]]] = None,
):

    plt.figure(figsize=figsize, facecolor="blue")

    rows, cols = np.indices(board.shape)
    for slot_value, color in [
        (constants.PLAYER1, "yellow"),
        (constants.PLAYER2, "red"),
        (constants.EMPTY, "grey"),
    ]:
        x = cols[board == slot_value].flatten()
        y = rows[board == slot_value].flatten()
        plt.scatter(x, y, c=color, s=slot_size)

    if agent_values:
        actions, actions_values = agent_values
        for action, value in zip(actions, actions_values):
            column = action
            try:
                row = get_fall_row(board, column)
                plt.text(
                    column,
                    row,
                    f"{value:.2f}",
                    horizontalalignment="center",
                    fontsize=14,
                    fontweight="bold",
                )
            except ValueError:
                continue

    plt.xlim(-0.5, board.shape[1] - 0.5)
    plt.ylim(-0.5, board.shape[0] - 0.5)
    plt.axis("off")
    plt.show()


def render_history(history, playback_speed=500, agent_values=None):
    """This is designed to be used in a notebook, be careful"""

    if len(history) == 0:
        raise ValueError("No history to render")

    play = widgets.Play(
        value=0,
        min=0,
        max=len(history) - 1,
        step=1,
        interval=playback_speed,
        disabled=False,
    )
    slider = widgets.IntSlider(min=0, max=len(history) - 1, step=1, value=0)
    widgets.jslink((play, "value"), (slider, "value"))
    hbox = widgets.HBox([play, slider])

    if agent_values:
        output = widgets.interactive_output(
            lambda turn: render_board(history[turn], agent_values=agent_values[turn]),
            {"turn": slider},
        )
    else:
        output = widgets.interactive_output(
            lambda turn: render_board(history[turn]), {"turn": slider}
        )

    display(hbox)
    display(output)
