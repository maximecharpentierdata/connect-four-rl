from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from ipywidgets import widgets

from constants import PLAYER2, PLAYER1, EMPTY
from connect_four_env.utils import get_fall_row

SLOT_COLORS = {
    PLAYER1: "yellow",
    PLAYER2: "red",
    EMPTY: "grey",
}


def render_board(
    board: np.ndarray,
    figsize: Tuple[float, float] = (10.5, 9),
    slot_size: int = 3000,
    agent_values: Optional[Tuple[List[int], List[float]]] = None,
):

    plt.figure(figsize=figsize, facecolor="blue")

    rows, cols = np.indices(board.shape)
    for slot_value, color in SLOT_COLORS.items():
        x = cols[board == slot_value].flatten()
        y = rows[board == slot_value].flatten()
        plt.scatter(x, y, c=color, s=slot_size)

    if agent_values:
        if np.isin(board, [PLAYER1, PLAYER2]).sum() % 2 == 0:
            text_color = SLOT_COLORS[PLAYER1]
        else:
            text_color = SLOT_COLORS[PLAYER2]
        actions, actions_values = agent_values
        best_value = max(actions_values)
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
                    fontweight="bold" if value == best_value else "normal",
                    color=text_color,
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

        def render_with_values(turn):
            if turn == len(history) - 1:
                return render_board(history[turn])
            else:
                return render_board(history[turn], agent_values=agent_values[turn])

        output = widgets.interactive_output(
            render_with_values,
            {"turn": slider},
        )
    else:
        output = widgets.interactive_output(
            lambda turn: render_board(history[turn]), {"turn": slider}
        )

    display(hbox)
    display(output)
