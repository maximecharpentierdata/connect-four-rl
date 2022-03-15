from typing import List, Tuple

import numpy as np


def get_fall_row(board, column: int) -> int:
    row = 0
    while board[row, column] != 0:
        row += 1
        if row >= board.shape[0]:
            raise ValueError(f"{column} is already full")
    return row


def modify_board(board, player_value, column):
    row = get_fall_row(board, column)
    board[row, column] = player_value
    return row


def get_next_actions_states(
    state: np.ndarray, player_value: int
) -> Tuple[List[int], List[np.ndarray]]:
    actions = []
    next_states = []
    for column in range(state.shape[1]):
        try:
            row = modify_board(state, player_value, column)
            actions.append(column)
            next_states.append(state.copy())  # Maybe optimize later
            state[row, column] = 0
        except ValueError:
            pass
    return actions, next_states
