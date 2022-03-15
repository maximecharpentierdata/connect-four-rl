import time
from typing import Callable

import ipywidgets as widgets
from IPython.display import clear_output, display

import constants
from agents.agent import Agent
from connect_four_env.connect_four_env import ConnectFourGymEnv


class ColumnButtons:
    def __init__(self, n_columns: int):
        """Creates a multi-button for selecting a column of the
        board.

        Args:
            n_columns (int): Number of columns
        """
        self.buttons = []
        for k in range(n_columns):
            self.buttons.append(
                widgets.Button(
                    description=str(k + 1),
                    layout=widgets.Layout(width="82px", height="30px"),
                )
            )

    def show(self):
        output = widgets.HBox(self.buttons)
        display(output)

    def on_click(self, handler: Callable):
        for button in self.buttons:
            button.on_click(handler)


def play_with_robot(env: ConnectFourGymEnv, position: str, agent: Agent):
    """Creates an interactive board for playing with the agent
    DO NOT use in a VSCode Notebook (only in Jupyter Notebook)

    Example:
            play_with_robot(env, "first", agent)
            env.render()

    Args:
        position (str): Must be either "first" or "second"
        agent (Agent): Agent to play against
    """
    if position == "first":
        human_player_number = constants.PLAYER1
        agent_player_number = constants.PLAYER2
    else:
        human_player_number = constants.PLAYER2
        agent_player_number = constants.PLAYER1

    agent.player_number = agent_player_number
    column_buttons = ColumnButtons(7)

    def handler(button: widgets.Button):
        clear_output(wait=True)
        action = int(button.description) - 1
        state, reward, done, _ = env.step([human_player_number, action])
        if done:
            print("Human player won!")
            return 0

        env.render()
        time.sleep(0.5)
        clear_output(wait=True)

        agent_action = agent.get_move(state)
        state, reward, done, _ = env.step([agent_player_number, agent_action])
        if done:
            print("Robot won!")
            return 0

        env.render()
        column_buttons.show()

    if position != "first":
        state = env.board
        agent_action = agent.get_move(state)
        state, reward, done, _ = env.step([agent_player_number, agent_action])

    column_buttons.on_click(handler)

    env.render()
    column_buttons.show()
