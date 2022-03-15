from typing import List, Tuple, Union

import numpy as np

import constants
from agents.agent import Agent
from connect_four_env.connect_four_env import ConnectFourGymEnv


def run_episode(
    agent_1: Agent,
    agent_2: Agent,
    env: ConnectFourGymEnv,
    keep_states: bool = False,
    keep_history: bool = False,
    for_evaluation: bool = False,
    get_values: bool = False,
) -> Union[None, Tuple[Tuple[List[np.ndarray]], Tuple[List[int]]]]:

    state = env.reset()
    done = False

    if keep_states:
        states = ([], [])
        rewards = ([], [])

    agent_1.player_number = constants.PLAYER1
    agent_2.player_number = constants.PLAYER2
    agents = [agent_1, agent_2]

    current_player = 0

    action_values = []

    while not done:
        agent = agents[current_player]

        if get_values:
            action, values = agent.get_move(state, explore=not for_evaluation, get_values=True)
            action_values.append(values)
        else:
            action = agent.get_move(state, explore=not for_evaluation, get_values=False)

        state, reward, done, _ = env.step((agent.player_number, action), keep_history)

        if keep_states:
            rewards[current_player].append(reward)
            states[current_player].append(state)

        current_player = (current_player + 1) % 2

    if keep_states:
        rewards[current_player][-1] = env.get_final_reward(agents[current_player].player_number)
        return states, rewards

    if get_values:
        return action_values


def run_episode_against_self(
    agent: Agent, env: ConnectFourGymEnv, keep_history: bool = False
) -> Tuple[List[np.ndarray], List[np.ndarray], List[int], List[int]]:
    """Makes the agent play against itself
    Returns player1_states, player2_states, player1_rewards, player2_rewards
    The state lists give the board that the player saw at its turn"""
    state = env.reset()
    done, opponent_turn = False, False
    states, rewards = [], []
    players = [constants.PLAYER1, constants.PLAYER2]
    while not done:
        player_number = players[opponent_turn]
        agent.player_number = player_number
        action_number = agent.get_move(state)

        state, reward, done, _ = env.step((player_number, action_number), keep_history)

        rewards.append(reward)
        states.append(state)
        opponent_turn = not opponent_turn

    before_last_player = players[opponent_turn]
    rewards[-2] = env.get_final_reward(before_last_player)

    return states[::2], states[1::2], rewards[::2], rewards[1::2]
