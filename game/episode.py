import os
from typing import List, Tuple, Union

import numpy as np

import constants
from agents.agent import Agent
from agents.deep_v_agent import DeepVAgent
from connect_four_env.connect_four_env import ConnectFourGymEnv
from connect_four_env.rendering import render_history


def run_episode(
    agent_1: Agent,
    agent_2: Agent,
    env: ConnectFourGymEnv,
    keep_states: bool = False,
    keep_history: bool = False,
    for_evaluation: bool = False,
    get_values: bool = False,
) -> Union[int, Tuple[Tuple[List[np.ndarray]], Tuple[List[int]], int]]:

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
            action, values = agent.get_move(
                state, explore=not for_evaluation, get_values=True
            )
            action_values.append(values)
        else:
            action = agent.get_move(state, explore=not for_evaluation, get_values=False)

        state, reward, done, _ = env.step((agent.player_number, action), keep_history)

        if keep_states:
            rewards[current_player].append(reward)
            states[current_player].append(state)

        current_player = (current_player + 1) % 2

    if keep_states:
        rewards[current_player][-1] = env.get_final_reward(
            agents[current_player].player_number
        )
        return states, rewards, env.result

    if get_values:
        return action_values, env.result

    return env.result


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


def watch_game(agent1: Union[str, Agent], agent2: Union[str, Agent]):
    """Watch a game between two agents"""

    env = ConnectFourGymEnv()

    agent1_path = None
    agent2_path = None

    if isinstance(agent1, str):
        agent1_path = agent1
        agent1 = DeepVAgent()
        agent1.load(os.path.join(agent1_path, "agent.pt"))

    if isinstance(agent2, str):
        agent2_path = agent2
        agent2 = DeepVAgent()
        agent2.load(os.path.join(agent2_path, "agent.pt"))

    action_values, _ = run_episode(
        agent1, agent2, env, for_evaluation=True, keep_history=True, get_values=True
    )

    if agent1_path is not None:
        print("Player 1 (yellow):", os.path.basename(agent1_path))
    if agent2_path is not None:
        print("Player 2 (red):", os.path.basename(agent2_path))
    render_history(env.history, agent_values=action_values)
