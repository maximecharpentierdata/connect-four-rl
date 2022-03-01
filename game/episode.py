from typing import List, Union, Tuple

import numpy as np

from agents.agent import Agent
from connect_four_env.connect_four_env import ConnectFourGymEnv


def run_episode(
    agent_1: Agent,
    agent_2: Agent,
    env: ConnectFourGymEnv,
    keep_states=False,
) -> Union[None, Tuple[Tuple[List[np.ndarray]], Tuple[List[int]]]]:

    state = env.reset()
    done = False

    if keep_states:
        states = ([], [])
        rewards = ([], [])

    agent_1.player_number = env.PLAYER1
    agent_2.player_number = env.PLAYER2
    agents = [agent_1, agent_2]

    current_player = 0

    while not done:

        agent = agents[current_player]

        action = agent.get_move(state)
        state, reward, done, _ = env.step(
            (agent.player_number, action), keep_history=True
        )

        if keep_states:
            rewards[current_player].append(reward)
            states[current_player].append(state)

        current_player = (current_player + 1) % 2

    if keep_states:
        rewards[current_player][-1] = env.get_final_reward(
            agents[current_player].player_number
        )

        return states, rewards

    return None


def run_episode_against_self(
    agent: Agent, env: ConnectFourGymEnv, keep_history: bool = False
) -> Tuple[List[np.ndarray], List[int], List[int]]:
    state = env.reset()
    done = False
    states = []
    rewards = []
    opponent_rewards = []
    opponent_reward = None
    opponent_turn = False
    while not done:
        state = -state if opponent_turn else state
        player_number = [env.PLAYER2, env.PLAYER1][opponent_turn]
        action_number = agent.get_move(state)

        state, reward, done, _ = env.step((player_number, action_number), keep_history)

        rewards_to_update = [opponent_rewards, rewards][opponent_turn]
        rewards_to_update.append(reward)
        states.append(state)
        opponent_turn = not opponent_turn

    before_last_player = [env.PLAYER2, env.PLAYER1][opponent_turn]
    rewards_to_update = [opponent_rewards, rewards][opponent_turn]
    rewards_to_update[-1] = env.get_final_reward(before_last_player)

    return states, rewards, opponent_rewards
