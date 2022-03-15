from copy import deepcopy
from typing import List, Tuple

import numpy as np
from tqdm.auto import tqdm

import constants
from agents.deep_v_agent import DeepVAgent
from agents.random_agent import RandomAgent
from connect_four_env.connect_four_env import ConnectFourGymEnv
from game.episode import run_episode, run_episode_against_self


def compute_gain_from_rewards(rewards: List[int], discount: float = 1.0) -> np.ndarray:
    gains = []
    for step in range(len(rewards)):
        dicounted_rewards = [
            rewards[i] * discount ** (i - step) for i in range(step, len(rewards))
        ]
        gains.append(np.sum(dicounted_rewards))
    return np.array(gains)


def win_rate_vs_opponent(
    agent: DeepVAgent, opponent, env: ConnectFourGymEnv, n_runs=10
):
    n_wins = 0
    index_agent = int(agent.player_number != env.PLAYER1)
    agent1 = [agent, opponent][index_agent]
    agent2 = [agent, opponent][1 - index_agent]
    for _ in range(n_runs):
        _, rewards = run_episode(
            agent1, agent2, env, keep_states=True, for_evaluation=True
        )
        n_wins += rewards[index_agent][-1] == env.WINNER_REWARD  # does not count draws
    return n_wins / n_runs


def make_opponent(agent: DeepVAgent):
    opponent = deepcopy(agent)
    opponent.stochastic = True
    return opponent


# Obsolete (for the moment at least)
def train_against_self(
    discount: float,
    n_episodes: int,
    agent: DeepVAgent,
    n_test_runs: int = 10,
    freq_change_opp: int = 1000,
) -> Tuple[List[float], List[float]]:
    win_rates, losses = [], []
    env = ConnectFourGymEnv()
    latest_opponent = make_opponent(agent)

    for i in tqdm(range(n_episodes)):
        if (i + 1) % freq_change_opp == 0:
            latest_opponent = make_opponent(agent)
        if (i + 1) % 100 == 0 or i == 0:
            win_rates.append(
                win_rate_vs_opponent(agent, env, latest_opponent, n_test_runs)
            )
        p1_states, p2_states, p1_rewards, p2_rewards = run_episode_against_self(
            agent, env
        )
        p1_gains = compute_gain_from_rewards(p1_rewards, discount)
        p2_gains = compute_gain_from_rewards(p2_rewards, discount)
        for (states, gains, player_number) in [
            (p1_states, p1_gains, constants.PLAYER1),
            (p2_states, p2_gains, constants.PLAYER2),
        ]:
            agent.player_number = player_number
            loss = agent.learn_from_episode(states, gains)
            losses.append(loss)

    return win_rates, losses


def train_both_agents(
    discount: float,
    n_episodes: int,
    agent1: DeepVAgent,
    agent2: DeepVAgent,
    n_test_runs: int = 10,
    period_change_opp: int = 1000,
):
    win_rates_1, losses_1 = [], []
    win_rates_2, losses_2 = [], []
    env = ConnectFourGymEnv()
    latest_opponent_1 = RandomAgent(env.PLAYER2, env.board.shape)
    latest_opponent_2 = RandomAgent(env.PLAYER1, env.board.shape)

    for i in tqdm(range(n_episodes)):
        if (i + 1) % period_change_opp == 0:
            latest_opponent_1 = make_opponent(agent2)
            latest_opponent_2 = make_opponent(agent1)
        if (i + 1) % 100 == 0 or i == 0:
            win_rates_1.append(
                win_rate_vs_opponent(agent1, latest_opponent_1, env, n_test_runs)
            )
            win_rates_2.append(
                win_rate_vs_opponent(agent2, latest_opponent_2, env, n_test_runs)
            )
        (p1_states, p2_states), (p1_rewards, p2_rewards) = run_episode(
            agent1, agent2, env, keep_states=True
        )
        p1_gains = compute_gain_from_rewards(p1_rewards, discount)
        p2_gains = compute_gain_from_rewards(p2_rewards, discount)

        loss_1 = agent1.learn_from_episode(p1_states, p1_gains)
        losses_1.append(loss_1)
        loss_2 = agent2.learn_from_episode(p2_states, p2_gains)
        losses_2.append(loss_2)

    return (win_rates_1, win_rates_2), (losses_1, losses_2)
