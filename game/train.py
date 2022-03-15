from typing import List, Tuple

import numpy as np
from copy import deepcopy
from tqdm.auto import tqdm

import constants
from agents.deep_v_agent import DeepVAgent
from agents.random_agent import RandomAgent
from connect_four_env.connect_four_env import ConnectFourGymEnv
from game.episode import run_episode, run_episode_against_self


def compute_gain_from_rewards(rewards: List[int], discount: float = 1.0) -> np.ndarray:
    gains = []
    for step in range(len(rewards)):
        dicounted_rewards = [rewards[i] * discount ** (i - step) for i in range(step, len(rewards))]
        gains.append(np.sum(dicounted_rewards))
    return np.array(gains)


def win_rate_vs_opponent(agent, env, opponent_agent, n_runs=10):
    n_wins = 0
    for i in range(n_runs):
        even = (i % 2 == 0)
        _, rewards = run_episode(
            agent if even else opponent_agent, 
            opponent_agent if even else agent, 
            env, keep_states=True, for_evaluation=True
        )
        n_wins += rewards[0][-1] == env.WINNER_REWARD  # does not count draws
    return n_wins / n_runs


def make_opponent(agent: DeepVAgent):
    opponent = deepcopy(agent)
    opponent.stochastic = True
    return opponent


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
            win_rates.append(win_rate_vs_opponent(agent, env, latest_opponent, n_test_runs))
        p1_states, p2_states, p1_rewards, p2_rewards = run_episode_against_self(agent, env)
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
