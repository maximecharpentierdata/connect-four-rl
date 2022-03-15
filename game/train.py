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
        dicounted_rewards = [rewards[i] * discount ** (i - step) for i in range(step, len(rewards))]
        gains.append(np.sum(dicounted_rewards))
    return np.array(gains)


def win_rate_vs_random(agent, env, random_agent, n_runs=10):
    n_wins = 0
    for _ in range(n_runs):
        _, rewards = run_episode(agent, random_agent, env, keep_states=True, for_evaluation=True)
        n_wins += rewards[0][-1] == constants.WINNER_REWARD  # does not count draws
    return n_wins / n_runs


def train_against_self(
    discount: float, n_episodes: int, agent: DeepVAgent, test_against_random: int = 10
) -> Tuple[List[float], List[float]]:
    win_rates, losses = [], []
    env = ConnectFourGymEnv()
    random_agent = RandomAgent(constants.PLAYER2, env.board.shape)

    for i in tqdm(range(n_episodes)):
        if (i + 1) % 100 == 0 or i == 0:
            win_rates.append(win_rate_vs_random(agent, env, random_agent, test_against_random))
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
