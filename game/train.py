from .episode import run_episode, run_episode_against_self
import numpy as np
from typing import List
from tqdm.auto import tqdm


def compute_gain_from_rewards(rewards: List[int], discount: float = 1.0) -> np.ndarray:
    gains = []
    for step in len(rewards):
        dicounted_rewards = [
            rewards[i] * discount ** (i - step) for i in range(step, len(rewards))
        ]
        gains.append(np.sum(dicounted_rewards))
    return gains


def compute_win_rate_random(agent, n_runs=10):
    pass


def train_against_self(discount, n_episodes, agent):
    win_rates = []
    losses = []
    for i in tqdm(range(n_episodes)):
        if (i + 1) % 100 == 0:
            win_rates.append(compute_win_rate_random(agent))
