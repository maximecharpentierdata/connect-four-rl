from copy import deepcopy
from typing import List, Tuple

import numpy as np
from tqdm.notebook import tqdm

import constants
from agents.agent import Agent
from agents.deep_v_agent import DeepVAgent
from agents.random_agent import RandomAgent
from connect_four_env.connect_four_env import ConnectFourGymEnv
from game.episode import run_episode, run_episode_against_self
import matplotlib.pyplot as plt
import pandas as pd


def compute_gain_from_rewards(rewards: List[int], discount: float = 1.0) -> np.ndarray:
    gains = []
    for step in range(len(rewards)):
        dicounted_rewards = [rewards[i] * discount ** (i - step) for i in range(step, len(rewards))]
        gains.append(np.sum(dicounted_rewards))
    return np.array(gains)


def win_rate_vs_opponent(agent: Agent, opponent: Agent, env: ConnectFourGymEnv, n_runs: int = 10):
    n_wins = 0
    index_agent = int(agent.player_number != constants.PLAYER1)
    agent1 = [agent, opponent][index_agent]
    agent2 = [agent, opponent][1 - index_agent]
    for _ in range(n_runs):
        _, rewards = run_episode(agent1, agent2, env, keep_states=True, for_evaluation=True)
        n_wins += rewards[index_agent][-1] == constants.WINNER_REWARD  # does not count draws
    return n_wins / n_runs


def make_opponent(agent: DeepVAgent):
    opponent = deepcopy(agent)
    opponent.stochastic = True
    return opponent


def evaluate_agent(
    agent: Agent,
    opponents: List[Agent],
    env: ConnectFourGymEnv,
    win_rates: List[List[float]],
    n_test_runs: int,
):
    for i, opponent in enumerate(opponents):
        win_rates[i].append(win_rate_vs_opponent(agent, opponent, env, n_test_runs))

    for i in range(len(opponents), len(win_rates)):
        win_rates[i].append(0)


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


def plot_win_rates(win_rates: List[List[float]], losses: List[float], path="progress.png"):
    fig, ax = plt.subplots(1, 2, figsize=(15, 8))

    fig.patch.set_facecolor("#f2f2f2")

    for win_rate in win_rates:
        ax[0].plot(win_rate)
    ax[0].set_title("Win rate against itself at different stages of training")

    pd.DataFrame(losses).rolling(500).mean().plot(ax=ax[1])
    ax[1].set_title("Loss")

    fig.savefig(path)
    plt.close(fig)


def train_both_agents(
    discount: float,
    n_episodes: int,
    agent1: DeepVAgent,
    agent2: DeepVAgent,
    n_test_runs: int = 10,
    num_opponents: int = 5,
    interval_test: int = 100,
):
    win_rates_1, losses_1 = [[] for _ in range(num_opponents)], []
    win_rates_2, losses_2 = [[] for _ in range(num_opponents)], []
    env = ConnectFourGymEnv()
    opponents_1 = [RandomAgent(constants.PLAYER2, env.board.shape)]
    opponents_2 = [RandomAgent(constants.PLAYER1, env.board.shape)]

    period_change_opponent = (n_episodes // num_opponents) + 1

    for i in tqdm(range(n_episodes)):

        if i > 0 and i % period_change_opponent == 0:
            opponents_1.append(make_opponent(agent2))
            opponents_2.append(make_opponent(agent1))

        if (i + 1) % interval_test == 0 or i == 0:
            evaluate_agent(agent1, opponents_1, env, win_rates_1, n_test_runs)
            evaluate_agent(agent2, opponents_2, env, win_rates_2, n_test_runs)
            if i > 0:
                plot_win_rates(win_rates_1, losses_1, "progress_1.png")
                plot_win_rates(win_rates_2, losses_2, "progress_2.png")

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
