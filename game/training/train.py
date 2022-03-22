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
from game.training.progress_plot import plot_win_rates
from game.win_rates import win_rate_vs_opponent, win_rate_vs_self


def compute_gain_from_rewards(rewards: List[int], discount: float = 1.0) -> np.ndarray:
    gains = []
    for step in range(len(rewards)):
        dicounted_rewards = [
            rewards[i] * discount ** (i - step) for i in range(step, len(rewards))
        ]
        gains.append(np.sum(dicounted_rewards))
    return np.array(gains)


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
    against_self: bool = False,
):
    for i, opponent in enumerate(opponents):
        if not against_self:
            win_rates[i].append(win_rate_vs_opponent(agent, opponent, env, n_test_runs))
        else:
            win_rates[i].append(win_rate_vs_self(agent, opponent, env, n_test_runs))


def train_against_self(
    discount: float,
    n_episodes: int,
    agent: DeepVAgent,
    n_test_runs: int = 10,
    num_opponents: int = 5,
    interval_test: int = 100,
) -> Tuple[List[float], List[float]]:
    win_rates, losses = [[]] + [[0] for _ in range(num_opponents - 1)], []
    env = ConnectFourGymEnv()
    opponents = [RandomAgent(constants.PLAYER1, env.board.shape)]

    period_change_opponent = (n_episodes // num_opponents) + 1

    for i in tqdm(range(n_episodes)):

        if i > 0 and i % period_change_opponent == 0:
            opponents.append(make_opponent(agent))

        if (i + 1) % interval_test == 0 or i == 0:
            evaluate_agent(
                agent, opponents, env, win_rates, n_test_runs, against_self=True
            )
            if i > 0:
                plot_win_rates(win_rates, losses, interval_test, "progress.png")

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
    num_opponents: int = 5,
    interval_test: int = 100,
):
    win_rates_1, losses_1 = [[]] + [[0] for _ in range(num_opponents - 1)], []
    win_rates_2, losses_2 = [[]] + [[0] for _ in range(num_opponents - 1)], []
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
                plot_win_rates(win_rates_1, losses_1, interval_test, "progress_1.png")
                plot_win_rates(win_rates_2, losses_2, interval_test, "progress_2.png")

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
