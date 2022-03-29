import os

import numpy as np
import pandas as pd

from agents.deep_v_agent import DeepVAgent
from agents.random_agent import RandomAgent
from game.episode import run_episode
import constants


def make_tournament(agents, env, agents_names=None):
    """
    Run a tournament between all agents.
    """

    results = [[0 for _ in range(len(agents))] for _ in range(len(agents))]

    for agent1_idx, agent1 in enumerate(agents):
        for agent2_idx, agent2 in enumerate(agents):
            if agent1_idx == agent2_idx:
                continue
            agent1_old_stochastic = agent1.stochastic
            agent2_old_stochastic = agent2.stochastic
            agent1.stochastic = False
            agent2.stochastic = False
            result = run_episode(
                agent1, agent2, env, keep_states=False, for_evaluation=True
            )
            results[agent1_idx][agent2_idx] = result
            agent1.stochastic = agent1_old_stochastic
            agent2.stochastic = agent2_old_stochastic

    wins_and_losses = []
    results = np.asarray(results)
    for agent_idx in range(len(agents)):
        wins = sum(results[agent_idx] == constants.PLAYER1) + sum(
            results[:, agent_idx] == constants.PLAYER2
        )
        losses = sum(results[agent_idx] == constants.PLAYER2) + sum(
            results[:, agent_idx] == constants.PLAYER1
        )
        ties = sum(results[agent_idx] == 0) + sum(results[:, agent_idx] == 0) - 2

        wins_and_losses.append((wins, losses, ties))

    wins_losses_df = pd.DataFrame(
        np.array(wins_and_losses),
        columns=["wins", "losses", "ties"],
        index=agents_names,
    ).sort_values(by=["wins", "ties"], ascending=False)

    results_df = pd.DataFrame(results, columns=agents_names, index=agents_names)

    return results_df, wins_losses_df


def make_tournament_from_files(agents_paths, env, agents_names=None, add_random=False):

    agents = []
    for agent_path in agents_paths:
        agent = DeepVAgent(board_shape=env.board.shape)
        agent.load(os.path.join(agent_path, "agent.pt"))
        agents.append(agent)

    if agents_names is None:
        agents_names = [os.path.basename(agent_path) for agent_path in agents_paths]

    if add_random:
        agents.append(RandomAgent())
        if agents_names is not None:
            agents_names.append("random")

    return make_tournament(agents, env, agents_names)
