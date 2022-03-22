import numpy as np
import pandas as pd

from agents.deep_v_agent import DeepVAgent
from agents.random_agent import RandomAgent
from game.episode import run_episode


def make_tournament(agents, env, agents_names=None):
    """
    Run a tournament between all agents.
    """

    results = [[0 for _ in range(len(agents))] for _ in range(len(agents))]

    for agent1_idx, agent1 in enumerate(agents):
        for agent2_idx, agent2 in enumerate(agents):
            if agent1_idx == agent2_idx:
                continue
            result = run_episode(
                agent1, agent2, env, keep_states=False, for_evaluation=True
            )
            results[agent1_idx][agent2_idx] += result

    wins_and_losses = []
    for agent_idx, _ in enumerate(agents):
        wins = sum(np.asarray(results[agent_idx]) == 1) + sum(
            [results[i][agent_idx] == -1 for i in range(len(agents))]
        )
        losses = sum(np.asarray(results[agent_idx]) == -1) + sum(
            [results[i][agent_idx] == 1 for i in range(len(agents))]
        )
        ties = sum(np.asarray(results[agent_idx]) == 0) + sum(
            [results[i][agent_idx] == 0 for i in range(len(agents))]
        )

        wins_and_losses.append((wins, losses, ties))

    wins_losses_df = pd.DataFrame(
        np.array(wins_and_losses),
        columns=["wins", "losses", "ties"],
        index=agents_names,
    ).sort_values(by=["wins", "ties"], ascending=False)

    return results, wins_losses_df


def make_tournament_from_files(agents_paths, env, agents_names=None, add_random=False):

    agents = []
    for agent_path in agents_paths:
        agent = DeepVAgent(board_shape=env.board.shape)
        agent.load(agent_path)
        agents.append(agent)

    if add_random:
        agents.append(RandomAgent())
        if agents_names is not None:
            agents_names.append("random")

    return make_tournament(agents, env, agents_names)
