import constants
from agents.agent import Agent
from connect_four_env.connect_four_env import ConnectFourGymEnv
from game.episode import run_episode


def win_rate_vs_opponent(
    agent: Agent, opponent: Agent, env: ConnectFourGymEnv, n_runs: int = 10
):
    n_wins = 0
    index_agent = int(agent.player_number != constants.PLAYER1)
    agent1 = [agent, opponent][index_agent]
    agent2 = [agent, opponent][1 - index_agent]
    for _ in range(n_runs):
        _, rewards, _ = run_episode(
            agent1, agent2, env, keep_states=True, for_evaluation=True
        )
        n_wins += (
            rewards[index_agent][-1] == constants.WINNER_REWARD
        )  # does not count draws
    return n_wins / n_runs


def win_rate_vs_self(
    agent: Agent, opponent: Agent, env: ConnectFourGymEnv, n_runs: int = 10
):
    n_wins = 0
    for i in range(n_runs):
        agent1 = [agent, opponent][i % 2]
        agent2 = [agent, opponent][1 - i % 2]
        _, rewards, _ = run_episode(
            agent1, agent2, env, keep_states=True, for_evaluation=True
        )
        n_wins += rewards[i % 2][-1] == constants.WINNER_REWARD  # does not count draws
    return n_wins / n_runs
