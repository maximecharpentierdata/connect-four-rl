from typing import List

import matplotlib.pyplot as plt
import pandas as pd


def plot_win_rates(
    win_rates: List[List[float]],
    losses: List[float],
    interval_test: int,
    path="progress.png",
):
    fig, ax = plt.subplots(1, 2, figsize=(15, 8))

    fig.patch.set_facecolor("#f2f2f2")
    num_tests = len(win_rates[0])
    for i, win_rate in enumerate(win_rates):
        if len(win_rate) <= 1:
            continue

        start = num_tests - len(win_rate)
        if i == 0:
            label = "Random Agent"
        else:
            label = f"Opponent trained {start * interval_test} episode"
        ax[0].plot(
            [i * interval_test for i in range(start, num_tests)], win_rate, label=label
        )
    ax[0].set_title("Win rate against opponent at different stages of training")
    ax[0].set_xlabel("Episodes")
    ax[0].set_ylim(0, 1)
    ax[0].legend()

    pd.DataFrame(losses).rolling(500).mean().plot(ax=ax[1])
    ax[1].set_title("Loss")
    ax[1].set_xlabel("Episodes")
    ax[1].get_legend().remove()

    fig.savefig(path)
    plt.close(fig)
