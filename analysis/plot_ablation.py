from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, PercentFormatter
import seaborn as sns

from utils import get_df


# Define env and algo names
ENV_LIST = [
    "ant_omni",
    "anttrap_omni",
    "humanoid_omni",
    "walker2d_uni",
    "halfcheetah_uni",
    "ant_uni",
    "humanoid_uni",
]
ENV_DICT = {
    "ant_omni": "Ant Omni",
    "anttrap_omni": "AntTrap Omni",
    "humanoid_omni": "Humanoid Omni",
    "walker2d_uni": "Walker Uni",
    "halfcheetah_uni": "HalfCheetah Uni",
    "ant_uni": "Ant Uni",
    "humanoid_uni": "Humanoid Uni",
}

ALGO_LIST = [
    "dcrl_me",
    "dcg_me",
    "ablation_actor",
    "ablation_ai",
]
ALGO_DICT = {
    "dcrl_me": "DCRL-MAP-Elites",
    "dcg_me": "DCG-MAP-Elites",
    "ablation_actor": "Ablation Actor",
    "ablation_ai": "Ablation AI",
}

XLABEL = "Evaluations"


def customize_axis(ax):
    # Remove spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Remove ticks
    # ax.tick_params(axis="y", length=0)

    # Add grid
    ax.grid(which="major", axis="y", color="0.9")
    return ax


def plot(df):
    # Create subplots
    fig, axes = plt.subplots(nrows=len(ENV_LIST), ncols=3, sharex=True, figsize=(15, 18))

    for row, env in enumerate(ENV_LIST):
        print(env)

        # Create formatter
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((0, 0))

        # Set title for the row
        axes[row, 0].set_ylabel(ENV_DICT[env])

        # Get df for the current env
        df_plot = df[df["env"] == env]

        # QD score
        axes[row, 0].yaxis.set_major_formatter(formatter)

        sns.lineplot(
            df_plot,
            x="num_evaluations",
            y="qd_score",
            hue="algo",
            hue_order=ALGO_LIST,
            estimator=np.median,
            errorbar=lambda x: (np.quantile(x, 0.25), np.quantile(x, 0.75)),
            legend=False,
            ax=axes[row, 0],
        )

        if row == 0:
            axes[row, 0].set_title("QD score")

        # Customize axis
        customize_axis(axes[row, 0])

        # Coverage
        axes[row, 1].set_ylim(0., 1.05)
        axes[row, 1].yaxis.set_major_formatter(PercentFormatter(1))

        sns.lineplot(
            df_plot,
            x="num_evaluations",
            y="coverage",
            hue="algo",
            hue_order=ALGO_LIST,
            estimator=np.median,
            errorbar=lambda x: (np.quantile(x, 0.25), np.quantile(x, 0.75)),
            legend=False,
            ax=axes[row, 1],
        )

        if row == 0:
            axes[row, 1].set_title("Coverage")

        # Customize axis
        customize_axis(axes[row, 1])

        # Max fitness
        ax = sns.lineplot(  # store ax for legend
            df_plot,
            x="num_evaluations",
            y="max_fitness",
            hue="algo",
            hue_order=ALGO_LIST,
            estimator=np.median,
            errorbar=lambda x: (np.quantile(x, 0.25), np.quantile(x, 0.75)),
            legend=False,
            ax=axes[row, 2],
        )

        if row == 0:
            axes[row, 2].set_title("Max Fitness")

        # Customize axis
        customize_axis(axes[row, 2])

        # Remove x-axis labels for all but the last row
        if row != len(ENV_LIST) - 1:
            for col in range(3):
                axes[row, col].set_xlabel('')

    # Set x-axis label only for the bottom row
    for col in range(3):
        axes[-1, col].set_xlabel(XLABEL)
        axes[-1, col].xaxis.set_major_formatter(formatter)

    # Remove y-axis labels for the second and third columns
    for row in range(len(ENV_LIST)):
        axes[row, 1].set_ylabel('')
        axes[row, 2].set_ylabel('')

    # Legend
    fig.legend(ax.get_lines(), [ALGO_DICT[algo] for algo in ALGO_LIST], loc="lower center", bbox_to_anchor=(0.5, -0.02), ncols=len(ALGO_LIST), frameon=False)

    # Aesthetic
    fig.align_ylabels(axes)
    fig.tight_layout()

    return fig


if __name__ == "__main__":
    # Avoid type3 fonts in matplotlib, see http://phyletica.org/matplotlib-fonts/
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    plt.rc("font", size=16)

    # Load repertoire
    results_dir = Path("/src/output/")
    df = get_df(results_dir)

    # Filter
    df = df[df["algo"].isin(ALGO_LIST)]
    df = df[df["num_evaluations"] <= 1_000_000]

    # Plot
    fig = plot(df)
    fig.savefig("/src/output/plot_ablation.pdf", bbox_inches="tight")
    plt.close()
