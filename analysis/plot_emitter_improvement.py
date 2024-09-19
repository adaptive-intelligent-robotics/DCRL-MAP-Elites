from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
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
    "ant_omni": "   Ant Omni   ",
    "anttrap_omni": "AntTrap Omni",
    "humanoid_omni": "Humanoid Omni",
    "walker2d_uni": "  Walker Uni  ",
    "halfcheetah_uni": "HalfCheetah Uni",
    "ant_uni": "     Ant Uni     ",
    "humanoid_uni": "Humanoid Uni",
}

ALGO_LIST = [
    "dcrl_me",
    "dcg_me",
    "pga_me",
    "ablation_ai",
]
ALGO_DICT = {
    "dcrl_me": "DCRL-MAP-Elites",
    "dcg_me": "DCG-MAP-Elites",
    "pga_me": "PGA-MAP-Elites",
    "qd_pg": "QD-PG",
    "me": "MAP-Elites",
    "me_es": "MAP-Elites-ES",
    "ablation_ai": "Ablation AI",
}

EMITTER_LIST = {
    "dcrl_me": ["ga_improvement", "qpg_ai_improvement", "total_improvement"],
    "dcg_me": ["ga_improvement", "qpg_ai_improvement", "total_improvement"],
    "ablation_ai": ["ga_improvement", "qpg_ai_improvement", "total_improvement"],
    "pga_me": ["ga_improvement", "qpg_ai_improvement", "total_improvement"],
    "qd_pg": ["ga_improvement", "qpg_ai_improvement", "total_improvement"],
    "me": ["ga_improvement", "total_improvement"],
    "me_es": ["es_improvement", "total_improvement"],
}
EMITTER_DICT = {
    "ga_improvement": "GA",
    "qpg_improvement": "PG",
    # "dpg_improvement": "DPG",
    # "es_improvement": "ES",
    "ai_improvement": "AI",
    "qpg_ai_improvement": "PG + AI",
    "total_improvement": "Total",
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
    ncols = len(EMITTER_LIST[ALGO_LIST[0]])
    fig, axes = plt.subplots(nrows=len(ENV_LIST), ncols=ncols, sharex=True, squeeze=False, figsize=(15, 17.5))

    # Create formatter x
    formatter_x = ScalarFormatter(useMathText=True)
    formatter_x.set_scientific(True)
    formatter_x.set_powerlimits((0, 0))

    # Create formatter y
    formatter_y = ScalarFormatter(useMathText=True)
    formatter_y.set_scientific(True)
    formatter_y.set_powerlimits((0, 0))

    for row, env in enumerate(ENV_LIST):
        print(env)

        # Set title for the row
        axes[row, 0].set_ylabel(ENV_DICT[env])

        # Get df for the current env
        df_plot = df[(df["env"] == env) & (df["algo"].isin(ALGO_LIST))]

        # Plot
        for col, emitter in enumerate(EMITTER_LIST[ALGO_LIST[0]]):
            ax = sns.lineplot(
                df_plot,
                x="num_evaluations",
                y=emitter,
                hue="algo",
                hue_order=ALGO_LIST,
                estimator=np.median,
                errorbar=lambda x: (np.quantile(x, 0.25), np.quantile(x, 0.75)),
                legend=False,
                ax=axes[row, col],
            )

            if row == 0:
                axes[row, col].set_title(f"{EMITTER_DICT[emitter]}")

            # Set the x label and formatter for the bottom row
            if row == len(ENV_LIST) - 1:
                axes[row, col].set_xlabel(XLABEL)
                axes[row, col].xaxis.set_major_formatter(formatter_x)
            else:
                axes[row, col].set_xlabel("")

            # Set the y formatter
            axes[row, col].yaxis.set_major_formatter(formatter_y)

            # Customize axis
            customize_axis(axes[row, col])

    # Remove y-axis labels for all but the first column
    for row in range(len(ENV_LIST)):
        for col in range(1, ncols):
            axes[row, col].set_ylabel("")

    # Legend
    fig.legend(ax.get_lines(), [ALGO_DICT[algo] for algo in ALGO_LIST], loc="lower center", bbox_to_anchor=(0.5, -0.02), ncols=len(ALGO_LIST), frameon=False)

    # Aesthetic
    fig.align_ylabels(axes[:, 0])
    fig.tight_layout()

    return fig


if __name__ == "__main__":
    # Avoid type3 fonts in matplotlib, see http://phyletica.org/matplotlib-fonts/
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    plt.rc("font", size=16)

    # Create the DataFrame
    results_dir = Path("/src/output/")
    df = get_df(results_dir)

    # Sum PG and AI emitters
    df["qpg_ai_improvement"] = df["qpg_improvement"] + df["ai_improvement"]

    # Total
    df["total_improvement"] = df["ga_improvement"] + df["qpg_improvement"] + df["ai_improvement"]

    # Get cumulative sum of elites
    for emitter in EMITTER_DICT:
        df[emitter] = df.groupby(['env', 'algo', 'run'])[emitter].cumsum()

    # Filter
    df = df[df["num_evaluations"] <= 1_000_000]

    # Plot
    fig = plot(df)
    fig.savefig("/src/output/plot_emitter_improvement.pdf", bbox_inches="tight")
    plt.close()
