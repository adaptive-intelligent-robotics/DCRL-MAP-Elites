from pathlib import Path
import pickle

import pandas as pd
import jax.numpy as jnp

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import matplotlib.patches as mpatches
import seaborn as sns


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
    "dcrl_me_actor",
    "dcg_me",
    "pga_me",
    "qd_pg",
    "me",
    "me_es",
]
ALGO_DICT = {
    "dcrl_me": "DCRL-MAP-Elites",
    "dcrl_me_actor": "DCRL-MAP-Elites Actor",
    "dcg_me": "DCG-MAP-Elites",
    "pga_me": "PGA-MAP-Elites",
    "qd_pg": "QD-PG",
    "me": "MAP-Elites",
    "me_es": "MAP-Elites-ES",
}


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

        ax = sns.boxplot(
            df_plot,
            x="algo",
            y="qd_score",
            order=ALGO_LIST,
            hue="algo",
            hue_order=ALGO_LIST,
            ax=axes[row, 0],
        )
        ax.set(xticklabels=[])
        ax.tick_params(bottom=False)

        if row == 0:
            axes[row, 0].set_title("Expected QD score")

        # Customize axis
        customize_axis(axes[row, 0])

        # Distance
        ax = sns.boxplot(
            df_plot,
            x="algo",
            y="distance",
            order=ALGO_LIST,
            hue="algo",
            hue_order=ALGO_LIST,
            ax=axes[row, 1],
        )
        ax.set(xticklabels=[])
        ax.tick_params(bottom=False)

        if row == 0:
            axes[row, 1].set_title("Expected Distance to Descriptor")

        # Customize axis
        customize_axis(axes[row, 1])

        # Max Fitness
        ax = sns.boxplot(
            df_plot,
            x="algo",
            y="max_fitness",
            order=ALGO_LIST,
            hue="algo",
            hue_order=ALGO_LIST,
            ax=axes[row, 2],
        )
        ax.set(xticklabels=[])
        ax.tick_params(bottom=False)

        if row == 0:
            axes[row, 2].set_title("Expected Max Fitness")

        # Customize axis
        customize_axis(axes[row, 2])

        # Remove x-axis label 'algo'
        for col in range(3):
            axes[row, col].set_xlabel("")

    # Remove y-axis labels for the second and third columns
    for row in range(len(ENV_LIST)):
        axes[row, 1].set_ylabel("")
        axes[row, 2].set_ylabel("")

    # Legend
    legend_patches = [mpatches.Patch(color=sns.color_palette()[i], label=ALGO_DICT[algo]) for i, algo in enumerate(ALGO_LIST)]
    fig.legend(handles=legend_patches, loc="lower center", bbox_to_anchor=(0.5, -0.04), ncols=4, frameon=False)

    # Aesthetic
    fig.align_ylabels(axes)
    fig.tight_layout()

    return fig


if __name__ == "__main__":
    # Avoid type3 fonts in matplotlib, see http://phyletica.org/matplotlib-fonts/
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    plt.rc("font", size=16)

    # Create the DataFrame
    results_dir = Path("/src/output/reproducibility/")
    df = pd.DataFrame(columns=["env", "algo", "run", "qd_score", "distance", "max_fitness"])

    # Construct reproducibility DataFrame
    reproducibility_dict = {}
    for env_dir in results_dir.iterdir():
        if env_dir.name not in ENV_LIST:
            continue
        for algo_dir in env_dir.iterdir():
            if algo_dir.name not in ALGO_LIST:
                continue
            with open(algo_dir / "repertoire_fitnesses.pickle", "rb") as repertoire_fitnesses_file:
                repertoire_fitnesses = pickle.load(repertoire_fitnesses_file)
            with open(algo_dir / "repertoire_distances.pickle", "rb") as repertoire_distances_file:
                repertoire_distances = pickle.load(repertoire_distances_file)
            entry = pd.DataFrame.from_dict({
                "env": env_dir.name,
                "algo": algo_dir.name,
                "run": jnp.arange(repertoire_fitnesses.shape[0]),
                "qd_score": jnp.nansum(jnp.nanmean(repertoire_fitnesses, axis=1), axis=-1),
                "distance": jnp.nanmean(jnp.nanmean(repertoire_distances, axis=1), axis=-1),
                "max_fitness": jnp.nanmax(jnp.nanmean(repertoire_fitnesses, axis=1), axis=-1),
            })
            df = pd.concat([df, entry], ignore_index=True)

    # Plot
    fig = plot(df)
    fig.savefig("/src/output/plot_reproducibility.pdf", bbox_inches="tight")
    plt.close()
