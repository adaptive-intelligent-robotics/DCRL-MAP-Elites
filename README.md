# Descriptor-Conditioned Reinforcement Learning MAP-Elites

This repository implements:
- **DCG-MAP-Elites**, introduced in [_MAP-Elites with Descriptor-Conditioned Gradients and Archive Distillation into a Single Policy_](https://dl.acm.org/doi/10.1145/3583131.3590503), GECCO 2023 _Best Paper Award_ 🏆.
- **DCRL-MAP-Elites**, introduced in [_Synergizing Quality-Diversity with Descriptor-Conditioned Reinforcement Learning_](https://arxiv.org/abs/2401.08632), ACM TELO Special Issue: Best of GECCO 2023 📝.

All experiments can be reproduced within a containerized environment, ensuring reproducibility 🔬.

⚠️ If you want to use DCRL-MAP-Elites for your own research projects, we recommend using the version available in [QDax](https://github.com/adaptive-intelligent-robotics/QDax), which is optimized for broader applications and regularly maintained for cutting-edge performance.

## Overview

<p align="center">
  <img src="https://github.com/user-attachments/assets/560665e4-ba1e-4dbc-95e2-3e737cbc3908" alt="DCRL-MAP-Elites">
  <br>
</p>

DCRL-MAP-Elites employs a standard Quality-Diversity loop comprising selection, variation, evaluation and addition. Concurrently, transitions generated during the evaluation step are stored in a replay buffer and used to train a descriptor-conditioned actor-critic model from reinforcement learning. Two complementary variation operators are used: a Genetic Algorithm (**GA**) variation operator for diversity and a Policy Gradient (**PG**) variation operator for quality. Additionally, the descriptor-conditioned actor is injected (**AI**) within the population to produce high-quality and diverse solutions.

DCRL-MAP-Elites builds upon PGA-MAP-Elites algorithm and introduces three key contributions:
1. The PG variation operator is enhanced with a descriptor-conditioned critic that reconciles diversity search with gradient-based methods coming from reinforcement learning.
2. During the actor-critic training, the diverse and high-performing policies from the archive are distilled into the generally capable actor, at no additional cost.
3. In turn, this descriptor-conditioned actor is utilized as a generative model to produce diverse solutions, which are then injected into the offspring batch at each generation.

### Baselines

The repository contains the code to run the following algorithms:
- [DCRL-MAP-Elites](https://arxiv.org/abs/2401.08632)
- [DCG-MAP-Elites](https://dl.acm.org/doi/10.1145/3583131.3590503)
- [PGA-MAP-Elites](https://dl.acm.org/doi/10.1145/3449639.3459304)
- [QD-PG](https://dl.acm.org/doi/10.1145/3512290.3528845)
- [MAP-Elites ES](https://dl.acm.org/doi/10.1145/3377930.3390217)
- [MAP-Elites](https://arxiv.org/abs/1504.04909)

and two ablation studies:
- DCRL-MAP-Elites without Actor Injection
- DCRL-MAP-Elites without a Descriptor-Conditioned Actor

## Installation

We provide an Apptainer definition file `apptainer/container.def`, that enables to create a containerized environment in which all the experiments and figures can be reproduced.

First, clone the repository:
```
git clone https://github.com/adaptive-intelligent-robotics/DCRL-MAP-Elites.git
```

Then, go at the root of the repository with `cd DCRL-MAP-Elites/` and build the container:
```bash
apptainer build --fakeroot --force apptainer/container.sif apptainer/container.def
```

Finally, you can shell within the container:
```bash
apptainer shell --bind $(pwd):/src/ --cleanenv --containall --home /tmp/ --no-home --nv --pwd /src/ --workdir apptainer/ apptainer/container.sif
```

Once you have a shell in the container, you can run experiments, see next section.

## Run main experiments

First, follow the previous section to build and shell into a container. Then, to run any algorithms `<algo>`, on any environments `<env>`, use:
```
python main.py env=<env> algo=<algo> seed=$RANDOM num_iterations=4000
```

For example, to run DCRL-MAP-Elites on Ant Omni:
```
python main.py env=ant_omni algo=dcrl_me seed=$RANDOM num_iterations=4000
```

During training, the metrics are logged in the `output/` directory.

The configurations for all algorithms and all environments can be found in the `configs/` directory. Alternatively, they can be modified directly in the command line. For example, to increase `num_critic_training_steps` to 5000 in PGA-MAP-Elites, you can run:
```bash
python main.py env=walker2d_uni algo=pga_me seed=$RANDOM num_iterations=4000 algo.num_critic_training_steps=5000
```

To faciliate the replication of all experiments, you can run the bash script `launch_experiments.sh`. This script will run one seed for each algorithm and each environment. Keep in mind that in the paper, we replicated all experiments with 20 independent seeds, so you would need to run `launch_experiments.sh` 20 times to replicate the results.

## Run reproducibility experiments

The reproducibility experiments load the saved archives from the main experiment (see previous section) and evaluate the expected QD score, expected distance to descriptor and expected max fitness of the populations of the different algorithms.

> :warning: Before running a reproducibility experiment, the main experiment for the corresponding environment and algorithm should be completed.

For example, to evaluate the reproducibility for QD-PG on AntTrap Omni, run:
```bash
python main_reproducibility.py env_name=anttrap_omni algo_name=qd_pg
```

The results will be saved in the `output/reproducibility/` directory.

## Figures

Once all the experiments are completed, any figures from the paper can be replicated with the scripts in the `analysis/` directory.

- Figure 3: `analysis/plot_main.py`
- Figure 4: `analysis/plot_archive.py`
- Figure 5: `analysis/plot_ablation.py`
- Figure 4: `analysis/plot_reproducibility.py`
- Figure 5: `analysis/plot_improvement.py`

## P-values

Once all the experiments are completed, any p-values from the paper can be replicated with the script `analysis/p_values.py`.
