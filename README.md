# GAIL: Learning Control Policies from Experts

**Reinforcement learning and generative adversarial imitation learning in PyTorch.**

Semester project for **Neurofuzzy Control and Applications**, School of Electrical and Computer Engineering, **National Technical University of Athens (NTUA)**. Author: **Ioannis Litsos** · February 2025.

This project investigates how **Generative Adversarial Imitation Learning (GAIL)** learns control behavior from expert policies trained with **TRPO**, **PPO**, and a **GAE-based policy optimization implementation**. The main study compares expert and imitation learning behavior across four control tasks, examining reward progression, convergence, and the influence of expert quality and stability.

[Read the final report](docs/report.pdf) · [Setup and commands](docs/setup.md) · [Checkpoint inventory](docs/checkpoints.md) · [Implementation notes](docs/implementation-notes.md)

## Experimental scope

| Environment | Control task | Action space | Main comparisons |
| --- | --- | --- | --- |
| `CartPole-v1` | Balance a pole on a moving cart | Discrete | TRPO / PPO / GAE-based experts and their GAIL imitators |
| `Pendulum-v0` | Swing up and stabilize a pendulum | Continuous | TRPO / PPO / GAE-based experts and their GAIL imitators |
| `BipedalWalker-v3` | Coordinate bipedal locomotion | Continuous | TRPO / PPO / GAE-based experts and their GAIL imitators |
| `LunarLander-v2` | Control a spacecraft landing | Discrete | TRPO / PPO / GAE-based experts and their GAIL imitators |

The source also includes policy gradient, actor-critic, and tabular Q-learning implementations, extra environment configurations, and a LunarLander run labeled `lambda=0`. These additional files are retained separately from the four-task study's scope. A configuration file alone does not mean a trained model is present or a task is supported end to end.

## Learning workflow

1. **Train an expert** using an RL algorithm and the environment's reward.
2. **Collect expert state-action samples** with the saved policy.
3. **Train GAIL** by alternating discriminator updates and policy optimization with a learned signal.
4. **Evaluate and compare** saved expert/GAIL reward traces and rendered rollouts.

The GAIL model contains a policy network, value network, and discriminator. The code uses categorical distributions for discrete actions and Gaussian distributions for continuous actions. The `gae` label denotes the supplied GAE-based optimizer, rather than claiming GAE is inherently a separate RL algorithm.

## Repository layout

| Directory | Contents |
| --- | --- |
| [`expert_policies/`](expert_policies/) | Expert training/evaluation scripts, six algorithm implementations, configuration, supplied checkpoints and reward logs |
| [`imitation_learning/`](imitation_learning/) | Main GAIL implementation, expert variants, imitation checkpoints, validation scripts, and plotting |
| [`docs/`](docs/) | Final report, setup guide, experiment/checkpoint inventory, and packaging notes |
| [`reference/`](reference/) | Earlier GAIL variant, original versions of adjusted scripts, and standalone validation experiment |
| [`third_party/`](third_party/) | Supplied third-party license and attribution |

Model weights and reward logs are retained with their corresponding implementations so the existing loader conventions remain understandable. Duplicate copies used by the original expert/GAIL workflows are deliberately preserved. Bytecode caches are excluded.

## Quick start

The scripts use **legacy OpenAI Gym**, not the Gymnasium API. They expect `reset()` to return an observation and `step()` to return four values. A compatible historical environment is required; installing the latest libraries is not a verified reproduction of the original runtime. See [setup](docs/setup.md) before running.

From the repository root, in a compatible Python environment:

```bash
python -m pip install -r requirements.txt
python expert_policies/run.py --env_name CartPole-v1 --model_name ppo --num_episodes 5 --render False
```

To train a new GAIL run from the bundled TRPO expert:

```bash
python imitation_learning/train.py --env_name CartPole-v1 --expert_name trpo --run_name cartpole-trpo-new
```

This writes to a new folder beneath `imitation_learning/ckpt/`. Existing output folders are rejected. Training can take substantial time; it has not been rerun during repository organization.

## Report and interpretation

The [final report](docs/report.pdf) develops the RL/GAIL background, describes the experiments, and discusses the relationship between expert behavior, task complexity, and imitation performance. Its conclusions are retained as the original study's analysis, without newly generated benchmark claims.

The plotting scripts shade rewards using the standard deviation of the plotted trace. Those bands are not confidence intervals across independent random seeds. See [implementation notes](docs/implementation-notes.md) for other reproduction limits.

## Attribution

The code builds on Hyungcheol Noh's GAIL / reinforcement-learning implementations. The supplied reference README and MIT license are retained; see [third-party attribution](third_party/README.md). This repository documents the coursework adaptation and experiments, rather than claiming the foundational GAIL implementation as original work.

## Validation

Python syntax, JSON configuration files, relative links in the new documentation, checkpoint file inventory, copied-file hashes, and ZIP integrity were checked. Algorithm/model files and binary artifacts remain unchanged. Packaging edits are limited to script paths and selection of expert/output folders. No training, checkpoint deserialization, or simulator rollout was performed.
