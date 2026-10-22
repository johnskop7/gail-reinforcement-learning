# Expert policy training

Contains `pg`, `ac`, `trpo`, `gae`, `ppo`, and `qlearning` implementations, along with training configuration and saved artifacts.

- `train.py`: train from `config.json`; saves under `ckpts/<algorithm>/<environment>/`.
- `run.py`: evaluate a saved policy and print episode/mean rewards.
- `visualize.py`: compare TRPO, PPO, and GAE-labeled reward traces.
- `models/`: policy/value networks and algorithm implementations.
- `utils/`: flat-parameter and optimization helpers.

The report's expert comparison uses TRPO, PPO, and GAE-labeled runs for four tasks. Extra configurations are retained as exploratory work. See [setup](../docs/setup.md) and [artifact inventory](../docs/checkpoints.md). The training script retains its original overwrite behavior.
