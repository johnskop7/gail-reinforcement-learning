# Generative Adversarial Imitation Learning

Main coursework GAIL implementation, taken from the original `gail/gail` folder.

- `models/gail.py`: discriminator/policy/value training loop.
- `models/nets.py`: policy, value, discriminator, and expert-policy networks.
- `train.py`: choose an environment, expert variant, and new output folder.
- `experts/`: bundled expert variants grouped by `<environment> <algorithm>`.
- `ckpt/`: saved imitation runs; each run keeps its configuration and available weights/logs.
- `validation*.py`: single-episode rendered evaluation examples.
- `visualize.py`: compare imitation reward traces by expert source.

Example from the repository root:

```bash
python imitation_learning/train.py --env_name CartPole-v1 --expert_name ppo --run_name cartpole-ppo-new
```

Existing output folders are rejected. The code requires the legacy Gym contract. See [setup](../docs/setup.md) before running and [implementation notes](../docs/implementation-notes.md) for experimental branches and compatibility limits.
