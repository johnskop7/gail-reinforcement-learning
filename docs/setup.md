# Setup and execution

## Runtime compatibility

The supplied source imports `gym`, uses observation-only `env.reset()`, calls four-value `env.step()`, and renders with `env.render()` without a construction-time render mode. Preserve a compatible legacy Gym runtime when reproducing the original work. Gymnasium is not a drop-in replacement for these scripts, and replacing environment IDs alone does not update the API or validate saved checkpoints.

Exact original Python, NumPy, Gym, and PyTorch versions were not recorded in the supplied requirements. Root `requirements.txt` lists imported packages and is not a reproducibility lockfile. No current-version combination has been certified here. Box2D tasks additionally require the matching Box2D/native environment dependencies; those are not installed by this package. Rendering requires a display-capable setup.

Create and activate an isolated Python environment suited to the original dependencies, then install:

```bash
python -m pip install -r requirements.txt
```

The original requirements omit PyTorch; the root list adds it. The source includes legacy Torch defaults and checkpoint-loading calls that may also need compatibility review in a newer runtime.

## Evaluate a bundled expert

Run from the repository root:

```bash
python expert_policies/run.py --env_name CartPole-v1 --model_name ppo --num_episodes 5 --render False
```

`run.py` reads `expert_policies/ckpts/<algorithm>/<environment>/`. Its `--render` parameter is a string: use exactly `True` or `False`. Algorithms with an existing policy/configuration can be selected; consult [the inventory](checkpoints.md). Training and evaluation allowlists are not identical, so configuration availability does not guarantee evaluation support for every entry.

## Train an expert

```bash
python expert_policies/train.py --env_name CartPole-v1 --model_name ppo
```

Parameters come from `expert_policies/config.json`. The original expert trainer writes to `expert_policies/ckpts/<algorithm>/<environment>/` and can overwrite a saved run. Preserve a separate copy of that run before retraining. Learning algorithms and this original save convention are retained.

## Train GAIL from a selected expert

```bash
python imitation_learning/train.py --env_name CartPole-v1 --expert_name trpo --run_name cartpole-trpo-new
python imitation_learning/train.py --env_name LunarLander-v2 --expert_name ppo --run_name lunarlander-ppo-new
```

Expert selection uses the existing `imitation_learning/experts/<environment> <expert_name>/` folders. Valid expert choices are `trpo`, `ppo`, and `gae`. Configuration comes from `imitation_learning/config.json`.

Outputs go to `imitation_learning/ckpt/<run_name>/`. If `--run_name` is omitted, the folder name is `<environment> <expert_name> new`. Existing output directories are rejected to protect supplied artifacts. A new folder can receive `model_config.json`, `results.pkl`, policy/value checkpoints, and a discriminator checkpoint.

The supported main-study examples are the four environments listed in the README. The inherited FrozenLake branch is incomplete: the source attempts to load a Q-table into a neural policy and lacks a complete discrete-observation path. It is retained as experimental code, not a working GAIL example.

## Validate saved imitation policies

```bash
python imitation_learning/validation.py
python imitation_learning/validation_pendulum.py
python imitation_learning/validation_bipedal_walker.py
python imitation_learning/validation_lunar_lander.py
```

Each original validation script selects a fixed saved run near its top: CartPole/TRPO, Pendulum/TRPO, BipedalWalker/TRPO, and LunarLander/GAE respectively. Change its package-relative `ckpt_policy_path` to inspect a different run. These scripts render and terminate after one episode; they are not a multi-seed benchmark. Some use `strict=False` during weight loading, so a successful call alone does not prove every network parameter matched.

## Plot supplied reward logs

```bash
python expert_policies/visualize.py
python imitation_learning/visualize.py
```

Change `ENVIRONMENT` near the top of the chosen script to select the task. The expert plot defaults to BipedalWalker; the GAIL plot defaults to LunarLander. The scripts expect TRPO/PPO/GAE traces to exist for that environment.

The scripts use pickle-format result files. They are preserved as experiment artifacts and were not deserialized during packaging. Their plotted standard-deviation bands describe variation within a trace, not uncertainty from independent repeated training runs.
