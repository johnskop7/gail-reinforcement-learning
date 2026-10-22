import os
from pathlib import Path
import json
import pickle
import argparse

import torch
import gym

from models.nets import Expert
from models.gail import GAIL


def main(env_name, expert_name="trpo", run_name=None):
    ckpt_path = str(Path(__file__).resolve().parent / "ckpt")
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)

    if env_name not in ["CartPole-v1", "Pendulum-v0", "BipedalWalker-v3", "LunarLander-v2", "FrozenLake-v0"]:
        print("The environment name is wrong!")
        return

    expert_ckpt_path = str(Path(__file__).resolve().parent / "experts")
    expert_ckpt_path = os.path.join(expert_ckpt_path, env_name + " " + expert_name)

    with open(os.path.join(expert_ckpt_path, "model_config.json")) as f:
        expert_config = json.load(f)

    ckpt_path = os.path.join(ckpt_path, run_name or (env_name + " " + expert_name + " new"))
    if os.path.exists(ckpt_path):
        raise FileExistsError("Choose a new --run_name; output already exists: " + ckpt_path)
    os.makedirs(ckpt_path)

    with open(Path(__file__).resolve().parent / "config.json") as f:
        config = json.load(f)[env_name]

    with open(os.path.join(ckpt_path, "model_config.json"), "w") as f:
        json.dump(config, f, indent=4)
        
    if env_name == "FrozenLake-v0":
        env = gym.make(env_name, is_slippery=False)
    else:
        env = gym.make(env_name)
    env.reset()

    if isinstance(env.observation_space, gym.spaces.Discrete):
        state_dim = env.observation_space.n  # Use number of discrete states
    else:
        state_dim = len(env.observation_space.high)  # For continuous spaces

    if env_name in ["CartPole-v1", "LunarLander-v2", "FrozenLake-v0"]:
        discrete = True
        action_dim = env.action_space.n
    else:
        discrete = False
        action_dim = env.action_space.shape[0]

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    expert = Expert(
        state_dim, action_dim, discrete, **expert_config
    ).to(device)
    if env_name == "FrozenLake-v0":
        expert.pi.load_state_dict(
            torch.load(
                os.path.join(expert_ckpt_path, "q_table.ckpt"),
                map_location=device
            )
        )
    else:
        expert.pi.load_state_dict(
            torch.load(
                os.path.join(expert_ckpt_path, "policy.ckpt"), map_location=device
            )
        )

    model = GAIL(state_dim, action_dim, discrete, config).to(device)

    results = model.train(env, expert)

    env.close()

    with open(os.path.join(ckpt_path, "results.pkl"), "wb") as f:
        pickle.dump(results, f)

    if hasattr(model, "pi"):
        torch.save(
            model.pi.state_dict(), os.path.join(ckpt_path, "policy.ckpt")
        )
    if hasattr(model, "v"):
        torch.save(
            model.v.state_dict(), os.path.join(ckpt_path, "value.ckpt")
        )
    if hasattr(model, "d"):
        torch.save(
            model.d.state_dict(), os.path.join(ckpt_path, "discriminator.ckpt")
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env_name",
        type=str,
        default="BipedalWalker-v3",
        help="Type the environment name to run. \
            The possible environments are \
                [CartPole-v1, Pendulum-v0, BipedalWalker-v3, LunarLander-v2, FrozenLake-v0]",
    )
    parser.add_argument("--expert_name", choices=["trpo", "ppo", "gae"], default="trpo",
                        help="Expert variant under experts/<environment> <variant>")
    parser.add_argument("--run_name", default=None,
                        help="New output folder name under ckpt; existing folders are rejected")
    args = parser.parse_args()

    main(**vars(args))
