import os
from pathlib import Path
import json
import pickle
import argparse

import torch
import gym

from models.pg import PolicyGradient
from models.ac import ActorCritic
from models.trpo import TRPO
from models.gae import GAE
from models.ppo import PPO
from models.qlearning import QLearning  # Add import for QLearning


def main(env_name, model_name):
    ckpt_path = str(Path(__file__).resolve().parent / "ckpts")
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)

    if model_name not in ["pg", "ac", "trpo", "gae", "ppo", "qlearning"]:
        print("The model name is wrong!")
        return

    if env_name not in ["CartPole-v1", "Pendulum-v0", "BipedalWalker-v3", "LunarLander-v2", "MountainCarContinuous-v0", "MountainCar-v0", "Acrobot-v1", "FrozenLake-v0", "CarRacing-v0", "Hopper-v3"]:
        print("The environment name is wrong!")
        return

    ckpt_path = os.path.join(ckpt_path, model_name)
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)

    ckpt_path = os.path.join(ckpt_path, env_name)
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)

    with open(Path(__file__).resolve().parent / "config.json") as f:
        config = json.load(f)[env_name][model_name]

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

        
    if env_name in ["CartPole-v1", "LunarLander-v2", "MountainCar-v0", "Acrobot-v1", "FrozenLake-v0"]:
        discrete = True
        action_dim = env.action_space.n
    else:
        discrete = False
        action_dim = env.action_space.shape[0]

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if model_name == "pg":
        model = PolicyGradient(
            state_dim, action_dim, discrete, **config
        ).to(device)
    elif model_name == "ac":
        model = ActorCritic(
            state_dim, action_dim, discrete, **config
        ).to(device)
    elif model_name == "trpo":
        model = TRPO(
            state_dim, action_dim, discrete, **config
        ).to(device)
    elif model_name == "gae":
        model = GAE(
            state_dim, action_dim, discrete, **config
        ).to(device)
    elif model_name == "ppo":
        model = PPO(
            state_dim, action_dim, discrete, **config
        ).to(device)
    elif model_name == "qlearning":  # Use QLearning model
        # qlearning_config = config[env_name][model_name]["train_config"]
        model = QLearning(
            state_dim, action_dim, **config
        ).to(device)


    results = model.train(env)

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
    if hasattr(model, "q_table"):  # For Q-learning, save the Q-table
        torch.save(
            model.q_table, os.path.join(ckpt_path, "q_table.ckpt")
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env_name",
        type=str,
        default="CartPole-v1",
        help="Type the environment name to run. \
            The possible environments are \
            [CartPole-v1, Pendulum-v0, BipedalWalker-v3, LunarLander-v2, MountainCarContinuous-v0, MountainCar-v0, Acrobot-v1, FrozenLake-v0, CarRacing-v0, Hopper-v3]"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="pg",
        help="Type the model name to train. \
            The possible models are [pg, ac, trpo, gae, ppo, qlearning]"
    )
    args = parser.parse_args()

    main(**vars(args))
