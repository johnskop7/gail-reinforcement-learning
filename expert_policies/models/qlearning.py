import numpy as np
import torch

from torch.nn import Module

if torch.cuda.is_available():
    from torch.cuda import FloatTensor
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    from torch import FloatTensor


class QLearning(Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        train_config=None
    ) -> None:
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.train_config = train_config

        # Extract training config parameters
        self.lr = train_config["lr"]
        self.gamma = train_config["discount"]
        self.epsilon = train_config["epsilon"]
        self.epsilon_decay = train_config.get("epsilon_decay", 0.99)
        self.epsilon_min = train_config.get("epsilon_min", 0.01)
        self.num_episodes = train_config["num_episodes"]
        self.max_steps = train_config["max_steps"]

        # Q-table initialized to zeros
        self.q_table = np.zeros((state_dim, action_dim))

    def act(self, state):
        """ Select an action using an epsilon-greedy policy. """
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_dim)  # Explore: random action
        else:
            return np.argmax(self.q_table[state])  # Exploit: best known action

    def train(self, env, render=False):
        """ Train the agent using Q-learning. """
        rewards_per_episode = []

        for episode in range(self.num_episodes):
            state = int(env.reset())  # Ensure state is an integer
            total_reward = 0

            for step in range(self.max_steps):
                action = self.act(state)

                next_state, reward, done, _ = env.step(action)
                next_state = int(next_state)  # Convert next_state to integer

                # Q-learning update rule
                best_next_action = np.argmax(self.q_table[next_state])
                self.q_table[state, action] += self.lr * (
                    reward + self.gamma * self.q_table[next_state, best_next_action] - self.q_table[state, action]
                )

                state = next_state
                total_reward += reward

                if render:
                    env.render()

                if done:
                    break

            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            rewards_per_episode.append(total_reward)
            print(f"Episode {episode+1}/{self.num_episodes}, Total Reward: {total_reward}")

        return rewards_per_episode
    
    def load_q_table(self, path):
        """Load Q-table from file"""
        try:
            self.q_table = torch.load(path)
            print(f"Loaded Q-table from {path}")
            return self.q_table
        except FileNotFoundError:
            print(f"Q-table file {path} not found. Initializing new Q-table.")
            self.q_table = np.zeros((self.state_dim, self.action_dim))


    def save_q_table(self, path):
        """Save Q-table to file"""
        torch.save(self.q_table, path)
        print(f"Q-table saved to {path}")
