import os
import torch
import gym
from models.nets import Expert  # Ensure this matches your project structure

# Define paths and device
env_name = "CartPole-v1"
ckpt_policy_path = os.path.join("C:/Users/giann/Desktop/gail-pytorch-main/ckpt", env_name, "policy.ckpt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create the Gym environment
env = gym.make(env_name)

# Determine state and action dimensions
state_dim = env.observation_space.shape[0]
discrete = True  # CartPole-v1 has a discrete action space
action_dim = env.action_space.n

# Load configuration if needed for model instantiation
# For simplicity, assuming expert_config or default parameters
expert_config = {}  # Replace with actual configuration parameters if available

# Initialize the Expert model (policy network) and load weights
policy_model = Expert(state_dim, action_dim, discrete, **expert_config).to(device)
policy_model.pi.load_state_dict(torch.load(ckpt_policy_path, map_location=device))
policy_model.eval()  # Set model to evaluation mode

# Run an episode using the loaded policy
state = env.reset()
done = False
total_reward = 0

while not done:
    # Convert state to tensor and send to device
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Get action probabilities or values from the policy network
        action_logits = policy_model.pi(state_tensor)
    
    # For a discrete action space like CartPole, select the best action
    action = torch.argmax(action_logits, dim=1).item()
    
    # Take a step in the environment
    state, reward, done, info = env.step(action)
    total_reward += reward
    
    # Optionally render the environment
    env.render()

env.close()
print(f"Total reward achieved: {total_reward}")
