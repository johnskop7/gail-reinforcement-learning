import os
import torch
import gym
import numpy as np
from models.nets import Expert  # Ensure this matches your project structure

# Define paths and device
env_name = "Pendulum-v0"
ckpt_policy_path = os.path.join("C:/Users/giann/Desktop/gail-pytorch-main/ckpt", env_name, "policy.ckpt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create the Gym environment
env = gym.make(env_name)

# Determine state and action dimensions
state_dim = env.observation_space.shape[0]
discrete = False  # Pendulum-v0 has a continuous action space
action_dim = env.action_space.shape[0]

# Load configuration if needed for model instantiation
expert_config = {}  # Replace with actual configuration parameters if available

# Initialize the Expert model (policy network) and load weights
policy_model = Expert(state_dim, action_dim, discrete, **expert_config).to(device)
policy_model.pi.load_state_dict(torch.load(ckpt_policy_path, map_location=device), strict=False)
policy_model.eval()  # Set model to evaluation mode

# Run an episode using the loaded policy
state = env.reset()
done = False
total_reward = 0

while not done:
    # Convert state to tensor and send to device
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # For a continuous action space, your model should output
        # something like a mean (and optionally std) for a distribution 
        # or direct action values. Let's assume 'action_logits' is 
        # actually the raw continuous action from your policy.
        # Adapt accordingly if your network structure differs.
        dist = policy_model.pi(state_tensor)
        action_tensor = dist.sample()               # draw a sample from the distribution
        action = action_tensor[0].cpu().numpy()  
        
        # Pendulum expects a shape (1,) and range [-2, 2].
        # You can clamp or just rely on your policy being well-trained:
        # action_numpy = np.clip(action_numpy, -2.0, 2.0)
        
    # Step the environment with a continuous action
    state, reward, done, info = env.step(action)
    total_reward += reward
    
    # Optionally render the environment
    env.render()

env.close()
print(f"Total reward achieved: {total_reward}")
