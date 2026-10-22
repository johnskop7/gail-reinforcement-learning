import os
import torch
import gym
import numpy as np
from models.nets import Expert  

env_name = "BipedalWalker-v3"
ckpt_policy_path = "C:/Users/giann/Desktop/neurofuzzy_control_semester_project/gail/gail/ckpt/BipedalWalker-v3 trpo/policy.ckpt"  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make(env_name)

state_dim = env.observation_space.shape[0]  
action_dim = env.action_space.shape[0]      
discrete = False 

expert_config = {}  


policy_model = Expert(state_dim, action_dim, discrete, **expert_config).to(device)
policy_model.pi.load_state_dict(
    torch.load(ckpt_policy_path, map_location=device), 
    strict=False
)
policy_model.eval()  

state = env.reset()
done = False
total_reward = 0.0

print("Running GAIL-trained policy on BipedalWalker...")
while not done:
    # Convert the state to a PyTorch tensor and move it to the appropriate device
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

    with torch.no_grad():
        # Get the action distribution from the policy
        dist = policy_model.pi(state_tensor)

        # Option 1: Sample an action from the distribution (stochastic policy)
        action_tensor = dist.sample()
        
        # Option 2: Use the mean of the distribution (deterministic policy)
        # action_tensor = dist.mean

        # Convert the action to a NumPy array and ensure it's in the correct range
        action = action_tensor.squeeze(0).cpu().numpy()
        action = np.clip(action, -1, 1)  # Ensure the action is in the valid range [-1, 1]

    # Take a step in the environment using the selected action
    state, reward, done, info = env.step(action)
    total_reward += reward

    # Render the environment (optional, comment out if running headless)
    env.render()

# Close the environment
env.close()
print(f"Total reward achieved: {total_reward}")
