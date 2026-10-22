import os
import torch
import gym
import numpy as np
from models.nets import Expert 

env_name = "LunarLander-v2"
ckpt_policy_path ="C:/Users/giann/Desktop/neurofuzzy_control_semester_project/gail/gail/ckpt/LunarLander-v2 gae/policy.ckpt"  


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make(env_name)

state_dim = env.observation_space.shape[0]  
if isinstance(env.action_space, gym.spaces.Discrete):
    action_dim = env.action_space.n  
    discrete = True  
else:
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

print("Running GAIL-trained policy on LunarLander...")
while not done:
    # Convert the state to a PyTorch tensor and move it to the appropriate device
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

    with torch.no_grad():
        dist = policy_model.pi(state_tensor)

        # Choose action based on environment type
        if discrete:
            action_tensor = dist.probs.argmax(dim=-1)
            action = action_tensor.cpu().numpy().item()
        else:
            action_tensor = dist.mean
            action = action_tensor.squeeze(0).cpu().numpy()

    # Take a step in the environment using the selected action
    state, reward, done, info = env.step(action)
    total_reward += reward

    # Render the environment (optional, comment out if running headless)
    env.render()

# Close the environment
env.close()
print(f"Total reward achieved: {total_reward}")
