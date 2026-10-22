import os
import torch
import gym
from models.nets import Expert 


env_name = "CartPole-v1"
ckpt_policy_path = "C:/Users/giann/Desktop/neurofuzzy_control_semester_project/gail/gail/ckpt/CartPole-v1 trpo/policy.ckpt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make(env_name)

state_dim = env.observation_space.shape[0]
discrete = True  
action_dim = env.action_space.n


expert_config = {}  

policy_model = Expert(state_dim, action_dim, discrete, **expert_config).to(device)
policy_model.pi.load_state_dict(torch.load(ckpt_policy_path, map_location=device))
policy_model.eval()  

state = env.reset()
done = False
total_reward = 0

while not done:
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    
    with torch.no_grad():
        action_logits = policy_model.pi(state_tensor)
        action_probs = action_logits.probs  # This is a tensor of probabilities.
    # For a discrete action space like CartPole, select the best action
    action = torch.argmax(action_probs, dim=1).item()
    
    # Take a step in the environment
    state, reward, done, info = env.step(action)
    total_reward += reward
    
    # Optionally render the environment
    env.render()

env.close()
print(f"Total reward achieved: {total_reward}")
