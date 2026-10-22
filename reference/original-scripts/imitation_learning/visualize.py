import pickle
import matplotlib.pyplot as plt
import numpy as np

BASE_URL = "C:/Users/giannis/Downloads/gail-pytorch-main/gail-pytorch-main/ckpt/"
ENVIRONMENT = "LunarLander-v2"

def load_results(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

# Load results files
results_trpo = load_results(BASE_URL + ENVIRONMENT + " trpo" + "/results.pkl")[1]
results_ppo = load_results(BASE_URL + ENVIRONMENT + " ppo" + "/results.pkl")[1]
results_gae = load_results(BASE_URL + ENVIRONMENT + " gae" + "/results.pkl")[1]

# Create episode indices
episodes_trpo = np.arange(len(results_trpo))
episodes_ppo = np.arange(len(results_ppo))
episodes_gae = np.arange(len(results_gae))

# Plot sets of results
plt.plot(episodes_trpo, results_trpo, label="trpo", color="blue")
plt.fill_between(episodes_trpo, np.array(results_trpo) - np.std(results_trpo), np.array(results_trpo) + np.std(results_trpo), alpha=0.3, color="blue")

plt.plot(episodes_ppo, results_ppo, label="ppo", color="orange")
plt.fill_between(episodes_ppo, np.array(results_ppo) - np.std(results_ppo), np.array(results_ppo) + np.std(results_ppo), alpha=0.3, color="orange")

plt.plot(episodes_gae, results_gae, label="gae", color="green")
plt.fill_between(episodes_gae, np.array(results_gae) - np.std(results_gae), np.array(results_gae) + np.std(results_gae), alpha=0.3, color="green")

plt.xlabel("Iterations")
plt.ylabel("Reward")
plt.title("GAIL Training Performance - " + ENVIRONMENT)
plt.legend()
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.show()
