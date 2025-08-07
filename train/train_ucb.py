from envs.workflow_env import WorkflowEnv
from agents.ucb_explorer import UCBExplorer
import matplotlib.pyplot as plt

task = {
    "name": "Summarize and Translate",
    "steps": ["Summarize", "Translate"],
    "tools": [
        {"name": "Tool A", "success_rate": 0.7},
        {"name": "Tool B", "success_rate": 0.5},
        {"name": "Tool C", "success_rate": 0.9}
    ]
}

env = WorkflowEnv(task)
action_size = len(task["tools"])
agent = UCBExplorer(action_size=action_size, c=2.0)

episodes = 100
rewards_per_episode = []

for ep in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.act()
        next_state, reward, done, info = env.step(action)
        agent.update(action, reward)
        total_reward += reward

    rewards_per_episode.append(total_reward)
    print(f"Episode {ep + 1}, Total Reward: {total_reward}")

# Plot
def moving_average(x, window=10):
    import numpy as np
    return np.convolve(x, np.ones(window)/window, mode='valid')

plt.figure(figsize=(10, 5))
plt.plot(rewards_per_episode, alpha=0.4, label="Raw Rewards")
plt.plot(moving_average(rewards_per_episode), label="Moving Avg", linewidth=2)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("UCB Learning Curve")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("ucb_learning_curve.png")
plt.show()
