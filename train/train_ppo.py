from envs.workflow_env import WorkflowEnv
from agents.ppo_agent import PPOAgent
import matplotlib.pyplot as plt

# Define the same workflow as before
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
state_size = 1
action_size = len(task["tools"])
agent = PPOAgent(state_dim=state_size, action_dim=action_size)

episodes = 100
rewards_per_episode = []

for ep in range(episodes):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        agent.store_reward(reward)
        state = next_state
        total_reward += reward

    agent.learn()
    rewards_per_episode.append(total_reward)
    print(f"Episode {ep + 1}, Total Reward: {total_reward}")

# --- Moving Average and Plotting ---
def moving_average(x, window=10):
    import numpy as np
    return np.convolve(x, np.ones(window)/window, mode='valid')

plt.figure(figsize=(10, 5))
plt.plot(rewards_per_episode, alpha=0.4, label="Raw Rewards")
plt.plot(moving_average(rewards_per_episode, window=10), label="Moving Avg", linewidth=2)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("PPO Learning Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("ppo_learning_curve.png")
plt.show()
