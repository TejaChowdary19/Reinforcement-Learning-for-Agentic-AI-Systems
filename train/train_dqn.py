import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.workflow_env import WorkflowEnv
from agents.dqn_agent import DQNAgent
import numpy as np

# Sample task setup
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
state_size = 1  # step number
action_size = len(task["tools"])
agent = DQNAgent(state_size, action_size)

episodes = 100
rewards_per_episode = []

for ep in range(episodes):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        agent.replay()
        state = next_state
        total_reward += reward

    rewards_per_episode.append(total_reward)
    print(f"Episode {ep+1}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.3f}")

def moving_average(x, window=10):
    return np.convolve(x, np.ones(window)/window, mode='valid')

# Plot reward per episode
plt.figure(figsize=(10, 5))
plt.plot(rewards_per_episode, alpha=0.4, label="Raw Rewards")
plt.plot(moving_average(rewards_per_episode, window=10), label="Moving Average", linewidth=2)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("DQN Learning Curve with Moving Average")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("dqn_learning_curve.png")
plt.show()