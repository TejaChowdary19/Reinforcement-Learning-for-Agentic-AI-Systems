from envs.workflow_env import WorkflowEnv
from agents.ppo_agent import PPOAgent
from train.transfer_tasks import task_b
import matplotlib.pyplot as plt

def train(agent, tag):
    env = WorkflowEnv(task_b)
    rewards = []

    for ep in range(100):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = agent.act(state)
            state, reward, done, _ = env.step(action)
            agent.store_reward(reward)
            total_reward += reward
        agent.learn()
        rewards.append(total_reward)
        print(f"[{tag}] Episode {ep+1}, Reward: {total_reward}")
    return rewards

# Train from scratch
agent_scratch = PPOAgent(1, len(task_b["tools"]))
rewards_scratch = train(agent_scratch, "Scratch")

# Train with transfer
agent_transfer = PPOAgent(1, len(task_b["tools"]))
agent_transfer.load_policy("ppo_task_a.pth")  # <-- Transfer weights
rewards_transfer = train(agent_transfer, "Transfer")

# Plot comparison
def moving_avg(x, w=10):
    import numpy as np
    return np.convolve(x, np.ones(w)/w, mode='valid')

plt.plot(rewards_scratch, alpha=0.3, label="Raw - Scratch")
plt.plot(rewards_transfer, alpha=0.3, label="Raw - Transfer")
plt.plot(moving_avg(rewards_scratch), label="Avg - Scratch", linewidth=2)
plt.plot(moving_avg(rewards_transfer), label="Avg - Transfer", linewidth=2)
plt.title("Meta-Learning via Transfer (PPO)")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("ppo_transfer_comparison.png")
plt.show()
