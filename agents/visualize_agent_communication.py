import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# Create a folder to store plots
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Parameters
episodes = 50
agents = ['PPO', 'DQN', 'UCB', 'Agent0', 'Agent1', 'Agent2']
comm_agents = ['Agent0', 'Agent1', 'Agent2']

# Simulated Rewards
np.random.seed(42)
rewards_data = {agent: np.random.randint(-5, 10, size=episodes).tolist() for agent in agents}

# Simulated Messages Sent
message_counts = {agent: np.random.randint(0, 5, size=episodes).tolist() for agent in comm_agents}


def plot_rewards_comparison(agent_rewards, save_name='all_rewards_plot.png'):
    plt.figure(figsize=(14, 6))
    for agent, rewards in agent_rewards.items():
        plt.plot(rewards, label=f'{agent} Reward', alpha=0.6)
        smooth = pd.Series(rewards).rolling(window=5).mean()
        plt.plot(smooth, linestyle='--', label=f'{agent} (MA)', linewidth=2)

    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward per Episode for All Agents (with Moving Average)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    full_path = os.path.join(RESULTS_DIR, save_name)
    plt.savefig(full_path)
    plt.show()
    print(f"[✔] Reward plot saved to: {full_path}")


def plot_message_counts(message_logs, save_name='message_counts_plot.png'):
    plt.figure(figsize=(12, 5))
    for agent, counts in message_logs.items():
        plt.plot(counts, label=f'{agent} Messages')

    plt.xlabel('Episode')
    plt.ylabel('Message Count')
    plt.title('Messages Sent per Episode')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    full_path = os.path.join(RESULTS_DIR, save_name)
    plt.savefig(full_path)
    plt.show()
    print(f"[✔] Message count plot saved to: {full_path}")


def plot_message_heatmap(message_logs, save_name='message_heatmap.png'):
    data = np.array(list(message_logs.values()))
    plt.figure(figsize=(10, 4))
    plt.imshow(data, aspect='auto', cmap='plasma')
    plt.colorbar(label='Messages Sent')
    plt.yticks(ticks=range(len(message_logs)), labels=message_logs.keys())
    plt.xticks(ticks=np.arange(0, episodes, 5))
    plt.xlabel('Episode')
    plt.title('Message Activity Heatmap per Agent')
    plt.tight_layout()

    full_path = os.path.join(RESULTS_DIR, save_name)
    plt.savefig(full_path)
    plt.show()
    print(f"[✔] Message heatmap saved to: {full_path}")


# Run All Plots
plot_rewards_comparison(rewards_data)
plot_message_counts(message_counts)
plot_message_heatmap(message_counts)
