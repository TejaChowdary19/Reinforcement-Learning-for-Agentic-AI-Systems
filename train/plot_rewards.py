import matplotlib.pyplot as plt
import os

def plot_agent_rewards(agent_rewards, filename=None):
    plt.figure(figsize=(12, 6))
    for i, rewards in enumerate(agent_rewards):
        plt.plot(rewards, label=f'Agent {i}')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Agent Rewards Over Time')
    plt.legend()
    plt.grid(True)

    if filename:
        os.makedirs("plots", exist_ok=True)
        filepath = os.path.join("plots", filename)
        plt.savefig(filepath)
        print(f"✅ Plot saved to {filepath}")
    else:
        plt.show()

    plt.close()
