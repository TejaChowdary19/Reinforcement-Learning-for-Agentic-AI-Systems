import matplotlib.pyplot as plt
import os

def plot_agent_rewards(all_rewards, save_path="rag_communicating_agent_rewards.png"):
    num_episodes = len(all_rewards)
    num_agents = len(all_rewards[0])

    for agent_id in range(num_agents):
        agent_rewards = [episode[agent_id] for episode in all_rewards]
        plt.plot(agent_rewards, label=f'Agent {agent_id}')

    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Agent Rewards Over Episodes')
    plt.legend()
    plt.grid(True)

    # Create output directory if not exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)

    # Show the plot in a window
    plt.show()

    plt.close()
