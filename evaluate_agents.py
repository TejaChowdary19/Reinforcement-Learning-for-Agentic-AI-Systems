import os
import numpy as np
import pandas as pd
from envs.workflow_env import SimpleMultiAgentEnv
from agents.communicating_agent import CommunicatingAgent
from train.plot_rewards import plot_agent_rewards

# === Config ===
EPISODES = 50
NUM_AGENTS = 3
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# === Setup ===
env = SimpleMultiAgentEnv(num_agents=NUM_AGENTS)
agents = [CommunicatingAgent(i) for i in range(NUM_AGENTS)]

episode_rewards = {f'Agent{i}': [] for i in range(NUM_AGENTS)}
messages_sent = {f'Agent{i}': [] for i in range(NUM_AGENTS)}

# === Run Evaluation Episodes ===
for episode in range(EPISODES):
    obs = env.reset()
    done = False
    messages = [None] * NUM_AGENTS
    total_rewards = [0] * NUM_AGENTS
    message_count = [0] * NUM_AGENTS

    while not done:
        actions = []
        new_messages = []
        for i, agent in enumerate(agents):
            action = agent.act(obs[i], messages)
            actions.append(action)
            new_messages.append(agent.send_message(i, action))
            if new_messages[i] is not None:
                message_count[i] += 1

        obs, rewards, done, _ = env.step(actions)
        for i in range(NUM_AGENTS):
            total_rewards[i] += rewards[i]
        messages = new_messages

    # Log per-episode reward + messages
    for i in range(NUM_AGENTS):
        episode_rewards[f'Agent{i}'].append(total_rewards[i])
        messages_sent[f'Agent{i}'].append(message_count[i])

# === Save CSV ===
df_rewards = pd.DataFrame(episode_rewards)
df_messages = pd.DataFrame(messages_sent)

df_rewards.to_csv(os.path.join(RESULTS_DIR, 'agent_rewards.csv'), index=False)
df_messages.to_csv(os.path.join(RESULTS_DIR, 'agent_messages.csv'), index=False)
print("[✔] Rewards and messages saved to CSV.")

# === Print Evaluation Summary ===
print("\n🎯 Final Evaluation Metrics (per agent):")
for i in range(NUM_AGENTS):
    rewards = episode_rewards[f'Agent{i}']
    print(f"Agent{i} - Mean: {np.mean(rewards):.2f}, Max: {np.max(rewards)}, Std: {np.std(rewards):.2f}")

# === Optional: Plot Rewards ===
plot_agent_rewards(episode_rewards, save_path=os.path.join(RESULTS_DIR, 'evaluation_plot.png'))
