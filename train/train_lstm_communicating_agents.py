# train_lstm_communicating_agents.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from envs.workflow_env import SimpleMultiAgentEnv
from agents.lstm_ppo_agent import LSTMPPOAgent

# Set random seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Initialize environment
num_agents = 3
env = SimpleMultiAgentEnv(num_agents=num_agents)

# Initialize agents
agents = [LSTMPPOAgent(agent_id=i, input_dim=env.observation_space, hidden_dim=64, output_dim=env.action_space) for i in range(num_agents)]

# Hyperparameters
epochs = 200
rollout_len = 10
lr = 3e-4
gamma = 0.99
clip_param = 0.2

# Logging
reward_logs = [[] for _ in range(num_agents)]

for episode in range(epochs):
    states = env.reset()
    h_states = [agent.init_hidden() for agent in agents]
    episode_rewards = [0 for _ in range(num_agents)]
    messages = [None for _ in range(num_agents)]

    for _ in range(rollout_len):
        actions, new_h_states, new_messages = [], [], []
        for i, agent in enumerate(agents):
            action, h = agent.select_action(states[i], h_states[i], messages)
            actions.append(action)
            new_h_states.append(h)
            new_messages.append(agent.send_message(action))

        next_states, rewards, done, _ = env.step(actions)

        for i in range(num_agents):
            agents[i].store_transition(states[i], actions[i], rewards[i], next_states[i], h_states[i], messages, done)
            episode_rewards[i] += rewards[i]

        states = next_states
        h_states = new_h_states
        messages = new_messages

    # Train after episode
    for agent in agents:
        agent.train()

    for i in range(num_agents):
        reward_logs[i].append(episode_rewards[i])

    if (episode + 1) % 10 == 0:
        print(f"Episode {episode+1} Rewards: {[round(r, 2) for r in episode_rewards]}")

# Save model and logs
for i, agent in enumerate(agents):
    torch.save(agent.policy.state_dict(), f"models/lstm_agent_{i}.pt")
    np.save(f"logs/lstm_agent_{i}_rewards.npy", reward_logs[i])

print("✅ Training completed and saved!")
