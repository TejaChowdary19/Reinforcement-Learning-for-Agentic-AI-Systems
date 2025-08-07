import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import random
import numpy as np
from envs.workflow_env import SimpleMultiAgentEnv
from agents.communicating_agent import CommunicatingAgent
from agents.lstm_ppo_agent import LSTMPPOAgent

# Set seed
random.seed(42)
np.random.seed(42)

env = SimpleMultiAgentEnv()
agents = [
    CommunicatingAgent(agent_id=0),
    CommunicatingAgent(agent_id=1),
    LSTMPPOAgent(agent_id=2, input_dim=env.state_size, hidden_dim=32, output_dim=env.num_actions)
]

rewards_history = [[] for _ in agents]

for episode in range(50):
    obs = env.reset()
    hidden_states = [None] * len(agents)
    done = False
    while not done:
        actions, messages = [], []
        for i, agent in enumerate(agents):
            if isinstance(agent, LSTMPPOAgent):
                action, hidden = agent.select_action(obs[i], hidden_states[i])
                hidden_states[i] = hidden
            else:
                action = agent.select_action(obs[i])
            actions.append(action)
            messages.append(agent.send_message(i, action))

        next_obs, rewards, done, _ = env.step(actions)
        for i, agent in enumerate(agents):
            if isinstance(agent, LSTMPPOAgent):
                agent.store_transition(obs[i], actions[i], rewards[i], next_obs[i], done)
        obs = next_obs

    for i, agent in enumerate(agents):
        if isinstance(agent, LSTMPPOAgent):
            agent.train()
        rewards_history[i].append(sum(env.agent_rewards[i]))

    print(f"Episode {episode+1} Rewards: {[sum(env.agent_rewards[i]) for i in range(len(agents))]}")

# Save training rewards
import matplotlib.pyplot as plt
for i, rewards in enumerate(rewards_history):
    plt.plot(rewards, label=f"Agent {i}")
plt.title("Training Rewards per Agent")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.legend()
plt.grid(True)
plt.savefig("multiagent_lstm_training_rewards.png")
