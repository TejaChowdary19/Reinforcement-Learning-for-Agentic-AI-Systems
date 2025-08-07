# communicating_agent.py (full with RAG retrieval logic)
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random
import numpy as np
from retrieval.memory_store import EpisodeMemory

class CommunicatingAgent:
    def __init__(self, agent_id, obs_dim, num_actions):
        self.agent_id = agent_id
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.memory = EpisodeMemory(obs_dim=obs_dim)

    def select_action(self, obs):
        # Retrieve similar past observations
        retrieved_obs, retrieved_actions = self.memory.retrieve(obs, k=3)
        if len(retrieved_actions) == 0:
            action = np.random.choice([0, 1])
        else:
            action = max(set(retrieved_actions), key=retrieved_actions.count)
        return action

    def store_experience(self, obs, action):
        self.memory.add(obs, action)

    def send_message(self, action):
        return f"Agent{self.agent_id}-action:{action}"


def run_episodes(num_episodes=50, num_agents=3, obs_dim=3):
    agents = [CommunicatingAgent(i, obs_dim=obs_dim) for i in range(num_agents)]
    all_rewards = []

    for episode in range(num_episodes):
        obs = np.random.rand(num_agents, obs_dim)  # Simulated observations
        rewards = []
        messages = []

        for i, agent in enumerate(agents):
            action = agent.select_action(obs[i])
            reward = simulate_environment_response(action, i)

            agent.store_experience(obs[i], action)
            rewards.append(reward)
            messages.append(agent.send_message(action))

        print(f"Episode {episode+1}, Rewards: {rewards}, Messages: {messages}")
        all_rewards.append(rewards)

    from utils.plot_rewards import plot_agent_rewards
    plot_agent_rewards(all_rewards, save_path="plots/rag_communicating_agent_rewards.png")


def simulate_environment_response(action, agent_id):
    # Simulated reward logic: reward depends on agent_id and action
    base_reward = random.randint(0, 100)
    modifier = 10 if action == agent_id % 2 else -10
    return base_reward + modifier


if __name__ == "__main__":
    run_episodes()
