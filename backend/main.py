from agents.communicating_agent import CommunicatingAgent
from environment.multi_agent_env import MultiAgentEnv
from utils.plot_rewards import get_logs_only

NUM_AGENTS = 3
EPISODE_LENGTH = 10
NUM_EPISODES = 1

all_logs = []

def run_episode():
    global all_logs
    env = MultiAgentEnv(num_agents=NUM_AGENTS)
    agents = [CommunicatingAgent(agent_id=i, obs_dim=env.obs_dim, action_dim=env.action_dim) for i in range(NUM_AGENTS)]
    
    logs = []
    
    for episode in range(NUM_EPISODES):
        obs = env.reset()
        episode_rewards = [0 for _ in range(NUM_AGENTS)]
        messages = ["" for _ in range(NUM_AGENTS)]

        for step in range(EPISODE_LENGTH):
            actions = []
            for i, agent in enumerate(agents):
                action = agent.select_action(obs[i])
                actions.append(action)
                agent.store_experience(obs[i], action)

            next_obs, rewards = env.step(actions)

            for i in range(NUM_AGENTS):
                episode_rewards[i] += rewards[i]
                messages[i] = f"Agent{i}-action:{actions[i]}"

            obs = next_obs

        logs.append({
            "episode": len(all_logs) + 1,
            "rewards": episode_rewards,
            "messages": messages
        })

        all_logs.append(logs[-1])

    return logs

def get_all_logs():
    return all_logs
