# api/simulator.py

from agents.communicating_agent import CommunicatingAgent
from envs.workflow_env import SimpleMultiAgentEnv 

NUM_AGENTS = 3
OBS_DIM = 3
NUM_ACTIONS = 2

env = SimpleMultiAgentEnv(num_agents=NUM_AGENTS)
agents = [CommunicatingAgent(agent_id=i, obs_dim=OBS_DIM, num_actions=NUM_ACTIONS) for i in range(NUM_AGENTS)]

log = []

def run_one_episode():
    obs = env.reset()
    done = False
    total_rewards = [0] * NUM_AGENTS
    messages = []

    while not done:
        actions = []
        for i, agent in enumerate(agents):
            action = agent.select_action(obs[i])
            agent.store_experience(obs[i], action)
            actions.append(action)
            messages.append(f"Agent{i}-action:{action}")

        obs, rewards, done, _ = env.step(actions)
        total_rewards = [r + tr for r, tr in zip(rewards, total_rewards)]

    log.append({"rewards": total_rewards, "messages": messages})
    return {"rewards": total_rewards, "messages": messages}

def get_log():
    return log
