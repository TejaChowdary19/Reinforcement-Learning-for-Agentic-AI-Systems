from envs.workflow_env import WorkflowEnv
from agents.ppo_agent import PPOAgent
from train.transfer_tasks import task_a
import matplotlib.pyplot as plt

env = WorkflowEnv(task_a)
agent = PPOAgent(state_dim=1, action_dim=len(task_a["tools"]))
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
    print(f"[Task A] Episode {ep+1}, Reward: {total_reward}")

# Save policy
agent.save_policy("ppo_task_a.pth")
