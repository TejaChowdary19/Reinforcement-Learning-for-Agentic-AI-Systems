import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class SimpleMultiAgentEnv:
    def __init__(self, num_agents=3):
        self.num_agents = num_agents
        self.step_count = 0

    def reset(self):
        self.step_count = 0
        return [0 for _ in range(self.num_agents)]  # Dummy observations

    def step(self, actions):
        rewards = []
        messages = []
        for i, action in enumerate(actions):
            # Simple reward logic:
            if action == 1:
                reward = random.choice([10, -5])
            else:
                reward = 0
            rewards.append(reward)
            messages.append(f'Agent{i}-action:{action}')
        self.step_count += 1
        done = self.step_count >= 50
        next_obs = [0 for _ in range(self.num_agents)]
        return next_obs, rewards, done, messages

class WorkflowEnv(gym.Env):
    def __init__(self, task_config):
        super(WorkflowEnv, self).__init__()
        self.task_name = task_config["name"]
        self.steps = task_config["steps"]
        self.tools = task_config["tools"]
        self.current_step = 0

        self.action_space = spaces.Discrete(len(self.tools))  # Choose a tool
        self.observation_space = spaces.Discrete(len(self.steps))  # Step index

    def reset(self):
        self.current_step = 0
        return self.current_step

    def step(self, action):
        done = self.current_step >= len(self.steps)
        if done:
            return self.current_step, 0, done, {}

        tool = self.tools[action]
        success_prob = tool["success_rate"]
        success = np.random.rand() < success_prob
        reward = 10 if success else -5

        self.current_step += 1
        done = self.current_step >= len(self.steps)
        return self.current_step, reward, done, {"tool_used": tool["name"]}

    def render(self):
        print(f"Current Step: {self.current_step}/{len(self.steps)}")
