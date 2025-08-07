import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.fc(x)


class ValueNetwork(nn.Module):
    def __init__(self, input_dim):
        super(ValueNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.fc(x)

class PPOAgent:
    def save_policy(self, path="ppo_policy.pth"):
        torch.save(self.policy_net.state_dict(), path)

    def load_policy(self, path="ppo_policy.pth"):
        self.policy_net.load_state_dict(torch.load(path))

    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, clip_epsilon=0.2):
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon

        self.policy_net = PolicyNetwork(state_dim, action_dim)
        self.value_net = ValueNetwork(state_dim)

        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)

        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []

    def act(self, state):
        state_tensor = torch.FloatTensor([[state]])  # shape: [1, 1]
        probs = self.policy_net(state_tensor)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        self.states.append(state)
        self.actions.append(action.item())
        self.log_probs.append(log_prob)
        return action.item()

    def store_reward(self, reward):
        self.rewards.append(reward)

    def learn(self):
        # Compute returns
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)

        states = torch.FloatTensor([[s] for s in self.states])  # shape: [batch_size, 1]
        actions = torch.LongTensor(self.actions)
        old_log_probs = torch.stack(self.log_probs)

        for _ in range(5):  # multiple epochs
            probs = self.policy_net(states)
            dist = torch.distributions.Categorical(probs)
            new_log_probs = dist.log_prob(actions)

            # Policy loss
            ratios = torch.exp(new_log_probs - old_log_probs.detach())
            advantages = returns - self.value_net(states).squeeze()
            policy_loss = -torch.min(
                ratios * advantages,
                torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            ).mean()

            # Value loss
            values = self.value_net(states).squeeze()
            value_loss = nn.MSELoss()(values, returns)

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

        # Clear memory
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []