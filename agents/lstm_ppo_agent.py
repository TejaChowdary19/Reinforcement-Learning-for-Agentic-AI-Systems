# agents/lstm_ppo_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque

class ActorCriticLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ActorCriticLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.actor = nn.Linear(hidden_dim, output_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x, hx, cx):
        x, (hx, cx) = self.lstm(x, (hx, cx))
        logits = self.actor(x[:, -1])
        value = self.critic(x[:, -1])
        return logits, value, (hx, cx)


class LSTMPPOAgent:
    def __init__(self, input_dim, action_dim, hidden_dim=64, lr=1e-3):
        self.model = ActorCriticLSTM(input_dim, hidden_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.hidden_dim = hidden_dim
        self.memory = []
        self.reset_memory()

    def reset_memory(self):
        self.hx = torch.zeros(1, 1, self.hidden_dim)
        self.cx = torch.zeros(1, 1, self.hidden_dim)

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)
        logits, value, (self.hx, self.cx) = self.model(state, self.hx, self.cx)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob, value

    def send_message(self, action):
        return f"LSTM-action:{action}"

    def store_transition(self, transition):
        self.memory.append(transition)

    def compute_returns(self, next_value, gamma=0.99):
        returns = []
        R = next_value
        for _, _, reward, _, _ in reversed(self.memory):
            R = reward + gamma * R
            returns.insert(0, R)
        return returns

    def learn(self, next_state):
        next_state = torch.FloatTensor(next_state).unsqueeze(0).unsqueeze(0)
        _, next_value, _ = self.model(next_state, self.hx.detach(), self.cx.detach())
        returns = self.compute_returns(next_value.detach().item())

        policy_loss = 0
        value_loss = 0
        for (log_prob, value, reward, _, _), R in zip(self.memory, returns):
            advantage = R - value.item()
            policy_loss -= log_prob * advantage
            value_loss += (value - R) ** 2

        loss = policy_loss + value_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.memory = []
