import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class ActorCriticLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_dim):
        super(ActorCriticLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x, hidden_state):
        x = x.unsqueeze(1)  # Add sequence dimension
        lstm_out, hidden_state = self.lstm(x, hidden_state)
        logits = self.actor(lstm_out.squeeze(1))
        value = self.critic(lstm_out.squeeze(1))
        return logits, value, hidden_state

    def init_hidden(self):
        return (torch.zeros(1, 1, 64), torch.zeros(1, 1, 64))  # (num_layers, batch, hidden_dim)

class PPOAgentLSTM:
    def __init__(self, input_dim, action_dim, lr=1e-3, gamma=0.99, eps_clip=0.2):
        self.model = ActorCriticLSTM(input_dim, 64, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.hidden_state = self.model.init_hidden()

    def select_action(self, state):
        state = torch.FloatTensor(state)
        logits, value, self.hidden_state = self.model(state, self.hidden_state)
        probs = torch.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value

    def evaluate(self, state, action, hidden_state):
        logits, value, _ = self.model(state, hidden_state)
        probs = torch.softmax(logits, dim=-1)
        dist = Categorical(probs)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, value, entropy

    def update(self, trajectories):
        states, actions, log_probs_old, returns, values = zip(*trajectories)

        for _ in range(5):  # PPO Epochs
            for i in range(len(states)):
                state = torch.FloatTensor(states[i])
                action = torch.tensor(actions[i])
                old_log_prob = log_probs_old[i].detach()
                return_val = torch.tensor(returns[i])
                value = values[i].detach()
                hidden_state = self.model.init_hidden()

                log_prob, new_value, entropy = self.evaluate(state, action, hidden_state)
                advantage = return_val - value

                ratio = torch.exp(log_prob - old_log_prob)
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
                loss = -torch.min(surr1, surr2) + 0.5 * (return_val - new_value).pow(2) - 0.01 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
