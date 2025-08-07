import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from utils.log_utils import init_logger, log_metrics

# Set seeds
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# Define DQN model
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.fc(x)

# Hyperparameters
env_name = "CartPole-v1"
num_episodes = 200
max_steps = 500
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
lr = 0.001
batch_size = 64
memory_size = 10000

# Environment and networks
env = gym.make(env_name)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy_net = DQN(state_dim, action_dim).to(device)
target_net = DQN(state_dim, action_dim).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=lr)
loss_fn = nn.MSELoss()
memory = deque(maxlen=memory_size)

# Init logger
init_logger()

# Select action
def select_action(state, epsilon):
    if random.random() < epsilon:
        return random.randrange(action_dim)
    else:
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = policy_net(state)
            return q_values.argmax().item()

# Experience replay
def replay():
    if len(memory) < batch_size:
        return 0

    batch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.FloatTensor(states).to(device)
    actions = torch.LongTensor(actions).unsqueeze(1).to(device)
    rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
    next_states = torch.FloatTensor(next_states).to(device)
    dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

    current_q = policy_net(states).gather(1, actions)
    next_q = target_net(next_states).max(1)[0].unsqueeze(1)
    expected_q = rewards + gamma * next_q * (1 - dones)

    loss = loss_fn(current_q, expected_q)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

# Train loop
for episode in range(num_episodes):
    state, _ = env.reset()
    total_reward = 0
    episode_loss = 0

    for t in range(max_steps):
        action = select_action(state, epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        memory.append((state, action, reward, next_state, float(done)))
        state = next_state
        total_reward += reward

        loss = replay()
        episode_loss += loss

        if done:
            break

    # Decay epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    target_net.load_state_dict(policy_net.state_dict())

    avg_loss = episode_loss / (t + 1)
    log_metrics(episode, total_reward, avg_loss)

    print(f"Episode {episode} | Reward: {total_reward:.2f} | Loss: {avg_loss:.4f} | Epsilon: {epsilon:.3f}")

env.close()
