import numpy as np

class UCBExplorer:
    def __init__(self, action_size, c=2.0):
        self.action_size = action_size
        self.c = c  # Exploration constant
        self.counts = np.zeros(action_size)  # N(a)
        self.values = np.zeros(action_size)  # Q(a)
        self.total_steps = 0

    def act(self):
        self.total_steps += 1
        ucb_scores = []

        for a in range(self.action_size):
            if self.counts[a] == 0:
                return a  # Explore untried action
            bonus = self.c * np.sqrt(np.log(self.total_steps) / self.counts[a])
            score = self.values[a] + bonus
            ucb_scores.append(score)

        return int(np.argmax(ucb_scores))

    def update(self, action, reward):
        self.counts[action] += 1
        step_size = 1 / self.counts[action]
        self.values[action] += step_size * (reward - self.values[action])
