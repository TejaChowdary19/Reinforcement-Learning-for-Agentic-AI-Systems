import numpy as np
import faiss

class EpisodeMemory:
    def __init__(self, obs_dim=3, max_size=1000):
        self.obs_dim = obs_dim
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(obs_dim)

        # Pre-allocate memory
        self.observations = [np.zeros((1, obs_dim), dtype='float32') for _ in range(max_size)]
        self.actions = [0 for _ in range(max_size)]

    def add(self, obs, action):
        obs = np.array(obs, dtype='float32')

        if obs.ndim == 0:
            obs = obs.reshape(1, 1)
        elif obs.ndim == 1:
            obs = obs.reshape(1, -1)
        elif obs.ndim == 2 and obs.shape[0] == 1:
            pass
        else:
            raise ValueError(f"Invalid obs shape: {obs.shape}")

        if obs.shape[1] != self.obs_dim:
            raise ValueError(f"Expected obs_dim={self.obs_dim}, got {obs.shape[1]}")

        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action

        if self.size < self.max_size:
            self.index.add(obs)
        else:
        # Full – rebuild index manually with all observations
            self.index.reset()
        self.index.add(np.vstack(self.observations))

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def retrieve(self, obs, k=3):
        if len(self.observations) == 0:
        # No memory yet — return dummy values
            return np.empty((0, self.obs_dim)), []

        obs = np.array(obs).reshape(1, -1).astype('float32')
        distances, indices = self.index.search(obs, k)

        retrieved_obs = np.array([self.observations[i] for i in indices[0]])
        retrieved_actions = [self.actions[i] for i in indices[0]]

        return retrieved_obs, retrieved_actions
    
    def retrieve(self, obs, k=3):
        if len(self.observations) == 0:
            return np.empty((0, self.obs_dim)), []

        obs = np.array(obs).reshape(1, -1).astype('float32')
        k = min(k, len(self.observations))  # avoid asking for more than available
        distances, indices = self.index.search(obs, k)

        valid_indices = [i for i in indices[0] if i < len(self.observations)]
        retrieved_obs = np.array([self.observations[i] for i in valid_indices])
        retrieved_actions = [self.actions[i] for i in valid_indices]

        return retrieved_obs, retrieved_actions
