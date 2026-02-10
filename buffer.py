import random
from collections import deque
import torch


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, next_state, reward, done):
        self.buffer.append((state, action, next_state, reward, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        if batch:
            states, actions, next_states, rewards, dones = zip(*batch)
        else:
            # Handle the case when the batch is empty
            states, actions, next_states, rewards, dones = [], [], [], [], []
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
        return states, actions, next_states, rewards, dones

    def __len__(self):
        return len(self.buffer)