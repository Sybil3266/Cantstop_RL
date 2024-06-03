import random
import numpy as np
import torch
from collections import deque, namedtuple

# Experience 네임드 튜플을 전역으로 이동해 pickling error 해결하도록
#Experience = namedtuple("Experience", field_names=["state_dim", "action", "reward", "next_state_dim", "done"])

class Cantstop_ReplayMemory:
    def __init__(self, action_size, buffer_size=1000, batch_size=64):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state_dim","action", "reward", "next_state_dim", "done"])
        #self.seed = random.seed(seed)

    def add(self, state_dim, action, reward, next_state_dim, done):
        #e = Experience(state_dim, action, reward, next_state_dim, done)
        e = self.experience(state_dim, action, reward, next_state_dim, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        return self.extract_tensors(experiences)

    def sample_by_indices(self, indices):
        experiences = [self.memory[i] for i in indices]
        return self.extract_tensors(experiences)

    def extract_tensors(self, experiences):
        #action_matrix = torch.from_numpy(np.vstack([e.action_matrix for e in experiences if e is not None])).float().to(self.device)
        #action_masking = torch.from_numpy(np.vstack([e.action_masking for e in experiences if e is not None])).float().to(self.device)
        state_dim = torch.from_numpy(np.vstack([e.state_dim for e in experiences if e is not None])).float().to(self.device)
        action = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        reward = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_state_dim = torch.from_numpy(np.vstack([e.next_state_dim for e in experiences if e is not None])).float().to(self.device)
        done = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.int8)).float().to(self.device)
        
        return (state_dim, action, reward, next_state_dim, done)

    def __len__(self):
        return len(self.memory)


    