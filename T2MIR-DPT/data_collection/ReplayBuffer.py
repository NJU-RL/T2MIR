import numpy as np
import pickle
import random
from collections import OrderedDict


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, action_dim: int, use_success_rate: bool = False) -> None:
        self.capacity = capacity
        self.use_success_rate = use_success_rate
        self.top = 0
        self.size = 0
        self.states = np.empty((capacity, state_dim), dtype=np.float32)
        self.actions = np.empty((capacity, action_dim), dtype=np.float32)
        self.rewards = np.empty((capacity,), dtype=np.float32)
        self.next_states = np.empty((capacity, state_dim), dtype=np.float32)
        self.dones = np.empty((capacity,), dtype=np.float32)
        self.masks = np.empty((capacity,), dtype=np.float32)
        if use_success_rate:
            self.success_rates = np.empty((capacity,), dtype=np.float32)

    def __len__(self):
        return self.size

    def push(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool, mask: bool, success_rate: float = 0.0) -> None:
        self.states[self.top] = state
        self.actions[self.top] = action
        self.rewards[self.top] = reward
        self.next_states[self.top] = next_state
        self.dones[self.top] = done
        self.masks[self.top] = mask
        if self.use_success_rate:
            self.success_rates[self.top] = success_rate
        self.top = (self.top + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        '''
        Sample transitions.

        :param batch_size: batch size.
        :return: (states, actions, rewards, next states, dones, masks).
        '''
        index = np.random.choice(self.size, size=batch_size, replace=(batch_size > self.size))
        return self.states[index], self.actions[index], self.rewards[index], self.next_states[index], self.dones[index], self.masks[index]

    def save_buffer(self, save_path: str) -> None:
        data = OrderedDict()
        data['states'] = self.states[:self.size]
        data['actions'] = self.actions[:self.size]
        data['next_states'] = self.next_states[:self.size]
        data['rewards'] = self.rewards[:self.size]
        data['dones'] = self.dones[:self.size]
        data['masks'] = self.masks[:self.size]
        if self.use_success_rate:
            data['success_rates'] = self.success_rates[:self.size]
        
        print(f"Buffer length: {len(self)}, save buffer to {save_path}")
        with open(save_path, 'wb') as fp:
            pickle.dump(data, fp)

    def load_buffer(self, load_path: str) -> None:
        with open(load_path, 'rb') as fp:
            data = pickle.load(fp)

        self.size = min(self.capacity, data['states'].shape[0])
        self.top = self.size % self.capacity
        self.states[:self.size] = data['states'][-self.size:]
        self.actions[:self.size] = data['actions'][-self.size:]
        self.next_states[:self.size] = data['next_states'][-self.size:]
        self.rewards[:self.size] = data['rewards'][-self.size:]
        self.dones[:self.size] = data['dones'][-self.size:]
        self.masks[:self.size] = data['masks'][-self.size:]
        if self.use_success_rate:
            self.success_rates[:self.size] = data['success_rates'][-self.size:]
