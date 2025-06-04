import gymnasium as gym
import itertools
import numpy as np
from typing import Optional


class DarkRoom(gym.Env):
    def __init__(self, tasks: Optional[np.ndarray] = None, dim: int = 10, max_episode_steps = 100):
        self.dim = dim
        self.horizon = max_episode_steps
        self._max_episode_steps = max_episode_steps
        self.state_dim = 2
        self.action_dim = 5
        self.observation_space = gym.spaces.Box(low=0., high=dim - 1., shape=(self.state_dim,))
        self.action_space = gym.spaces.Discrete(self.action_dim)

        if tasks is not None:
            self.tasks = tasks
            assert self.tasks.ndim == 2
        else:
            raise NotImplementedError
        
        self.reset_task(0)

    def sample_state(self):
        return np.random.randint(0, self.dim, 2)

    def sample_action(self):
        i = np.random.randint(0, 5)
        a = np.zeros(self.action_space.n)
        a[i] = 1
        return a

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.state = np.array([0, 0], dtype=np.float32)
        return self.state, {}
    
    def reset_task(self, idx):
        self.goal = self.tasks[idx].copy()
        self.reset()

    def transit(self, state, action):
        action = np.argmax(action)
        assert action in np.arange(self.action_space.n)
        state = np.array(state)
        if action == 0:
            state[0] += 1
        elif action == 1:
            state[0] -= 1
        elif action == 2:
            state[1] += 1
        elif action == 3:
            state[1] -= 1
        state = np.clip(state, 0, self.dim - 1)

        if np.all(state.astype(np.int64) == self.goal):
            reward = 1.
        else:
            reward = 0.
        return state, reward

    def step(self, action):
        if self.current_step >= self.horizon:
            raise ValueError("Episode has already ended")

        self.state, r = self.transit(self.state, action)
        self.current_step += 1
        done = (self.current_step >= self.horizon)
        return self.state.copy().astype(np.float32), r, done, False, {}

    def get_obs(self):
        return self.state.copy().astype(np.float32)

    def opt_action(self, state):
        if state[0] < self.goal[0]:
            action = 0
        elif state[0] > self.goal[0]:
            action = 1
        elif state[1] < self.goal[1]:
            action = 2
        elif state[1] > self.goal[1]:
            action = 3
        else:
            action = 4
        zeros = np.zeros(self.action_space.n)
        zeros[action] = 1
        return zeros


class DarkRoomPermuted(DarkRoom):
    """
    Darkroom environment with permuted actions. The goal is always the bottom right corner.
    """

    def __init__(self, dim, perm_index, H):
        goal = np.array([dim - 1, dim - 1])
        super().__init__(dim, goal, H)

        self.perm_index = perm_index
        assert perm_index < 120     # 5! permutations in darkroom
        actions = np.arange(self.action_space.n)
        permutations = list(itertools.permutations(actions))
        self.perm = permutations[perm_index]

    def transit(self, state, action):
        perm_action = np.zeros(self.action_space.n)
        perm_action[self.perm[np.argmax(action)]] = 1
        return super().transit(state, perm_action)

    def opt_action(self, state):
        action = super().opt_action(state)
        action = np.argmax(action)
        perm_action = np.where(self.perm == action)[0][0]
        zeros = np.zeros(self.action_space.n)
        zeros[perm_action] = 1
        return zeros