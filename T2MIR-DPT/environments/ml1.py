import metaworld
import numpy as np

from gymnasium import spaces, Env
from typing import List


class ML1Env(Env):
    def __init__(self, task_name: str, n_tasks: int = 50, tasks: List = None, max_episode_steps: int = 100) -> None:
        self._max_episode_steps = max_episode_steps
        ml1_env = metaworld.ML1(task_name, 0)
        self._env = ml1_env.train_classes[task_name]()
        self._env.max_path_length = max_episode_steps

        if tasks is None:
            self.n_tasks = n_tasks
            self.tasks = []
            for i in range(n_tasks):
                task = self.ml1_env.train_tasks[i]
                self.tasks.append(task)
        else:
            self.tasks = tasks
            self.n_tasks = len(tasks)

        self.reset_task(0)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(18,))
        self.action_space = spaces.Box(low=-1., high=1., shape=(4,))

    def seed(self, seed: int):
        self._env.seed(seed)

    def reset_task(self, idx):
        ''' reset goal AND reset the agent '''
        self._goal_idx = idx
        self._env.set_task(self.tasks[idx])
        self._env.max_path_length = self._max_episode_steps
        self.reset()

    def reset(self, seed=None, options=None):
        return self.reset_model()

    def reset_model(self):
        state, info = self._env.reset()
        return state.astype(np.float32)[:18], info

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        next_state, reward, _, _, info = self._env.step(action)
        return next_state.astype(np.float32)[:18], reward, False, False, info

    def get_all_task_idx(self):
        return range(self.n_tasks)