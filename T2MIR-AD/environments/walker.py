import numpy as np

from typing import List, Dict, Optional

from environments.tp_envs.walker_rand_params_wrapper import Walker2DRandParamsEnv


class WalkerRandParamsEnv(Walker2DRandParamsEnv):
    def __init__(self, tasks: List[Dict] = None, n_tasks: Optional[int] = None, include_goal: bool = False, max_episode_steps: int = 200):
        self.include_goal = include_goal
        self.n_tasks = len(tasks) if tasks is not None else n_tasks
        
        super(WalkerRandParamsEnv, self).__init__(tasks, n_tasks)

        self.set_task_idx(0)
        self._max_episode_steps = max_episode_steps

    def _get_obs(self):
        if self.include_goal:
            idx = 0
            try:
                idx = self._goal
            except:
                pass
            one_hot = np.zeros(self.n_tasks, dtype=np.float32)
            one_hot[idx] = 1.0
            obs = super()._get_obs()
            obs = np.concatenate([obs, one_hot])
        else:
            obs = super()._get_obs()
        return obs

    def set_task_idx(self, idx):
        self._task = self.tasks[idx]
        self._goal = idx
        self.set_task(self._task)
        self.reset()

    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        if done:
            reward -= 1.0
            done = False
        return obs, reward, done, truncated, info