import numpy as np

from environments.tp_envs.half_cheetah_vel import HalfCheetahVelEnv as HalfCheetahVelEnv_
from typing import List


class HalfCheetahVelEnv(HalfCheetahVelEnv_):
    def __init__(self, tasks: List[dict] = None, include_goal: bool = False, one_hot_goal: bool = False, n_tasks: int = None):
        self.include_goal = include_goal
        self.one_hot_goal = one_hot_goal
        if tasks is None:
            assert n_tasks is not None, "Either tasks or n_tasks must be non-None"
            tasks = self.sample_tasks(n_tasks)
        self.n_tasks = len(tasks)
        super().__init__(tasks)
        self.set_task_idx(0)
        self._max_episode_steps = 200

    def _get_obs(self):
        if self.include_goal:
            obs = super()._get_obs()
            if self.one_hot_goal:
                goal = np.zeros((self.n_tasks,))
                goal[self.tasks.index(self._task)] = 1
            else:
                goal = np.array([self._goal_vel])
            obs = np.concatenate([obs, goal])
        else:
            obs = super()._get_obs()

        return obs

    def set_task(self, task):
        self._task = task
        self._goal_vel = self._task['velocity']
        self.reset()

    def set_task_idx(self, idx):
        self.task_idx = idx
        self.set_task(self.tasks[idx])

    def print_task(self):
        print(f'Task information: Goal vel {self._goal}')

    def reset(self, **kwargs):
        return super().reset(**kwargs)