import numpy as np

from gymnasium import utils, spaces
from typing import List, Dict, Optional

from environments.rand_param_envs.base import RandomEnv


class Walker2DRandParamsEnv(RandomEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 125,
    }

    def __init__(self, tasks: Optional[List[Dict]] = None, n_tasks: Optional[int] = None, randomize_tasks: bool = True, log_scale_limit: float = 3.0):
        RandomEnv.__init__(self, log_scale_limit, 'walker2d.xml', 5, observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(17,)))
        utils.EzPickle.__init__(self)

        if tasks is None and n_tasks is None:
            raise Exception("Either tasks or n_tasks must be specified")

        if tasks is not None:
            self.tasks = tasks
        else:
            self.tasks = self.sample_tasks(n_tasks)
        self.reset_task(0)

    def _step(self, a):
        posbefore = self.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.data.qpos[0:3]
        alive_bonus = 1.0
        reward = ((posafter - posbefore) / self.dt)
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        done = not (height > 0.8 and height < 2.0 and
                    ang > -1.0 and ang < 1.0)
        ob = self._get_obs()
        return ob, reward, done, False, {}

    def _get_obs(self):
        qpos = self.data.qpos
        qvel = self.data.qvel
        return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)], dtype=np.float32).ravel()

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        )
        return self._get_obs()
    
    def get_all_task_idx(self):
        return range(len(self.tasks))

    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self._goal = idx
        self.set_task(self._task)
        self.reset()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] += .8
        self.viewer.cam.elevation = -20