import numpy as np
from gymnasium import utils, spaces
from gymnasium.envs.mujoco import MujocoEnv


class HalfCheetahEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 20,
    }
    
    def __init__(self):
        MujocoEnv.__init__(self, "half_cheetah.xml", 5, observation_space=spaces.Box(-np.inf, np.inf, shape=(20,)))
        utils.EzPickle.__init__(self)

    def step(self, action):
        xposbefore = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.data.qpos[0]
        ob = self._get_obs()
        reward_ctrl = -0.1 * np.square(action).sum()
        reward_run = (xposafter - xposbefore) / self.dt
        reward = reward_ctrl + reward_run
        done = False
        return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        return np.concatenate([
            self.data.qpos.flat[1:],
            self.data.qvel.flat,
            self.get_body_com("torso").flat,
        ]).astype(np.float32).flatten()

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            low=-0.1, high=0.1, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.standard_normal(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def render(self, mode='human'):
        if mode == 'rgb_array':
            self._get_viewer(mode).render()
            # window size used for old mujoco-py:
            width, height = 500, 500
            data = self._get_viewer().read_pixels(width, height, depth=False)
            return data
        elif mode == 'human':
            self._get_viewer(mode).render()