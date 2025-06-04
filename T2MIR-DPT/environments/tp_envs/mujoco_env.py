import os
from os import path

import numpy as np
from gymnasium.envs.mujoco import mujoco_env
from gymnasium import spaces


ENV_ASSET_DIR = os.path.join(os.path.dirname(__file__), 'assets')


class MujocoEnv(mujoco_env.MujocoEnv):
    """
    My own wrapper around MujocoEnv.

    The caller needs to declare
    """
    def __init__(
            self,
            model_path,
            frame_skip=1,
            model_path_is_local=True,
            automatically_set_obs_and_action_space=False,
    ):
        if model_path_is_local:
            model_path = get_asset_xml(model_path)
        if automatically_set_obs_and_action_space:
            mujoco_env.MujocoEnv.__init__(self, model_path, frame_skip, observation_space=spaces.Box(-np.inf, np.inf, shape=(27,)))
        else:
            raise NotImplementedError

    # def init_serialization(self, locals):
    #     Serializable.quick_init(self, locals)

    def log_diagnostics(self, paths):
        pass


def get_asset_xml(xml_name):
    return os.path.join(ENV_ASSET_DIR, xml_name)
