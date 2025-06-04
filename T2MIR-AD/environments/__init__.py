# import warnings
# warnings.filterwarnings("ignore")

from gymnasium.envs.registration import register

# point robot
register(
    id='PointRobot-v0',
    entry_point='environments.point_robot:PointEnv',
    max_episode_steps=20,
)


# mujoco
# half cheetah vel
register(
    id='HalfCheetahVel-v0',
    entry_point='environments.halfcheetah_vel:HalfCheetahVelEnv',
    max_episode_steps=200,
)

# half cheetah vel
register(
    id='HalfCheetahVel-v3',
    entry_point='environments.halfcheetah_vel:HalfCheetahVelEnv',
    max_episode_steps=200,
)

# walker
register(
    id='WalkerRandParams-v0',
    entry_point='environments.walker:WalkerRandParamsEnv',
    max_episode_steps=200,
)


# meta world
# reach
register(
    id='Reach-v2',
    entry_point='environments.ml1:ML1Env',
    max_episode_steps=100,
    kwargs={'task_name': 'reach-v2'},
)

# push
register(
    id='Push-v2',
    entry_point='environments.ml1:ML1Env',
    max_episode_steps=100,
    kwargs={'task_name': 'push-v2'},
)

# discrete enviroments
register(
    id='DarkRoom-v0',
    entry_point='environments.darkroom:DarkRoom',
    max_episode_steps=100,
)