# EXPERIMENTAL: all may be removed soon

from environments.rand_param_envs.gym.benchmarks import scoring
from environments.rand_param_envs.gym.benchmarks.registration import benchmark_spec, register_benchmark, registry, register_benchmark_view  # imports used elsewhere

register_benchmark(
    id='Atari200M',
    scorer=scoring.TotalReward(),
    name='Atari200M',
    view_group="Atari",
    description='7 Atari games, with pixel observations',
    tasks=[
        {
            'env_id': 'BeamRiderNoFrameskip-v3',
            'trials': 2,
            'max_timesteps': int(2e8),
            'reward_floor':   363.9,
            'reward_ceiling': 60000.0,
        },
        {
            'env_id': 'BreakoutNoFrameskip-v3',
            'trials': 2,
            'max_timesteps': int(2e8),
            'reward_floor':   1.7,
            'reward_ceiling': 800.0,
        },
        {
            'env_id': 'EnduroNoFrameskip-v3',
            'trials': 2,
            'max_timesteps': int(2e8),
            'reward_floor':   0.0,
            'reward_ceiling': 5000.0,
        },
        {
            'env_id': 'PongNoFrameskip-v3',
            'trials': 2,
            'max_timesteps': int(2e8),
            'reward_floor':  -20.7,
            'reward_ceiling': 21.0,
        },
        {
            'env_id': 'QbertNoFrameskip-v3',
            'trials': 2,
            'max_timesteps': int(2e8),
            'reward_floor':   163.9,
            'reward_ceiling': 40000.0,
        },
        {
            'env_id': 'SeaquestNoFrameskip-v3',
            'trials': 2,
            'max_timesteps': int(2e8),
            'reward_floor':   68.4,
            'reward_ceiling': 100000.0,
        },
        {
            'env_id': 'SpaceInvadersNoFrameskip-v3',
            'trials': 2,
            'max_timesteps': int(2e8),
            'reward_floor':   148.0,
            'reward_ceiling': 30000.0,
        },
    ])

register_benchmark(
    id='Atari40M',
    scorer=scoring.TotalReward(),
    name='Atari40M',
    view_group="Atari",
    description='7 Atari games, with pixel observations',
    tasks=[
        {
            'env_id': 'BeamRiderNoFrameskip-v3',
            'trials': 2,
            'max_timesteps': int(4e7),
            'reward_floor':   363.9,
            'reward_ceiling': 60000.0,
        },
        {
            'env_id': 'BreakoutNoFrameskip-v3',
            'trials': 2,
            'max_timesteps': int(4e7),
            'reward_floor':   1.7,
            'reward_ceiling': 800.0,
        },
        {
            'env_id': 'EnduroNoFrameskip-v3',
            'trials': 2,
            'max_timesteps': int(4e7),
            'reward_floor':   0.0,
            'reward_ceiling': 5000.0,
        },
        {
            'env_id': 'PongNoFrameskip-v3',
            'trials': 2,
            'max_timesteps': int(4e7),
            'reward_floor':  -20.7,
            'reward_ceiling': 21.0,
        },
        {
            'env_id': 'QbertNoFrameskip-v3',
            'trials': 2,
            'max_timesteps': int(4e7),
            'reward_floor':   163.9,
            'reward_ceiling': 40000.0,
        },
        {
            'env_id': 'SeaquestNoFrameskip-v3',
            'trials': 2,
            'max_timesteps': int(4e7),
            'reward_floor':   68.4,
            'reward_ceiling': 100000.0,
        },
        {
            'env_id': 'SpaceInvadersNoFrameskip-v3',
            'trials': 2,
            'max_timesteps': int(4e7),
            'reward_floor':   148.0,
            'reward_ceiling': 30000.0,
        }
    ])

register_benchmark(
    id='AtariExploration40M',
    scorer=scoring.TotalReward(),
    name='AtariExploration40M',
    view_group="Atari",
    description='7 Atari games, with pixel observations',
    tasks=[
        {
            'env_id': 'FreewayNoFrameskip-v3',
            'trials': 2,
            'max_timesteps': int(4e7),
            'reward_floor':   0.1,
            'reward_ceiling': 31.0,
        },
        {
            'env_id': 'GravitarNoFrameskip-v3',
            'trials': 2,
            'max_timesteps': int(4e7),
            'reward_floor':   245.5,
            'reward_ceiling': 1000.0,
        },
        {
            'env_id': 'MontezumaRevengeNoFrameskip-v3',
            'trials': 2,
            'max_timesteps': int(4e7),
            'reward_floor':   25.0,
            'reward_ceiling': 10000.0,
        },
        {
            'env_id': 'PitfallNoFrameskip-v3',
            'trials': 2,
            'max_timesteps': int(4e7),
            'reward_floor':  -348.8,
            'reward_ceiling': 1000.0,
        },
        {
            'env_id': 'PrivateEyeNoFrameskip-v3',
            'trials': 2,
            'max_timesteps': int(4e7),
            'reward_floor':   662.8,
            'reward_ceiling': 100.0,
        },
        {
            'env_id': 'SolarisNoFrameskip-v3',
            'trials': 2,
            'max_timesteps': int(4e7),
            'reward_floor':   2047.2,
            'reward_ceiling': 5000.0,
        },
        {
            'env_id': 'VentureNoFrameskip-v3',
            'trials': 2,
            'max_timesteps': int(4e7),
            'reward_floor':   18.0,
            'reward_ceiling': 100.0,
        }
    ])


register_benchmark(
    id='ClassicControl2-v0',
    name='ClassicControl2',
    view_group="Control",
    description='Simple classic control benchmark',
    scorer=scoring.ClipTo01ThenAverage(),
    tasks=[
        {'env_id': 'CartPole-v0',
         'trials': 1,
         'max_timesteps': 2000,
        },
        {'env_id': 'Pendulum-v0',
         'trials': 1,
         'max_timesteps': 1000,
        },
    ])

register_benchmark(
    id='ClassicControl-v0',
    name='ClassicControl',
    view_group="Control",
    description='Simple classic control benchmark',
    scorer=scoring.ClipTo01ThenAverage(),
    tasks=[
        {'env_id': 'CartPole-v1',
         'trials': 3,
         'max_timesteps': 100000,
         'reward_floor':   0.0,
         'reward_ceiling': 500.0,
        },
        {'env_id': 'Acrobot-v1',
         'trials': 3,
         'max_timesteps': 100000,
         'reward_floor': -500.0,
         'reward_ceiling': 0.0,
        },
        {'env_id': 'MountainCar-v0',
         'trials': 3,
         'max_timesteps': 100000,
         'reward_floor': -200.0,
         'reward_ceiling': -100.0,
        },
        {'env_id': 'Pendulum-v0',
         'trials': 3,
         'max_timesteps': 200000,
         'reward_floor': -1400.0,
         'reward_ceiling': 0.0,
        },
    ])

### Autogenerated by tinkerbell.benchmark.convert_benchmark.py

register_benchmark(
    id='Mujoco10M-v0',
    name='Mujoco10M',
    view_group="Control",
    description='Mujoco benchmark with 10M steps',
    scorer=scoring.ClipTo01ThenAverage(),
    tasks=[
        {'env_id': 'Ant-v1',
         'trials': 1,
         'max_timesteps': 1000000,
        },
        {'env_id': 'Hopper-v1',
         'trials': 1,
         'max_timesteps': 1000000,
        },
        {'env_id': 'Humanoid-v1',
         'trials': 1,
         'max_timesteps': 1000000,
        },
        {'env_id': 'HumanoidStandup-v1',
         'trials': 1,
         'max_timesteps': 1000000,
        },
        {'env_id': 'Walker2d-v1',
         'trials': 1,
         'max_timesteps': 1000000,
        }
    ])

register_benchmark(
    id='Mujoco1M-v0',
    name='Mujoco1M',
    view_group="Control",
    description='Mujoco benchmark with 1M steps',
    scorer=scoring.ClipTo01ThenAverage(),
    tasks=[
        {'env_id': 'HalfCheetah-v1',
         'trials': 3,
         'max_timesteps': 1000000,
         'reward_floor':  -280.0,
         'reward_ceiling': 4000.0,
        },
        {'env_id': 'Hopper-v1',
         'trials': 3,
         'max_timesteps': 1000000,
         'reward_floor':  16.0,
         'reward_ceiling': 4000.0,
        },
        {'env_id': 'InvertedDoublePendulum-v1',
         'trials': 3,
         'max_timesteps': 1000000,
         'reward_floor':  53.0,
         'reward_ceiling': 10000.0,
        },
        {'env_id': 'InvertedPendulum-v1',
         'trials': 3,
         'max_timesteps': 1000000,
         'reward_floor':  5.6,
         'reward_ceiling': 1000.0,
        },
        {'env_id': 'Reacher-v1',
         'trials': 3,
         'max_timesteps': 1000000,
         'reward_floor':  -43.0,
         'reward_ceiling': -0.5,
        },
        {'env_id': 'Swimmer-v1',
         'trials': 3,
         'max_timesteps': 1000000,
         'reward_floor':  0.23,
         'reward_ceiling': 500.0,
        },
        {'env_id': 'Walker2d-v1',
         'trials': 3,
         'max_timesteps': 1000000,
         'reward_floor':  1.6,
         'reward_ceiling': 5500.0,
        }
    ])

register_benchmark(
    id='MinecraftEasy-v0',
    name='MinecraftEasy',
    view_group="Minecraft",
    description='Minecraft easy benchmark',
    scorer=scoring.ClipTo01ThenAverage(),
    tasks=[
        {'env_id': 'MinecraftBasic-v0',
         'trials': 2,
         'max_timesteps': 600000,
         'reward_floor': -2200.0,
         'reward_ceiling': 1000.0,
        },
        {'env_id': 'MinecraftDefaultFlat1-v0',
         'trials': 2,
         'max_timesteps': 2000000,
         'reward_floor': -500.0,
         'reward_ceiling': 0.0,
        },
        {'env_id': 'MinecraftTrickyArena1-v0',
         'trials': 2,
         'max_timesteps': 300000,
         'reward_floor': -1000.0,
         'reward_ceiling': 2800.0,
        },
        {'env_id': 'MinecraftEating1-v0',
         'trials': 2,
         'max_timesteps': 300000,
         'reward_floor': -300.0,
         'reward_ceiling': 300.0,
        },
    ])

register_benchmark(
    id='MinecraftMedium-v0',
    name='MinecraftMedium',
    view_group="Minecraft",
    description='Minecraft medium benchmark',
    scorer=scoring.ClipTo01ThenAverage(),
    tasks=[
        {'env_id': 'MinecraftCliffWalking1-v0',
         'trials': 2,
         'max_timesteps': 400000,
         'reward_floor': -100.0,
         'reward_ceiling': 100.0,
        },
        {'env_id': 'MinecraftVertical-v0',
         'trials': 2,
         'max_timesteps': 900000,
         'reward_floor': -1000.0,
         'reward_ceiling': 8040.0,
        },
        {'env_id': 'MinecraftMaze1-v0',
         'trials': 2,
         'max_timesteps': 600000,
         'reward_floor': -1000.0,
         'reward_ceiling': 1000.0,
        },
        {'env_id': 'MinecraftMaze2-v0',
         'trials': 2,
         'max_timesteps': 2000000,
         'reward_floor': -1000.0,
         'reward_ceiling': 1000.0,
        },
    ])

register_benchmark(
    id='MinecraftHard-v0',
    name='MinecraftHard',
    view_group="Minecraft",
    description='Minecraft hard benchmark',
    scorer=scoring.ClipTo01ThenAverage(),
    tasks=[
        {'env_id': 'MinecraftObstacles-v0',
         'trials': 1,
         'max_timesteps': 900000,
         'reward_floor': -1000.0,
         'reward_ceiling': 2080.0,
        },
        {'env_id': 'MinecraftSimpleRoomMaze-v0',
         'trials': 1,
         'max_timesteps': 900000,
         'reward_floor': -1000.0,
         'reward_ceiling': 4160.0,
        },
        {'env_id': 'MinecraftAttic-v0',
         'trials': 1,
         'max_timesteps': 600000,
         'reward_floor': -1000.0,
         'reward_ceiling': 1040.0,
        },
        {'env_id': 'MinecraftComplexityUsage-v0',
         'trials': 1,
         'max_timesteps': 600000,
         'reward_floor': -1000.0,
         'reward_ceiling': 1000.0,
        },
    ])

register_benchmark(
    id='MinecraftVeryHard-v0',
    name='MinecraftVeryHard',
    view_group="Minecraft",
    description='Minecraft very hard benchmark',
    scorer=scoring.ClipTo01ThenAverage(),
    tasks=[
        {'env_id': 'MinecraftMedium-v0',
         'trials': 2,
         'max_timesteps': 1800000,
         'reward_floor': -10000.0,
         'reward_ceiling': 16280.0,
        },
        {'env_id': 'MinecraftHard-v0',
         'trials': 2,
         'max_timesteps': 2400000,
         'reward_floor': -10000.0,
         'reward_ceiling': 32640.0,
        },
    ])

register_benchmark(
    id='MinecraftImpossible-v0',
    name='MinecraftImpossible',
    view_group="Minecraft",
    description='Minecraft impossible benchmark',
    scorer=scoring.ClipTo01ThenAverage(),
    tasks=[
        {'env_id': 'MinecraftDefaultWorld1-v0',
         'trials': 2,
         'max_timesteps': 6000000,
         'reward_floor': -1000.0,
         'reward_ceiling': 1000.0,
        },
    ])
