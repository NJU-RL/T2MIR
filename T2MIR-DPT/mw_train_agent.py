import argparse
import environments  # register environments
import os
import pickle
import gymnasium as gym
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CallbackList, EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from algorithms.tools import set_seed, download_config, copy_files


class CustomEvalCallback(EvalCallback):
    def __init__(self, task_id, best_checkpoint_path, **kwargs):
        super(CustomEvalCallback, self).__init__(**kwargs)
        self.task_id = task_id
        self.best_checkpoint_path = best_checkpoint_path
        self._best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        result = super()._on_step()

        if self.n_calls % self.eval_freq == 0:
            mean_reward = self.last_mean_reward
            # std_reward = np.std(self.evaluations_results[-1])
            print(f"Eval @ task {self.task_id} | step: {self.num_timesteps} | mean reward: {mean_reward:.2f}")
            if mean_reward > self._best_mean_reward:
                self._best_mean_reward = mean_reward
                self.model.save(self.best_checkpoint_path)

        return result


class CustomCheckpointCallback(CheckpointCallback):
    def _checkpoint_path(self, checkpoint_type: str = "", extension: str = "") -> str:
        """
        Helper to get checkpoint path for each type of checkpoint.

        :param checkpoint_type: empty for the model, "replay_buffer_"
            or "vecnormalize_" for the other checkpoints.
        :param extension: Checkpoint file extension (zip for model, pkl for others)
        :return: Path to the checkpoint
        """
        num_timesteps_plus = local_args.load_ckpt if local_args.load_ckpt is not None else 0
        return os.path.join(self.save_path, f"{self.name_prefix}_{checkpoint_type}{self.num_timesteps+num_timesteps_plus}_steps.{extension}")


################################## parameters ##################################
parser = argparse.ArgumentParser()
parser.add_argument('env', type=str, default='reach-v2', choices=['reach-v2', 'push-v2'], help='environment')
parser.add_argument('--seed', type=int, default=123456, help='0 means no seed')
parser.add_argument('--task-id-start', type=int, default=0, help='from task_id_start to task_id_end-1')
parser.add_argument('--task-id-end', type=int, default=5)
parser.add_argument('--load-ckpt', type=int, default=None, help='load ckpt from this step and continue training')
local_args = parser.parse_args()

############################### download config ################################
args, config_path = download_config(local_args.env, return_config_path=True)
args['save_path'] = Path('../datasets') / f"{args['env_name']}/checkpoints"  # save path
args['save_path'].mkdir(exist_ok=True, parents=True)
copy_files(f"../datasets/{args['env_name']}", folders=['algorithms', 'data_collection', 'environments'], files=['mw_train_agent.py', config_path])

############################### make environment ###############################
with open(f"../datasets/{args['env_name']}/task_goals.pkl", 'rb') as fp:
    tasks = pickle.load(fp)
train_env = gym.make(args['env_name'], tasks=tasks)
eval_env = gym.make(args['env_name'], tasks=tasks)

################################### set seed ###################################
set_seed(local_args.seed, train_env)

################################### training ###################################
for task_id in range(local_args.task_id_start, local_args.task_id_end):
    train_env.unwrapped.reset_task(task_id)
    eval_env.unwrapped.reset_task(task_id)
    save_path = args['save_path'] / f"task_{task_id}"

    eval_callback = CustomEvalCallback(
        task_id=task_id,
        best_checkpoint_path = save_path / 'agent_best.zip',
        eval_env=Monitor(eval_env),
        eval_freq=args['model_save_freq'],
        deterministic=True,
        render=False,
        verbose=0
    )

    if local_args.load_ckpt is None:
        model = PPO("MlpPolicy", train_env, n_steps=args['n_steps'], batch_size=args['ppo_batch_size'], n_epochs=args['n_epochs'], verbose=0, tensorboard_log=f"../datasets/{args['env_name']}/data_collection/task_{task_id}/", seed=local_args.seed, device='cpu')
        # model = SAC("MlpPolicy", train_env, batch_size=args['sac_batch_size'], learning_starts=args['start_step'], verbose=0, tensorboard_log=f"../datasets/{args['env_name']}/data_collection/task_{task_id}/")
    else:
        model = PPO.load(path=save_path / f"agent_{local_args.load_ckpt}_steps.zip", env=train_env, n_steps=args['n_steps'], batch_size=args['ppo_batch_size'], n_epochs=args['n_epochs'], verbose=0, tensorboard_log=f"../datasets/{args['env_name']}/data_collection/task_{task_id}/", seed=local_args.seed)
    callback = CallbackList([
        CustomCheckpointCallback(save_freq=args['model_save_freq'], save_path=save_path, name_prefix='agent'),
        eval_callback
    ])
    model.learn(total_timesteps=args['num_steps'], callback=callback)