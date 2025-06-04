import argparse
import environments  # register environments
import json
import pickle
import torch
import wandb
import gymnasium as gym
import numpy as np
from tqdm import tqdm

from algorithms.evaluation import eval_policy
from algorithms.policy import DPTTransformerMOE
from algorithms.tools import set_seed, copy_files, download_config, make_dir


parser = argparse.ArgumentParser()
parser.add_argument('env', type=str, choices=['darkroom-v0', 'point-robot-v0', 'cheetah-vel-v0', 'cheetah-vel-v3', 'walker-v0', 'reach-v2', 'push-v2'], help="environment")
parser.add_argument('id', type=int)
parser.add_argument('--exp', type=str, help='experiment name')
parser.add_argument('--start-ckpt', type=int, default=0, help='start evaluating from this checkpoint')
parser.add_argument('--stop-ckpt', type=int, default=1000000, help='stop evaluating at this checkpoint')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--seed-eval', type=int, default=None, help='seed for reset environment in evaluation')
parser.add_argument('--cuda', type=int, default=0)
local_args = parser.parse_args()
checkpoints = []

args, config_path = download_config(local_args.env, return_config_path=True)
args['device'] = torch.device('cuda', index=local_args.cuda) if torch.cuda.is_available() else torch.device('cpu')
args['exp_name'] = local_args.exp
args['eval_reset_seed'] = local_args.seed_eval
assert args['eval_horizon'] % args['max_episode_steps'] == 0, "eval_horizon should be divisible by max_episode_steps"

with open(f"../datasets/{args['env_name']}/{args['data_type']}/return_scale.json", 'r') as fp:
    task_info = json.load(fp)
# compute target return
train_task_ids = sorted((set(range(args['num_tasks']))) - set(args['eval_tasks']))[:5]
eval_task_ids = args['eval_tasks']
# target_returns = []
# for task_id in train_task_ids:
#     target_returns.append(task_info[f"task_{task_id}"][1])
# args['target_return_train'] = np.mean(target_returns)
# target_returns = []
# for task_id in eval_task_ids:
#     target_returns.append(task_info[f"task_{task_id}"][1])
# args['target_return_eval'] = np.mean(target_returns)

logdir = make_dir(f"./runs/{args['env_name']}/{args['data_type']}/dpt-prompt-{args['prompt_horizon']}/{local_args.exp}-seed_{local_args.seed}", ignore_exist=True)
assert logdir.exists(), f"logdir {logdir} does not exist"
(logdir / f"eval_{local_args.id}-seed_{local_args.seed_eval}").mkdir()
wandb.init(project=f"T2MIR-DPT-{args['env_name']}", group=args['data_type'], name=f"{local_args.exp}-seed_{local_args.seed}-eval_{local_args.id}-seed_{local_args.seed_eval}", config=args)
copy_files(logdir / f"eval_{local_args.id}-seed_{local_args.seed_eval}", files=['eval_online.py', 'algorithms/evaluation.py', config_path], parent=False)
print(f"Logging at: {logdir}")
return_writer_train = open(logdir / f"eval_{local_args.id}-seed_{local_args.seed_eval}/train_returns.csv", 'w')
return_writer_eval = open(logdir / f"eval_{local_args.id}-seed_{local_args.seed_eval}/eval_returns.csv", 'w')
success_writer_train = open(logdir / f"eval_{local_args.id}-seed_{local_args.seed_eval}/train_success.csv", 'w')
success_writer_eval = open(logdir / f"eval_{local_args.id}-seed_{local_args.seed_eval}/eval_success.csv", 'w')
return_writer_train.write(','.join(['timesteps', 'task_id'] + [f"episode_{i}" for i in range(args['eval_episodes'])]) + '\n')
return_writer_eval.write(','.join(['timesteps', 'task_id'] + [f"episode_{i}" for i in range(args['eval_episodes'])]) + '\n')
success_writer_train.write(','.join(['timesteps', 'task_id'] + [f"episode_{i}" for i in range(args['eval_episodes'])]) + '\n')
success_writer_eval.write(','.join(['timesteps', 'task_id'] + [f"episode_{i}" for i in range(args['eval_episodes'])]) + '\n')

with open(f"../datasets/{args['env_name']}/task_goals.pkl", 'rb') as fp:
    tasks = pickle.load(fp)
env = gym.make(args['env_name'], tasks=tasks).unwrapped
state_dim = np.prod(env.observation_space.shape)
action_dim = np.prod(env.action_space.shape)

# print(f"Evaluation on train tasks target return: {args['target_return_train']}")
# print(f"Evaluation on eval tasks target return: {args['target_return_eval']}")

set_seed(local_args.seed, env)

policy = DPTTransformerMOE(
    state_dim=state_dim,
    action_dim=action_dim,
    config=args,
    action_tanh=args.get('action_tanh', True),
).to(args['device'])

start_ckpt = args['eval_freq'] if local_args.start_ckpt == 0 else local_args.start_ckpt
for ckpt in tqdm(sorted(checkpoints + list(range(start_ckpt, min(args['total_steps'], local_args.stop_ckpt)+1, args['eval_freq']))), leave=True, ncols=100):
    ckpt_dict = torch.load(logdir / f"checkpoints/policy_{ckpt}.pt", map_location=args['device'], weights_only=True)
    policy.load_state_dict(ckpt_dict['policy'], strict=True)
    state_mean, state_std = ckpt_dict['state_mean'], ckpt_dict['state_std']
    eval_policy(policy, return_writer_train, return_writer_eval, success_writer_train, success_writer_eval, ckpt, train_task_ids, eval_task_ids, tasks, args, state_mean, state_std)

return_writer_train.close()
return_writer_eval.close()
success_writer_train.close()
success_writer_eval.close()
wandb.finish()