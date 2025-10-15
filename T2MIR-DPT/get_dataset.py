import argparse
import environments  # register environments
import json
import pickle
import re
import torch
import gymnasium as gym
import numpy as np
from collections import OrderedDict
from glob import glob
from pathlib import Path

from data_collection.sac import SAC
from data_collection.ReplayBuffer import ReplayBuffer
from typing import List
from algorithms.tools import set_seed, download_config


def roll_out(env: gym.Env, replaybuffer: ReplayBuffer, returns: List[float]):
    total_timestep = 0
    while total_timestep < local_args.capacity:
        episode_return = 0.
        state, _ = env.reset()
        for step in range(args['max_episode_steps']):
            action = agent.select_action(state, False)
            next_state, reward, done, truncated, info = env.step(action)
            mask = True if (step == args['max_episode_steps'] - 1) else (not done)
            replaybuffer.push(state, action, reward, next_state, done, mask, info.get('success', 0.0))
            
            state = next_state
            total_timestep += 1
            episode_return += reward
            
            if done:
                break
        
        returns.append(episode_return)

    return returns, replaybuffer


################################## parameters ##################################
parser = argparse.ArgumentParser()
parser.add_argument('env', type=str, default='point-robot-v0', choices=['point-robot-v0', 'cheetah-vel-v0', 'cheetah-vel-v3', 'walker-v0'], help='environment')
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--seed', type=int, default=123456)
parser.add_argument('--task-id-start', type=int, default=0, help='from task_id_start to task_id_end-1')
parser.add_argument('--task-id-end', type=int, default=5)
parser.add_argument('--capacity', type=int, default=800, help='total timesteps of one checkpoint')
parser.add_argument('--max-ckpt', type=int, default=None, help='maximum checkpoint to use')
local_args, rest_args = parser.parse_known_args()
has_success_rate = local_args.env in ['reach-v2']

############################### download config ################################
args = download_config(local_args.env)
args['device'] = torch.device('cuda', index=local_args.cuda) if torch.cuda.is_available() else torch.device('cpu')
args['save_path'] = Path('../datasets') / f"{args['env_name']}" / f"{args['data_type']}"  # save path
args['save_path'].mkdir(exist_ok=True, parents=True)

############################### make environment ###############################
with open(f"../datasets/{args['env_name']}/task_goals.pkl", 'rb') as fp:
    tasks = pickle.load(fp)
env = gym.make(args['env_name'], tasks=tasks)
state_dim = np.prod(env.observation_space.shape)
action_dim = np.prod(env.action_space.shape)

################################### set seed ###################################
set_seed(local_args.seed, env)

################################### roll out ###################################
return_scale_info = OrderedDict()
agent = SAC(env, args['sac_hidden_dim'], args['alpha'], args['sac_lr'], args['gamma'], args['tau'], args['device'])
for task_id in range(local_args.task_id_start, local_args.task_id_end):
    model_paths = glob(f"../datasets/{args['env_name']}/checkpoints/task_{task_id}/*.pt")
    # model_paths = glob(f"../datasets/{args['env_name']}/checkpoints/{tasks[task_id]}/*.pt")
    model_paths = list(filter(lambda x: 'agent_best' not in x, model_paths))
    model_paths = sorted(model_paths, key=lambda x: int(re.findall(r'\d+', Path(x).name)[0]))
    if local_args.max_ckpt is not None:
        model_paths = list(filter(lambda model_path: int(re.findall(r'\d+', Path(model_path).name)[0]) <= local_args.max_ckpt, model_paths))
    replaybuffer = ReplayBuffer(local_args.capacity * len(model_paths), state_dim, action_dim, has_success_rate)
    episode_returns = []

    env.unwrapped.reset_task(task_id)
    for model_path in model_paths:
        agent.load(model_path)
        episode_returns, replaybuffer = roll_out(env, replaybuffer, episode_returns)

    replaybuffer.save_buffer(args['save_path'] / f"dataset_task_{task_id}.pkl")
    return_scale_info[f"task_{task_id}"] = [min(episode_returns), max(episode_returns)]

with open(args['save_path'] / f"return_scale_info_{local_args.task_id_start}.json", 'w') as fp:
    json.dump(return_scale_info, fp, indent=4)