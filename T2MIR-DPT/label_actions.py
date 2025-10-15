import argparse
import environments  # register the environments
import pickle
import re
import torch
import gymnasium as gym
import numpy as np
from glob import glob
from pathlib import Path
from shutil import copyfile

from data_collection import SAC, evaluate
from algorithms.tools import set_seed, download_config


################################## parameters ##################################
parser = argparse.ArgumentParser()
parser.add_argument('env', type=str, default='point-robot-v0', choices=['point-robot-v0', 'cheetah-vel-v0', 'cheetah-vel-v3', 'walker-v0'], help='environment')
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--seed', type=int, default=123456, help='0 means no seed')
parser.add_argument('--eval-episodes', type=int, default=10, help="number of episodes for evaluation to decide the best model")
parser.add_argument('--max-ckpt', type=int, default=1000000, help='max timestep of checkpoints to evaluate')
local_args = parser.parse_args()

############################### download config ################################
args = download_config(local_args.env)
args['device'] = torch.device('cuda', index=local_args.cuda) if torch.cuda.is_available() else torch.device('cpu')
args['save_path'] = Path('../datasets') / f"{args['env_name']}" / f"{args['data_type']}"
local_args.max_ckpt = min(args['num_steps'], local_args.max_ckpt)

############################### make environment ###############################
with open(f"../datasets/{args['env_name']}/task_goals.pkl", 'rb') as fp:
    tasks = pickle.load(fp)
env = gym.make(args['env_name'], tasks=tasks)
state_dim = np.prod(env.observation_space.shape)
action_dim = np.prod(env.action_space.shape)
has_success_rate = hasattr(env.unwrapped, 'discount')  # metaworld environments use action repeat

################################### set seed ###################################
set_seed(local_args.seed, env)

# sac model
agent = SAC(env, args['sac_hidden_dim'], args['alpha'], args['sac_lr'], args['gamma'], args['tau'], args['device'])

# use best or specific model to label actions
for task_id in range(args['num_tasks']):
    ckpt_path = Path(f"../datasets/{args['env_name']}/checkpoints/task_{task_id}")
    env.unwrapped.reset_task(task_id)

    # get best model
    if not (ckpt_path / f'agent_best_{local_args.max_ckpt}.pt').exists():
        best_episode_return = -np.inf
        best_model_path = None
        model_paths = glob(f"../datasets/{args['env_name']}/checkpoints/task_{task_id}/*.pt")
        model_paths = list(filter(lambda x: 'agent_best' not in x, model_paths))
        model_paths = sorted(model_paths, key=lambda x: int(re.findall(r'\d+', Path(x).name)[0]))
        if local_args.max_ckpt is not None:
            model_paths = list(filter(lambda model_path: int(re.findall(r'\d+', Path(model_path).name)[0]) <= local_args.max_ckpt, model_paths))

        for model_path in model_paths:
            agent.load(model_path)
            episode_return_mean, _ = evaluate(env, agent, num_eval_episodes=local_args.eval_episodes)
            if episode_return_mean > best_episode_return:
                best_episode_return = episode_return_mean
                best_model_path = model_path
                print(f"task_{task_id}: {best_episode_return}, model_path: {model_path}")

        copyfile(best_model_path, ckpt_path / f'agent_best_{local_args.max_ckpt}.pt')

    # label actions
    agent.load(ckpt_path / f'agent_best_{local_args.max_ckpt}.pt')

    with open(f"../datasets/{args['env_name']}/{args['data_type']}/dataset_task_{task_id}.pkl", 'rb') as fp:
        dataset = pickle.load(fp)
    states = dataset['states'].reshape(-1, args['max_episode_steps'], state_dim)  # shape: [num_episodes, max_episode_steps, state_dim]

    actions = []
    for i in range(states.shape[0]):
        state = states[i]
        action = agent.select_action(state, evaluate=False)
        actions.append(action)
    actions = np.stack(actions, axis=0)

    query_dataset = {'states': states.reshape(-1, state_dim), 'actions': actions.reshape(-1, action_dim)}
    with open(f"../datasets/{args['env_name']}/{args['data_type']}/query_dataset_task_{task_id}.pkl", 'wb') as fp:
        pickle.dump(query_dataset, fp)