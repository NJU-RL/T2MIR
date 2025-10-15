import argparse
import environments  # register the environments
import gymnasium as gym
import numpy as np
import pickle
from pathlib import Path
from stable_baselines3 import PPO

from algorithms.tools import set_seed, download_config


################################## parameters ##################################
parser = argparse.ArgumentParser()
parser.add_argument('env', type=str, default='reach-v2', choices=['reach-v2', 'push-v2'], help='environment')
parser.add_argument('--seed', type=int, default=123456, help='0 means no seed')
local_args = parser.parse_args()

############################### download config ################################
args = download_config(local_args.env)
args['save_path'] = Path('../datasets') / f"{args['env_name']}" / f"{args['data_type']}"

############################ generate query dataset ############################
# use actions from the dataset instead of from expert policy
for i in range(args['num_tasks']):
    with open(f"../datasets/{args['env_name']}/{args['data_type']}/dataset_task_{i}.pkl", 'rb') as fp:
        dataset = pickle.load(fp)
    query_dataset = {}
    query_dataset['states'] = dataset['states']
    query_dataset['actions'] = dataset['actions']
    with open(f"../datasets/{args['env_name']}/{args['data_type']}/query_dataset_task_{i}.pkl", 'wb') as fp:
        pickle.dump(query_dataset, fp)

# ############################### make environment ###############################
# with open(f"../datasets/{args['env_name']}/task_goals.pkl", 'rb') as fp:
#     tasks = pickle.load(fp)
# env = gym.make(args['env_name'], tasks=tasks)
# state_dim = np.prod(env.observation_space.shape)
# action_dim = np.prod(env.action_space.shape)

# ################################### set seed ###################################
# set_seed(local_args.seed, env)

# # use best or specific model to label actions
# for task_id in range(args['num_tasks']):
#     model_path = f"../datasets/{args['env_name']}/checkpoints/task_{task_id}/agent_best.zip"

#     # label actions
#     agent = PPO.load(model_path, device='cpu')

#     with open(f"../datasets/{args['env_name']}/{args['data_type']}/dataset_task_{task_id}.pkl", 'rb') as fp:
#         dataset = pickle.load(fp)
#     states = dataset['states'].reshape(-1, args['max_episode_steps'], state_dim)  # shape: [num_episodes, max_episode_steps, state_dim]

#     actions = []
#     for i in range(states.shape[0]):
#         state = states[i]
#         action = agent.predict(state, deterministic=False)[0]
#         actions.append(action)
#     actions = np.stack(actions, axis=0)

#     query_dataset = {'states': states.reshape(-1, state_dim), 'actions': actions.reshape(-1, action_dim)}
#     with open(f"../datasets/{args['env_name']}/{args['data_type']}/query_dataset_task_{task_id}.pkl", 'wb') as fp:
#         pickle.dump(query_dataset, fp)