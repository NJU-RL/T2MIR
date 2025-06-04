import argparse
import environments  # register environments
import gymnasium as gym
import json
import math
import numpy as np
import pickle
import torch
import torch.nn.functional as F
import wandb
from pathlib import Path
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from algorithms.datasets import AD_Dataset
from algorithms.evaluation import eval_policy
from algorithms.policy import DecisionTransformerMoE
from algorithms.tools import set_seed, copy_files, data_loader, download_config, make_dir, model_structure


def warmup_lr(steps: int):
    if steps < args['warmup_steps']:
        return (steps + 1) / args['warmup_steps']
    else:
        if args.get('cosine_lr', True):
            multiplier = (steps - args['warmup_steps']) / (args['total_steps'] - args['warmup_steps'])
            return 0.5 * (1 + math.cos(math.pi * multiplier))
        else:
            return 1.


parser = argparse.ArgumentParser()
parser.add_argument('env', type=str, choices=['darkroom-v0', 'point-robot-v0', 'cheetah-vel-v0', 'cheetah-vel-v3', 'walker-v0', 'reach-v2', 'push-v2'], default="cheetah-vel-v0")
parser.add_argument('--exp', type=str, default='test', help='experiment name')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--log-freq', type=int, default=100)
local_args = parser.parse_args()

args, config_path = download_config(local_args.env, return_config_path=True)
args['device'] = torch.device('cuda', index=local_args.cuda) if torch.cuda.is_available() else torch.device('cpu')
args['exp_name'] = local_args.exp
assert args['training_tasks'] + len(args['eval_tasks']) == args['num_tasks']

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

logdir = make_dir(f"./runs/{args['env_name']}/{args['data_type']}/horizon-{args['train_episode_horizon']}/{local_args.exp}-seed_{local_args.seed}")
copy_files(logdir, ['algorithms'], ['train.py', config_path])
wandb.init(project=f"T2MIR-AD-{args['env_name']}", group=args['data_type'], name=f"{local_args.exp}-seed_{local_args.seed}", config=args)
print(f"Logging at: {logdir}")
return_writer_train = open(logdir / 'train_returns.csv', 'w')
return_writer_eval = open(logdir / 'eval_returns.csv', 'w')
success_writer_train = open(logdir / 'train_success.csv', 'w')
success_writer_eval = open(logdir / 'eval_success.csv', 'w')
return_writer_train.write(','.join(['timesteps', 'task_id'] + [f"episode_{i}" for i in range(args['eval_episodes'])]) + '\n')
return_writer_eval.write(','.join(['timesteps', 'task_id'] + [f"episode_{i}" for i in range(args['eval_episodes'])]) + '\n')
success_writer_train.write(','.join(['timesteps', 'task_id'] + [f"episode_{i}" for i in range(args['eval_episodes'])]) + '\n')
success_writer_eval.write(','.join(['timesteps', 'task_id'] + [f"episode_{i}" for i in range(args['eval_episodes'])]) + '\n')

with open(f"../datasets/{args['env_name']}/task_goals.pkl", 'rb') as fp:
    tasks = pickle.load(fp)
env = gym.make(args['env_name'], tasks=tasks)
discrete_environment = isinstance(env.action_space, gym.spaces.Discrete)
state_dim = np.prod(env.observation_space.shape)
action_dim = env.action_space.n if discrete_environment else np.prod(env.action_space.shape)

set_seed(local_args.seed, env)

dataset_path = Path(f"../datasets/{args['env_name']}/{args['data_type']}")
train_data, train_length, train_idx = data_loader(dataset_path, args['training_tasks'], args['num_tasks'], eval_task_ids=args['eval_tasks'], load_eval_tasks=False)[0]
train_data = {key: np.array(np.split(value, train_idx[1:], axis=0)) for key, value in train_data.items()}
train_dataset = AD_Dataset(train_data, args, state_norm=not discrete_environment)
state_mean, state_std = map(lambda x: torch.from_numpy(x).float().to(args['device']), train_dataset.get_norm_params()) if not discrete_environment else [None, None]
del train_data, train_length, train_idx
# print(f"Evaluation on train tasks target return: {args['target_return_train']}")
# print(f"Evaluation on test tasks target return: {args['target_return_eval']}")

policy = DecisionTransformerMoE(
    state_dim=state_dim,
    action_dim=action_dim,
    config=args,
    action_tanh=(not discrete_environment) and args.get('action_tanh', True),
    n_layer=args['n_layer'],
    n_head=args['n_head'],
    activation_function=args['activation_function'],
    emb_pdrop=args['emb_pdrop'],
    ff_pdrop=args['ff_pdrop'],
    ff_moe_pdrop=args['ff_moe_pdrop'],
    attn_pdrop=args['attn_pdrop'],
    pos_emb=args.get('positional_embedding', 'learnable'),
    attention=args.get('attention', 'flash'),
    sigma_reparam=args.get('sigma_reparam', True),
).to(args['device'])
optimizer = AdamW(policy.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
scheduler = LambdaLR(optimizer, lr_lambda=warmup_lr)
model_structure(policy)

global_steps = 0
policy.train()
pbar = tqdm(total=args['total_steps'], leave=True, ncols=100)
loss_func = F.cross_entropy if discrete_environment else F.mse_loss

while global_steps < args['total_steps']:
    batch = train_dataset.sample_batch_contrastive(args['batch_size'])
    states, actions, rewards = map(lambda x: torch.from_numpy(x).to(args['device']), batch)  # shape: [batch_size, num_episode * max_episode_steps, dim]
    pred_actions, balance_loss, contrastive_loss = policy(states, actions, rewards)
    if discrete_environment:
        pred_actions = pred_actions.reshape(-1, action_dim)
        actions = actions.reshape(-1, action_dim)
    pred_loss = loss_func(pred_actions, actions)
    loss = pred_loss + balance_loss + contrastive_loss

    optimizer.zero_grad()
    loss.backward()
    clip_grad_norm_(policy.parameters(), args['max_grad_norm'])
    optimizer.step()
    scheduler.step()

    policy.update_target_network()

    global_steps += 1

    # logging
    if global_steps % local_args.log_freq == 0:
        wandb.log(
            {
                'train/lr': scheduler.get_last_lr()[0],
                'train/pred_loss': pred_loss,
                'train/balance_loss': balance_loss / args['moe_config']['gate_balance_loss_weight'],
                'train/contrastive_loss': contrastive_loss / args['moe_config']['contrastive_loss_weight'],
            },
            step=global_steps
        )
        pbar.update(local_args.log_freq)

    # save model
    if global_steps % args['eval_freq'] == 0:
        torch.save(
            {
                'policy': policy.state_dict(),
                'state_mean': state_mean,
                'state_std': state_std,
            },
            logdir / 'checkpoints' / f"policy_{global_steps}.pt"
        )
        eval_policy(policy, return_writer_train, return_writer_eval, success_writer_train, success_writer_eval, global_steps, train_task_ids, eval_task_ids, tasks, args, state_mean, state_std, sample_by_timestep=args.get('sample_by_timestep', False))

    if global_steps == args['total_steps']:
        break

pbar.close()
return_writer_train.close()
return_writer_eval.close()
success_writer_train.close()
success_writer_eval.close()
wandb.finish()