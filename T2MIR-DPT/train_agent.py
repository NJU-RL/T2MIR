import argparse
import environments  # register the environments
import pickle
import torch
import gymnasium as gym
import numpy as np
from itertools import count
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from data_collection import SAC, ReplayBuffer, evaluate
from algorithms.tools import set_seed, download_config, copy_files


################################## parameters ##################################
parser = argparse.ArgumentParser()
parser.add_argument('env', type=str, default='point-robot-v0', choices=['point-robot-v0', 'cheetah-vel-v0', 'cheetah-vel-v3', 'walker-v0'], help='environment')
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--seed', type=int, default=123456, help='0 means no seed')
parser.add_argument('--log-freq', type=int, default=100)
parser.add_argument('--task-id-start', type=int, default=0, help='from task_id_start to task_id_end-1')
parser.add_argument('--task-id-end', type=int, default=5)
parser.add_argument('--load-ckpt', type=int, default=None, help='load the checkpoint from the specified step, and continue training')
local_args = parser.parse_args()
has_success_rate = local_args.env in ['reach-v2', 'push-v2']  # metaworld environments use action repeat

############################### download config ################################
args, config_path = download_config(local_args.env, return_config_path=True)
args['device'] = torch.device('cuda', index=local_args.cuda) if torch.cuda.is_available() else torch.device('cpu')
args['save_path'] = Path('../datasets') / f"{args['env_name']}"  # save path
args['save_path'].mkdir(exist_ok=True, parents=True)
copy_files(args['save_path'], folders=['algorithms', 'data_collection', 'environments'], files=['train_agent.py', config_path])

############################### make environment ###############################
with open(f"../datasets/{args['env_name']}/task_goals.pkl", 'rb') as fp:
    tasks = pickle.load(fp)
env = gym.make(args['env_name'], tasks=tasks)
state_dim = np.prod(env.observation_space.shape)
action_dim = np.prod(env.action_space.shape)

################################### set seed ###################################
set_seed(local_args.seed, env)

################################### training ###################################
for task_id in range(local_args.task_id_start, local_args.task_id_end):
    env.unwrapped.reset_task(task_id)
    
    agent = SAC(env, args['sac_hidden_dim'], args['alpha'], args['sac_lr'], args['gamma'], args['tau'], args['device'])
    # replay buffer
    memory = ReplayBuffer(args['capacity'], state_dim, action_dim, use_success_rate=has_success_rate)
    logger = SummaryWriter(args['save_path'] / f"data_collection/task_{task_id}")

    # training loop
    total_step = 0
    epi_returns = []  # episode returns
    
    # save model path
    save_path = args['save_path'] / f"checkpoints/task_{task_id}"
    save_path.mkdir(parents=True, exist_ok=True)
    
    if local_args.load_ckpt is not None:
        total_step = local_args.load_ckpt
        agent.load(save_path / f"agent_{local_args.load_ckpt}.pt")
        memory.load_buffer(args['save_path'] / f"replay_buffer/dataset_task_{task_id}.pkl")
    
    for i_episode in count(1):
        epi_return = 0.
        state, _ = env.reset()
        for step in range(args['max_episode_steps']):
            if total_step < args['start_step']:
                action = env.action_space.sample()  # sample random action
            else:
                action = agent.select_action(state)  # sample action from policy
            
            next_state, reward, done, truncated, info = env.step(action)
            # Ignore the "done" signal if it comes from hitting the time horizon.
            mask = True if (step == args['max_episode_steps'] - 1) else (not done)  # mask 为 True 表示考虑未来价值
            memory.push(state, action, reward, next_state, done, mask, success_rate=info.get('success', 0.0))

            state = next_state
            epi_return += reward
            total_step += 1
            
            # update sac agent
            if len(memory) > args['start_step']:
                q1_loss, q2_loss, policy_loss = agent.update(memory, args['sac_batch_size'])

                if total_step % local_args.log_freq == 0:
                    logger.add_scalar('loss/q1_loss', q1_loss, total_step)
                    logger.add_scalar('loss/q2_loss', q2_loss, total_step)
                    logger.add_scalar('loss/policy_loss', policy_loss, total_step)

            # save models
            if total_step % args['model_save_freq'] == 0:
                agent.save(save_path / f"agent_{total_step}.pt")

            if done:
                break

        # evaluate the sac agent
        eval_return, success_rate = evaluate(env, agent, args['num_eval_episodes'])
        print(f"Task {task_id}, episode {i_episode}, eval return {eval_return:.4f}, success rate {success_rate:.2f}")
        logger.add_scalar('return/train', epi_return, total_step)
        logger.add_scalar('return/eval', eval_return, total_step)
        logger.add_scalar('success_rate/eval', success_rate, total_step)
        
        epi_returns.append(epi_return)
        if total_step >= args['num_steps']:
            break

    logger.close()
    # save the replay buffer
    save_buffer_path = args['save_path'] / 'replay_buffer'
    save_buffer_path.mkdir(exist_ok=True)
    memory.save_buffer(save_buffer_path / f'dataset_task_{task_id}.pkl')