import gymnasium as gym
import numpy as np
import torch
import wandb
from gymnasium.vector import SyncVectorEnv
from torch import Tensor
from typing import Dict, Any, List

from algorithms.policy import DPTTransformerMOE


def write_returns(returns: List[float], fp, timeteps: int, task_id: int):
    fp.write(','.join(map(str, [timeteps, task_id] + returns)) + '\n')
    fp.flush()

@torch.no_grad()
def evaluate_online(
    task_ids: List[int],
    tasks: Any,
    policy,
    config: Dict[str, Any],
    state_mean: Tensor,
    state_std: Tensor,
) -> List[np.ndarray]:
    num_tasks = len(task_ids)
    vec_envs = SyncVectorEnv([lambda: gym.make(config['env_name'], tasks=tasks) for _ in task_ids])
    for i, env in enumerate(vec_envs.envs):
        env.unwrapped.reset_task(task_ids[i])

    avg_epi_returns = []
    avg_epi_successes = []
    eval_horizon = config['eval_horizon']
    max_episode_steps = config['max_episode_steps']
    maintain_horizon = eval_horizon + max_episode_steps

    state_dim = np.prod(env.observation_space.shape)
    action_dim = np.prod(env.action_space.shape)

    if policy.env_discrete:
        states_buffer = torch.zeros([maintain_horizon, num_tasks], dtype=torch.long, device=config['device'])
        actions_buffer = torch.zeros([maintain_horizon, num_tasks], dtype=torch.long, device=config['device'])
    else:
        states_buffer = torch.zeros([maintain_horizon, num_tasks, state_dim], dtype=torch.float, device=config['device'])
        actions_buffer = torch.zeros([maintain_horizon, num_tasks, action_dim], dtype=torch.float, device=config['device'])

    rewards_buffer = torch.zeros([maintain_horizon, num_tasks], dtype=torch.float, device=config['device'])
    timesteps_buffer = torch.arange(eval_horizon + 1, dtype=torch.long, device=config['device']).reshape(-1, 1).repeat(1, num_tasks)
    buffer_point = 0
    
    for _ in range(config['eval_episodes']):
        avg_epi_return = np.zeros([num_tasks,])
        avg_epi_success = np.zeros([num_tasks,])
        state, _ = vec_envs.reset(seed=config.get('eval_reset_seed', None))
        reward = np.zeros(num_tasks)

        if buffer_point == maintain_horizon:
            buffer_point = eval_horizon

        prompt_horizon = min((buffer_point // max_episode_steps) * max_episode_steps, eval_horizon)
        prompt_states = states_buffer[:prompt_horizon]
        prompt_actions = actions_buffer[:prompt_horizon]
        prompt_rewards = rewards_buffer[:prompt_horizon]

        for t in range(max_episode_steps):
            states_buffer[buffer_point] = torch.tensor(state, dtype=torch.float, device=config['device'])
            query_state = states_buffer[buffer_point]
            timesteps = timesteps_buffer[:prompt_horizon+1]

            if policy.env_discrete:
                action = policy.forward(
                    prompt_states.permute(1, 0),
                    prompt_actions.permute(1, 0),
                    prompt_rewards.permute(1, 0).unsqueeze(-1),
                    query_state.unsqueeze(1),
                    timesteps.permute(1, 0),
                    eval=True
                )
                dist = torch.distributions.Categorical(logits=action)
                action = dist.mode
            else:
                action = policy.forward(
                    ((prompt_states - state_mean) / state_std).permute(1, 0, 2),
                    prompt_actions.permute(1, 0, 2),
                    prompt_rewards.permute(1, 0).unsqueeze(-1),
                    ((query_state - state_mean) / state_std).unsqueeze(1),
                    timesteps.permute(1, 0),
                    eval=True
                )

            state, reward, _, _, info = vec_envs.step(action.cpu().numpy())
            avg_epi_return += reward
            avg_epi_success += info.get('success', np.zeros_like(avg_epi_success))

            actions_buffer[buffer_point] = action.clone().detach()
            rewards_buffer[buffer_point] = torch.tensor(reward, dtype=torch.float, device=config['device'])
            buffer_point += 1

        avg_epi_returns.append(avg_epi_return)
        avg_epi_successes.append(avg_epi_success)

        if buffer_point > eval_horizon:
            states_buffer = states_buffer.roll(-max_episode_steps, dims=0)
            actions_buffer = actions_buffer.roll(-max_episode_steps, dims=0)
            rewards_buffer = rewards_buffer.roll(-max_episode_steps, dims=0)

    return np.array(avg_epi_returns).transpose(1, 0), np.array(avg_epi_successes).transpose(1, 0)

def eval_policy(policy: DPTTransformerMOE, return_writer_train, return_writer_eval, success_writer_train, success_writer_eval, timestep: int, train_task_ids: List[int], eval_task_ids: List[int], tasks: Any, configs: Dict[str, Any], state_mean: Tensor, state_std: Tensor):
    policy.eval()

    # evaluation on training tasks
    # avg_returns, avg_successes = evaluate_online(
    #     train_task_ids,
    #     tasks,
    #     policy,
    #     configs,
    #     state_mean,
    #     state_std,
    # )
    # for i, id in enumerate(train_task_ids):
    #     write_returns(avg_returns[i].tolist(), return_writer_train, timestep, id)
    #     write_returns(avg_successes[i].tolist(), success_writer_train, timestep, id)
    # wandb.log({'train/average return': avg_returns.mean()}, step=timestep)
    # wandb.log({'train/average success': avg_successes.mean()}, step=timestep)

    # evaluation on test tasks
    avg_returns, avg_successes = evaluate_online(
        eval_task_ids,
        tasks,
        policy,
        configs,
        state_mean,
        state_std,
    )
    for i, id in enumerate(eval_task_ids):
        write_returns(avg_returns[i].tolist(), return_writer_eval, timestep, id)
        write_returns(avg_successes[i].tolist(), success_writer_eval, timestep, id)
    wandb.log({'eval/average return': avg_returns.mean()}, step=timestep)
    wandb.log({'eval/average success': avg_successes.mean()}, step=timestep)

    policy.train()