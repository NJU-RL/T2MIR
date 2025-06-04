import torch
import numpy as np
import gymnasium as gym
import wandb
from gymnasium.vector import SyncVectorEnv
from torch import Tensor
from typing import Dict, Any, List
from algorithms.policy import DecisionTransformerMoE


@torch.no_grad()
def evaluate(
    task_ids: List[int],
    tasks: Any,
    policy: DecisionTransformerMoE,
    config: Dict[str, Any],
    original_rtg: np.ndarray,
    state_mean: Tensor,
    state_std: Tensor,
    rand_first: bool = False,
) -> List[float]:
    # environments
    num_tasks = len(task_ids)
    vec_envs = SyncVectorEnv([lambda: gym.make(config['env_name'], tasks=tasks) for _ in task_ids])
    for i, env in enumerate(vec_envs.envs):
        env.unwrapped.reset_task(task_ids[i])

    avg_epi_returns = []
    eval_horizon = config['eval_horizon']
    max_episode_steps = config['max_episode_steps']
    return_scale = config['return_scale']
    maintain_horizon = eval_horizon + max_episode_steps

    # environment parameters
    state_dim = np.prod(env.observation_space.shape)
    action_dim = np.prod(env.action_space.shape)

    rtgs_buffer = torch.zeros([maintain_horizon, num_tasks], dtype=torch.float, device=config['device'])
    states_buffer = torch.zeros([maintain_horizon, num_tasks, state_dim], dtype=torch.float, device=config['device'])
    actions_buffer = torch.zeros([maintain_horizon, num_tasks, action_dim], dtype=torch.float, device=config['device'])
    timesteps_buffer = torch.zeros([maintain_horizon, num_tasks], dtype=torch.long, device=config['device'])
    buffer_point = 0
    
    for _ in range(config['eval_episodes']):
        avg_epi_return = np.zeros([num_tasks,])
        state, _ = vec_envs.reset()
        reward = np.zeros(num_tasks)
        rtg = original_rtg.copy()
        for t in range(max_episode_steps):
            if buffer_point == maintain_horizon:
                buffer_point = eval_horizon
            rtg -= reward
            rtgs_buffer[buffer_point] = torch.tensor(rtg, dtype=torch.float, device=config['device'])
            states_buffer[buffer_point] = torch.tensor(state, dtype=torch.float, device=config['device'])
            timesteps_buffer[buffer_point] = torch.full([num_tasks,], t, dtype=torch.long, device=config['device'])

            if t == 0 and rand_first:
                action = torch.tensor(vec_envs.action_space.sample())
            else:
                action = policy.get_action(
                    (rtgs_buffer[:buffer_point+1] / return_scale).permute(1, 0).unsqueeze(-1),
                    ((states_buffer[:buffer_point+1] - state_mean) / state_std).permute(1, 0, 2),
                    (actions_buffer[:buffer_point+1]).permute(1, 0, 2),
                    (timesteps_buffer[:buffer_point+1]).permute(1, 0)
                )
            state, reward, _, _, _ = vec_envs.step(action.cpu().numpy())
            avg_epi_return += reward

            actions_buffer[buffer_point] = action.clone().float().to(config['device'])
            buffer_point += 1

        avg_epi_returns.append(avg_epi_return)

        rtgs_buffer[buffer_point-max_episode_steps:buffer_point] += torch.tensor(avg_epi_return - original_rtg, dtype=torch.float, device=config['device'])
        
        returns = rtgs_buffer[:buffer_point:max_episode_steps]
        sorted_indices = torch.argsort(returns, dim=0)
        for i in range(num_tasks):
            rtgs_buffer[:buffer_point, i] = rtgs_buffer[:buffer_point, i].view(-1, max_episode_steps)[sorted_indices[:, i]].view(-1)
            states_buffer[:buffer_point, i] = states_buffer[:buffer_point, i].view(-1, max_episode_steps, state_dim)[sorted_indices[:, i]].view(-1, state_dim)
            actions_buffer[:buffer_point, i] = actions_buffer[:buffer_point, i].view(-1, max_episode_steps, action_dim)[sorted_indices[:, i]].view(-1, action_dim)

        if buffer_point > eval_horizon:
            rtgs_buffer = torch.cat([rtgs_buffer[-eval_horizon:], rtgs_buffer[:max_episode_steps]], dim=0)
            states_buffer = torch.cat([states_buffer[-eval_horizon:], states_buffer[:max_episode_steps]], dim=0)
            actions_buffer = torch.cat([actions_buffer[-eval_horizon:], actions_buffer[:max_episode_steps]], dim=0)

    return np.array(avg_epi_returns).transpose(1, 0)

@torch.no_grad()
def evaluate_with_random(
    task_ids: List[int],
    tasks: Any,
    policy: DecisionTransformerMoE,
    config: Dict[str, Any],
    original_rtg: np.ndarray,
    state_mean: Tensor,
    state_std: Tensor,
) -> List[float]:
    # environments
    num_tasks = len(task_ids)
    vec_envs = SyncVectorEnv([lambda: gym.make(config['env_name'], tasks=tasks) for _ in task_ids])
    for i, env in enumerate(vec_envs.envs):
        env.reset_task(task_ids[i])

    avg_epi_returns = []
    eval_horizon = config['eval_horizon']
    max_episode_steps = config['max_episode_steps']
    return_scale = config['return_scale']
    maintain_horizon = eval_horizon + max_episode_steps

    # environment parameters
    state_dim = np.prod(env.observation_space.shape)
    action_dim = np.prod(env.action_space.shape)

    rtgs_buffer = torch.zeros([maintain_horizon, num_tasks], dtype=torch.float, device=config['device'])
    states_buffer = torch.zeros([maintain_horizon, num_tasks, state_dim], dtype=torch.float, device=config['device'])
    actions_buffer = torch.zeros([maintain_horizon, num_tasks, action_dim], dtype=torch.float, device=config['device'])
    timesteps_buffer = torch.zeros([maintain_horizon, num_tasks], dtype=torch.long, device=config['device'])
    buffer_point = 0
    
    for episode in range(config['eval_episodes'] + config['random_episodes']):
        avg_epi_return = np.zeros([num_tasks,])
        state, _ = vec_envs.reset()
        reward = np.zeros(num_tasks)
        rtg = original_rtg.copy()
        for t in range(max_episode_steps):
            if buffer_point == maintain_horizon:
                buffer_point = eval_horizon
            rtg -= reward
            rtgs_buffer[buffer_point] = torch.tensor(rtg, dtype=torch.float, device=config['device'])
            states_buffer[buffer_point] = torch.tensor(state, dtype=torch.float, device=config['device'])
            timesteps_buffer[buffer_point] = torch.full([num_tasks,], t, dtype=torch.long, device=config['device'])

            if episode < config['random_episodes']:
                action = torch.tensor(vec_envs.action_space.sample())
            else:
                action = policy.get_action(
                    (rtgs_buffer[:buffer_point+1] / return_scale).permute(1, 0).unsqueeze(-1),
                    ((states_buffer[:buffer_point+1] - state_mean) / state_std).permute(1, 0, 2),
                    (actions_buffer[:buffer_point+1]).permute(1, 0, 2),
                    (timesteps_buffer[:buffer_point+1]).permute(1, 0)
                )
            state, reward, _, _, _ = vec_envs.step(action.cpu().numpy())
            avg_epi_return += reward

            actions_buffer[buffer_point] = torch.tensor(action, dtype=torch.float, device=config['device'])
            buffer_point += 1

        if episode >= config['random_episodes']:
            avg_epi_returns.append(avg_epi_return)

        rtgs_buffer[buffer_point-max_episode_steps:buffer_point] += torch.tensor(avg_epi_return - original_rtg, dtype=torch.float, device=config['device'])
        
        returns = rtgs_buffer[:buffer_point:max_episode_steps]
        sorted_indices = torch.argsort(returns, dim=0)
        for i in range(num_tasks):
            rtgs_buffer[:buffer_point, i] = rtgs_buffer[:buffer_point, i].view(-1, max_episode_steps)[sorted_indices[:, i]].view(-1)
            states_buffer[:buffer_point, i] = states_buffer[:buffer_point, i].view(-1, max_episode_steps, state_dim)[sorted_indices[:, i]].view(-1, state_dim)
            actions_buffer[:buffer_point, i] = actions_buffer[:buffer_point, i].view(-1, max_episode_steps, action_dim)[sorted_indices[:, i]].view(-1, action_dim)

        if buffer_point > eval_horizon:
            rtgs_buffer = torch.cat([rtgs_buffer[-eval_horizon:], rtgs_buffer[:max_episode_steps]], dim=0)
            states_buffer = torch.cat([states_buffer[-eval_horizon:], states_buffer[:max_episode_steps]], dim=0)
            actions_buffer = torch.cat([actions_buffer[-eval_horizon:], actions_buffer[:max_episode_steps]], dim=0)

    return np.array(avg_epi_returns).transpose(1, 0)


@torch.no_grad()
def evaluate_reward(
    task_ids: List[int],
    tasks: Any,
    policy: DecisionTransformerMoE,
    config: Dict[str, Any],
    state_mean: Tensor,
    state_std: Tensor,
    rand_first: bool = False,
) -> List[float]:
    # environments
    num_tasks = len(task_ids)
    vec_envs = SyncVectorEnv([lambda: gym.make(config['env_name'], tasks=tasks) for _ in task_ids])
    for i, env in enumerate(vec_envs.envs):
        env.unwrapped.reset_task(task_ids[i])

    avg_epi_returns = []
    eval_horizon = config['eval_horizon']
    max_episode_steps = config['max_episode_steps']
    maintain_horizon = eval_horizon + max_episode_steps

    # environment parameters
    state_dim = np.prod(env.observation_space.shape)
    action_dim = np.prod(env.action_space.shape)

    states_buffer = torch.zeros([maintain_horizon, num_tasks, state_dim], dtype=torch.float, device=config['device'])
    actions_buffer = torch.zeros([maintain_horizon, num_tasks, action_dim], dtype=torch.float, device=config['device'])
    rewards_buffer = torch.zeros([maintain_horizon, num_tasks], dtype=torch.float, device=config['device'])
    timesteps_buffer = torch.zeros([maintain_horizon, num_tasks], dtype=torch.long, device=config['device'])
    buffer_point = 0
    
    for _ in range(config['eval_episodes']):
        avg_epi_return = np.zeros([num_tasks,])
        state, _ = vec_envs.reset()
        for t in range(max_episode_steps):
            if buffer_point == maintain_horizon:
                buffer_point = eval_horizon
            states_buffer[buffer_point] = torch.tensor(state, dtype=torch.float, device=config['device'])
            timesteps_buffer[buffer_point] = torch.full([num_tasks,], t, dtype=torch.long, device=config['device'])

            if t == 0 and rand_first:
                action = torch.tensor(vec_envs.action_space.sample())
            else:
                action = policy.get_action(
                    rewards_buffer[:buffer_point+1].permute(1, 0).unsqueeze(-1),
                    ((states_buffer[:buffer_point+1] - state_mean) / state_std).permute(1, 0, 2),
                    actions_buffer[:buffer_point+1].permute(1, 0, 2),
                    timesteps_buffer[:buffer_point+1].permute(1, 0)
                )
            state, reward, _, _, _ = vec_envs.step(action.cpu().numpy())
            avg_epi_return += reward

            actions_buffer[buffer_point] = torch.tensor(action, dtype=torch.float, device=config['device'])
            rewards_buffer[buffer_point] = torch.tensor(reward, dtype=torch.float, device=config['device'])
            buffer_point += 1

        avg_epi_returns.append(avg_epi_return)

        returns = rewards_buffer[:buffer_point].transpose(1, 0).view(num_tasks, -1, max_episode_steps).sum(-1).transpose(1, 0)
        sorted_indices = torch.argsort(returns, dim=0)
        for i in range(num_tasks):
            states_buffer[:buffer_point, i] = states_buffer[:buffer_point, i].view(-1, max_episode_steps, state_dim)[sorted_indices[:, i]].view(-1, state_dim)
            actions_buffer[:buffer_point, i] = actions_buffer[:buffer_point, i].view(-1, max_episode_steps, action_dim)[sorted_indices[:, i]].view(-1, action_dim)
            rewards_buffer[:buffer_point, i] = rewards_buffer[:buffer_point, i].view(-1, max_episode_steps)[sorted_indices[:, i]].view(-1)

        if buffer_point > eval_horizon:
            states_buffer = torch.cat([states_buffer[-eval_horizon:], states_buffer[:max_episode_steps]], dim=0)
            actions_buffer = torch.cat([actions_buffer[-eval_horizon:], actions_buffer[:max_episode_steps]], dim=0)
            rewards_buffer = torch.cat([rewards_buffer[-eval_horizon:], rewards_buffer[:max_episode_steps]], dim=0)

    return np.array(avg_epi_returns).transpose(1, 0)


@torch.no_grad()
def evaluate_latest(
    task_ids: List[int],
    tasks: Any,
    policy: DecisionTransformerMoE,
    config: Dict[str, Any],
    state_mean: Tensor,
    state_std: Tensor,
    sorted_by_return: bool = True,
) -> List[float]:
    # environments
    num_tasks = len(task_ids)
    vec_envs = SyncVectorEnv([lambda: gym.make(config['env_name'], tasks=tasks) for _ in task_ids])
    for i, env in enumerate(vec_envs.envs):
        env.unwrapped.reset_task(task_ids[i])

    discrete_environments = isinstance(env.action_space, gym.spaces.Discrete)
    avg_epi_returns = []
    avg_epi_successes = []
    eval_horizon = config['eval_horizon']
    max_episode_steps = config['max_episode_steps']
    maintain_horizon = eval_horizon + max_episode_steps

    # environment parameters
    state_dim = np.prod(env.observation_space.shape)
    action_dim = env.action_space.n if discrete_environments else np.prod(env.action_space.shape)

    states_buffer = torch.zeros([maintain_horizon, num_tasks, state_dim], dtype=torch.float, device=config['device'])
    actions_buffer = torch.zeros([maintain_horizon, num_tasks, action_dim], dtype=torch.float, device=config['device'])
    rewards_buffer = torch.zeros([maintain_horizon, num_tasks], dtype=torch.float, device=config['device'])
    timesteps_buffer = torch.zeros([maintain_horizon, num_tasks], dtype=torch.long, device=config['device'])
    buffer_point = 0

    for _ in range(config['eval_episodes']):
        avg_epi_return = np.zeros([num_tasks,])
        avg_epi_success = np.zeros([num_tasks,])
        state, _ = vec_envs.reset(seed=config.get('eval_reset_seed', None))
        for t in range(max_episode_steps):
            states_buffer[buffer_point] = torch.tensor(state, device=config['device'])
            timesteps_buffer[buffer_point] = torch.full([num_tasks,], t, dtype=torch.long, device=config['device'])

            action = policy.get_action(
                ((states_buffer[:buffer_point+1] - state_mean) / state_std).permute(1, 0, 2) if state_mean is not None else states_buffer[:buffer_point+1].permute(1, 0, 2),
                actions_buffer[:buffer_point+1].permute(1, 0, 2),
                rewards_buffer[:buffer_point+1].permute(1, 0).unsqueeze(-1),
                timesteps_buffer[:buffer_point+1].permute(1, 0)
            )
            if discrete_environments:
                a = action.argmax(dim=-1)
                action = torch.nn.functional.one_hot(a, num_classes=action_dim)
            state, reward, _, _, info = vec_envs.step(action.cpu().numpy())
            avg_epi_return += reward
            avg_epi_success += info.get('success', np.zeros_like(avg_epi_success))

            actions_buffer[buffer_point] = action.clone()
            rewards_buffer[buffer_point] = torch.tensor(reward, dtype=torch.float, device=config['device'])
            buffer_point += 1

        avg_epi_returns.append(avg_epi_return)
        avg_epi_successes.append(avg_epi_success)

        if buffer_point > eval_horizon:
            buffer_point = eval_horizon
            states_buffer = states_buffer.roll(-max_episode_steps, dims=0)
            actions_buffer = actions_buffer.roll(-max_episode_steps, dims=0)
            rewards_buffer = rewards_buffer.roll(-max_episode_steps, dims=0)
            timesteps_buffer = timesteps_buffer.roll(-max_episode_steps, dims=0)

        if sorted_by_return:
            returns = rewards_buffer[:buffer_point].reshape(-1, max_episode_steps, num_tasks).sum(1)
            sorted_indices = torch.argsort(returns, dim=0)
            for i in range(num_tasks):
                states_buffer[:buffer_point, i] = states_buffer[:buffer_point, i].view(-1, max_episode_steps, state_dim)[sorted_indices[:, i]].view(-1, state_dim)
                actions_buffer[:buffer_point, i] = actions_buffer[:buffer_point, i].view(-1, max_episode_steps, action_dim)[sorted_indices[:, i]].view(-1, action_dim)
                rewards_buffer[:buffer_point, i] = rewards_buffer[:buffer_point, i].view(-1, max_episode_steps)[sorted_indices[:, i]].view(-1)

    return np.array(avg_epi_returns).transpose(1, 0), np.array(avg_epi_successes).transpose(1, 0)

@torch.no_grad()
def evaluate_step(
    task_ids: List[int],
    tasks: Any,
    policy: DecisionTransformerMoE,
    config: Dict[str, Any],
    state_mean: Tensor,
    state_std: Tensor,
) -> List[float]:
    # environments
    num_tasks = len(task_ids)
    vec_envs = SyncVectorEnv([lambda: gym.make(config['env_name'], tasks=tasks) for _ in task_ids])
    for i, env in enumerate(vec_envs.envs):
        env.unwrapped.reset_task(task_ids[i])

    discrete_environments = isinstance(env.action_space, gym.spaces.Discrete)
    avg_epi_returns = []
    avg_epi_successes = []
    eval_horizon = config['eval_horizon']
    max_episode_steps = config['max_episode_steps']

    # environment parameters
    state_dim = np.prod(env.observation_space.shape)
    action_dim = env.action_space.n if discrete_environments else np.prod(env.action_space.shape)

    states_buffer = torch.zeros([eval_horizon, num_tasks, state_dim], dtype=torch.float, device=config['device'])
    actions_buffer = torch.zeros([eval_horizon, num_tasks, action_dim], dtype=torch.float, device=config['device'])
    rewards_buffer = torch.zeros([eval_horizon, num_tasks], dtype=torch.float, device=config['device'])
    timesteps_buffer = torch.zeros([eval_horizon, num_tasks], dtype=torch.long, device=config['device'])
    buffer_size = 0

    for _ in range(config['eval_episodes']):
        avg_epi_return = np.zeros([num_tasks,])
        avg_epi_success = np.zeros([num_tasks,])
        state, _ = vec_envs.reset(seed=config.get('eval_reset_seed', None))
        for t in range(max_episode_steps):
            buffer_size += 1
            states_buffer[-1] = torch.tensor(state, device=config['device'])
            timesteps_buffer[-1] = torch.full([num_tasks,], t, dtype=torch.long, device=config['device'])

            action = policy.get_action(
                ((states_buffer[-buffer_size:] - state_mean) / state_std).permute(1, 0, 2) if state_mean is not None else states_buffer[-buffer_size:].permute(1, 0, 2),
                actions_buffer[-buffer_size:].permute(1, 0, 2),
                rewards_buffer[-buffer_size:].permute(1, 0).unsqueeze(-1),
                timesteps_buffer[-buffer_size:].permute(1, 0)
            )
            if discrete_environments:
                a = action.argmax(dim=-1)
                action = torch.nn.functional.one_hot(a, num_classes=action_dim)
            state, reward, _, _, info = vec_envs.step(action.cpu().numpy())
            avg_epi_return += reward
            # avg_epi_success += np.array([inf.get('success', 0) for inf in info])

            actions_buffer[-1] = action.clone()
            rewards_buffer[-1] = torch.tensor(reward, dtype=torch.float, device=config['device'])

            states_buffer = states_buffer.roll(-1, dims=0)
            actions_buffer = actions_buffer.roll(-1, dims=0)
            rewards_buffer = rewards_buffer.roll(-1, dims=0)
            timesteps_buffer = timesteps_buffer.roll(-1, dims=0)

        avg_epi_returns.append(avg_epi_return)
        avg_epi_successes.append(avg_epi_success)

    return np.array(avg_epi_returns).transpose(1, 0), np.array(avg_epi_successes).transpose(1, 0)

def write_returns(returns: List[float], fp, timeteps: int, task_id: int):
    fp.write(','.join(map(str, [timeteps, task_id] + returns)) + '\n')
    fp.flush()

def eval_policy(policy: DecisionTransformerMoE, return_writer_train, return_writer_eval, success_writer_train, success_writer_eval, timestep: int, train_task_ids: List[int], eval_task_ids: List[int], tasks: Any, configs: Dict[str, Any], state_mean: Tensor, state_std: Tensor, sample_by_timestep: bool = False):
    policy.eval()

    eval_func = evaluate_step if sample_by_timestep else evaluate_latest

    # evaluation on training tasks
    # avg_returns, avg_successes = eval_func(
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
    avg_returns, avg_successes = eval_func(
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