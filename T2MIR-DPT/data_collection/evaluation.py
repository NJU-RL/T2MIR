import numpy as np
from gymnasium import Env

from data_collection.sac import SAC


def evaluate(env: Env, agent: SAC, num_eval_episodes: int = 10):
    epi_returns = []
    success_rate = []
    for _ in range(num_eval_episodes):
        episode_success = []
        epi_return = 0.
        state, _ = env.reset()
        for _ in range(env._max_episode_steps):
            action = agent.select_action(state, True)
            next_state, reward, done, truncated, info = env.step(action)
            epi_return += reward
            state = next_state
            episode_success.append(info.get('success', 0.0))

        epi_returns.append(epi_return)
        success_rate.append(np.mean(episode_success[-40:]))

    return np.mean(epi_returns).item(), np.mean(success_rate)