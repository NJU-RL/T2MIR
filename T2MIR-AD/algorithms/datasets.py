import numpy as np
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Any, Optional

from algorithms.tools import discount_cumsum


class AD_Dataset(Dataset):
    def __init__(self, dataset: Dict[str, np.ndarray], config: Dict[str, Any], state_norm: bool = True, use_rewards: bool = True, norm_params: Tuple[np.ndarray, np.ndarray] = [None, None]) -> None:
        super(AD_Dataset, self).__init__()
        self.dataset_size = config['total_steps'] * config['batch_size']
        self.max_episode_steps = config['max_episode_steps']
        self.train_episode_horizon = config['train_episode_horizon']
        self.return_scale = config['return_scale']
        self.last_expert = config.get('last_expert', False)
        self.sample_by_timestep = config.get('sample_by_timestep', False)
        self.horizon = self.max_episode_steps * self.train_episode_horizon
        assert (self.sample_by_timestep and self.last_expert) == False

        if use_rewards:
            self.keys = ['states', 'actions','rewards']
        else:
            self.keys = ['rtgs' ,'states', 'actions']

        del dataset['masks'], dataset['dones'], dataset['next_states']
        assert dataset['states'].shape[1] % self.max_episode_steps == 0
        self.num_tasks, _, _ = dataset['states'].shape

        if not use_rewards:
            dataset = self.add_rtg(dataset)
        if state_norm:
            dataset = self.normalize_states(dataset, *norm_params)
        self.datasets: Dict[str, np.ndarray] = {key: dataset[key].reshape(self.num_tasks, -1, self.max_episode_steps, dataset[key].shape[-1]).astype(np.float32) for key in dataset.keys()}
        self.num_episodes = self.datasets['states'].shape[1]
        self.expert_split_id = int(self.num_episodes * 0.8)

        if config.get('sort_data', True):
            self.dataset_sort()

    def sample_batch_contrastive(self, batch_size: int):
        task_ids = np.random.choice(self.num_tasks, batch_size // 2, replace=(batch_size // 2 > self.num_tasks)).reshape(1, -1).repeat(2, axis=0).reshape(-1)
        datasets = {key: [] for key in self.keys}
        for task_id in task_ids:
            if self.sample_by_timestep:
                timestep = np.random.randint(0, self.num_episodes * self.max_episode_steps - self.horizon + 1)
                for key in self.keys:
                    datasets[key].append(self.datasets[key].reshape(self.num_tasks, -1, self.datasets[key].shape[-1])[task_id, timestep:timestep+self.horizon])
            else:
                if self.last_expert:
                    episode_id = np.sort(np.random.choice(self.expert_split_id, self.train_episode_horizon - 1, replace=False)).tolist() + [np.random.choice(self.num_episodes - self.expert_split_id, 1).item() + self.expert_split_id]
                else:
                    episode_id = np.sort(np.random.choice(self.num_episodes, self.train_episode_horizon, replace=False))
                for key in self.keys:
                    datasets[key].append(self.datasets[key][task_id, episode_id])
        return [np.stack(datasets[key], axis=0).reshape(batch_size, self.horizon, -1) for key in self.keys]

    def __len__(self) -> int:
        return self.dataset_size

    def __getitem__(self, index) -> List[np.ndarray]:
        task_id = self.task_index[index]
        episode_id = self.episode_index[index]
        return [self.datasets[key][task_id, episode_id].reshape(self.horizon, -1) for key in self.keys]

    def sample_all_tasks(self, batch_size: int, expert_episodes: bool = True, range_episodes: bool = False):
        if range_episodes:
            assert batch_size == 1

        dataset = {key: [] for key in self.keys}
        for task_id in range(self.num_tasks):
            for _ in range(batch_size):
                if expert_episodes:
                    start_episode_id = int(self.num_episodes * 0.8)
                    episode_ids = np.sort(np.random.choice(self.num_episodes - start_episode_id, self.train_episode_horizon, replace=False)) + start_episode_id
                elif range_episodes:
                    episode_ids = np.linspace(0, self.num_episodes, self.train_episode_horizon, endpoint=False, dtype=int)
                else:
                    episode_ids = np.sort(np.random.choice(self.num_episodes, self.train_episode_horizon, replace=False))
                for key in self.keys:
                    dataset[key].append(self.datasets[key][task_id, episode_ids])
        return [np.stack(dataset[key], axis=0).reshape(self.num_tasks, batch_size, self.train_episode_horizon*self.max_episode_steps, dataset[key][0].shape[-1]) for key in self.keys]

    def generate_index(self) -> None:
        self.task_index = np.arange(self.num_tasks).reshape(1, -1).repeat(self.dataset_size // self.num_tasks + 1, axis=0).reshape(-1)[:self.dataset_size]
        self.episode_index = np.zeros([0, self.train_episode_horizon], dtype=int)
        size = self.dataset_size
        while size != 0:
            index = np.sort(np.random.randint(0, self.num_episodes, [max(size, 1000), self.train_episode_horizon]))
            # check
            for i in range(self.train_episode_horizon - 1):
                index = index[index[:, i] != index[:, i+1]]
            if index.shape[0] > size:
                index = index[:size]
            size -= index.shape[0]
            self.episode_index = np.concatenate([self.episode_index, index], axis=0)

    def normalize_states(self, dataset, state_mean: Optional[np.ndarray] = None, state_std: Optional[np.ndarray] = None) -> None:
        if state_mean is None:
            states = dataset['states'].reshape(-1, dataset['states'].shape[-1])
            state_mean = np.mean(states, axis=0)
            state_std = np.std(states, axis=0) + 1e-6
        dataset['states'] = (dataset['states'] - state_mean) / state_std
        self.state_mean, self.state_std = state_mean, state_std
        return dataset

    def get_norm_params(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.state_mean, self.state_std

    def dataset_sort(self) -> None:
        returns = self.datasets['rewards'].sum(axis=2).reshape(self.num_tasks, self.num_episodes)  # shape: [num_tasks, num_episodes]
        sorted_indices = np.argsort(returns, axis=1)
        for _, value in self.datasets.items():
            for i in range(sorted_indices.shape[0]):
                value[i] = value[i][sorted_indices[i]]

    def add_rtg(self, dataset: Dict[str, np.ndarray]) -> None:
        rtgs = []
        rewards = dataset['rewards'].reshape(-1, self.max_episode_steps)
        for i in range(rewards.shape[0]):
            rtgs.append(discount_cumsum(rewards[i]))
        dataset['rtgs'] = np.stack(rtgs, axis=0).reshape(self.num_tasks, -1, 1) / self.return_scale
        return dataset