import numpy as np
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Any, Optional


class DPT_Dataset(Dataset):
    def __init__(self, dataset: Dict[str, np.ndarray], query_dataset: Dict[str, np.ndarray], config:  Dict[str, Any], state_norm: bool = True, norm_params: Optional[Tuple[np.ndarray, np.ndarray]] = [None, None]) -> None:
        super(DPT_Dataset, self).__init__()
        self.dataset_size = query_dataset['states'].shape[0] * query_dataset['states'].shape[1]
        self.max_episode_steps = config['max_episode_steps']
        self.prompt_episode_horizon = config['prompt_episode_horizon']
        self.horizon = self.max_episode_steps * self.prompt_episode_horizon
        self.keys = ['states', 'actions','rewards']
        self.query_keys = ['states', 'actions']

        del dataset['masks'], dataset['dones']
        assert dataset['states'].shape[1] % self.max_episode_steps == 0
        self.num_tasks, _, _ = dataset['states'].shape

        if state_norm:
            dataset = self.normalize_states(dataset, *norm_params)
            query_dataset = self.normalize_states(query_dataset, state_mean=self.state_mean, state_std=self.state_std)

        self.datasets: Dict[str, np.ndarray] = {key: dataset[key].reshape(self.num_tasks, -1, self.max_episode_steps, dataset[key].shape[-1]).astype(np.float32) for key in dataset.keys()}
        self.query_datasets: Dict[str, np.ndarray] = {key: query_dataset[key].reshape(self.num_tasks, -1, query_dataset[key].shape[-1]).astype(np.float32) for key in query_dataset.keys()}

        self.num_episodes = self.datasets['states'].shape[1]
        self.num_query = self.query_datasets['states'].shape[1]
        self.batch_per_task = self.num_episodes - self.prompt_episode_horizon + 1

    def sample_batch_contrastive(self, batch_size: int):
        task_ids = np.random.choice(self.num_tasks, batch_size // 2, replace=(batch_size // 2 > self.num_tasks)).reshape(1, -1).repeat(2, axis=0).reshape(-1)
        datasets = {key: [] for key in self.keys}
        query_datasets = {key: [] for key in self.query_keys}
        for task_id in task_ids:
            episode_id = np.sort(np.random.choice(self.num_episodes, self.prompt_episode_horizon, replace=False))
            query_id = np.random.choice(self.num_query)
            for key in self.keys:
                datasets[key].append(self.datasets[key][task_id, episode_id])
            for key in self.query_keys:
                query_datasets[key].append(self.query_datasets[key][task_id, query_id])
        return [np.stack(datasets[key], axis=0).reshape(batch_size, self.horizon, -1) for key in self.keys], [np.stack(query_datasets[key], axis=0).reshape(batch_size, 1, -1) for key in self.query_keys]

    def __len__(self) -> int:
        return self.dataset_size

    def normalize_states(self, dataset, state_mean: Optional[np.ndarray] = None, state_std: Optional[np.ndarray] = None) -> None:
        if state_mean is None:
            states = dataset['states'].reshape(-1, dataset['states'].shape[-1])
            state_mean = np.mean(states, axis=0)
            state_std = np.std(states, axis=0) + 1e-6
        self.state_mean, self.state_std = state_mean, state_std
        dataset['states'] = (dataset['states'] - state_mean) / state_std
        if 'next_states' in dataset:
            dataset['next_states'] = (dataset['next_states'] - state_mean) / state_std
        return dataset

    def get_norm_params(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.state_mean, self.state_std