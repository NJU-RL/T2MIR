import pickle
import shutil
import torch
import yaml
import numpy as np

from collections import OrderedDict
from os.path import join
from pathlib import Path
from torch.nn import Module, Linear, init, Embedding, LayerNorm
from typing import Dict, Any, List, Optional


def copy_files(to_path: str, folders: List = [], files: List = [], parent: bool = True) -> None:
    path = Path(to_path) / 'Codes' if parent else Path(to_path)
    path.mkdir(parents=True, exist_ok=True)

    # copy files
    for folder in folders:
        destiantion = path / folder
        if destiantion.exists():
            shutil.rmtree(destiantion)
        shutil.copytree(folder, destiantion, ignore=shutil.ignore_patterns('*.pyc', '__pycache__'))

    for file in files:
        shutil.copy(file, path)
        
        
def soft_update(target: Module, source: Module, tau: float) -> None:
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data * tau + target_param.data * (1 - tau))
        
        
def hard_update(target: Module, source: Module) -> None:
    target.load_state_dict(source.state_dict())


def weight_init_(m, gain: float = 1.) -> None:
    if isinstance(m, Linear):
        init.xavier_uniform_(m.weight, gain=gain)
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, Embedding):
        init.normal_(m.weight, mean=0, std=0.01)
    elif isinstance(m, LayerNorm):
        m.weight.data.fill_(1.0)
        if m.bias is not None:
            m.bias.data.zero_()


def freeze(m: Module) -> None:
    for param in m.parameters():
        param.requires_grad = False
        

def unfreeze(m: Module) -> None:
    for param in m.parameters():
        param.requires_grad = True
        

def data_loader(dir_path: str, train_tasks: int, total_tasks: int, eval_task_ids: Optional[List[int]] = None, load_eval_tasks: bool = True):
    keys = ['states', 'actions', 'rewards', 'next_states', 'dones', 'masks']
    train_lengths = []
    test_lengths = []
    train_datasets = {key: [] for key in keys}
    test_datasets = {key: [] for key in keys}
    eval_task_ids = list(range(train_tasks, total_tasks)) if eval_task_ids is None else sorted(eval_task_ids)
    train_task_ids = sorted((set(range(total_tasks))) - set(eval_task_ids))

    for i in train_task_ids:
        file = join(dir_path, f"dataset_task_{i}.pkl")
        with open(file, 'rb') as fp:
            dataset: OrderedDict = pickle.load(fp)
        
        dataset['rewards'] = dataset['rewards'].reshape(-1, 1)
        dataset['dones'] = dataset['dones'].reshape(-1, 1)
        dataset['masks'] = dataset['masks'].reshape(-1, 1)

        for key in keys:
            train_datasets[key].append(dataset[key])

        train_lengths.append(dataset['states'].shape[0])

    if load_eval_tasks:
        for i in eval_task_ids:
            file = join(dir_path, f"dataset_task_{i}.pkl")
            with open(file, 'rb') as fp:
                dataset: OrderedDict = pickle.load(fp)

            dataset['rewards'] = dataset['rewards'].reshape(-1, 1)
            dataset['dones'] = dataset['dones'].reshape(-1, 1)
            dataset['masks'] = dataset['masks'].reshape(-1, 1)

            for key in keys:
                test_datasets[key].append(dataset[key])

            test_lengths.append(dataset['states'].shape[0])
    
    train_lengths = np.array(train_lengths)
    train_idxs = np.concatenate([[0], np.cumsum(train_lengths[:-1])], axis=0)  # index of each training task's start
    train_datasets: Dict[str, np.ndarray] = {key: np.concatenate(train_datasets[key], axis=0) for key in keys}  # shape: [size, dim]
    if load_eval_tasks:
        test_lengths = np.array(test_lengths)
        test_idxs = np.concatenate([[0], np.cumsum(test_lengths[:-1])], axis=0)  # index of each test task's start
        test_datasets: Dict[str, np.ndarray] = {key: np.concatenate(test_datasets[key], axis=0) for key in keys}
    else:
        test_idxs = None

    return (train_datasets, train_lengths, train_idxs), (test_datasets, test_lengths, test_idxs)


def set_seed(seed: int, env: Any) -> None:
    if seed != 0:
        env.action_space.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            

def download_config(env_type: str, return_config_path: bool = False):
    config_path = './configs/args_' + env_type.replace('-', '_') + '.yaml'
    with open(config_path, 'r') as fp:
        args: Dict[str, Any] = yaml.load(fp, Loader=yaml.FullLoader)
    if return_config_path:
        return args, config_path
    else:
        return args


def discount_cumsum(x: np.ndarray, gamma: float = 1.):
    discount_cumsum = np.empty_like(x)
    discount_cumsum[-1] = x[-1]
    for t in range(x.shape[0]-2, -1, -1):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum


def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-6):
    m = torch.rsqrt(torch.square(x).sum(dim=dim, keepdim=True) + eps)
    return x * m


def make_dir(dir_path_: str, ignore_exist: bool = False):
    dir_path = Path(dir_path_)
    if (dir_path.exists()) and (not ignore_exist):
        del_dir = input(f"{dir_path} already exists. Do you want to delete it? (y/n): ")
        if del_dir == 'y':
            shutil.rmtree(dir_path)
        else:
            raise FileExistsError(f"{dir_path} already exists.")
    dir_path.mkdir(parents=True, exist_ok=ignore_exist)
    (dir_path / 'checkpoints').mkdir(exist_ok=True)

    return dir_path

def make_dir_ignore_exist(dir_path_: str):
    dir_path = Path(dir_path_)
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / 'checkpoints').mkdir(exist_ok=True)

    return dir_path

def model_structure(model):
    blank = ' '
    # print('-' * 120)
    # print('|' + ' ' * 31 + 'weight name' + ' ' * 30 + '|' \
    #       + ' ' * 10 + 'weight shape' + ' ' * 10 + '|' \
    #       + ' ' * 3 + 'number' + ' ' * 3 + '|')
    # print('-' * 120)
    num_para = 0
    type_size = 1

    for index, (key, w_variable) in enumerate(model.named_parameters()):
        if len(key) <= 70:
            key = key + (70 - len(key)) * blank
        else:
            key = key[:67] + '...'
        shape = str(w_variable.shape)
        if len(shape) <= 30:
            shape = shape + (30 - len(shape)) * blank
        each_para = 1
        for k in w_variable.shape:
            each_para *= k
        num_para += each_para
        str_num = str(each_para)
        if len(str_num) <= 10:
            str_num = str_num + (10 - len(str_num)) * blank

        # print('| {} | {} | {} |'.format(key, shape, str_num))
    print('-' * 120)
    print('The total number of parameters: ' + str(num_para))
    print('The parameters of Model {}: {:4f}M'.format(model._get_name(), num_para * type_size / 1000 / 1000))
    print('-' * 120)


class GlobalValue:
    def __init__(self):
        self.value = None

global_value = GlobalValue()