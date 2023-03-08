import csv
from dataclasses import dataclass
from datetime import datetime
from email import policy
import json
from pathlib import Path
import random
import string
import sys
import os

import numpy as np
from sklearn import datasets
import torch
import torch.nn as nn
from IPython import embed


DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Squeeze(nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(dim=self.dim)


def mlp(dims, activation=nn.ReLU, output_activation=None, squeeze_output=False):
    n_dims = len(dims)
    assert n_dims >= 2, 'MLP requires at least two dims (input and output)'

    layers = []
    for i in range(n_dims - 2):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        layers.append(activation())
    layers.append(nn.Linear(dims[-2], dims[-1]))
    if output_activation is not None:
        layers.append(output_activation())
    if squeeze_output:
        assert dims[-1] == 1
        layers.append(Squeeze(-1))
    net = nn.Sequential(*layers)
    net.to(dtype=torch.float32)
    return net


def compute_batched(f, xs):
    return f(torch.cat(xs, dim=0)).split([len(x) for x in xs])


def update_exponential_moving_average(target, source, alpha):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.mul_(1. - alpha).add_(source_param.data, alpha=alpha)


def torchify(x):
    x = torch.from_numpy(x)
    if x.dtype is torch.float64:
        x = x.float()
    x = x.to(device=DEFAULT_DEVICE)
    return x


def return_range(dataset, max_episode_steps):
    returns, lengths = [], []
    ep_ret, ep_len = 0., 0
    for r, d in zip(dataset['rewards'], dataset['terminals']):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0., 0
    # returns.append(ep_ret)    # incomplete trajectory
    lengths.append(ep_len)      # but still keep track of number of steps
    assert sum(lengths) == len(dataset['rewards'])
    return min(returns), max(returns)


def extract_done_makers(dones):
    (ends, ) = np.where(dones)
    starts = np.concatenate(([0], ends[:-1] + 1))
    length = ends - starts + 1
    return starts, ends, length


def _sample_indces(dataset, batch_size):
    try: 
        dones = dataset["timeouts"].cpu().numpy()
    except:
        dones = dataset["terminals"].cpu().numpy()
    starts, ends, lengths = extract_done_makers(dones)
    # credit to Dibya Ghosh's GCSL codebase
    trajectory_indces = np.random.choice(len(starts), batch_size)
    proportional_indices_1 = np.random.rand(batch_size)
    proportional_indices_2 = np.random.rand(batch_size)
    # proportional_indices_2 = 1
    time_dinces_1 = np.floor(
        proportional_indices_1 * (lengths[trajectory_indces] - 1)
    ).astype(int)
    time_dinces_2 = np.floor(
        proportional_indices_2 * (lengths[trajectory_indces])
    ).astype(int)
    start_indices = starts[trajectory_indces] + np.minimum(
        time_dinces_1,
        time_dinces_2
    )
    goal_indices = starts[trajectory_indces] + np.maximum(
        time_dinces_1,
        time_dinces_2
    )

    return start_indices, goal_indices


# dataset is a dict, values of which are tensors of same first dimension
def sample_batch(dataset, batch_size):
    k = list(dataset.keys())[0]
    n, device = len(dataset[k]), dataset[k].device
    for v in dataset.values():
        assert len(v) == n, 'Dataset values must have same length'
    indices = torch.randint(low=0, high=n, size=(batch_size,), device=device)
    return {k: v[indices] for k, v in dataset.items()}
        

def rvs_sample_batch(dataset, batch_size):
    start_indices, goal_indices = _sample_indces(dataset, batch_size)
    dict = {}
    for k, v in dataset.items():
        if (k == "observations") or (k == "actions"):
            dict[k] = v[start_indices]
    dict["next_observations"] = dataset["observations"][goal_indices]
    dict["rewards"] = 0
    dict["terminals"] = 0
    return dict


def evaluate_iql(env, policy, mean, std, deterministic=True):
    obs = env.reset()
    total_reward = 0.
    done, i = False, 0
    while not done:
        obs = (obs - mean)/std
        with torch.no_grad():
            action = policy.act(torchify(obs), deterministic=deterministic).cpu().numpy()
        obs, reward, done, info = env.step(action)
        total_reward += reward
        i += 1
    return total_reward


def evaluate_por(env, policy, goal_policy, mean, std, deterministic=True):
    obs = env.reset()
    total_reward = 0.
    done, i = False, 0
    while not done:
        obs = (obs - mean)/std
        with torch.no_grad():
            g = goal_policy.act(torchify(obs), deterministic=deterministic).cpu().numpy()
            action = policy.act(torchify(np.concatenate([obs, g])), deterministic=deterministic).cpu().numpy()
        obs, reward, done, info = env.step(action)
        total_reward += reward
        i += 1
    return total_reward


def evaluate_rvs(env, policy, mean, std, deterministic=True):
    obs = env.reset()
    goal = np.array(env.target_goal)
    goal = (goal - mean[:2])/std[:2]
    total_reward = 0.
    done, i = False, 0
    while not done:
        obs = (obs - mean)/std
        with torch.no_grad():
            if i % 100 == 0:
                print('current location:', obs[:2])
            action = policy.act(torchify(np.concatenate([obs, goal])), deterministic=deterministic).cpu().numpy()
        obs, reward, done, info = env.step(action)
        total_reward += reward
        i += 1
    return total_reward


def set_seed(seed, env=None):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if env is not None:
        env.seed(seed)


def save(dir ,filename, env_name, network_model):
    if not os.path.exists(dir):
        os.mkdir(dir)
    file = dir + env_name + "-" + filename 
    torch.save(network_model.state_dict(), file)
    print(f"***save the {network_model} model to {file}***")
    

def load(dir, filename, env_name, network_model):
    file = dir + env_name + "-" + filename
    if not os.path.exists(file):
        raise FileExistsError("Doesn't exist the model")
    network_model.load_state_dict(torch.load(file, map_location=torch.device('cpu')))
    print(f"***load the model from {file}***")


def _gen_dir_name():
    now_str = datetime.now().strftime('%m-%d-%y_%H.%M.%S')
    rand_str = ''.join(random.choices(string.ascii_lowercase, k=4))
    return f'{now_str}_{rand_str}'

def generate_test_generlaization_data(dataset, env_name, env_idx = None):
    locations_range = [[[5, 10], [2, 7]], [[10, 15], [10, 15]], [[26, 30], [14, 18]]]
    obs = dataset["observations"]
    if "umaze" in env_name:
        env_idx = 0
    elif "medium" in env_name:
        env_idx = 1
    else:
        env_idx = 2
    
    delete_range = locations_range[env_idx]
    x_range = delete_range[0]
    y_range = delete_range[1]
    x_index = np.where(np.logical_and( obs[:, 0]>= x_range[0], obs[:, 0] <= x_range[1]))
    y_index = np.where(np.logical_and( obs[:, 1]>= y_range[0], obs[:, 1] <= y_range[1]))
    index = np.intersect1d(x_index, y_index)
    for k in dataset.keys():
        dataset[k] = np.delete(dataset[k], index, 0)
    
    return dataset


class Log:
    def __init__(self, root_log_dir, cfg_dict,
                 txt_filename='log.txt',
                 csv_filename='progress.csv',
                 cfg_filename='config.json',
                 flush=True):
        self.dir = Path(root_log_dir)/_gen_dir_name()
        self.dir.mkdir(parents=True)
        self.txt_file = open(self.dir/txt_filename, 'w')
        self.csv_file = None
        (self.dir/cfg_filename).write_text(json.dumps(cfg_dict))
        self.txt_filename = txt_filename
        self.csv_filename = csv_filename
        self.cfg_filename = cfg_filename
        self.flush = flush

    def write(self, message, end='\n'):
        now_str = datetime.now().strftime('%H:%M:%S')
        message = f'[{now_str}] ' + message
        for f in [sys.stdout, self.txt_file]:
            print(message, end=end, file=f, flush=self.flush)

    def __call__(self, *args, **kwargs):
        self.write(*args, **kwargs)

    def row(self, dict):
        if self.csv_file is None:
            self.csv_file = open(self.dir/self.csv_filename, 'w', newline='')
            self.csv_writer = csv.DictWriter(self.csv_file, list(dict.keys()))
            self.csv_writer.writeheader()

        self(str(dict))
        self.csv_writer.writerow(dict)
        if self.flush:
            self.csv_file.flush()

    def close(self):
        self.txt_file.close()
        if self.csv_file is not None:
            self.csv_file.close()

