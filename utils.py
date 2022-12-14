import numpy as np
import torch


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, next_action, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        # ind = np.random.randint(0, self.size, size=batch_size)
        ind = np.random.randint(0, self.size-1, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind+1]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )

    def convert_D4RL(self, dataset):
        self.state = dataset['observations']
        self.action = dataset['actions']
        self.next_state = dataset['next_observations']
        self.reward = dataset['rewards'].reshape(-1, 1)
        self.not_done = 1. - dataset['terminals'].reshape(-1, 1)
        self.size = self.state.shape[0]

    def normalize_states_meanstd(self, eps=1e-3, mean=None, std=None):
        if mean is None and std is None:
            mean = self.state.mean(0, keepdims=True)
            std = self.state.std(0, keepdims=True) + eps
        self.state = (self.state - mean) / std
        self.next_state = (self.next_state - mean) / std
        return mean, std

    def normalize_states_minmax(self, eps=1e-3, min=None, max=None):
        self.state = 2 * (self.state - min) / (max - min + eps) - 1
        self.next_state = 2 * (self.next_state - min) / (max - min + eps) - 1

    def normalize_rewards(self, env_name, max_episode_steps):
        if any(s in env_name for s in ('halfcheetah', 'hopper', 'walker2d')):
            min_ret, max_ret = self.return_range(max_episode_steps)
            print(f'Dataset returns have range [{min_ret}, {max_ret}]')
            self.reward /= (max_ret - min_ret)
            self.reward *= max_episode_steps
        elif 'antmaze' in env_name:
            self.reward -= 1.

    def return_range(self, max_episode_steps):
        returns, lengths = [], []
        ep_ret, ep_len = 0., 0
        for r, d in zip(self.reward, self.not_done):
            ep_ret += float(r)
            ep_len += 1
            if not d or ep_len == max_episode_steps:
                returns.append(ep_ret)
                lengths.append(ep_len)
                ep_ret, ep_len = 0., 0
        # returns.append(ep_ret)    # incomplete trajectory
        lengths.append(ep_len)  # but still keep track of number of steps
        assert sum(lengths) == len(self.reward)
        return min(returns), max(returns)
