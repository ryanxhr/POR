import math
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import TanhTransform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MEAN_MIN = -9.0
MEAN_MAX = 9.0
LOG_STD_MIN = -5
LOG_STD_MAX = 2
EPS = 1e-7


class Guide_policy(nn.Module):
    def __init__(self, state_dim, log_std_multiplier=1.0, log_std_offset=-1.0, hidden_num=256):
        super(Guide_policy, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_num)
        self.fc2 = nn.Linear(hidden_num, hidden_num)
        self.mu_head = nn.Linear(hidden_num, state_dim)
        self.sigma_head = nn.Linear(hidden_num, state_dim)

        self.log_sigma_multiplier = log_std_multiplier
        self.log_sigma_offset = log_std_offset

    def _get_outputs(self, state):
        a = F.relu(self.fc1(state))
        a = F.relu(self.fc2(a))
        mu = self.mu_head(a)
        mu = torch.clip(mu, MEAN_MIN, MEAN_MAX)
        log_sigma = self.sigma_head(a)
        # log_sigma = self.log_sigma_multiplier * log_sigma + self.log_sigma_offset

        log_sigma = torch.clip(log_sigma, LOG_STD_MIN, LOG_STD_MAX)
        sigma = torch.exp(log_sigma)

        a_distribution = TransformedDistribution(
            Normal(mu, sigma), TanhTransform(cache_size=1)
        )
        a_tanh_mode = torch.tanh(mu)
        return a_distribution, a_tanh_mode

    def forward(self, state):
        a_dist, a_tanh_mode = self._get_outputs(state)
        action = a_dist.rsample()
        logp_pi = a_dist.log_prob(action).sum(axis=-1)
        return action, logp_pi, a_tanh_mode

    def get_log_density(self, state, action):
        a_dist, _ = self._get_outputs(state)
        action_clip = torch.clip(action, -1. + EPS, 1. - EPS)
        logp_action = a_dist.log_prob(action_clip)
        return logp_action


class Execute_policy(nn.Module):
    def __init__(self, state_dim, action_dim, log_std_multiplier=1.0, log_std_offset=-1.0, hidden_num=512):
        super(Execute_policy, self).__init__()

        self.fc1 = nn.Linear(state_dim*2, hidden_num)
        self.fc2 = nn.Linear(hidden_num, hidden_num)
        self.mu_head = nn.Linear(hidden_num, action_dim)
        self.sigma_head = nn.Linear(hidden_num, action_dim)

        self.log_sigma_multiplier = log_std_multiplier
        self.log_sigma_offset = log_std_offset

    def _get_outputs(self, state, goal):
        concat_state = torch.concat([state, goal], dim=1)
        a = F.relu(self.fc1(concat_state))
        a = F.relu(self.fc2(a))

        mu = self.mu_head(a)
        mu = torch.clip(mu, MEAN_MIN, MEAN_MAX)
        log_sigma = self.sigma_head(a)
        # log_sigma = self.log_sigma_multiplier * log_sigma + self.log_sigma_offset

        log_sigma = torch.clip(log_sigma, LOG_STD_MIN, LOG_STD_MAX)
        sigma = torch.exp(log_sigma)

        a_distribution = TransformedDistribution(
            Normal(mu, sigma), TanhTransform(cache_size=1)
        )
        a_tanh_mode = torch.tanh(mu)
        return a_distribution, a_tanh_mode

    def forward(self, state, goal):
        a_dist, a_tanh_mode = self._get_outputs(state, goal)
        action = a_dist.rsample()
        logp_pi = a_dist.log_prob(action).sum(axis=-1)
        return action, logp_pi, a_tanh_mode

    def get_log_density(self, state, goal, action):
        a_dist, _ = self._get_outputs(state, goal)
        action_clip = torch.clip(action, -1. + EPS, 1. - EPS)
        logp_action = a_dist.log_prob(action_clip)
        return logp_action


class Double_Critic(nn.Module):
    def __init__(self, state_dim):
        super(Double_Critic, self).__init__()

        # V1 architecture
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # V2 architecture
        self.l4 = nn.Linear(state_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state):
        v1 = F.relu(self.l1(state))
        v1 = F.relu(self.l2(v1))
        v1 = self.l3(v1)

        v2 = F.relu(self.l4(state))
        v2 = F.relu(self.l5(v2))
        v2 = self.l6(v2)
        return v1, v2

    def V1(self, state):
        v1 = F.relu(self.l1(state))
        v1 = F.relu(self.l2(v1))
        v1 = self.l3(v1)
        return v1


def loss(diff, expectile=0.8):
    weight = torch.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)


class POR(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            discount=0.99,
            eta=0.005,
            tau=0.9,
            alpha=0.5,
            g_weight=True,
            e_weight=True,
    ):

        self.policy_e = Execute_policy(state_dim, action_dim).to(device)
        self.policy_e_optimizer = torch.optim.Adam(self.policy_e.parameters(), lr=3e-4)

        self.policy_g = Guide_policy(state_dim).to(device)
        self.policy_g_optimizer = torch.optim.Adam(self.policy_g.parameters(), lr=3e-4)

        self.critic = Double_Critic(state_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action

        self.tau = tau
        self.alpha = alpha
        self.g_weight = g_weight
        self.e_weight = e_weight

        self.discount = discount
        self.eta = eta
        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        _, _, goal = self.policy_g(state)
        _, _, action = self.policy_e(state, goal)
        return action.cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, _, reward, not_done = replay_buffer.sample(batch_size)

        # Update V
        with torch.no_grad():
            next_v1, next_v2 = self.critic_target(next_state)
            next_v = torch.minimum(next_v1, next_v2).detach()
            # next_v = next_v1
            target_v = (reward + self.discount * not_done * next_v).detach()

        v1, v2 = self.critic(state)
        critic_loss = (loss(target_v - v1, self.tau) + loss(target_v - v2, self.tau)).mean()
        # critic_loss = loss(target_v - v1, self.tau).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update guide-policy
        with torch.no_grad():
            next_v1, next_v2 = self.critic_target(next_state)
            next_v = torch.minimum(next_v1, next_v2).detach()
            target_v = (reward + self.discount * not_done * next_v).detach()
            v1, v2 = self.critic(state)
            residual = target_v - v1
            weight = torch.exp(residual * 10)
            weight = torch.clamp(weight, max=100.0).squeeze(-1).detach()

        log_pi_g = self.policy_g.get_log_density(state, next_state)
        log_pi_g = torch.sum(log_pi_g, 1)

        if self.g_weight:
            p_g_loss = -(weight * log_pi_g).mean()
        else:
            g, _, _ = self.policy_g(state)
            v1_g, v2_g = self.critic(g)
            min_v_g = torch.squeeze(torch.min(v1_g, v2_g))
            log_pi_g = self.policy_g.get_log_density(state, next_state)
            log_pi_g = torch.sum(log_pi_g, 1)
            p_g_loss = (self.alpha * log_pi_g - min_v_g).mean()
        
        self.policy_g_optimizer.zero_grad()
        p_g_loss.backward()
        self.policy_g_optimizer.step()

        # Update execute-policy
        log_pi_a = self.policy_e.get_log_density(state, next_state, action)
        log_pi_a = torch.sum(log_pi_a, 1)

        if self.g_weight:
            p_e_loss = -(weight * log_pi_a).mean()
        else:
            p_e_loss = -log_pi_a.mean()

        self.policy_e_optimizer.zero_grad()
        p_e_loss.backward()
        self.policy_e_optimizer.step()

        if self.total_it % 5000 == 0:
            print(f'mean target v value is {target_v.mean()}')
            print(f'mean v1 value is {v1.mean()}')
            print(f'mean residual is {residual.mean()}')

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.eta * param.data + (1 - self.eta) * target_param.data)

    def save(self, filename):
        torch.save(self.policy_g.state_dict(), filename + "_policy_g")
        torch.save(self.policy_g_optimizer.state_dict(), filename + "_policy_g_optimizer")
        torch.save(self.policy_e.state_dict(), filename + "_policy_e")
        torch.save(self.policy_e_optimizer.state_dict(), filename + "_policy_e_optimizer")

    def load(self, filename):
        self.policy_g.load_state_dict(torch.load(filename + "_policy_g"))
        self.policy_g_optimizer.load_state_dict(torch.load(filename + "_policy_g_optimizer"))
        self.policy_e.load_state_dict(torch.load(filename + "_policy_e"))
        self.policy_e_optimizer.load_state_dict(torch.load(filename + "_policy_e_optimizer"))