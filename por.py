import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
from IPython import embed

from util import DEFAULT_DEVICE, compute_batched, update_exponential_moving_average


EXP_ADV_MAX = 100.


def asymmetric_l2_loss(u, tau):
    # from paper: "Offline Reinforcement Learning with Implicit Q-Learning" by Ilya et al.
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)


class POR(nn.Module):
    def __init__(self, vf, policy, goal_policy, max_steps,
                 tau, alpha, value_lr=1e-4, policy_lr=1e-4, discount=0.99, beta=0.005):
        super().__init__()
        self.vf = vf.to(DEFAULT_DEVICE)
        self.v_target = copy.deepcopy(vf).requires_grad_(False).to(DEFAULT_DEVICE)
        self.policy = policy.to(DEFAULT_DEVICE)
        self.goal_policy = goal_policy.to(DEFAULT_DEVICE)
        self.v_optimizer = torch.optim.Adam(self.vf.parameters(), lr=value_lr)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=policy_lr)
        self.policy_lr_schedule = CosineAnnealingLR(self.policy_optimizer, max_steps)
        self.goal_policy_optimizer = torch.optim.Adam(self.goal_policy.parameters(), lr=policy_lr)
        self.goal_lr_schedule = CosineAnnealingLR(self.goal_policy_optimizer, max_steps)
        self.tau = tau
        self.alpha = alpha
        self.discount = discount
        self.beta = beta
        self.step = 0
        self.pretrain_step = 0

    def por_residual_update(self, observations, actions, next_observations, rewards, terminals):
        # the network will NOT update
        with torch.no_grad():
            next_v = self.v_target(next_observations)

        # Update value function
        target_v = rewards + (1. - terminals.float()) * self.discount * next_v
        vs = self.vf.both(observations)
        v_loss = sum(asymmetric_l2_loss(target_v - v, self.tau) for v in vs) / len(vs)
        self.v_optimizer.zero_grad(set_to_none=True)
        v_loss.backward()
        self.v_optimizer.step()

        # Update target V network
        update_exponential_moving_average(self.v_target, self.vf, self.beta)

        # Update goal policy
        v = self.vf(observations)
        adv = target_v - v
        weight = torch.exp(self.alpha * adv)
        weight = torch.clamp_max(weight, EXP_ADV_MAX).detach()
        goal_out = self.goal_policy(observations)
        g_loss = -goal_out.log_prob(next_observations)
        g_loss = torch.mean(weight * g_loss)
        self.goal_policy_optimizer.zero_grad(set_to_none=True)
        g_loss.backward()
        self.goal_policy_optimizer.step()
        self.goal_lr_schedule.step()

        # Update policy
        policy_out = self.policy(torch.concat([observations, next_observations], dim=1))
        bc_losses = -policy_out.log_prob(actions)
        policy_loss = torch.mean(bc_losses)
        self.policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.policy_optimizer.step()
        self.policy_lr_schedule.step()

        # wandb
        if (self.step+1) % 100000 == 0:
            wandb.log({"v_loss": v_loss, "v_value": v.mean()}, step=self.step)
        self.step += 1

    def pretrain_init(self, b_goal_policy):
        self.b_goal_policy = b_goal_policy.to(DEFAULT_DEVICE)
        self.b_goal_policy_optimizer = torch.optim.Adam(self.b_goal_policy.parameters(), lr=0.0001)

    def pretrain(self, observations, actions, next_observations, rewards, terminals):
        # Update behavior goal policy
        b_goal_out = self.b_goal_policy(observations)
        b_g_loss = -b_goal_out.log_prob(next_observations).mean()
        b_g_loss = torch.mean(b_g_loss)
        self.b_goal_policy_optimizer.zero_grad(set_to_none=True)
        b_g_loss.backward()
        self.b_goal_policy_optimizer.step()

        if (self.pretrain_step+1) % 10000 == 0:
            wandb.log({"b_g_loss": b_g_loss}, step=self.pretrain_step)

        self.pretrain_step += 1

    def por_qlearning_update(self, observations, actions, next_observations, rewards, terminals):
        # the network will NOT update
        with torch.no_grad():
            next_v = self.v_target(next_observations)

        # Update value function
        target_v = rewards + (1. - terminals.float()) * self.discount * next_v
        vs = self.vf.both(observations)
        v_loss = sum(asymmetric_l2_loss(target_v - v, self.tau) for v in vs) / len(vs)
        self.v_optimizer.zero_grad(set_to_none=True)
        v_loss.backward()
        self.v_optimizer.step()

        # Update target V network
        update_exponential_moving_average(self.v_target, self.vf, self.beta)

        # Update goal policy
        v = self.vf(observations)
        goal_out = self.goal_policy(observations)
        b_goal_out = self.b_goal_policy(observations)
        g_sample = goal_out.rsample()
        g_loss1 = -self.vf(g_sample)
        g_loss2 = -b_goal_out.log_prob(g_sample).mean()
        lmbda = self.alpha/g_loss1.abs().mean().detach()
        g_loss = torch.mean(lmbda * g_loss1 + g_loss2)
        self.goal_policy_optimizer.zero_grad(set_to_none=True)
        g_loss.backward()
        self.goal_policy_optimizer.step()
        self.goal_lr_schedule.step()

        # Update policy
        policy_out = self.policy(torch.concat([observations, next_observations], dim=1))
        bc_losses = -policy_out.log_prob(actions)
        policy_loss = torch.mean(bc_losses)
        self.policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.policy_optimizer.step()
        self.policy_lr_schedule.step()

        # wandb
        if (self.step+1) % 100000 == 0:
            wandb.log({"v_loss": v_loss, "v_value": v.mean(), "g_loss1": g_loss1.mean(), "g_loss2": g_loss2.mean()}, step=self.step)

        self.step += 1

    def save_pretrain(self, filename):
        torch.save(self.b_goal_policy.state_dict(), filename + "-behavior_goal_network")
        print(f"***save models to {filename}***")

    def load_pretrain(self, filename):
        self.b_goal_policy.load_state_dict(torch.load(filename + "-behavior_goal_network", map_location=DEFAULT_DEVICE))
        print(f"***load models from {filename}***")

    def save(self, filename):
        torch.save(self.policy.state_dict(), filename + "-policy_network")
        torch.save(self.goal_policy.state_dict(), filename + "-goal_network")
        print(f"***save models to {filename}***")

    def load(self, filename):
        self.policy.load_state_dict(torch.load(filename + "-policy_network", map_location=torch.device('cpu')))
        print(f"***load the RvS policy model from {filename}***")
