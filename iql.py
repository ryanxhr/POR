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


class IQL(nn.Module):
    def __init__(self, qf, vf, policy, max_steps,
                 tau, alpha, value_lr=1e-4, policy_lr=1e-4, discount=0.99, beta=0.005):
        super().__init__()
        self.qf = qf.to(DEFAULT_DEVICE)
        self.q_target = copy.deepcopy(qf).requires_grad_(False).to(DEFAULT_DEVICE)
        self.vf = vf.to(DEFAULT_DEVICE)
        self.policy = policy.to(DEFAULT_DEVICE)
        self.v_optimizer = torch.optim.Adam(self.vf.parameters(), lr=value_lr)
        self.q_optimizer = torch.optim.Adam(self.qf.parameters(), lr=value_lr)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=policy_lr)
        self.policy_lr_schedule = CosineAnnealingLR(self.policy_optimizer, max_steps)
        self.tau = tau
        self.alpha = alpha
        self.discount = discount
        self.beta = beta
        self.step = 0
        self.pretrain_step = 0

    def iql_update(self, observations, actions, next_observations, rewards, terminals):
        # the network will NOT update
        with torch.no_grad():
            target_q = self.q_target(observations, actions)
            next_v = self.vf(next_observations)

        v = self.vf(observations)
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, self.tau)
        self.v_optimizer.zero_grad(set_to_none=True)
        v_loss.backward()
        self.v_optimizer.step()

        # Update Q function
        targets = rewards + (1. - terminals.float()) * self.discount * next_v
        qs = self.qf.both(observations, actions)
        q_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)
        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_optimizer.step()

        # Update target Q network
        update_exponential_moving_average(self.q_target, self.qf, self.beta)

        # Update policy
        weight = torch.exp(self.alpha * adv)
        weight = torch.clamp_max(weight, EXP_ADV_MAX).detach()
        policy_out = self.policy(observations)
        bc_losses = -policy_out.log_prob(actions)
        policy_loss = torch.mean(weight * bc_losses)
        self.policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.policy_optimizer.step()
        # self.policy_lr_schedule.step()
        wandb.log({"p_loss": policy_loss}, step=self.step)

        # wandb
        if (self.step+1) % 10000 == 0:
            wandb.log({"v_loss": v_loss, "v_value": v.mean(), "q_loss": q_loss, "q_value": qs[0].mean()}, step=self.step)
        self.step += 1

    def save(self, filename):
        torch.save(self.policy.state_dict(), filename + "-policy_network")
        print(f"***save models to {filename}***")

    def load(self, filename):
        self.policy.load_state_dict(torch.load(filename + "-policy_network", map_location=torch.device('cpu')))
        print(f"***load the RvS policy model from {filename}***")
