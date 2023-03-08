from dataclasses import dataclass
from pathlib import Path

import gym
import os
import d4rl
import sys
import numpy as np
import torch
from tqdm import trange

from por import POR
from policy import GaussianPolicy
from value_functions import TwinQ, ValueFunction, TwinV
from util import return_range, set_seed, Log, sample_batch, torchify, evaluate_iql, evaluate_por
import wandb
import time


def get_env_and_dataset(env_name, max_episode_steps, normalize):
    env = gym.make(env_name)
    dataset = d4rl.qlearning_dataset(env)
    if any(s in env_name for s in ('halfcheetah', 'hopper', 'walker2d')):
        min_ret, max_ret = return_range(dataset, max_episode_steps)
        print(f'Dataset returns have range [{min_ret}, {max_ret}]')
        dataset['rewards'] /= (max_ret - min_ret)
        dataset['rewards'] *= max_episode_steps
    elif 'antmaze' in env_name:
        dataset['rewards'] -= 1.

    # dones = dataset["timeouts"]
    print("***********************************************************************")
    print(f"Normalize for the state: {normalize}")
    print("***********************************************************************")
    if normalize:
        mean = dataset['observations'].mean(0)
        std = dataset['observations'].std(0) + 1e-3
        dataset['observations'] = (dataset['observations'] - mean)/std
        dataset['next_observations'] = (dataset['next_observations'] - mean)/std
    else:
        obs_dim = dataset['observations'].shape[1]
        mean, std = np.zeros(obs_dim), np.ones(obs_dim)

    for k, v in dataset.items():
        dataset[k] = torchify(v)

    return env, dataset, mean, std


def main(args):
    wandb.init(project="POR_reproduce",
               entity="ryanxhr",
               name=f"{args.env_name}",
               config={
                   "env_name": args.env_name,
                   "normalize": args.normalize,
                   "tau": args.tau,
                   "alpha": args.alpha,
                   "seed": args.seed,
                   "type": args.type,
               })
    torch.set_num_threads(1)

    env, dataset, mean, std = get_env_and_dataset(args.env_name,
                                                  args.max_episode_steps,
                                                  args.normalize)
    obs_dim = dataset['observations'].shape[1]
    act_dim = dataset['actions'].shape[1]   # this assume continuous actions
    set_seed(args.seed, env=env)

    policy = GaussianPolicy(obs_dim + obs_dim, act_dim, hidden_dim=1024, n_hidden=2)
    goal_policy = GaussianPolicy(obs_dim, obs_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden)

    por = POR(
        qf=TwinQ(obs_dim, obs_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden),
        vf=TwinV(obs_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden),
        policy=policy,
        goal_policy=goal_policy,
        max_steps=args.n_steps,
        tau=args.tau,
        alpha=args.alpha,
        discount=args.discount,
        lr=args.lr
    )

    def eval_por(step):
        eval_returns = np.array([evaluate_por(env, policy, goal_policy, mean, std) \
                                 for _ in range(args.n_eval_episodes)])
        normalized_returns = d4rl.get_normalized_score(args.env_name, eval_returns) * 100.0
        wandb.log({
            'return mean': eval_returns.mean(),
            'normalized return mean': normalized_returns.mean(),
        }, step=step)

        return normalized_returns.mean()

    # pretrain behavior goal policy if needed
    if any(s in args.env_name for s in ('halfcheetah', 'hopper', 'walker2d')):
        b_goal_policy = GaussianPolicy(obs_dim, obs_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden)
        por.pretrain_init(b_goal_policy)
        if args.pretrain:
            for _ in trange(args.n_steps):
                por.pretrain(**sample_batch(dataset, args.batch_size))
            algo_name = f"pretrain_step-{args.n_steps}_normalize-{args.normalize}"
            os.makedirs(f"{args.model_dir}/{args.env_name}", exist_ok=True)
            por.save_pretrain(f"{args.model_dir}/{args.env_name}/{algo_name}")
        else:
            algo_name = f"pretrain_step-{args.n_steps}_normalize-{args.normalize}"
            por.load_pretrain(f"{args.model_dir}/{args.env_name}/{algo_name}")

    # train por
    if not args.pretrain:
        algo_name = f"{args.type}_alpha-{args.alpha}_tau-{args.tau}_alpha-{args.alpha}_normalize-{args.normalize}"
        os.makedirs(f"{args.log_dir}/{args.env_name}/{algo_name}", exist_ok=True)
        eval_log = open(f"{args.log_dir}/{args.env_name}/{algo_name}/seed-{args.seed}.txt", 'w')
        for step in trange(args.n_steps):
            if args.type == 'por':  # learn goal policy by q-learning (mujoco results in the paper)
                por.por_update(**sample_batch(dataset, args.batch_size))
            elif args.type == 'por_r':  # learn goal policy by weighted BC (antmaze reuslts in the paper)
                por.por_update_residual(**sample_batch(dataset, args.batch_size))

            if (step+1) % args.eval_period == 0:
                average_returns = eval_por(step)
                eval_log.write(f'{step + 1}\t{average_returns}\n')
                eval_log.flush()
        eval_log.close()
        os.makedirs(f"{args.model_dir}/{args.env_name}", exist_ok=True)
        por.save(f"{args.model_dir}/{args.env_name}/{algo_name}")


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--env_name', type=str, default="antmaze-medium-play-v2")
    parser.add_argument('--log_dir', type=str, default="./results/")
    parser.add_argument('--model_dir', type=str, default="./models/")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--n_hidden', type=int, default=2)
    parser.add_argument('--n_steps', type=int, default=10**6)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--tau', type=float, default=0.7)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--alpha', type=float, default=10.0)
    parser.add_argument('--eval_period', type=int, default=5000)
    parser.add_argument('--n_eval_episodes', type=int, default=10)
    parser.add_argument('--max_episode_steps', type=int, default=1000)
    parser.add_argument("--normalize", action='store_true')
    parser.add_argument("--type", type=str, choices=['por', 'por_r'], default='por_r')
    parser.add_argument("--pretrain", action='store_true')
    # parser.add_argument("--ablation_type", type=str, required=True, choices=['None', 'generlization'])
    now = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    args = parser.parse_args()
    
    main(args)