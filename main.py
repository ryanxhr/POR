import numpy as np
import torch
import gym
import argparse
import os
import d4rl
import time

import utils
import get_dataset
from algos import TD3BC, CQL, POR


# Runs policy for X episodes and returns D4RL score
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, mean, std, seed_offset=100, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + seed_offset)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            state = (np.array(state).reshape(1, -1) - mean) / std
            action = policy.select_action(state)
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes
    d4rl_score = eval_env.get_normalized_score(avg_reward) * 100

    print("---------------------------------------")
    print(f"Env: {env_name}, Evaluation over {eval_episodes} episodes: {avg_reward:.3f}, D4RL score: {d4rl_score:.3f}")
    print("---------------------------------------")
    return d4rl_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument("--root_dir", default="results_add_data")  # Policy name
    parser.add_argument("--algorithm", default="CQL")  # Policy name
    parser.add_argument('--env', default="hopper-medium-v2")  # environment name
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--max_timesteps", default=5e5, type=int)  # Max time steps to run environment
    # Algo
    parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
    parser.add_argument("--alpha", default=5.0, type=float)
    parser.add_argument("--tau", default=0.9, type=float)
    parser.add_argument("--no_state_normalize", action='store_true')
    parser.add_argument("--no_reward_normalize", action='store_true')
    args = parser.parse_args()

    # Set seeds
    env = gym.make(args.env)
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    dataset_name = f"{args.env}_norm_s-{1-args.no_state_normalize}_norm_r-{1-args.no_reward_normalize}"

    # Set evaluation episode and interval
    if 'antmaze' in args.env:
        eval_freq = 100000
        eval_episodes = 100
    else:
        eval_freq = 5000
        eval_episodes = 10

    # Initialize policy
    if args.algorithm == 'TD3BC':
        policy = TD3BC.TD3BC(state_dim, action_dim, max_action, alpha=args.alpha)
        algo_name = f"{args.algorithm}_alpha-{args.alpha}"
    elif args.algorithm == 'CQL':
        policy = CQL.CQL(state_dim, action_dim, max_action, alpha=args.alpha)
        algo_name = f"{args.algorithm}_alpha-{args.alpha}"
    elif args.algorithm == 'IQL':
        policy = IQL.IQL(state_dim, action_dim, max_action, alpha=args.alpha, tau=args.tau)
        algo_name = f"{args.algorithm}_alpha-{args.alpha}_tau-{args.tau}"

    # checkpoint dir
    os.makedirs(f"{args.root_dir}/{dataset_name}/{algo_name}", exist_ok=True)
    save_dir = f"{args.root_dir}/{dataset_name}/{algo_name}/seed-{args.seed}.txt"
    print("---------------------------------------")
    print(f"Dataset: {dataset_name}, Algorithm: {algo_name}, Seed: {args.seed}")
    print("---------------------------------------")

    # Load dataset
    raw_dataset = env.get_dataset()
    dataset = get_dataset.qlearning_dataset(raw_dataset)
    states = dataset['observations']
    print('# {} of demonstraions'.format(states.shape[0]))

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
    replay_buffer.convert_D4RL(dataset)

    if args.no_state_normalize:
        shift, scale = 0, 1
    else:
        shift = np.mean(states, 0)
        scale = np.std(states, 0) + 1e-3
    replay_buffer.normalize_states(mean=shift, std=scale)

    if not args.no_reward_normalize:
        replay_buffer.normalize_rewards(args.env)

    eval_log = open(save_dir, 'w')
    # Start training
    for t in range(int(args.max_timesteps)):
        policy.train(replay_buffer, args.batch_size)
        # Evaluate episode
        if (t + 1) % eval_freq == 0:
            print(f"Time steps: {t + 1}")
            average_returns = eval_policy(policy, args.env, args.seed, shift, scale, eval_episodes=eval_episodes)
            eval_log.write(f'{t + 1}\t{average_returns}\n')
            eval_log.flush()
    eval_log.close()