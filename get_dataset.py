import numpy as np


def qlearning_dataset(dataset, terminate_on_end=False):
    """
    Returns datasets formatted for use by standard Q-learning algorithms,
    with observations, actions, next_observations, rewards, and a terminal
    flag.
    """
    N = dataset['rewards'].shape[0]
    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []

    episode_step = 0
    for i in range(N-1):
        obs = dataset['observations'][i].astype(np.float32)
        new_obs = dataset['observations'][i+1].astype(np.float32)
        action = dataset['actions'][i].astype(np.float32)
        reward = dataset['rewards'][i].astype(np.float32)
        done_bool = bool(dataset['terminals'][i])

        final_timestep = dataset['timeouts'][i]
        if (not terminate_on_end) and final_timestep:
            # Skip this transition and don't apply terminals on the last step of an episode
            episode_step = 0
            continue
        if done_bool or final_timestep:
            episode_step = 0

        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        reward_.append(reward)
        done_.append(done_bool)
        episode_step += 1

    return {
        'observations': np.array(obs_),
        'actions': np.array(action_),
        'next_observations': np.array(next_obs_),
        'rewards': np.array(reward_),
        'terminals': np.array(done_),
    }


def dataset_add_data(dataset, terminate_on_end=False):
    """
    Returns D_e and D_b.
    """
    N = dataset['rewards'].shape[0]

    # D_b
    start_b_num = int(0.2*N)
    end_b_num = int(0.8*N)
    obs_b = []
    next_obs_b = []
    action_b = []
    reward_b = []
    done_b = []

    episode_step = 0
    for i in range(start_b_num, end_b_num):
        obs = dataset['observations'][i].astype(np.float32)
        new_obs = dataset['observations'][i+1].astype(np.float32)
        action = dataset['actions'][i].astype(np.float32)
        reward = dataset['rewards'][i].astype(np.float32)
        done_bool = bool(dataset['terminals'][i])

        final_timestep = dataset['timeouts'][i]
        if (not terminate_on_end) and final_timestep:
            # Skip this transition and don't apply terminals on the last step of an episode
            episode_step = 0
            continue
        if done_bool or final_timestep:
            episode_step = 0

        obs_b.append(obs)
        next_obs_b.append(new_obs)
        action_b.append(action)
        reward_b.append(reward)
        done_b.append(done_bool)
        episode_step += 1

    dataset_b = {
        'observations': np.array(obs_b),
        'actions': np.array(action_b),
        'next_observations': np.array(next_obs_b),
        'rewards': np.array(reward_b),
        'terminals': np.array(done_b),
    }

    # D_e
    start_e_num = int(0.3*N)
    end_e_num = int(0.7*N)
    obs_e = []
    next_obs_e = []
    action_e = []
    reward_e = []
    done_e = []

    episode_step = 0
    for i in range(start_e_num, end_e_num):
        obs = dataset['observations'][i].astype(np.float32)
        new_obs = dataset['observations'][i+1].astype(np.float32)
        action = dataset['actions'][i].astype(np.float32)
        reward = dataset['rewards'][i].astype(np.float32)
        done_bool = bool(dataset['terminals'][i])

        final_timestep = dataset['timeouts'][i]
        if (not terminate_on_end) and final_timestep:
            # Skip this transition and don't apply terminals on the last step of an episode
            episode_step = 0
            continue
        if done_bool or final_timestep:
            episode_step = 0

        obs_e.append(obs)
        next_obs_e.append(new_obs)
        action_e.append(action)
        reward_e.append(reward)
        done_e.append(done_bool)
        episode_step += 1

    dataset_e = {
        'observations': np.array(obs_e),
        'actions': np.array(action_e),
        'next_observations': np.array(next_obs_e),
        'rewards': np.array(reward_e),
        'terminals': np.array(done_e),
    }

    return dataset_e, dataset_b
