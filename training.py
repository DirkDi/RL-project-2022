import random
import numpy as np
import torch

import gym
from env import CityEnv
from sarsa import sarsa, save_q


def set_seeds(seed: int):
    """
    Set all random generators used to a specific seed

        Parameters:
            seed: an integer
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.use_deterministic_algorithms(True)


def train_sarsa_small(num_episodes: int, seed: int = 1111, rand: bool = True, save: bool = False):
    """
    Trains the SARSA agent on the environment of field 3 x 3

        Parameters:
            num_episodes: amount of episodes the agents should learn
            seed: an integer
            rand: boolean flag for randomness of the environment
            save: boolean flag for saving the Q table

        Returns:
            r: list of rewards the agent gained per training episode
            l: list of lengths of the episodes the agent trained
            q: Q table
    """
    set_seeds(seed)
    env = CityEnv(init_random=rand, height=3, width=3, packages=[(2, 2), (2, 0)])
    env.draw_map()
    r, l, q = sarsa(env, num_episodes)
    if save:
        file_name = "q_sarsa_small_" + str(seed)
        save_q(q, file_name)
    return r, l, q


def train_sarsa_medium(num_episodes: int, seed: int = 2222, rand: bool = True, save: bool = False):
    """
    Trains the SARSA agent on the environment of field 5 x 5

        Parameters:
            num_episodes: amount of episodes the agents should learn
            seed: an integer
            rand: boolean flag for randomness of the environment
            save: boolean flag for saving the Q table

        Returns:
            r: list of rewards the agent gained per training episode
            l: list of lengths of the episodes the agent trained
            q: Q table
    """
    set_seeds(seed)
    env = CityEnv(init_random=rand, height=5, width=5, packages=[(0, 4), (2, 0), (4, 2)])
    env.draw_map()
    r, l, q = sarsa(env, num_episodes)
    if save:
        file_name = "q_sarsa_medium_" + str(seed)
        save_q(q, file_name)
    return r, l, q


def train_sarsa_large(num_episodes: int, seed: int = 3333, rand: bool = True, save: bool = False):
    """
    Trains the SARSA agent on the environment of field 10 x 10

        Parameters:
            num_episodes: amount of episodes the agents should learn
            seed: an integer
            rand: boolean flag for randomness of the environment
            save: boolean flag for saving the Q table

        Returns:
            r: list of rewards the agent gained per training episode
            l: list of lengths of the episodes the agent trained
            q: Q table
    """
    set_seeds(seed)
    env = CityEnv(init_random=rand, height=10, width=10, packages=[(0, 2), (7, 2), (5, 5), (3, 8)])
    env.draw_map()
    r, l, q = sarsa(env, num_episodes)
    if save:
        file_name = "q_sarsa_large_" + str(seed)
        save_q(q, file_name)
    return r, l, q
