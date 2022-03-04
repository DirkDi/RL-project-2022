import random
import numpy as np
import torch

from env import CityEnv


def set_seeds(seed):
    """
    Set all random generators used to a specific seed.

    :param seed: an integer to set the random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def create_small_env(seed, rand, draw_map):
    """
    Creates the small environment (3 x 3) with defined packages and a specific random seed.

    :param seed: an integer to set the random seed
    :param rand: boolean flag for the randomness of the environment

    :param draw_map: boolean flag for showing the map
    :return: env: created environment
    """
    set_seeds(seed)
    env = CityEnv(init_random=rand, height=3, width=3, packages=[(2, 2), (2, 0)])
    if draw_map:
        env.draw_map()
    return env


def create_medium_env(seed, rand, draw_map):
    """
    Creates the medium environment (5 x 5) with defined packages and a specific random seed.

    :param seed: an integer to set the random seed
    :param rand: boolean flag for the randomness of the environment
    :param draw_map: boolean flag for showing the map

    :return: env: created environment
    """
    set_seeds(seed)
    env = CityEnv(init_random=rand, height=5, width=5, packages=[(0, 4), (2, 0), (4, 2)])
    if draw_map:
        env.draw_map()
    return env


def create_large_env(seed, rand, draw_map):
    """
    Creates the large environment (10 x 10) with defined packages and a specific random seed.

    :param seed: an integer to set the random seed
    :param rand: boolean flag for the randomness of the environment
    :param draw_map: boolean flag for showing the map

    :return: env: created environment
    """
    set_seeds(seed)
    env = CityEnv(init_random=rand, height=10, width=10, packages=[(0, 2), (7, 2), (5, 5), (3, 8)])
    if draw_map:
        env.draw_map()
    return env
