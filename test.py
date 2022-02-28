import random
import numpy as np
import torch

from env import CityEnv
from baselines import random_agent, min_weight_agent, max_weight_agent


def set_seeds(seed):
    """
    Set all random generators used to a specific seed

        Parameters:
            seed: an integer
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.use_deterministic_algorithms(True)


def test_random_small(rand: bool = True):
    """
    Runs the randomly acting agent on the environment of field 3 x 3

        Parameters:
            rand: randomness of the environment, default is True

        Returns:
            cum_r: cumulative reward the random agent gained
            actions: action sequence the random agent performed
    """
    set_seeds(1111)
    env = CityEnv(init_random=rand, height=3, width=3, packages=[(2, 2), (2, 0)])
    env.draw_map()
    cum_r, actions = random_agent(env)
    return cum_r, actions


def test_random_medium(rand: bool = True):
    """
    Runs the randomly acting agent on the environment of field 3 x 3

        Parameters:
            rand: randomness of the environment, default is True

        Returns:
            cum_r: cumulative reward the random agent gained
            actions: action sequence the random agent performed
    """
    set_seeds(2222)
    env = CityEnv(init_random=rand, height=5, width=5, packages=[(0, 4), (2, 0), (4, 2)])
    env.draw_map()
    cum_r, actions = random_agent(env)
    return cum_r, actions


def test_random_large(rand: bool = True):
    """
    Runs the randomly acting agent on the environment of field 3 x 3

        Parameters:
            rand: randomness of the environment, default is True

        Returns:
            cum_r: cumulative reward the random agent gained
            actions: action sequence the random agent performed
    """
    set_seeds(3333)
    env = CityEnv(init_random=rand, height=10, width=10, packages=[(0, 2), (7, 2), (5, 5), (3, 8)])
    env.draw_map()
    cum_r, actions = random_agent(env)
    return cum_r, actions


def test_min_weight_small(rand: bool = True):
    """
    Runs the minimal weight searching agent on the environment of field 3 x 3

        Parameters:
            rand: randomness of the environment, default is True

        Returns:
            cum_r: cumulative reward the random agent gained
            actions: action sequence the random agent performed
    """
    set_seeds(1111)
    env = CityEnv(init_random=rand, height=3, width=3, packages=[(2, 2), (2, 0)])
    env.draw_map()
    cum_r, actions = min_weight_agent(env)
    return cum_r, actions


def test_min_weight_medium(rand: bool = True):
    """
    Runs the minimal weight searching agent on the environment of field 3 x 3

        Parameters:
            rand: randomness of the environment, default is True

        Returns:
            cum_r: cumulative reward the random agent gained
            actions: action sequence the random agent performed
    """
    set_seeds(2222)
    env = CityEnv(init_random=rand, height=5, width=5, packages=[(0, 4), (2, 0), (4, 2)])
    env.draw_map()
    cum_r, actions = min_weight_agent(env)
    return cum_r, actions


def test_min_weight_large(rand: bool = True):
    """
    Runs the minimal weight searching agent on the environment of field 3 x 3

        Parameters:
            rand: randomness of the environment, default is True

        Returns:
            cum_r: cumulative reward the random agent gained
            actions: action sequence the random agent performed
    """
    set_seeds(3333)
    env = CityEnv(init_random=rand, height=10, width=10, packages=[(0, 2), (7, 2), (5, 5), (3, 8)])
    env.draw_map()
    cum_r, actions = min_weight_agent(env)
    return cum_r, actions


def test_max_weight_small(rand: bool = True):
    """
    Runs the maximal weight searching agent on the environment of field 3 x 3

        Parameters:
            rand: randomness of the environment, default is True

        Returns:
            cum_r: cumulative reward the random agent gained
            actions: action sequence the random agent performed
    """
    set_seeds(1111)
    env = CityEnv(init_random=rand, height=3, width=3, packages=[(2, 2), (2, 0)])
    env.draw_map()
    cum_r, actions = max_weight_agent(env)
    return cum_r, actions


def test_max_weight_medium(rand: bool = True):
    """
    Runs the maximal weight searching agent on the environment of field 3 x 3

        Parameters:
            rand: randomness of the environment, default is True

        Returns:
            cum_r: cumulative reward the random agent gained
            actions: action sequence the random agent performed
    """
    set_seeds(2222)
    env = CityEnv(init_random=rand, height=5, width=5, packages=[(0, 4), (2, 0), (4, 2)])
    env.draw_map()
    cum_r, actions = max_weight_agent(env)
    return cum_r, actions


def test_max_weight_large(rand: bool = True):
    """
    Runs the maximal weight searching agent on the environment of field 3 x 3

        Parameters:
            rand: randomness of the environment, default is True

        Returns:
            cum_r: cumulative reward the random agent gained
            actions: action sequence the random agent performed
    """
    set_seeds(3333)
    env = CityEnv(init_random=rand, height=10, width=10, packages=[(0, 2), (7, 2), (5, 5), (3, 8)])
    env.draw_map()
    cum_r, actions = max_weight_agent(env)
    return cum_r, actions
