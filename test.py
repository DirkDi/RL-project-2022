import random
import numpy as np
import torch

from collections import defaultdict
from typing import DefaultDict, Tuple

from env import CityEnv
from sarsa import evaluate_sarsa_policy, load_q
from baselines import random_agent, min_weight_agent, max_weight_agent
from training import train_sarsa_small, train_sarsa_medium, train_sarsa_large


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


def test_random_small(seed: int = 1111, rand: bool = True):
    """
    Runs the randomly acting agent on the environment of field 3 x 3

        Parameters:
            seed: an integer
            rand: boolean flag for the randomness of the environment

        Returns:
            cum_r: cumulative reward the random agent gained
            actions: action sequence the random agent performed
    """
    set_seeds(seed)
    env = CityEnv(init_random=rand, height=3, width=3, packages=[(2, 2), (2, 0)])
    env.draw_map()
    cum_r, actions = random_agent(env)
    return cum_r, actions


def test_random_medium(seed: int = 2222, rand: bool = True):
    """
    Runs the randomly acting agent on the environment of field 5 x 5

        Parameters:
            seed: an integer
            rand: boolean flag for the randomness of the environment

        Returns:
            cum_r: cumulative reward the random agent gained
            actions: action sequence the random agent performed
    """
    set_seeds(seed)
    env = CityEnv(init_random=rand, height=5, width=5, packages=[(0, 4), (2, 0), (4, 2)])
    env.draw_map()
    cum_r, actions = random_agent(env)
    return cum_r, actions


def test_random_large(seed: int = 3333, rand: bool = True):
    """
    Runs the randomly acting agent on the environment of field 10 x 10

        Parameters:
            seed: an integer
            rand: boolean flag for the randomness of the environment

        Returns:
            cum_r: cumulative reward the random agent gained
            actions: action sequence the random agent performed
    """
    set_seeds(seed)
    env = CityEnv(init_random=rand, height=10, width=10, packages=[(0, 2), (7, 2), (5, 5), (3, 8)])
    env.draw_map()
    cum_r, actions = random_agent(env)
    return cum_r, actions


def test_min_weight_small(seed: int = 1111, rand: bool = True):
    """
    Runs the minimal weight searching agent on the environment of field 3 x 3

        Parameters:
            seed: an integer
            rand: boolean flag for the randomness of the environment

        Returns:
            cum_r: cumulative reward the random agent gained
            actions: action sequence the random agent performed
    """
    set_seeds(seed)
    env = CityEnv(init_random=rand, height=3, width=3, packages=[(2, 2), (2, 0)])
    env.draw_map()
    cum_r, actions = min_weight_agent(env)
    return cum_r, actions


def test_min_weight_medium(seed: int = 2222, rand: bool = True):
    """
    Runs the minimal weight searching agent on the environment of field 5 x 5

        Parameters:
            seed: an integer
            rand: boolean flag for the randomness of the environment

        Returns:
            cum_r: cumulative reward the random agent gained
            actions: action sequence the random agent performed
    """
    set_seeds(seed)
    env = CityEnv(init_random=rand, height=5, width=5, packages=[(0, 4), (2, 0), (4, 2)])
    env.draw_map()
    cum_r, actions = min_weight_agent(env)
    return cum_r, actions


def test_min_weight_large(seed: int = 3333, rand: bool = True):
    """
    Runs the minimal weight searching agent on the environment of field 10 x 10

        Parameters:
            seed: an integer
            rand: boolean flag for the randomness of the environment

        Returns:
            cum_r: cumulative reward the random agent gained
            actions: action sequence the random agent performed
    """
    set_seeds(seed)
    env = CityEnv(init_random=rand, height=10, width=10, packages=[(0, 2), (7, 2), (5, 5), (3, 8)])
    env.draw_map()
    cum_r, actions = min_weight_agent(env)
    return cum_r, actions


def test_max_weight_small(seed: int = 1111, rand: bool = True):
    """
    Runs the maximal weight searching agent on the environment of field 3 x 3

        Parameters:
            seed: an integer
            rand: boolean flag for the randomness of the environment

        Returns:
            cum_r: cumulative reward the random agent gained
            actions: action sequence the random agent performed
    """
    set_seeds(seed)
    env = CityEnv(init_random=rand, height=3, width=3, packages=[(2, 2), (2, 0)])
    env.draw_map()
    cum_r, actions = max_weight_agent(env)
    return cum_r, actions


def test_max_weight_medium(seed: int = 2222, rand: bool = True):
    """
    Runs the maximal weight searching agent on the environment of field 5 x 5

        Parameters:
            seed: an integer
            rand: boolean flag for the randomness of the environment

        Returns:
            cum_r: cumulative reward the random agent gained
            actions: action sequence the random agent performed
    """
    set_seeds(seed)
    env = CityEnv(init_random=rand, height=5, width=5, packages=[(0, 4), (2, 0), (4, 2)])
    env.draw_map()
    cum_r, actions = max_weight_agent(env)
    return cum_r, actions


def test_max_weight_large(seed: int = 3333, rand: bool = True):
    """
    Runs the maximal weight searching agent on the environment of field 10 x 10

        Parameters:
            seed: an integer
            rand: boolean flag for the randomness of the environment

        Returns:
            cum_r: cumulative reward the random agent gained
            actions: action sequence the random agent performed
    """
    set_seeds(seed)
    env = CityEnv(init_random=rand, height=10, width=10, packages=[(0, 2), (7, 2), (5, 5), (3, 8)])
    env.draw_map()
    cum_r, actions = max_weight_agent(env)
    return cum_r, actions


def test_sarsa_small(q: DefaultDict[Tuple, np.ndarray], seed: int = 1111, rand: bool = True, load: bool = False):
    """
    Runs the SARSA agent on the environment of field 3 x 3

        Parameters:
            q: Q table
            seed: an integer
            rand: boolean flag for the randomness of the environment
            load: boolean flag for loading the Q table

        Returns:
            cum_r: cumulative reward the SARSA agent gained
            actions: action sequence the SARSA agent performed
    """
    # Check that the Q table is not empty or that the load flag is set to True
    assert q is not None or load

    set_seeds(seed)
    env = CityEnv(init_random=rand, height=3, width=3, packages=[(2, 2), (2, 0)])
    env.draw_map()
    if load:
        if q is None:
            q = defaultdict(lambda: np.zeros(env.action_space.n))
        file_name = "q_sarsa_small_" + str(seed)
        q = load_q(q, file_name)
    cum_r, actions = evaluate_sarsa_policy(env, q)
    return cum_r, actions


def test_sarsa_medium(q: DefaultDict[Tuple, np.ndarray], seed: int = 2222, rand: bool = True, load: bool = False):
    """
    Runs the SARSA agent on the environment of field 5 x 5

        Parameters:
            q: Q table
            seed: an integer
            rand: boolean flag for the randomness of the environment
            load: boolean flag for loading the Q table

        Returns:
            cum_r: cumulative reward the SARSA agent gained
            actions: action sequence the SARSA agent performed
    """
    # Check that the Q table is not empty or that the load flag is set to True
    assert q is not None or load

    set_seeds(seed)
    env = CityEnv(init_random=rand, height=5, width=5, packages=[(0, 4), (2, 0), (4, 2)])
    env.draw_map()
    if load:
        if q is None:
            q = defaultdict(lambda: np.zeros(env.action_space.n))
        file_name = "q_sarsa_medium_" + str(seed)
        q = load_q(q, file_name)
    cum_r, actions = evaluate_sarsa_policy(env, q)
    return cum_r, actions


def test_sarsa_large(q: DefaultDict[Tuple, np.ndarray], seed: int = 3333, rand: bool = True, load: bool = False):
    """
    Runs the SARSA agent on the environment of field 10 x 10

        Parameters:
            q: Q table
            seed: an integer
            rand: boolean flag for the randomness of the environment
            load: boolean flag for loading the Q table

        Returns:
            cum_r: cumulative reward the SARSA agent gained
            actions: action sequence the SARSA agent performed
    """
    # Check that the Q table is not empty or that the load flag is set to True
    assert q is not None or load

    set_seeds(seed)
    env = CityEnv(init_random=rand, height=10, width=10, packages=[(0, 2), (7, 2), (5, 5), (3, 8)])
    env.draw_map()
    if load:
        if q is None:
            q = defaultdict(lambda: np.zeros(env.action_space.n))
        file_name = "q_sarsa_large_" + str(seed)
        q = load_q(q, file_name)
    cum_r, actions = evaluate_sarsa_policy(env, q)
    return cum_r, actions
