import random
import numpy as np
import torch

from typing import Tuple, List, DefaultDict

from env import CityEnv
from sarsa import sarsa, save_q


def set_seeds(seed: int):
    """
    Set all random generators used to a specific seed

    :param seed: an integer
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.use_deterministic_algorithms(True)


def train_sarsa_small(num_episodes: int, seed: int = 1111,
                      rand: bool = True, save: bool = False, draw_map: bool = False
                      ) -> Tuple[List[float], List[int], DefaultDict[Tuple, np.ndarray]]:
    """
    Trains the SARSA agent on the environment of field 3 x 3

    :param num_episodes: amount of episodes the agents should learn
    :param seed: an integer
    :param rand: boolean flag for randomness of the environment
    :param save: boolean flag for saving the Q table
    :param draw_map: boolean flag for showing the map

    :return r: list of rewards the agent gained per training episode
    :return l: list of lengths of the episodes the agent trained
    :return q: Q table
    """
    set_seeds(seed)
    env = CityEnv(init_random=rand, height=3, width=3, packages=[(2, 2), (2, 0)])
    if draw_map:
        env.draw_map()
    r, l, q = sarsa(env, num_episodes)
    if save:
        file_name = f"q_sarsa_small_{seed}"
        save_q(q, file_name)
    return r, l, q


def train_sarsa_medium(num_episodes: int, seed: int = 2222,
                       rand: bool = True, save: bool = False, draw_map: bool = False
                       ) -> Tuple[List[float], List[int], DefaultDict[Tuple, np.ndarray]]:
    """
    Trains the SARSA agent on the environment of field 5 x 5

    :param num_episodes: amount of episodes the agents should learn
    :param seed: an integer
    :param rand: boolean flag for randomness of the environment
    :param save: boolean flag for saving the Q table
    :param draw_map: boolean flag for showing the map

    :return r: list of rewards the agent gained per training episode
    :return l: list of lengths of the episodes the agent trained
    :return q: Q table
    """
    set_seeds(seed)
    env = CityEnv(init_random=rand, height=5, width=5, packages=[(0, 4), (2, 0), (4, 2)])
    if draw_map:
        env.draw_map()
    r, l, q = sarsa(env, num_episodes)
    if save:
        file_name = f"q_sarsa_medium_{seed}"
        save_q(q, file_name)
    return r, l, q


def train_sarsa_large(num_episodes: int, seed: int = 3333,
                      rand: bool = True, save: bool = False, draw_map: bool = False
                      ) -> Tuple[List[float], List[int], DefaultDict[Tuple, np.ndarray]]:
    """
    Trains the SARSA agent on the environment of field 10 x 10

    :param num_episodes: amount of episodes the agents should learn
    :param seed: an integer
    :param rand: boolean flag for randomness of the environment
    :param save: boolean flag for saving the Q table
    :param draw_map: boolean flag for showing the map

    :return r: list of rewards the agent gained per training episode
    :return l: list of lengths of the episodes the agent trained
    :return q: Q table
    """
    set_seeds(seed)
    env = CityEnv(init_random=rand, height=10, width=10, packages=[(0, 2), (7, 2), (5, 5), (3, 8)])
    if draw_map:
        env.draw_map()
    r, l, q = sarsa(env, num_episodes)
    if save:
        file_name = f"q_sarsa_large_{seed}"
        save_q(q, file_name)
    return r, l, q


def train():
    """
    Trains the SARSA agents and saves the Q tables.
    """
    seeds = [1111, 2222, 3333]

    # Train on the small field (3 x 3)
    num_episodes_small = 10000
    for seed in seeds:
        train_sarsa_small(num_episodes_small, seed, save=True)

    # Train on the medium field (5 x 5)
    num_episodes_medium = 100000
    for seed in seeds:
        train_sarsa_medium(num_episodes_medium, seed, save=True)

    # Train on the large field (10 x 10)
    num_episodes_large = 1000000
    for seed in seeds:
        train_sarsa_small(num_episodes_large, seed, save=True)


if __name__ == '__main__':
    train()
