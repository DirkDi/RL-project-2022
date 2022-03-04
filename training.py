import numpy as np

from typing import Tuple, List, DefaultDict

from env_creator import create_small_env, create_medium_env, create_large_env
from sarsa import sarsa, save_q


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
    env = create_small_env(seed, rand, draw_map)
    r, l, q = sarsa(env, num_episodes)
    print("sarsa done")
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
    env = create_medium_env(seed, rand, draw_map)
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
    env = create_large_env(seed, rand, draw_map)
    r, l, q = sarsa(env, num_episodes)
    if save:
        file_name = f"q_sarsa_large_{seed}"
        save_q(q, file_name)
    return r, l, q


def train():
    """
    Trains SARSA agents for different environments (3x3, 5x5, 10x10) and saves the Q tables.
    """
    seeds = [1111, 2222, 3333]
    num_episodes_small = 10000
    num_episodes_medium = 100000
    num_episodes_large = 1000000
    # train environments with different seeds
    for seed in seeds:
        # Train on the small field (3 x 3)
        train_sarsa_small(num_episodes_small, seed, save=True)
        # Train on the medium field (5 x 5)
        train_sarsa_medium(num_episodes_medium, seed, save=True)
        # Train on the large field (10 x 10)
        train_sarsa_small(num_episodes_large, seed, save=True)


if __name__ == '__main__':
    train()
