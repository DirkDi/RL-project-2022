import random
import logging
import numpy as np
import torch

from collections import defaultdict
from typing import DefaultDict, Tuple, List, Optional

from env import CityEnv
from sarsa import evaluate_sarsa_policy, load_q
from baselines import random_agent, min_weight_agent, max_weight_agent


def set_seeds(seed):
    """
    Set all random generators used to a specific seed

    :param seed: an integer
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.use_deterministic_algorithms(True)


def test_random_small(seed: int = 1111, rand: bool = True, draw_map: bool = False) -> Tuple[float, List[int]]:
    """
    Runs the randomly acting agent on the environment of field 3 x 3

    :param seed: an integer
    :param rand: boolean flag for the randomness of the environment
    :param draw_map: boolean flag for showing the map

    :return cum_r: cumulative reward the random agent gained
    :return actions: action sequence the random agent performed
    """
    set_seeds(seed)
    env = CityEnv(init_random=rand, height=3, width=3, packages=[(2, 2), (2, 0)])
    if draw_map:
        env.draw_map()
    cum_r, actions = random_agent(env)
    return cum_r, actions


def test_random_medium(seed: int = 2222, rand: bool = True, draw_map: bool = False) -> Tuple[float, List[int]]:
    """
    Runs the randomly acting agent on the environment of field 5 x 5

    :param seed: an integer
    :param rand: boolean flag for the randomness of the environment
    :param draw_map: boolean flag for showing the map

    :return cum_r: cumulative reward the random agent gained
    :return actions: action sequence the random agent performed
    """
    set_seeds(seed)
    env = CityEnv(init_random=rand, height=5, width=5, packages=[(0, 4), (2, 0), (4, 2)])
    if draw_map:
        env.draw_map()
    cum_r, actions = random_agent(env)
    return cum_r, actions


def test_random_large(seed: int = 3333, rand: bool = True, draw_map: bool = False) -> Tuple[float, List[int]]:
    """
    Runs the randomly acting agent on the environment of field 10 x 10

    :param seed: an integer
    :param rand: boolean flag for the randomness of the environment
    :param draw_map: boolean flag for showing the map

    :return cum_r: cumulative reward the random agent gained
    :return actions: action sequence the random agent performed
    """
    set_seeds(seed)
    env = CityEnv(init_random=rand, height=10, width=10, packages=[(0, 2), (7, 2), (5, 5), (3, 8)])
    if draw_map:
        env.draw_map()
    cum_r, actions = random_agent(env)
    return cum_r, actions


def test_min_weight_small(seed: int = 1111, rand: bool = True, draw_map: bool = False) -> Tuple[float, List[int]]:
    """
    Runs the minimal weight searching agent on the environment of field 3 x 3

    :param seed: an integer
    :param rand: boolean flag for the randomness of the environment
    :param draw_map: boolean flag for showing the map

    :return cum_r: cumulative reward the random agent gained
    :return actions: action sequence the random agent performed
    """
    set_seeds(seed)
    env = CityEnv(init_random=rand, height=3, width=3, packages=[(2, 2), (2, 0)])
    if draw_map:
        env.draw_map()
    cum_r, actions = min_weight_agent(env)
    return cum_r, actions


def test_min_weight_medium(seed: int = 2222, rand: bool = True, draw_map: bool = False) -> Tuple[float, List[int]]:
    """
    Runs the minimal weight searching agent on the environment of field 5 x 5

    :param seed: an integer
    :param rand: boolean flag for the randomness of the environment
    :param draw_map: boolean flag for showing the map

    :return cum_r: cumulative reward the random agent gained
    :return actions: action sequence the random agent performed
    """
    set_seeds(seed)
    env = CityEnv(init_random=rand, height=5, width=5, packages=[(0, 4), (2, 0), (4, 2)])
    if draw_map:
        env.draw_map()
    cum_r, actions = min_weight_agent(env)
    return cum_r, actions


def test_min_weight_large(seed: int = 3333, rand: bool = True, draw_map: bool = False) -> Tuple[float, List[int]]:
    """
    Runs the minimal weight searching agent on the environment of field 10 x 10

    :param seed: an integer
    :param rand: boolean flag for the randomness of the environment
    :param draw_map: boolean flag for showing the map

    :return cum_r: cumulative reward the random agent gained
    :return actions: action sequence the random agent performed
    """
    set_seeds(seed)
    env = CityEnv(init_random=rand, height=10, width=10, packages=[(0, 2), (7, 2), (5, 5), (3, 8)])
    if draw_map:
        env.draw_map()
    cum_r, actions = min_weight_agent(env)
    return cum_r, actions


def test_max_weight_small(seed: int = 1111, rand: bool = True, draw_map: bool = False) -> Tuple[float, List[int]]:
    """
    Runs the maximal weight searching agent on the environment of field 3 x 3

    :param seed: an integer
    :param rand: boolean flag for the randomness of the environment
    :param draw_map: boolean flag for showing the map

    :return cum_r: cumulative reward the random agent gained
    :return actions: action sequence the random agent performed
    """
    set_seeds(seed)
    env = CityEnv(init_random=rand, height=3, width=3, packages=[(2, 2), (2, 0)])
    if draw_map:
        env.draw_map()
    cum_r, actions = max_weight_agent(env)
    return cum_r, actions


def test_max_weight_medium(seed: int = 2222, rand: bool = True, draw_map: bool = False) -> Tuple[float, List[int]]:
    """
    Runs the maximal weight searching agent on the environment of field 5 x 5

    :param seed: an integer
    :param rand: boolean flag for the randomness of the environment
    :param draw_map: boolean flag for showing the map

    :return cum_r: cumulative reward the random agent gained
    :return actions: action sequence the random agent performed
    """
    set_seeds(seed)
    env = CityEnv(init_random=rand, height=5, width=5, packages=[(0, 4), (2, 0), (4, 2)])
    if draw_map:
        env.draw_map()
    cum_r, actions = max_weight_agent(env)
    return cum_r, actions


def test_max_weight_large(seed: int = 3333, rand: bool = True, draw_map: bool = False) -> Tuple[float, List[int]]:
    """
    Runs the maximal weight searching agent on the environment of field 10 x 10

    :param seed: an integer
    :param rand: boolean flag for the randomness of the environment
    :param draw_map: boolean flag for showing the map

    :return cum_r: cumulative reward the random agent gained
    :return actions: action sequence the random agent performed
    """
    set_seeds(seed)
    env = CityEnv(init_random=rand, height=10, width=10, packages=[(0, 2), (7, 2), (5, 5), (3, 8)])
    if draw_map:
        env.draw_map()
    cum_r, actions = max_weight_agent(env)
    return cum_r, actions


def test_sarsa_small(q: Optional[DefaultDict[Tuple, np.ndarray]], seed: int = 1111,
                     rand: bool = True, load: bool = False, draw_map: bool = False) -> Tuple[float, List[int]]:
    """
    Runs the SARSA agent on the environment of field 3 x 3

    :param q: Q table
    :param seed: an integer
    :param rand: boolean flag for the randomness of the environment
    :param load: boolean flag for loading the Q table
    :param draw_map: boolean flag for showing the map

    :return cum_r: cumulative reward the SARSA agent gained
    :return actions: action sequence the SARSA agent performed
    """
    # Check that the Q table is not empty or that the load flag is set to True
    assert q is not None or load

    set_seeds(seed)
    env = CityEnv(init_random=rand, height=3, width=3, packages=[(2, 2), (2, 0)])
    if draw_map:
        env.draw_map()
    if load:
        if q is None:
            q = defaultdict(lambda: np.zeros(env.action_space.n))
        file_name = f"q_sarsa_small_{seed}"
        q = load_q(q, file_name)
    cum_r, actions = evaluate_sarsa_policy(env, q)
    return cum_r, actions


def test_sarsa_medium(q: Optional[DefaultDict[Tuple, np.ndarray]], seed: int = 1111,
                      rand: bool = True, load: bool = False, draw_map: bool = False) -> Tuple[float, List[int]]:
    """
    Runs the SARSA agent on the environment of field 5 x 5

    :param q: Q table
    :param seed: an integer
    :param rand: boolean flag for the randomness of the environment
    :param load: boolean flag for loading the Q table
    :param draw_map: boolean flag for showing the map

    :return cum_r: cumulative reward the SARSA agent gained
    :return actions: action sequence the SARSA agent performed
    """
    # Check that the Q table is not empty or that the load flag is set to True
    assert q is not None or load

    set_seeds(seed)
    env = CityEnv(init_random=rand, height=5, width=5, packages=[(0, 4), (2, 0), (4, 2)])
    if draw_map:
        env.draw_map()
    if load:
        if q is None:
            q = defaultdict(lambda: np.zeros(env.action_space.n))
        file_name = f"q_sarsa_medium_{seed}"
        q = load_q(q, file_name)
    cum_r, actions = evaluate_sarsa_policy(env, q)
    return cum_r, actions


def test_sarsa_large(q: Optional[DefaultDict[Tuple, np.ndarray]], seed: int = 1111,
                     rand: bool = True, load: bool = False, draw_map: bool = False) -> Tuple[float, List[int]]:
    """
    Runs the SARSA agent on the environment of field 10 x 10

    :param q: Q table
    :param seed: an integer
    :param rand: boolean flag for the randomness of the environment
    :param load: boolean flag for loading the Q table
    :param draw_map: boolean flag for showing the map

    :return cum_r: cumulative reward the SARSA agent gained
    :return actions: action sequence the SARSA agent performed
    """
    # Check that the Q table is not empty or that the load flag is set to True
    assert q is not None or load

    set_seeds(seed)
    env = CityEnv(init_random=rand, height=10, width=10, packages=[(0, 2), (7, 2), (5, 5), (3, 8)])
    if draw_map:
        env.draw_map()
    if load:
        if q is None:
            q = defaultdict(lambda: np.zeros(env.action_space.n))
        file_name = f"q_sarsa_large_{seed}"
        q = load_q(q, file_name)
    cum_r, actions = evaluate_sarsa_policy(env, q)
    return cum_r, actions


def test():
    """
    Tests the agents.
    """
    for seed in [1111, 2222, 3333]:
        logging.info("Testing with seed:", seed)

        logging.info("Testing the random agent")

        cum_r, actions = test_random_small(seed)
        logging.info(f"Results for the small field (3 x 3):\ncumulative reward: {cum_r}\naction sequence: {actions}")

        cum_r, actions = test_random_medium(seed)
        logging.info(f"Results for the medium field (5 x 5):\ncumulative reward: {cum_r}\naction sequence: {actions}")

        cum_r, actions = test_random_large(seed)
        logging.info(f"Results for the large field (10 x 10):\ncumulative reward: {cum_r}\naction sequence: {actions}")

        logging.info("Testing the min-weight-agent")

        cum_r, actions = test_min_weight_small(seed)
        logging.info(f"Results for the small field (3 x 3):\ncumulative reward: {cum_r}\naction sequence: {actions}")

        cum_r, actions = test_min_weight_medium(seed)
        logging.info(f"Results for the medium field (5 x 5):\ncumulative reward: {cum_r}\naction sequence: {actions}")

        cum_r, actions = test_min_weight_large(seed)
        logging.info(f"Results for the large field (10 x 10):\ncumulative reward: {cum_r}\naction sequence: {actions}")

        logging.info("Testing the max-weight-agent")

        cum_r, actions = test_max_weight_small(seed)
        logging.info(f"Results for the small field (3 x 3):\ncumulative reward: {cum_r}\naction sequence: {actions}")

        cum_r, actions = test_max_weight_medium(seed)
        logging.info(f"Results for the medium field (5 x 5):\ncumulative reward: {cum_r}\naction sequence: {actions}")

        cum_r, actions = test_max_weight_large(seed)
        logging.info(f"Results for the large field (10 x 10):\ncumulative reward: {cum_r}\naction sequence: {actions}")

        logging.info("Testing the SARSA agent")

        cum_r, actions = test_sarsa_small(None, seed, load=True)
        logging.info(f"Results for the small field (3 x 3):\ncumulative reward: {cum_r}\naction sequence: {actions}")

        cum_r, actions = test_sarsa_medium(None, seed, load=True)
        logging.info(f"Results for the medium field (5 x 5):\ncumulative reward: {cum_r}\naction sequence: {actions}")

        cum_r, actions = test_sarsa_large(None, seed, load=True)
        logging.info(f"Results for the large field (10 x 10):\ncumulative reward: {cum_r}\naction sequence: {actions}")


if __name__ == '__main__':
    test()
