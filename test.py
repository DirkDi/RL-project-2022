import logging
import numpy as np

from collections import defaultdict
from typing import DefaultDict, Tuple, List, Optional

from env_creator import create_small_env, create_medium_env, create_large_env
from sarsa import evaluate_sarsa_policy, load_q
from baselines import random_agent, min_weight_agent, max_weight_agent


def test_random_small(seed: int = 1111, rand: bool = True, draw_map: bool = False) -> Tuple[float, List[int]]:
    """
    Runs the randomly acting agent on the environment of field 3 x 3

    :param seed: an integer to set the random seed
    :param rand: boolean flag for the randomness of the environment
    :param draw_map: boolean flag for showing the map

    :return cum_r: cumulative reward the random agent gained
    :return actions: action sequence the random agent performed
    """
    env = create_small_env(seed, rand, draw_map)
    cum_r, actions = random_agent(env)
    return cum_r, actions


def test_random_medium(seed: int = 2222, rand: bool = True, draw_map: bool = False) -> Tuple[float, List[int]]:
    """
    Runs the randomly acting agent on the environment of field 5 x 5

    :param seed: an integer to set the random seed
    :param rand: boolean flag for the randomness of the environment
    :param draw_map: boolean flag for showing the map

    :return cum_r: cumulative reward the random agent gained
    :return actions: action sequence the random agent performed
    """
    env = create_medium_env(seed, rand, draw_map)
    cum_r, actions = random_agent(env)
    return cum_r, actions


def test_random_large(seed: int = 3333, rand: bool = True, draw_map: bool = False) -> Tuple[float, List[int]]:
    """
    Runs the randomly acting agent on the environment of field 10 x 10

    :param seed: an integer to set the random seed
    :param rand: boolean flag for the randomness of the environment
    :param draw_map: boolean flag for showing the map

    :return cum_r: cumulative reward the random agent gained
    :return actions: action sequence the random agent performed
    """
    env = create_large_env(seed, rand, draw_map)
    cum_r, actions = random_agent(env)
    return cum_r, actions


def test_min_weight_small(seed: int = 1111, rand: bool = True, draw_map: bool = False) -> Tuple[float, List[int]]:
    """
    Runs the minimal weight searching agent on the environment of field 3 x 3

    :param seed: an integer to set the random seed
    :param rand: boolean flag for the randomness of the environment
    :param draw_map: boolean flag for showing the map

    :return cum_r: cumulative reward the random agent gained
    :return actions: action sequence the random agent performed
    """
    env = create_small_env(seed, rand, draw_map)
    cum_r, actions = min_weight_agent(env)
    return cum_r, actions


def test_min_weight_medium(seed: int = 2222, rand: bool = True, draw_map: bool = False) -> Tuple[float, List[int]]:
    """
    Runs the minimal weight searching agent on the environment of field 5 x 5

    :param seed: an integer to set the random seed
    :param rand: boolean flag for the randomness of the environment
    :param draw_map: boolean flag for showing the map

    :return cum_r: cumulative reward the random agent gained
    :return actions: action sequence the random agent performed
    """
    env = create_medium_env(seed, rand, draw_map)
    cum_r, actions = min_weight_agent(env)
    return cum_r, actions


def test_min_weight_large(seed: int = 3333, rand: bool = True, draw_map: bool = False) -> Tuple[float, List[int]]:
    """
    Runs the minimal weight searching agent on the environment of field 10 x 10

    :param seed: an integer to set the random seed
    :param rand: boolean flag for the randomness of the environment
    :param draw_map: boolean flag for showing the map

    :return cum_r: cumulative reward the random agent gained
    :return actions: action sequence the random agent performed
    """
    env = create_large_env(seed, rand, draw_map)
    cum_r, actions = min_weight_agent(env)
    return cum_r, actions


def test_max_weight_small(seed: int = 1111, rand: bool = True, draw_map: bool = False) -> Tuple[float, List[int]]:
    """
    Runs the maximal weight searching agent on the environment of field 3 x 3

    :param seed: an integer to set the random seed
    :param rand: boolean flag for the randomness of the environment
    :param draw_map: boolean flag for showing the map

    :return cum_r: cumulative reward the random agent gained
    :return actions: action sequence the random agent performed
    """
    env = create_small_env(seed, rand, draw_map)
    cum_r, actions = max_weight_agent(env)
    return cum_r, actions


def test_max_weight_medium(seed: int = 2222, rand: bool = True, draw_map: bool = False) -> Tuple[float, List[int]]:
    """
    Runs the maximal weight searching agent on the environment of field 5 x 5

    :param seed: an integer to set the random seed
    :param rand: boolean flag for the randomness of the environment
    :param draw_map: boolean flag for showing the map

    :return cum_r: cumulative reward the random agent gained
    :return actions: action sequence the random agent performed
    """
    env = create_medium_env(seed, rand, draw_map)
    cum_r, actions = max_weight_agent(env)
    return cum_r, actions


def test_max_weight_large(seed: int = 3333, rand: bool = True, draw_map: bool = False) -> Tuple[float, List[int]]:
    """
    Runs the maximal weight searching agent on the environment of field 10 x 10

    :param seed: an integer to set the random seed
    :param rand: boolean flag for the randomness of the environment
    :param draw_map: boolean flag for showing the map

    :return cum_r: cumulative reward the random agent gained
    :return actions: action sequence the random agent performed
    """
    env = create_large_env(seed, rand, draw_map)
    cum_r, actions = max_weight_agent(env)
    return cum_r, actions


def test_sarsa_small(q: Optional[DefaultDict[Tuple, np.ndarray]], seed: int = 1111,
                     rand: bool = True, load: bool = False, draw_map: bool = False) -> Tuple[float, List[int]]:
    """
    Runs the SARSA agent on the environment of field 3 x 3

    :param q: Q table
    :param seed: an integer to set the random seed
    :param rand: boolean flag for the randomness of the environment
    :param load: boolean flag for loading the Q table
    :param draw_map: boolean flag for showing the map

    :return cum_r: cumulative reward the SARSA agent gained
    :return actions: action sequence the SARSA agent performed
    """
    # Check that the Q table is not empty or that the load flag is set to True
    assert q is not None or load

    env = create_small_env(seed, rand, draw_map)
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
    :param seed: an integer to set the random seed
    :param rand: boolean flag for the randomness of the environment
    :param load: boolean flag for loading the Q table
    :param draw_map: boolean flag for showing the map

    :return cum_r: cumulative reward the SARSA agent gained
    :return actions: action sequence the SARSA agent performed
    """
    # Check that the Q table is not empty or that the load flag is set to True
    assert q is not None or load

    env = create_medium_env(seed, rand, draw_map)
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
    :param seed: an integer to set the random seed
    :param rand: boolean flag for the randomness of the environment
    :param load: boolean flag for loading the Q table
    :param draw_map: boolean flag for showing the map

    :return cum_r: cumulative reward the SARSA agent gained
    :return actions: action sequence the SARSA agent performed
    """
    # Check that the Q table is not empty or that the load flag is set to True
    assert q is not None or load

    env = create_large_env(seed, rand, draw_map)
    if load:
        if q is None:
            q = defaultdict(lambda: np.zeros(env.action_space.n))
        file_name = f"q_sarsa_large_{seed}"
        q = load_q(q, file_name)
    cum_r, actions = evaluate_sarsa_policy(env, q)
    return cum_r, actions


def test_agent(name, seed, tests):
    """
    Tests specific agent with the corresponding test functions.
    :param name: name of the agent
    :param seed: an integer to set the random seed
    :param tests: array which contains the functions for tests
    """
    logging.info(f"Testing the {name}")
    cum_r, actions = tests[0](seed) if name != "sarsa agent" else tests[0](None, seed, load=True)
    logging.info(f"Results for the small field (3 x 3):\ncumulative reward: {cum_r}\naction sequence: {actions}")
    cum_r, actions = tests[1](seed) if name != "sarsa agent" else tests[1](None, seed, load=True)
    logging.info(f"Results for the medium field (5 x 5):\ncumulative reward: {cum_r}\naction sequence: {actions}")
    cum_r, actions = tests[2](seed) if name != "sarsa agent" else tests[2](None, seed, load=True)
    logging.info(f"Results for the large field (10 x 10):\ncumulative reward: {cum_r}\naction sequence: {actions}")


def test():
    """
    Tests the agents.
    """
    agent_func_dict = {
        "random agent": [test_random_small, test_random_medium, test_random_large],
        "min-weight-agent": [test_min_weight_small, test_min_weight_medium, test_min_weight_large],
        "max-weight-agent": [test_max_weight_small, test_max_weight_medium, test_max_weight_large],
        "sarsa agent": [test_sarsa_small, test_sarsa_medium, test_sarsa_large],
    }
    for seed in [1111, 2222, 3333]:
        logging.info(f"Testing with seed: {seed}")

        test_agent("random agent", seed, agent_func_dict["random agent"])
        test_agent("min-weight-agent", seed, agent_func_dict["min-weight-agent"])
        test_agent("max-weight-agent", seed, agent_func_dict["max-weight-agent"])
        test_agent("sarsa agent", seed, agent_func_dict["sarsa agent"])


if __name__ == '__main__':
    log = logging.INFO
    logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=log)
    test()
