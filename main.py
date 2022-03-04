import torch
import random
import argparse
import numpy as np
from pathlib import Path
from sarsa import load_q
from test import test_random_small, test_random_medium, test_random_large, \
    test_min_weight_small, test_min_weight_medium, test_min_weight_large, \
    test_max_weight_small, test_max_weight_medium, test_max_weight_large, \
    test_sarsa_small, test_sarsa_medium, test_sarsa_large
from training import train_sarsa_small, train_sarsa_medium, train_sarsa_large
from env import CityEnv
from env_creator import *
from sb_agents import *
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env


def args_parser():
    """
    Creates an argument parser to choose between several options.

    :return: created argument parser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug", action="store_true", help="debug output")
    parser.add_argument('--mode', '-m', type=str, default='normal',
                        help='choose the mode normal or experimental (experimental env)')
    parser.add_argument("--static", action="store_true", help="disables random distance generation")
    parser.add_argument("-o", "--bidirectional", action="store_true", help="deactivates one way streets")
    parser.add_argument("-c", "--interconnected", action="store_true", help="deactivates construction sites")
    parser.add_argument("-t", "--notrafficlights", action="store_true", help="deactivates traffic lights")
    parser.add_argument("-s", "--seed", type=list, default=[1111, 2222, 3333], help="set the environment seed")
    parser.add_argument("-g", "--graph", action="store_true", help="show graphic representation of env")
    args = parser.parse_args()
    return args


def show_result(grid_name, cum_r, actions):
    """
    Shows the cumulative reward and action sequence results of an agent for a specific grid

    :param grid_name: the name of the grid
    :param cum_r: the cumulative reward
    :param actions: the chosen action sequence
    """
    logging.info(f'Results for {grid_name}:')
    logging.info(f'The cumulative reward is {cum_r}')
    logging.info(f'The optimal action sequence is {actions}')


def show_average_results(agent_name, reward_s, reward_m, reward_l):
    """
    Shows the average reward values for the small environment (3x3), medium environment (5x5) and large one (10x10)
    :param agent_name: the name of the used agent
    :param reward_s: the reward list of the small environment over all seeds
    :param reward_m: the reward list of the medium environment over all seeds
    :param reward_l: the reward list of the large environment over all seeds
    """
    # get amount of specific environment rewards
    len_small = len(reward_s)
    len_medium = len(reward_m)
    len_large = len(reward_l)
    # get standard deviation of rewards for specific environment sizes
    std_s = round(np.std(reward_s), 2)
    std_m = round(np.std(reward_m), 2)
    std_l = round(np.std(reward_l), 2)
    # get average reward for specific environment size
    average_reward_s = round(sum(reward_s) / len_small, 2) if len_small else 0.00
    average_reward_m = round(sum(reward_m) / len_medium, 2) if len_medium else 0.00
    average_reward_l = round(sum(reward_l) / len_large, 2) if len_medium else 0.00
    # show the results
    logging.info(f'Average rewards for {agent_name}:')
    logging.info(f'Average reward for 3x3 with {len_small} seeds is {average_reward_s} (-/+ {std_s})')
    logging.info(f'Average reward for 5x5 with {len_medium} seeds is {average_reward_m} (-/+ {std_m})')
    logging.info(f'Average reward for 10x10 with {len_large} seeds is {average_reward_l} (-/+ {std_l})')


def main():
    """
    Performs training and testing with different seeds and environments for a chosen agent.
    """
    args = args_parser()
    # use logging to get prints for debug/info/error mode
    log = logging.INFO
    if args.debug:
        log = logging.DEBUG
    logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=log)
    # check which method should be used
    mode = args.mode.lower()
    if mode not in ['normal', 'experimental', 'random', 'min_weight', 'max_weight', 'a2c', 'ppo', 'dqn']:
        logging.error('The mode has to be normal or experimental.')
        return

    seeds = args.seed
    show_graph = args.graph
    average_reward_s = []
    average_reward_m = []
    average_reward_l = []
    if mode == 'normal':
        episodes = [100000, 100000, 250000]
        for seed in seeds:
            num_episodes = episodes[0]
            if not Path(f"q_sarsa_small_{seed}.csv").is_file():
                train_sarsa_small(num_episodes, seed, save=True)
            cum_r, actions = test_sarsa_small(q=None, seed=seed, load=True, draw_map=True)
            average_reward_s.append(cum_r)
            show_result("3x3 grid", cum_r, actions)

            num_episodes = episodes[1]
            if not Path(f"q_sarsa_medium_{seed}.csv").is_file():
                train_sarsa_medium(num_episodes, seed, save=True)
            cum_r, actions = test_sarsa_medium(q=None, seed=seed, load=True, draw_map=True)
            average_reward_m.append(cum_r)
            show_result("5x5 grid", cum_r, actions)

            num_episodes = episodes[2]
            if not Path(f"q_sarsa_large_{seed}.csv").is_file():
                train_sarsa_large(num_episodes, seed, save=True)
            cum_r, actions = test_sarsa_large(q=None, seed=seed, load=True, draw_map=show_graph)
            average_reward_l.append(cum_r)
            show_result("10x10 grid", cum_r, actions)
        show_average_results("SARSA", average_reward_s, average_reward_m, average_reward_l)
    elif mode == 'random':
        for seed in seeds:
            # calculate small environment and store value
            cum_r, actions = test_random_small(seed)
            show_result("3x3 grid", cum_r, actions)
            average_reward_s.append(cum_r)
            # calculate medium environment and store value
            cum_r, actions = test_random_medium(seed)
            show_result("5x5 grid", cum_r, actions)
            average_reward_m.append(cum_r)
            # calculate large environment and store value
            cum_r, actions = test_random_large(seed)
            show_result("10x10 grid", cum_r, actions)
            average_reward_l.append(cum_r)
        show_average_results("random", average_reward_s, average_reward_m, average_reward_l)
    elif mode == 'min_weight':
        for seed in seeds:
            # calculate small environment and store value
            cum_r, actions = test_min_weight_small(seed)
            show_result("3x3 grid", cum_r, actions)
            average_reward_s.append(cum_r)
            # calculate medium environment and store value
            cum_r, actions = test_min_weight_medium(seed)
            show_result("5x5 grid", cum_r, actions)
            average_reward_m.append(cum_r)
            # calculate large environment and store value
            cum_r, actions = test_min_weight_large(seed)
            show_result("10x10 grid", cum_r, actions)
            average_reward_l.append(cum_r)
        show_average_results("minimum weight agent", average_reward_s, average_reward_m, average_reward_l)
    elif mode == 'max_weight':
        for seed in seeds:
            # calculate small environment and store value
            cum_r, actions = test_max_weight_small(seed)
            show_result("3x3 grid", cum_r, actions)
            average_reward_s += cum_r
            # calculate medium environment and store value
            cum_r, actions = test_max_weight_medium(seed)
            show_result("5x5 grid", cum_r, actions)
            average_reward_m += cum_r
            # calculate large environment and store value
            cum_r, actions = test_max_weight_large(seed)
            show_result("10x10 grid", cum_r, actions)
            average_reward_l += cum_r
        show_average_results("maximum weight agent", average_reward_s, average_reward_m, average_reward_l)
    elif mode == 'a2c':  # not worked above 3x3 grid size (no useful policy)
        for seed in seeds:
            env = create_small_env(seed, True, show_graph)
            check_env(env, warn=True)
            model = a2c_agent(env, total_timesteps=1000000, log_interval=100)
            cum_r, actions = run_agent(env, model)
            show_result("3x3 grid", cum_r, actions)
    elif mode == 'ppo':  # not worked above 3x3 grid size (no useful policy)
        for seed in seeds:
            env = create_small_env(seed, True, show_graph)
            check_env(env, warn=True)
            model = ppo_agent(env, total_timesteps=100000)
            cum_r, actions = run_agent(env, model)
            show_result("3x3 grid", cum_r, actions)
    elif mode == 'dqn':  # no useful policy
        for seed in seeds:
            env = create_small_env(seed, True, show_graph)
            check_env(env, warn=True)
            model = dqn_agent(env, total_timesteps=100000)
            cum_r, actions = run_agent(env, model)
            show_result("3x3 grid", cum_r, actions)


if __name__ == '__main__':
    main()
