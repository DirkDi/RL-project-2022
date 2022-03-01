import torch
import random
import argparse
import logging
from env import CityEnv
from sarsa import load_q
from test import *
from training import train_sarsa_small, train_sarsa_medium, train_sarsa_large
from experimental.CityEnv import CityEnv as Env
from sb_agents import *
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env


def argsparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug", action="store_true", help="debug output")
    parser.add_argument('--mode', '-m', type=str, default='normal',
                        help='choose the mode normal or experimental (experimental env)')
    parser.add_argument("--static", "-s", action="store_true", help="disables random distance generation")
    parser.add_argument("-o", "--bidirectional", action="store_true", help="deactivates one way streets")
    parser.add_argument("-c", "--interconnected", action="store_true", help="deactivates construction sites")
    parser.add_argument("-t", "--notrafficlights", action="store_true", help="deactivates traffic lights")
    args = parser.parse_args()
    return args


def set_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    #torch.use_deterministic_algorithms(True)


def test():
    """
    To experiment with some things
    """
    """
    NOTE: Important!
        Indexing scheme: row entry := start vertex
                     column entry := target vertex
    """
    dist_matrix = np.array([
        [0, 20, 0, 1, 0, 0, 0, 0, 0],
        [20, 0, 1, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 20, 0, 0, 0],
        [1, 0, 0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 1, 0, 1, 0],
        [0, 0, 20, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 20, 0, 1, 0]
    ])

    dist_matrix = np.array([
        [0, 113, 0, 92, 0, 0, 0, 0, 0],
        [113, 0, 43, 0, 139, 0, 0, 0, 0],
        [0, 43, 0, 0, 0, 17, 0, 0, 0],
        [92, 0, 0, 0, 0, 0, 130, 0, 0],
        [0, 139, 0, 0, 0, 23, 0, 0, 0],
        [0, 0, 0, 0, 23, 0, 0, 0, 77],
        [0, 0, 0, 130, 0, 0, 0, 123, 0],
        [0, 0, 0, 0, 126, 0, 123, 0, 141],
        [0, 0, 0, 0, 0, 77, 0, 141, 0]
    ])

    env = Env(height=3, width=3, packages=[(0, 2), (2, 2)], dist_matrix=dist_matrix, traffic_lights=[(1, 0), (0, 1)])
    env.reset()
    # hyper_parameter_grid_search(env)
    # logging.info(env.packages)
    # env.draw_map()
    env.close()
    return 0, []

    r, l, Q = sarsa(env, 25000)

    pi = np.zeros((3, env.height, env.width))
    for (x, y, c), actions in Q.items():
        pi[c, x, y] = np.argmax(actions)
    logging.debug(pi)

    cum_r, actions = evaluate_sarsa_policy(Q, env)
    return cum_r, actions


def main():
    args = argsparser()
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

    # logging.info(f'Start experiments with the seed {seed} and mode {mode}')
    if mode == 'normal':
        # logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
        # logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=log)
        seeds = [1111]
        episodes = [2000]
        for seed, num_episodes in zip(seeds, episodes):
            # _, _, q = train_sarsa_small(num_episodes, seed, save=True)
            cum_r, actions = test_sarsa_small(q=None, seed=seed, load=True)
            logging.info(f'The cummulative reward is {cum_r}')
            logging.info(f'The optimal action sequence is {actions}')
    elif mode == 'random':
        cum_r, actions = test_random_small()
        # cum_r, actions = test_random_medium()
        # cum_r, actions = test_random_large()
    elif mode == 'min_weight':
        cum_r, actions = test_min_weight_small()
        # cum_r, actions = test_max_weight_medium()
        # cum_r, actions = test_min_weight_large()
    elif mode == 'max_weight':
        # cum_r, actions = test_max_weight_small()
        # cum_r, actions = test_max_weight_medium()
        cum_r, actions = test_max_weight_large()
    elif mode == 'a2c':
        env = CityEnv(init_random=not args.static, height=3, width=3, packages=[(2, 0), (2, 2)])
        env.draw_map()
        check_env(env, warn=True)
        model = a2c_agent(env, total_timesteps=1000000, log_interval=100)
        logging.info("training done")
        cum_r, actions = run_agent(env, model)
    elif mode == 'ppo':
        env = CityEnv(init_random=not args.static, height=5, width=5, packages=[(0, 4), (2, 0), (4, 2)])
        env.draw_map()
        # check_env(env, warn=True)
        model = ppo_agent(env, total_timesteps=1000000)
        logging.info("training done")
        cum_r, actions = run_agent(env, model)
    elif mode == 'dqn':
        env = CityEnv(init_random=not args.static, height=3, width=3, packages=[(2, 2)])
        check_env(env, warn=True)
        model = dqn_agent(env)
        logging.info("training done")
        cum_r, actions = run_agent(env, model)
    else:
        # cum_r, actions = test()
        env = CityEnv(init_random=not args.static, height=5, width=5, packages=[(2, 2), (2, 0)],
                      one_way=not args.bidirectional, construction_sites=not args.interconnected,
                      traffic_lights=not args.notrafficlights)
        # hyper_parameter_grid_search(env)
        cum_r, actions = 0, []


if __name__ == '__main__':
    main()
