import torch
import random
import argparse
import baselines
from env import CityEnv
from sarsa import *
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
    hyper_parameter_grid_search(env)
    # print(env.packages)
    # env.draw_map()
    env.close()
    return 0, []

    r, l, Q = sarsa(env, 25000)

    pi = np.zeros((3, env.height, env.width))
    for (x, y, c), actions in Q.items():
        pi[c, x, y] = np.argmax(actions)
    print(pi)

    cum_r, actions = evaluate_sarsa_policy(Q, env)
    return cum_r, actions


def random_agent(env):
    state = env.reset()
    done = False
    r_acc = 0
    actions = []
    k = 1
    while not done:
        action = env.action_space.sample()
        actions.append(action)
        new_state, reward, done, _ = env.step(action)
        r_acc += reward
        state = new_state
        k += 1
    return r_acc, actions


def main():
    args = argsparser()

    # use logging to get prints for debug/info/error mode
    log = logging.INFO
    if args.debug:
        log = logging.DEBUG
    logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=log)
    # check which method should be used
    mode = args.mode.lower()
    if mode not in ['normal', 'experimental', 'random', 'min_weight', 'max_weight', 'a2c', 'ppo1', 'dqn']:
        logging.error('The mode has to be normal or experimental.')
        return
    seeds = [2222]  # list of seeds for experiments
    for seed in seeds:
        logging.info(f'Start experiments with the seed {seed} and mode {mode}')
        set_seeds(seed)
        if mode == 'normal':
            env = CityEnv(init_random=not args.static, height=10, width=10, packages=[(7, 2), (5, 5), (0, 2), (3, 8)],  #, (4, 2), (0, 3)],
                          one_way=not args.bidirectional, construction_sites=not args.interconnected,
                          traffic_lights=not args.notrafficlights)
            # print(env.vertices_matrix[2, 2], env.vertices_matrix[2, 0])
            logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
            env.draw_map()
            logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=log)
            # env.reset()
            # print(env.step(1))
            r, l, Q = sarsa(env, 100000)
            cum_r, actions = evaluate_sarsa_policy(Q, env)
        elif mode == 'random':
            env = CityEnv(init_random=not args.static, height=3, width=3, packages=[(2, 2), (0, 2)])
            cum_r, actions = baselines.random_agent(env)
        elif mode == 'min_weight':
            env = CityEnv(init_random=not args.static, height=3, width=3, packages=[(2, 2), (0, 2)])
            env.draw_map()
            cum_r, actions = baselines.min_weight_agent(env)
        elif mode == 'max_weight':
            env = CityEnv(init_random=not args.static, height=3, width=3, packages=[(2, 2), (0, 2)])
            env.draw_map()
            cum_r, actions = baselines.max_weight_agent(env)
        elif mode == 'a2c':
            env = CityEnv(init_random=not args.static, height=5, width=5, packages=[(0, 2), (2, 2)])
            check_env(env, warn=True)
            model = a2c_agent(env, total_timesteps=100000, log_interval=1000, seed=seed)
            logging.info("training done")
            cum_r, actions = run_agent(env, model)
        elif mode == 'ppo1':
            env = CityEnv(init_random=not args.static, height=3, width=3, packages=[(2, 2)])
            check_env(env, warn=True)
            model = ppo_agent(env)
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
            hyper_parameter_grid_search(env)
            cum_r, actions = 0, []
        logging.info(f'The cummulative reward is {cum_r}')
        logging.info(f'The optimal action sequence is {actions}')


if __name__ == '__main__':
    main()
