import torch
import random
import argparse
import logging
from env import CityEnv
from sarsa import *
from experimental.CityEnv import CityEnv as Env


def argsparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug", action="store_true", help="debug output")
    parser.add_argument('--mode', '-m', type=str, default='normal',
                        help='choose the mode normal or experimental (experimental env)')
    parser.add_argument("--static", "-s", action="store_true", help="disables random distance generation")
    args = parser.parse_args()
    return args


def set_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


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
        [0, 1, 0, 1, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 9000, 0, 0, 0],
        [1, 0, 0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 1, 0, 1, 0],
        [0, 0, 9000, 0, 1, 0, 0, 0, 9000],
        [0, 0, 0, 1, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 9000, 0, 1, 0]
    ])
    env = Env(height=3, width=3, packages=[(0, 2), (2, 2)], dist_matrix=dist_matrix)
    env.reset()
    """
    print(env.dist_matrix)
    cum_r = 0
    _, r, d, _ = env.step(3)
    cum_r += r
    print(cum_r, d)
    _, r, d, _ = env.step(3)
    cum_r += r
    print(cum_r, d)
    _, r, d, _ = env.step(1)
    cum_r += r
    print(cum_r, d)
    _, r, d, _ = env.step(1)
    cum_r += r
    print(cum_r, d)

    print(env.reset())
    cum_r = 0
    s, r, d, _ = env.step(3)
    cum_r += r
    print(s, cum_r, d)
    s, r, d, _ = env.step(3)
    cum_r += r
    print(s, cum_r, d)
    s, r, d, _ = env.step(2)
    cum_r += r
    print(s, cum_r, d)
    s, r, d, _ = env.step(1)
    cum_r += r
    print(s, cum_r, d)
    s, r, d, _ = env.step(1)
    cum_r += r
    print(s, cum_r, d)
    s, r, d, _ = env.step(3)
    cum_r += r
    print(s, cum_r, d)
    return
    """
    r, l, Q = sarsa(env, 100)

    pi = np.zeros((3, env.height, env.width))
    for ((x, y), c), actions in Q.items():
        pi[c, x, y] = np.argmax(actions)
    print(pi)

    cum_r, actions = evaluate_sarsa_policy(Q, env)
    print(cum_r)
    return actions


def main():
    args = argsparser()

    # use logging to get prints for debug/info/error mode
    log = logging.INFO
    if args.debug:
        log = logging.DEBUG
    logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=log)
    # check which method should be used
    mode = args.mode.lower()
    if mode not in ['normal', 'experimental']:
        logging.error('The mode has to be normal or experimental.')
        return
    seeds = [1234]  # list of seeds for experiments
    for seed in seeds:
        logging.info(f'Start experiments with the seed {seed} and mode {mode}')
        set_seeds(seed)
        if mode == 'normal':
            env = CityEnv(init_random=not args.static, height=3, width=3, packages=[(0, 2), (2, 2)])
            # env.reset()
            # print(env.step(1))
            r, l, Q = sarsa(env, 100)
            cum_r, actions = evaluate_sarsa_policy(Q, env)
        else:
            actions = test()
        logging.info(f'The optimal action sequence is {actions}')


if __name__ == '__main__':
    main()
