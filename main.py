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
    env = Env(height=3, width=3, packages=[(0, 2), (2, 2)])
    env.reset()
    """
    print(env.position)
    print(env.packages)
    print(env.step(1))
    print(env.step(2))
    print(env.step(1))
    print(env.step(3))
    """
    r, l, Q = sarsa(env, 1000)
    policy = make_epsilon_greedy_policy(Q, 0, env.action_space.n)
    pi = np.zeros_like(env.vertices_matrix)
    # for state, actions in Q.items():
    #     print(state, np.argmax(actions))
    for l in range(2, 0, -1):
        for i in range(env.height):
            for j in range(env.width):
                pi[i, j] = choose_action(policy((i, j, l)))
        logging.debug(f'Current policy: {pi}')

    cum_r, actions = evaluate_sarsa_policy(Q, env)
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
            env = CityEnv(init_random=True)
            # env.reset()
            # print(env.step(1))
            r, l, Q = sarsa(env, 100)
            cum_r, actions = evaluate_sarsa_policy(Q, env)
        else:
            actions = test()
        logging.info(f'The optimal action sequence is {actions}')


if __name__ == '__main__':
    main()
