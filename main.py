import numpy as np
import torch
import random
from env import CityEnv
from sarsa import *


def set_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def main():
    set_seeds(1234)
    env = CityEnv(init_random=True)
    # env.reset()
    # print(env.step(1))
    r, l, Q = sarsa(env, 100)
    cum_r, actions = evaluate_sarsa_policy(Q, env)
    print(actions)


if __name__ == '__main__':
    main()
