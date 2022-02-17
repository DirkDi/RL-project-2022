import torch
import random
from env import CityEnv
from sarsa import *
from experimental.CityEnv import CityEnv as Env


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
    r, l, Q = sarsa(env, 200)
    policy = make_epsilon_greedy_policy(Q, 0, env.action_space.n)
    pi = np.zeros_like(env.vertices_matrix)
    # for state, actions in Q.items():
    #     print(state, np.argmax(actions))
    for l in range(2, 0, -1):
        for i in range(env.height):
            for j in range(env.width):
                pi[i, j] = choose_action(policy((i, j, l)))
        print(pi)

    cum_r, actions = evaluate_sarsa_policy(Q, env)
    print(actions)


def main():
    test()
    return
    set_seeds(1234)
    env = CityEnv(init_random=True)
    # env.reset()
    # print(env.step(1))
    r, l, Q = sarsa(env, 100)
    cum_r, actions = evaluate_sarsa_policy(Q, env)
    print(actions)


if __name__ == '__main__':
    main()
