from env import CityEnv
from sarsa import *


def main():
    env = CityEnv()
    # env.reset()
    # print(env.step(1))
    r, l, Q = sarsa(env, 100)
    cum_r, actions = evaluate_sarsa_policy(Q, env)
    print(actions)


if __name__ == '__main__':
    main()
