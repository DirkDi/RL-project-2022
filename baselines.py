import gym
from env import CityEnv


def random_agent(env: gym.Env):
    """
    Solves the environment by performing randomly sampled actions per step.
    Note that the environment has an own random generator which is not seeded.

        Parameters:
            env: an environment which meets the openAI gym specification

        Returns:
            r_acc: cumulative reward
            actions: action sequence
    """
    env.reset()
    done = False
    r_acc = 0
    actions = []
    k = 1
    while not done:
        action = env.action_space.sample()
        actions.append(action)
        _, reward, done, _ = env.step(action)
        r_acc += reward
        k += 1
    return r_acc, actions


def min_weight_agent(env: CityEnv):
    """
    Solves the environment by taking the action with the lowest edge weight per step.

        Parameters:
            env: an environment of class CityEnv

        Returns:
            r_acc: cumulative reward
            actions: action sequence
    """
    env.reset()
    done = False
    r_acc = 0
    actions = []
    k = 1
    while not done:
        action = env.get_min_emission_action()
        actions.append(action)
        _, reward, done, _ = env.step(action)
        r_acc += reward
        k += 1
    return r_acc, actions


def max_weight_agent(env: CityEnv):
    """
    Solves the environment by taking the action with the highest edge weight per step.

        Parameters:
            env: an environment of class CityEnv

        Returns:
            r_acc: cumulative reward
            actions: action sequence
    """
    env.reset()
    done = False
    r_acc = 0
    actions = []
    k = 1
    while not done:
        action = env.get_max_emission_action()
        actions.append(action)
        _, reward, done, _ = env.step(action)
        r_acc += reward
        k += 1
    return r_acc, actions
