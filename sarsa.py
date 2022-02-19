import numpy as np
from tqdm import tqdm
import gym
import logging
from typing import DefaultDict, Callable, Tuple, List, Hashable
from _collections import defaultdict


def make_epsilon_greedy_policy(Q: DefaultDict[int, np.ndarray], epsilon: float, n_actions: int) -> Callable[[int], np.ndarray]:
    def policy_fn(observation: int) -> np.ndarray:
        new_policy = np.ones((n_actions,)) * (epsilon / n_actions)
        policy = Q[observation]
        indices = np.where(policy == policy.max())[0]
        new_policy[np.random.choice(indices)] += 1 - epsilon
        return new_policy

    return policy_fn


def choose_action(probability_distribution: np.ndarray) -> int:
    return np.random.choice(np.arange(0, len(probability_distribution)), p=probability_distribution)


def td_update(Q: DefaultDict[int, np.ndarray], state: int, action: int, reward: float, next_state: int,
              next_action: int, gamma: float, alpha: float, done: bool) -> float:
    current_q_value = Q[state][action]
    td_target = reward
    if not done:
        td_target += gamma * Q[next_state][next_action]
    return current_q_value + alpha * (td_target - current_q_value)


def sarsa(env: gym.Env, num_episodes: int, gamma: float = 1.0, alpha: float = 0.5, epsilon: float = 0.1) -> Tuple[List[float], List[int], DefaultDict[Hashable, np.ndarray]]:
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    rewards = []
    lens = []
    train_steps_list = []
    num_performed_steps = 0
    pbar = tqdm(total=num_episodes)
    for episode in range(1, num_episodes + 1):
        policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
        state = env.reset()
        done = False
        episode_length = 0
        cumulative_reward = 0
        while not done:
            num_performed_steps += 1
            action = choose_action(policy(state))
            next_state, reward, done, _ = env.step(action)
            next_action = choose_action(policy(next_state))
            Q[state][action] = td_update(Q, state, action, reward, next_state, next_action, gamma, alpha, done)
            policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
            cumulative_reward += reward
            #print(cumulative_reward)
            episode_length += 1
            state = next_state
        rewards.append(cumulative_reward)
        lens.append(episode_length)
        train_steps_list.append(num_performed_steps)
        num_performed_steps = 0
        # logging.info(f'{episode:4d}/{num_episodes:4d} episodes done, episodes total reward: {cumulative_reward}')
        pbar.set_postfix({"episodes total reward": cumulative_reward})
        pbar.update(1)
    logging.info(f'mean training steps for each run: {np.mean(train_steps_list)}')
    return rewards, lens, Q


def evaluate_sarsa_policy(Q, env):
    state = env.reset()
    done = False
    policy = make_epsilon_greedy_policy(Q, 0, env.action_space.n)
    """
    pi = np.zeros_like(env.vertices_matrix)
    for i in range(env.length):
        for j in range(env.width):
            pi[i, j] = choose_action(policy((i, j)))
    print(pi)
    """
    r_acc = 0
    actions = []
    k = 1
    while not done:
        action = choose_action(policy(state))
        actions.append(action)
        new_state, reward, done, _ = env.step(action)
        r_acc += reward
        state = new_state
        k += 1
    return r_acc, actions