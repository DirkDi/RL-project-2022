import numpy as np
from tqdm import tqdm
import csv
import gym
import logging
from typing import DefaultDict, Callable, Tuple, List
from collections import defaultdict


def make_epsilon_greedy_policy(q: DefaultDict[Tuple, np.ndarray], epsilon: float, n_actions: int) -> Callable[[np.ndarray], int]:
    def policy_fn(observation: np.ndarray) -> int:
        obs = tuple(observation.tolist())
        new_policy = np.ones((n_actions,)) * (epsilon / n_actions)
        policy = q[obs]
        indices = np.where(policy == policy.max())[0]
        new_policy[np.random.choice(indices)] += 1 - epsilon
        return np.random.choice(np.arange(0, n_actions), p=new_policy)

    return policy_fn


def td_update(q: DefaultDict[Tuple, np.ndarray], state: np.ndarray, action: int, reward: float,
              next_state: np.ndarray, next_action: int, gamma: float, alpha: float, done: bool):
    state = tuple(state.tolist())
    current_q_value = q[state][action]
    td_target = reward
    if not done:
        td_target += gamma * q[tuple(next_state.tolist())][next_action]
    q[state][action] = current_q_value + alpha * (td_target - current_q_value)


def sarsa(env: gym.Env, num_episodes: int, q: DefaultDict[Tuple, np.ndarray] = None,
          gamma: float = 1.0, alpha: float = 0.5, epsilon: float = 0.1
          ) -> Tuple[List[float], List[int], DefaultDict[Tuple, np.ndarray]]:
    """
    Performs a training with the SARSA algorithm on the environment
    """
    if q is None:
        q = defaultdict(lambda: np.zeros(env.action_space.n))
    rewards = []
    lens = []
    train_steps_list = []
    num_performed_steps = 0
    pbar = tqdm(total=num_episodes)
    for episode in range(1, num_episodes + 1):
        policy = make_epsilon_greedy_policy(q, epsilon, env.action_space.n)
        state = env.reset()
        done = False
        episode_length = 0
        cumulative_reward = 0
        while not done:
            num_performed_steps += 1
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            next_action = policy(next_state)
            td_update(q, state, action, reward, next_state, next_action, gamma, alpha, done)
            policy = make_epsilon_greedy_policy(q, epsilon, env.action_space.n)
            cumulative_reward += reward
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
    return rewards, lens, q


def evaluate_sarsa_policy(q, env):
    state = env.reset()
    done = False
    policy = make_epsilon_greedy_policy(q, 0, env.action_space.n)
    r_acc = 0
    actions = []
    k = 1
    while not done:
        action = policy(state)
        actions.append(action)
        new_state, reward, done, _ = env.step(action)
        r_acc += reward
        state = new_state
        k += 1
        if k >= 1000:
            break
    return r_acc, actions


def save_q(q: DefaultDict[Tuple, np.ndarray], file_name: str = "q_sarsa"):
    """
    Saves the Q table into a csv-file.
    """
    print("Saving Q table")
    with open(file_name + ".csv", "w+") as fd:
        w = csv.writer(fd)
        pbar = tqdm(total=len(q))
        for key, value in q.items():
            w.writerow([key, value.tolist()])
            pbar.update(1)
    print("Q table saved")


def load_q(q: DefaultDict[Tuple, np.ndarray], file_name: str = "q_sarsa") -> DefaultDict[Tuple, np.ndarray]:
    """
    Loads the Q table from the csv-file.
    The policy will be extracted from this Q table.
    """
    print("Loading Q table")
    with open(file_name + ".csv") as fd:
        r = list(csv.reader(fd))
        pbar = tqdm(total=len(r))
        for key, value in r:
            q[eval(key)] = np.array(eval(value))
            pbar.update(1)
    print("Q table loaded")
    return q


def hyper_parameter_grid_search(env: gym.Env):
    best_r = float("-inf")
    best_q = None
    for alpha in np.round(np.arange(0.1, 1, 0.2), 1):
        for gamma in np.round(np.arange(0.1, 1, 0.2), 1):
            for epsilon in np.round(np.arange(0.1, 1, 0.2), 1):
                r, l, q = sarsa(env, num_episodes=2000, alpha=alpha, gamma=gamma, epsilon=epsilon)
                cum_r, actions = evaluate_sarsa_policy(q, env)
                if cum_r > best_r:
                    best_r = cum_r
                    print(alpha, gamma, epsilon)
                    print(cum_r)
                    if len(actions) <= 50:
                        print(actions)
    return
