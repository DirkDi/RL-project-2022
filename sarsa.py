import numpy as np
from tqdm import tqdm
import csv
import gym
import logging
from typing import DefaultDict, Callable, Tuple, List
from collections import defaultdict


def make_epsilon_greedy_policy(q: DefaultDict[Tuple, np.ndarray],
                               epsilon: float, n_actions: int) -> Callable[[np.ndarray], int]:
    """
    Creates an epsilon-greedy policy.

    :param q: Q table
    :param epsilon: exploration rate
    :param n_actions: size of action space

    :return: function that expects a state and returns an action based on the policy
    """
    def policy_fn(observation: np.ndarray) -> int:
        """
        Returns an action depending on an observation

        :param observation: observed state

        :return: an action
        """
        obs = tuple(observation.tolist())
        new_policy = np.ones((n_actions,)) * (epsilon / n_actions)
        policy = q[obs]
        indices = np.where(policy == policy.max())[0]
        new_policy[np.random.choice(indices)] += 1 - epsilon
        return np.random.choice(np.arange(0, n_actions), p=new_policy)

    return policy_fn


def td_update(q: DefaultDict[Tuple, np.ndarray], state: np.ndarray, action: int, reward: float,
              next_state: np.ndarray, next_action: int, gamma: float, alpha: float, done: bool):
    """
    Updates the Q table using the temporal difference formula

    :param q: Q table
    :param state: current state
    :param action: action for the current state
    :param reward: reward gained for the current state and action
    :param next_state:
    :param next_action:
    :param gamma: discount factor
    :param alpha: learning rate
    :param done: boolean flag
    """
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

    :param env: an environment
    :param num_episodes: amount of episodes the agent should learn
    :param q: Q table, if None a new one will be created
    :param gamma: discount factor
    :param alpha: learning rate
    :param epsilon: exploration rate

    :return rewards: list of rewards the agent gained per episode
    :return lens: list of lengths of the episodes the agent trained
    :return q: Q table
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

        pbar.set_postfix({"episodes total reward": cumulative_reward})
        pbar.update(1)
    logging.info(f'mean training steps for each run: {np.mean(train_steps_list)}')
    return rewards, lens, q


def evaluate_sarsa_policy(env: gym.Env, q: DefaultDict[Tuple, np.ndarray]) -> Tuple[float, List[int]]:
    """
    Evaluates the Q table on the environment.

    Note that the test loop will be stopped after 500 steps
    if it seems that the SARSA agent has not learned
    a useful policy (coded in the Q table.)

    :param q: Q table
    :param env: an environment

    :return cum_r: cumulative reward the SARSA agent gained
    :return actions: action sequence the SARSA agent performed
    """
    state = env.reset()
    done = False
    policy = make_epsilon_greedy_policy(q, 0, env.action_space.n)
    cum_r = 0
    actions = []
    k = 1
    while not done:
        action = policy(state)
        actions.append(action)
        new_state, reward, done, _ = env.step(action)
        cum_r += reward
        state = new_state
        k += 1
        if k >= 500:
            logging.info("No solution found after 500 steps.")
            break
    return cum_r, actions


def save_q(q: DefaultDict[Tuple, np.ndarray], file_name: str = "q_sarsa"):
    """
    Saves the Q table into a csv-file.

    :param q: Q table
    :param file_name: a string
    """
    logging.info("Saving Q table")
    # NOTE: On Windows you have to set the newline flag
    #       on the empty string to correctly write line by line.
    with open(file_name + ".csv", "w+", newline="") as fd:
        w = csv.writer(fd)
        pbar = tqdm(total=len(q))
        for key, value in q.items():
            w.writerow([key, value.tolist()])
            pbar.update(1)
    logging.info("Q table saved")


def load_q(q: DefaultDict[Tuple, np.ndarray], file_name: str = "q_sarsa") -> DefaultDict[Tuple, np.ndarray]:
    """
    Loads the Q table from the csv-file.

    :param q: Q table to write in
    :param file_name: a string

    :return q: the updated Q table
    """
    logging.info("Loading Q table")
    with open(file_name + ".csv") as fd:
        r = list(csv.reader(fd))
        pbar = tqdm(total=len(r))
        for key, value in r:
            q[eval(key)] = np.array(eval(value))
            pbar.update(1)
    logging.info("Q table loaded")
    return q


def hyper_parameter_grid_search(env: gym.Env):
    """
    Function to find the best Q table with hyper parameter search.
    Deprecated and not used.

    :param env: the environment to test for
    """
    best_r = float("-inf")
    best_q = None
    best_alpha = None
    best_gamma = None
    best_epsilon = None
    for alpha in np.round(np.arange(0.1, 1, 0.2), 1):
        for gamma in np.round(np.arange(0.1, 1, 0.2), 1):
            for epsilon in np.round(np.arange(0.1, 1, 0.2), 1):
                r, l, q = sarsa(env, num_episodes=2000, alpha=alpha, gamma=gamma, epsilon=epsilon)
                cum_r, actions = evaluate_sarsa_policy(q, env)
                if cum_r > best_r:
                    best_r = cum_r
                    best_q = q
                    best_alpha = alpha
                    best_gamma = gamma
                    best_epsilon = epsilon
                    logging.debug(alpha, gamma, epsilon)
                    logging.debug(cum_r, q)
                    if len(actions) <= 50:
                        logging.debug(actions)