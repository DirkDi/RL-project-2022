import logging
from stable_baselines3 import SAC, PPO, DQN, A2C


def sac_agent(env, total_timesteps=1000, log_interval=10, verbose=1, seed=1234):
    """
    Trains an agent with the SAC algorithm.

    :param: env: The training environment where the agent should learn.
    :param: total_timesteps: The total amount of timesteps for learning.
    :param: log_interval: The amount of logging the current learning state.
    :param: verbose: Integer to set the verbose mode for seeing specific information.
    :param: seed: Integer to realise reproducibility and avoid randomness.

    :return: the trained model/agent
    """
    model = SAC("MlpPolicy", env, verbose=verbose, seed=seed)
    model.learn(total_timesteps=total_timesteps, log_interval=log_interval)
    # model.save('sac_city')
    return model


def ppo_agent(env, total_timesteps=1000, log_interval=10, verbose=1, seed=1234):
    """
    Trains an agent with the PPO algorithm.

    :param: env: The training environment where the agent should learn.
    :param: total_timesteps: The total amount of timesteps for learning.
    :param: log_interval: The amount of logging the current learning state.
    :param: verbose: Integer to set the verbose mode for seeing specific information.
    :param: seed: Integer to realise reproducibility and avoid randomness.

    :return: the trained model/agent
    """
    model = PPO("MlpPolicy", env, verbose=verbose, n_steps=1000, seed=seed)
    model.learn(total_timesteps=total_timesteps, log_interval=log_interval)
    # model.save('ppo_city')
    return model


def dqn_agent(env, total_timesteps=1000, log_interval=10, verbose=1, seed=1234):
    """
    Trains an agent with the DQN algorithm.

    :param: env: The training environment where the agent should learn.
    :param: total_timesteps: The total amount of timesteps for learning.
    :param: log_interval: The amount of logging the current learning state.
    :param: verbose: Integer to set the verbose mode for seeing specific information.
    :param: seed: Integer to realise reproducibility and avoid randomness.

    :return: the trained model/agent
    """
    model = DQN("MlpPolicy", env, verbose=verbose, seed=seed)
    model.learn(total_timesteps=total_timesteps, log_interval=log_interval)
    # model.save('dqno_city')
    return model


def a2c_agent(env, total_timesteps=1000, log_interval=10, verbose=1, seed=1234):
    """
    Trains an agent with the A2C algorithm.

    :param: env: The training environment where the agent should learn.
    :param: total_timesteps: The total amount of timesteps for learning.
    :param: log_interval: The amount of logging the current learning state.
    :param: verbose: Integer to set the verbose mode for seeing specific information.
    :param: seed: Integer to realise reproducibility and avoid randomness.

    :return: the trained model/agent
    """
    model = A2C("MlpPolicy", env, verbose=verbose, n_steps=1000, seed=seed)
    model.learn(total_timesteps=total_timesteps, log_interval=log_interval)
    # model.save('a2c_city')
    return model


def run_agent(env, model):
    """
    Runs the neural network driven agent on the environment.

    Note that the test loop will be left after 500 steps
    if it seems that the NN-driven agent have not learned
    a useful policy (coded in the model.)

    :param: env: an environment
    :param: model: a neural network

    :return: cum_r: cumulative reward the agent gained
    :return: actions: action sequence the agent performed
    """
    obs = env.reset()
    actions = []
    cum_r = 0
    k = 0
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        actions.append(action)
        obs, reward, done, info = env.step(action)
        cum_r += reward
        logging.debug(reward)
        k += 1
        # terminate agent to avoid endless loop
        if k >= 500:
            logging.info("probably no solution found")
            break
    return cum_r, actions
