from stable_baselines3 import SAC, PPO, DQN, A2C
from stable_baselines3.common.callbacks import BaseCallback


def sac_agent(env, total_timesteps=1000, log_interval=10, verbose=1, seed=1234):
    model = SAC("MlpPolicy", env, verbose=verbose, seed=seed)
    model.learn(total_timesteps=total_timesteps, log_interval=log_interval)
    # model.save('sac_city')
    return model


def ppo_agent(env, total_timesteps=1000, log_interval=10, verbose=1, seed=1234):
    model = PPO("MlpPolicy", env, verbose=verbose, seed=seed)
    model.learn(total_timesteps=total_timesteps, log_interval=log_interval)
    # model.save('ppo_city')
    return model


def dqn_agent(env, total_timesteps=1000, log_interval=10, verbose=1, seed=1234):
    model = DQN("MlpPolicy", env, verbose=verbose, seed=seed)
    model.learn(total_timesteps=total_timesteps, log_interval=log_interval)
    # model.save('dqno_city')
    return model


def a2c_agent(env, total_timesteps=1000, log_interval=10, verbose=1, seed=1234):
    model = A2C("MlpPolicy", env, verbose=verbose, n_steps=10000, seed=seed)
    model.learn(total_timesteps=total_timesteps, log_interval=log_interval)
    # model.save('a2c_city')
    return model


def run_agent(env, model):
    obs = env.reset()
    actions = []
    cum_r = 0
    while True:  # TODO check if loop is correct for all agents
        action, _states = model.predict(obs, deterministic=True)
        actions.append(action)
        obs, reward, done, info = env.step(action)
        cum_r += reward
        # print(reward)
        if done:
            return cum_r, actions
