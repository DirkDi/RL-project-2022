
def random_agent(env):
    state = env.reset()
    done = False
    r_acc = 0
    actions = []
    k = 1
    while not done:
        action = env.action_space.sample()
        actions.append(action)
        new_state, reward, done, _ = env.step(action)
        r_acc += reward
        state = new_state
        k += 1
    return r_acc, actions


def max_weight_agent(env):
    state = env.reset()
    done = False
    r_acc = 0
    actions = []
    k = 1
    while not done:
        action = env.get_max_emission_action()
        actions.append(action)
        new_state, reward, done, _ = env.step(action)
        r_acc += reward
        state = new_state
        k += 1
    return r_acc, actions


def min_weight_agent(env):
    state = env.reset()
    done = False
    r_acc = 0
    actions = []
    k = 1
    while not done:
        action = env.get_min_emission_action()
        actions.append(action)
        new_state, reward, done, _ = env.step(action)
        r_acc += reward
        state = new_state
        k += 1
    return r_acc, actions