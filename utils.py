import numpy as np


def cv_reward_func(state, next_state):
    mean = np.mean(state)
    std = np.std(state)
    before_cv = (std/mean) * 100

    mean = np.mean(next_state)
    std = np.std(next_state)
    after_cv = (std/mean) * 100

    cv_reward = before_cv - after_cv
    return cv_reward


def reward_func(state, next_state, coef_cv, cv=False):
    reward = np.mean(next_state) - np.mean(state)
    if cv:
        cv_reward = cv_reward_func(state, next_state)
        reward = reward + coef_cv * cv_reward

    return reward