from __future__ import annotations

import os
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
# from A2C import *
# from train import *

import gymnasium as gym

#
# n_showcase_episodes = 10

# agent = A2C(obs_shape, action_shape, device, critic_lr, actor_lr)
# actor_weights_path = "weights/actor_weights.h5"
# critic_weights_path = "weights/critic_weights.h5"
#
# agent.actor.load_state_dict(torch.load(actor_weights_path))
# agent.critic.load_state_dict(torch.load(critic_weights_path))
# agent.actor.eval()
# agent.critic.eval()


# def gen_eval_envs_para(n_showcase_episodes=10, seed=92):
#     random.seed(seed)
#     np.random.seed(seed)
#     eval_envs_para = []
#     for episode in range(n_showcase_episodes):
#         gravity_para = np.clip(
#                 np.random.normal(loc=-1.2*episode, scale=2.0), a_min=-11.99, a_max=-0.01
#             )
#         enable_wind_para = np.random.choice([True, False])
#         wind_power_para = np.clip(
#                 np.random.normal(loc=2*episode, scale=2.0), a_min=0.01, a_max=19.99
#         )
#         turbulence_power_para = np.clip(
#                 np.random.normal(loc=0.2*episode, scale=1.0), a_min=0.01, a_max=1.99
#             )
#         max_episode_steps_para = 500
#         eval_envs_para.append((gravity_para, enable_wind_para, wind_power_para, turbulence_power_para, max_episode_steps_para))
#     return eval_envs_para


def gen_eval_envs_para(n_showcase_episodes=10, seed=92):
    random.seed(seed)
    np.random.seed(seed)
    eval_envs_para = []
    for episode in range(n_showcase_episodes):
        gravity_para = np.clip(
                np.random.normal(loc=-0.9*episode, scale=2.0), a_min=-9.99, a_max=-0.01
            )
        enable_wind_para = np.random.choice([True, False])
        wind_power_para = np.clip(
                np.random.normal(loc=0.7*episode, scale=2.0), a_min=0.01, a_max=10.00
        )
        turbulence_power_para = np.clip(
                np.random.normal(loc=0.1*episode, scale=1.0), a_min=0.01, a_max=1.5
            )
        max_episode_steps_para = 500
        eval_envs_para.append((gravity_para, enable_wind_para, wind_power_para, turbulence_power_para, max_episode_steps_para))
    return eval_envs_para

# def gen_test_envs_para(n_showcase_episodes=10, seed=92):
#     random.seed(seed)
#     np.random.seed(seed)
#     eval_envs_para = []
#     for episode in range(n_showcase_episodes):
#         gravity_para = np.clip(
#                 np.random.normal(loc=-0.8*episode, scale=2.0), a_min=-9.99, a_max=-0.01
#             )
#         enable_wind_para = np.random.choice([True, False])
#         wind_power_para = np.clip(
#                 np.random.normal(loc=0.65*episode, scale=2.0), a_min=0.01, a_max=10.00
#         )
#         turbulence_power_para = np.clip(
#                 np.random.normal(loc=0.09*episode, scale=1.0), a_min=0.01, a_max=1.5
#             )
#         max_episode_steps_para = 500
#         eval_envs_para.append((gravity_para, enable_wind_para, wind_power_para, turbulence_power_para, max_episode_steps_para))
#     return eval_envs_para

def gen_test_envs_para(n_showcase_episodes=10, seed=92):
    random.seed(seed)
    np.random.seed(seed)
    eval_envs_para = []
    for episode in range(n_showcase_episodes):
        # gravity_para = np.clip(
        #         np.random.normal(loc=-0.8*episode, scale=2.0), a_min=-9.99, a_max=-0.01
        #     )
        gravity_para = np.clip(
                -0.8*episode/n_showcase_episodes, a_min=-9.99, a_max=-0.01
            )
        enable_wind_para = np.random.choice([True, False])
        wind_power_para = np.clip(
                np.random.normal(loc=0.65*episode, scale=2.0), a_min=0.01, a_max=10.00
        )
        turbulence_power_para = np.clip(
                np.random.normal(loc=0.09*episode, scale=1.0), a_min=0.01, a_max=1.5
            )
        max_episode_steps_para = 500
        eval_envs_para.append((gravity_para, enable_wind_para, wind_power_para, turbulence_power_para, max_episode_steps_para))
    return eval_envs_para


def gen_test_envs_para_fixed(n_showcase_episodes=10, seed=92):
    test_envs_para = []
    for episode in range(n_showcase_episodes):
        # gravity_para = np.clip(
        #         np.random.normal(loc=-0.8*episode, scale=2.0), a_min=-9.99, a_max=-0.01
        #     )
        gravity_para = np.clip(
                -0.8*episode/n_showcase_episodes, a_min=-9.99, a_max=-0.01
            )
        enable_wind_para = np.random.choice([True, False])
        wind_power_para = np.clip(
                np.random.normal(loc=0.65*episode, scale=2.0), a_min=0.01, a_max=10.00
        )
        turbulence_power_para = np.clip(
                np.random.normal(loc=0.09*episode, scale=1.0), a_min=0.01, a_max=1.5
            )
        max_episode_steps_para = 500
        test_envs_para.append((gravity_para, enable_wind_para, wind_power_para, turbulence_power_para, max_episode_steps_para))
        test_envs_para = [
            (-2.0, True, 1.0, 0.2, 500),
            (-2.9, False, 0.6, 0.9, 500),
            (-1.1, True, 2.1, 0.3, 500),
            (-1.9, False, 3.1, 0.4, 500),
            (-4.9, True, 4.2, 0.5, 500),
            (-4.2, False, 1.9, 0.6, 500),
            (-7.9, True, 5.8, 0.7, 500),
            (-8.9, False, 6.1, 0.8, 500),
            (-6.0, True, 7.1, 0.99, 500),
            (-6.3, False, 7.3, 0.22, 500),
        ]
    return test_envs_para


def gen_evaluate_envs_with_para(eval_envs_para, n_showcase_episodes=10):
    eval_envs = []
    for episode in range(n_showcase_episodes):
        env = gym.make(
            "LunarLander-v2",
            # render_mode="human",
            gravity=np.clip(
                np.random.normal(loc=-1.2*episode, scale=2.0), a_min=-11.99, a_max=-0.01
            ),
            enable_wind=np.random.choice([True, False]),
            wind_power=np.clip(
                np.random.normal(loc=2*episode, scale=2.0), a_min=0.01, a_max=19.99
            ),
            turbulence_power=np.clip(
                np.random.normal(loc=0.2*episode, scale=1.0), a_min=0.01, a_max=1.99
            ),
            max_episode_steps=500,
        )
        eval_envs.append(env)
    return eval_envs


def gen_evaluate_envs(n_showcase_episodes=10, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    eval_envs = []
    for episode in range(n_showcase_episodes):
        env = gym.make(
            "LunarLander-v2",
            # render_mode="human",
            gravity=np.clip(
                np.random.normal(loc=-1.2*episode, scale=2.0), a_min=-11.99, a_max=-0.01
            ),
            enable_wind=np.random.choice([True, False]),
            wind_power=np.clip(
                np.random.normal(loc=2*episode, scale=2.0), a_min=0.01, a_max=19.99
            ),
            turbulence_power=np.clip(
                np.random.normal(loc=0.2*episode, scale=1.0), a_min=0.01, a_max=1.99
            ),
            max_episode_steps=500,
        )
        eval_envs.append(env)
    return eval_envs


def eval_agent_performance(agent, envs, gamma, envs_para=None):
    n_showcase_episodes = len(envs)
    eval_stats = []
    for episode in range(n_showcase_episodes):
        # print(f"starting episode {episode}...")
        if envs_para:
            env = gym.make(
                "LunarLander-v2",
                # render_mode="human",
                gravity=envs_para[episode][0],
                enable_wind=envs_para[episode][1],
                wind_power=envs_para[episode][2],
                turbulence_power=envs_para[episode][3],
                max_episode_steps=envs_para[episode][4],
            )
            # print("generating eval environment with parameters")
        else:
            env = envs[episode]

        rewards = 0
        # get an initial state
        state, info = env.reset()

        # play one episode
        done = False
        while not done:
            # select an action A_{t} using S_{t} as input for the agent
            with torch.no_grad():
                action, _, _, _ = agent.select_action(state[None, :])

            # perform the action A_{t} in the environment to get S_{t+1} and R_{t+1}
            state, reward, terminated, truncated, info = env.step(action.item())

            rewards = rewards + gamma * reward
            # update if the environment is done
            done = terminated or truncated
        eval_stats.append(rewards)
        # print("environment return during testing:", rewards)
    # print("### finish evaluation")
    # env.close()
    return eval_stats

