from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

from A2C import A2C
from eval import eval_agent_performance, gen_evaluate_envs, gen_eval_envs_para, gen_test_envs_para, gen_test_envs_para_fixed
from arguments import parser
from teacher import *
from teacher_DDPG import DDPG
from buffer import ReplayBuffer
from util.utils import reward_func
from train_diffusion import train_diffusion_model

# Diffusion model
from agent import Agent_Diffusion


if __name__ == "__main__":

    args = parser.parse_args()
    randomize_domain = args.domain_randomization
    ACCEL = args.accel
    SEED = args.seed

    torch.manual_seed(args.diffusion_seed)
    np.random.seed(args.diffusion_seed)

    n_envs = args.num_env
    n_updates = args.num_udpates_per_env
    n_budget = args.num_budget
    n_steps_per_update = args.num_steps_per_update
    n_num_eval_envs = args.num_eval_envs

    # agent hyperparams
    gamma = args.gamma
    lam = args.lam  # hyperparameter for GAE
    ent_coef = args.ent_coef  # coefficient for the entropy bonus (to encourage exploration)
    actor_lr = args.actor_lr
    critic_lr = args.critic_lr
    log_dir = args.log_dir

    new_MDP = args.newMDP
    use_diffusion = args.use_diffusion

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir)

    eval_envs = gen_evaluate_envs(n_num_eval_envs)
    eval_envs_para = gen_eval_envs_para(n_num_eval_envs)

    test_envs = gen_evaluate_envs(n_num_eval_envs)
    test_envs_para = gen_test_envs_para_fixed(n_num_eval_envs)

    save_weights = args.save_weights
    load_weights = args.load_weights

    actor_weights_path = f"{args.log_dir}/{args.actor_weights_path}"

    critic_weights_path = f"{args.log_dir}/{args.critic_weights_path}"

    # create a fake environment to get the parameters of agent
    envs = gym.vector.AsyncVectorEnv(
        [
            lambda: gym.make(
                "LunarLander-v2",
                gravity=-10.0,
                enable_wind=True,
                wind_power=15.0,
                turbulence_power=1.5,
                max_episode_steps=600,
            ),
            lambda: gym.make(
                "LunarLander-v2",
                gravity=-9.8,
                enable_wind=True,
                wind_power=10.0,
                turbulence_power=1.3,
                max_episode_steps=600,
            ),
            lambda: gym.make(
                "LunarLander-v2", gravity=-7.0, enable_wind=False, max_episode_steps=600
            ),
        ]
    )
    obs_shape = envs.single_observation_space.shape[0]
    action_shape = envs.single_action_space.n
    # set the device
    use_cuda = args.cuda
    if use_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    # init the agent
    agent = A2C(obs_shape, action_shape, device, critic_lr, actor_lr, n_envs)
    if save_weights:
        # print("save model...")
        torch.save(agent.actor.state_dict(), actor_weights_path)
        torch.save(agent.critic.state_dict(), critic_weights_path)

    # Initialize the evaluate performance:
    eval_agent = A2C(obs_shape, action_shape, device, critic_lr, actor_lr, 1)
    eval_agent.actor.load_state_dict(torch.load(actor_weights_path))
    eval_agent.critic.load_state_dict(torch.load(critic_weights_path))
    # eval_agent.actor.eval()
    # eval_agent.critic.eval()
    stats = eval_agent_performance(eval_agent, eval_envs, gamma, eval_envs_para)
    test_stats = eval_agent_performance(eval_agent, test_envs, gamma, test_envs_para)


    if new_MDP:
        teacher_action_shape = 3 * n_envs
        teacher_max_action = np.tile(args.teacher_max_action, (n_envs,))
        teacher_agent = DDPG(n_num_eval_envs, teacher_action_shape, teacher_max_action, n_envs)
        # teacher_agent = TD3(n_num_eval_envs, teacher_action_shape, teacher_max_action)
        # teacher_agent = SAC(n_num_eval_envs, teacher_action_shape, device, teacher_max_action)
        teacher_buffer = ReplayBuffer(capacity=args.buffer_size)
        teacher_synth_buffer = ReplayBuffer(capacity=args.diffusion_synth_buffer_size)
        state = np.array(stats)

    if use_diffusion:
        agent_diffusion = Agent_Diffusion(state_dim=n_num_eval_envs,
                      action_dim=teacher_action_shape,
                      max_action=teacher_max_action,
                      max_state=args.diffusion_max_state,
                      device=device,
                      discount=args.diffusion_discount,
                      tau=args.diffusion_tau,
                      beta_schedule=args.diffusion_beta_schedule,
                      n_timesteps=args.diffusion_T,
                      lr=args.diffusion_lr)

    # ============================train the agent===================================
    num_update = 0
    if new_MDP:
        for ep in range(200):
            print("=======================New MDP=======================")
            agent = A2C(obs_shape, action_shape, device, critic_lr, actor_lr, n_envs)
            if save_weights:
                # print("save model...")
                torch.save(agent.actor.state_dict(), actor_weights_path)
                torch.save(agent.critic.state_dict(), critic_weights_path)
            eval_agent = A2C(obs_shape, action_shape, device, critic_lr, actor_lr, 1)
            eval_agent.actor.load_state_dict(torch.load(actor_weights_path))
            eval_agent.critic.load_state_dict(torch.load(critic_weights_path))
            stats = eval_agent_performance(eval_agent, eval_envs, gamma, eval_envs_para)
            test_stats = eval_agent_performance(eval_agent, test_envs, gamma, test_envs_para)

            state = np.array(stats)
            num_update = 0
            for budget in tqdm(range(n_budget)):
                action = teacher_agent.select_action(state)
                envs = gym.vector.AsyncVectorEnv(
                    [
                        lambda: gym.make(
                            "LunarLander-v2",
                            gravity=action[3*i],
                            enable_wind=True,
                            wind_power=action[3*i+1],
                            turbulence_power=action[3*i+2],
                            max_episode_steps=600,
                        )
                        for i in range(n_envs)
                    ]
                )

                # we don't have to reset the envs, they just continue playing
                # until the episode is over and then reset automatically

                # create a wrapper environment to save episode returns and episode lengths
                envs_wrapper = gym.wrappers.RecordEpisodeStatistics(envs, deque_size=n_envs * n_updates)

                critic_losses = []
                actor_losses = []
                entropies = []

                for sample_phase in tqdm(range(n_updates)):
                    # we don't have to reset the envs, they just continue playing
                    # until the episode is over and then reset automatically

                    # reset lists that collect experiences of an episode (sample phase)
                    ep_value_preds = torch.zeros(n_steps_per_update, n_envs, device=device)
                    ep_rewards = torch.zeros(n_steps_per_update, n_envs, device=device)
                    ep_action_log_probs = torch.zeros(n_steps_per_update, n_envs, device=device)
                    masks = torch.zeros(n_steps_per_update, n_envs, device=device)

                    # at the start of training reset all envs to get an initial state
                    if sample_phase == 0:
                        states, info = envs_wrapper.reset()
                    for step in range(n_steps_per_update):
                        # select an action A_{t} using S_{t} as input for the agent
                        actions, action_log_probs, state_value_preds, entropy = agent.select_action(
                            states
                        )

                        # perform the action A_{t} in the environment to get S_{t+1} and R_{t+1}
                        states, rewards, terminated, truncated, infos = envs_wrapper.step(
                            actions.cpu().numpy()
                        )
                        ep_value_preds[step] = torch.squeeze(state_value_preds)
                        ep_rewards[step] = torch.tensor(rewards, device=device)
                        ep_action_log_probs[step] = action_log_probs

                        # add a mask (for the return calculation later);
                        # for each env the mask is 1 if the episode is ongoing and 0 if it is terminated (not by truncation!)
                        masks[step] = torch.tensor([not term for term in terminated])

                    # calculate the losses for actor and critic
                    critic_loss, actor_loss = agent.get_losses(
                        ep_rewards,
                        ep_action_log_probs,
                        ep_value_preds,
                        entropy,
                        masks,
                        gamma,
                        lam,
                        ent_coef,
                        device,
                    )

                    # update the actor and critic networks
                    agent.update_parameters(critic_loss, actor_loss)

                    # log the losses and entropy
                    critic_losses.append(critic_loss.detach().cpu().numpy())
                    actor_losses.append(actor_loss.detach().cpu().numpy())
                    entropies.append(entropy.detach().mean().cpu().numpy())

                    num_update += 1
                    writer.add_scalar('training_critic_loss', critic_loss.detach().cpu().numpy(), num_update)
                    writer.add_scalar('training_actor_loss', actor_loss.detach().cpu().numpy(), num_update)
                    writer.add_scalar('training_entropy', entropy.detach().mean().cpu().numpy(), num_update)
                    if save_weights:
                        # print("save model after training...")
                        torch.save(agent.actor.state_dict(), actor_weights_path)
                        torch.save(agent.critic.state_dict(), critic_weights_path)

                    # =================================== Start Evaluation ===============================
                    # eval_agent = A2C(obs_shape, action_shape, device, critic_lr, actor_lr, 1)
                    eval_agent.actor.load_state_dict(torch.load(actor_weights_path))
                    eval_agent.critic.load_state_dict(torch.load(critic_weights_path))
                    # eval_agent.actor.eval()
                    # eval_agent.critic.eval()
                    stats = eval_agent_performance(eval_agent, eval_envs, gamma, eval_envs_para)
                    test_stats = eval_agent_performance(eval_agent, test_envs, gamma, test_envs_para)

                    writer.add_scalar('Mean_agent_Loss/Eval_overall', np.mean(stats), num_update)
                    writer.add_scalar('Mean_agent_Loss/Test_overall', np.mean(test_stats), num_update)
                    for idx in range(n_num_eval_envs):
                        writer.add_scalar('agent_value_loss/Eval_{}'.format(idx), stats[idx], num_update)
                        writer.add_scalar('agent_value_loss/Eval_{}_{}'.format(ep, idx), stats[idx], num_update)

                        writer.add_scalar('agent_value_loss/Test_{}'.format(idx), test_stats[idx], num_update)
                        writer.add_scalar('agent_value_loss/Test_{}_{}'.format(ep, idx), test_stats[idx], num_update)
                    # writer.add_scalar('agent_value_loss/Eval_0', stats[0], num_update)
                    # writer.add_scalar('agent_value_loss/Eval_1', stats[1], num_update)
                    # writer.add_scalar('agent_value_loss/Eval_2', stats[2], num_update)
                    # writer.add_scalar('agent_value_loss/Eval_3', stats[3], num_update)
                    # writer.add_scalar('agent_value_loss/Eval_4', stats[4], num_update)
                    # writer.add_scalar('agent_value_loss/Eval_5', stats[5], num_update)
                    # writer.add_scalar('agent_value_loss/Eval_6', stats[6], num_update)
                    # writer.add_scalar('agent_value_loss/Eval_7', stats[7], num_update)
                    # writer.add_scalar('agent_value_loss/Eval_8', stats[8], num_update)
                    # writer.add_scalar('agent_value_loss/Eval_9', stats[9], num_update)

                next_state = np.array(stats)
                reward = reward_func(state, next_state, args.coef_cv, args.cv)
                done = False
                if budget == n_budget - 1:
                    done = True
                teacher_buffer.add(state, action, next_state, reward, float(done))
                state = next_state
                # print("current teacher buffer size: ", len(teacher_buffer))
                if use_diffusion and len(teacher_buffer) > args.diffusion_batch_size:
                    print("========================== training diffusion model ==========================")
                    train_diffusion_model(agent_diffusion, teacher_buffer, teacher_synth_buffer, n_num_eval_envs, teacher_action_shape, teacher_max_action, args.diffusion_max_state,
                                          device, args, writer)

                if use_diffusion:
                    if len(teacher_synth_buffer) > args.diffusion_size_use_syn_buffer and len(teacher_buffer) > args.batch_size and len(teacher_synth_buffer) > args.batch_size:
                        for i in range(2):
                            teacher_agent.train_with_diffusion(teacher_buffer, teacher_synth_buffer, batch_size=args.batch_size, ratio=args.diffusion_syn_ratio)
                    elif len(teacher_buffer) > args.batch_size:
                        for i in range(3):
                            teacher_agent.train(teacher_buffer, batch_size=args.batch_size)
                elif len(teacher_buffer) > args.batch_size:
                    # print("==============training teacher================")
                    # teacher_agent.train(teacher_buffer, iterations=args.iterations, batch_size=args.batch_size)
                    for i in range(2):
                        teacher_agent.train(teacher_buffer, batch_size=args.batch_size)

            writer.close()

    else:
        for budget in tqdm(range(n_budget)):
            if randomize_domain:
                envs = gym.vector.AsyncVectorEnv(
                    [
                        lambda: gym.make(
                            "LunarLander-v2",
                            gravity=np.clip(
                                np.random.normal(loc=-10.0, scale=1.0), a_min=-11.99, a_max=-0.01
                            ),
                            enable_wind=np.random.choice([True, False]),
                            wind_power=np.clip(
                                np.random.normal(loc=15.0, scale=1.0), a_min=0.01, a_max=19.99
                            ),
                            turbulence_power=np.clip(
                                np.random.normal(loc=1.5, scale=0.5), a_min=0.01, a_max=1.99
                            ),
                            max_episode_steps=600,
                        )
                        for i in range(n_envs)
                    ]
                )
            elif ACCEL:
                interval_1 = (12-2)/n_budget
                interval_2 = (20-4)/n_budget
                interval_3 = (2-0.4)/n_budget
                print("=======================ACCEL=======================")
                envs = gym.vector.AsyncVectorEnv(
                    [
                        lambda: gym.make(
                            "LunarLander-v2",
                            gravity=np.clip(
                                np.random.normal(loc=-interval_1*budget, scale=2.0), a_min=-1.99 - interval_1*budget, a_max=-0.01 - interval_1*budget
                            ),
                            enable_wind=np.random.choice([True, False]),
                            wind_power=np.clip(
                                np.random.normal(loc=interval_2*budget, scale=2.0), a_min=0.01+interval_2*budget, a_max=3.99 + interval_2 * budget
                            ),
                            turbulence_power=np.clip(
                                np.random.normal(loc=interval_3*budget, scale=0.5), a_min=0.01+interval_3*budget, a_max=0.399 + interval_3 * budget
                            ),
                            max_episode_steps=600,
                        )
                        for i in range(n_envs)
                    ]
                )
            else:
                envs = gym.vector.make("LunarLander-v2", num_envs=n_envs, max_episode_steps=600)

            # we don't have to reset the envs, they just continue playing
            # until the episode is over and then reset automatically

            # create a wrapper environment to save episode returns and episode lengths
            envs_wrapper = gym.wrappers.RecordEpisodeStatistics(envs, deque_size=n_envs * n_updates)

            critic_losses = []
            actor_losses = []
            entropies = []

            for sample_phase in tqdm(range(n_updates)):
                # we don't have to reset the envs, they just continue playing
                # until the episode is over and then reset automatically

                # reset lists that collect experiences of an episode (sample phase)
                ep_value_preds = torch.zeros(n_steps_per_update, n_envs, device=device)
                ep_rewards = torch.zeros(n_steps_per_update, n_envs, device=device)
                ep_action_log_probs = torch.zeros(n_steps_per_update, n_envs, device=device)
                masks = torch.zeros(n_steps_per_update, n_envs, device=device)

                # at the start of training reset all envs to get an initial state
                if sample_phase == 0:
                    states, info = envs_wrapper.reset()
                for step in range(n_steps_per_update):


                    # select an action A_{t} using S_{t} as input for the agent
                    actions, action_log_probs, state_value_preds, entropy = agent.select_action(
                        states
                    )

                    # perform the action A_{t} in the environment to get S_{t+1} and R_{t+1}
                    states, rewards, terminated, truncated, infos = envs_wrapper.step(
                        actions.cpu().numpy()
                    )
                    ep_value_preds[step] = torch.squeeze(state_value_preds)
                    ep_rewards[step] = torch.tensor(rewards, device=device)
                    ep_action_log_probs[step] = action_log_probs

                    # add a mask (for the return calculation later);
                    # for each env the mask is 1 if the episode is ongoing and 0 if it is terminated (not by truncation!)
                    masks[step] = torch.tensor([not term for term in terminated])

                # calculate the losses for actor and critic
                critic_loss, actor_loss = agent.get_losses(
                    ep_rewards,
                    ep_action_log_probs,
                    ep_value_preds,
                    entropy,
                    masks,
                    gamma,
                    lam,
                    ent_coef,
                    device,
                )

                # update the actor and critic networks
                agent.update_parameters(critic_loss, actor_loss)
                num_update += 1

                # log the losses and entropy
                critic_losses.append(critic_loss.detach().cpu().numpy())
                actor_losses.append(actor_loss.detach().cpu().numpy())
                entropies.append(entropy.detach().mean().cpu().numpy())

                writer.add_scalar('training_critic_loss', critic_loss.detach().cpu().numpy(), num_update)
                writer.add_scalar('training_actor_loss', actor_loss.detach().cpu().numpy(), num_update)
                writer.add_scalar('training_entropy', entropy.detach().mean().cpu().numpy(), num_update)
                if save_weights:
                    # print("save model...")
                    torch.save(agent.actor.state_dict(), actor_weights_path)
                    torch.save(agent.critic.state_dict(), critic_weights_path)

                # =================================== Start Evaluation ===============================
                eval_agent = A2C(obs_shape, action_shape, device, critic_lr, actor_lr, 1)
                eval_agent.actor.load_state_dict(torch.load(actor_weights_path))
                eval_agent.critic.load_state_dict(torch.load(critic_weights_path))
                # eval_agent.actor.eval()
                # eval_agent.critic.eval()
                stats = eval_agent_performance(eval_agent, eval_envs, gamma, eval_envs_para)
                test_stats = eval_agent_performance(eval_agent, test_envs, gamma, test_envs_para)


                writer.add_scalar('Mean_agent_Loss/Eval_overall', np.mean(stats), num_update)
                writer.add_scalar('Mean_agent_Loss/Test_overall', np.mean(test_stats), num_update)
                for idx in range(n_num_eval_envs):
                    writer.add_scalar('agent_value_loss/Eval_{}'.format(idx), stats[idx], num_update)
                    writer.add_scalar('agent_value_loss/Test_{}'.format(idx), test_stats[idx], num_update)
                # writer.add_scalar('agent_value_loss/Eval_0', stats[0], num_update)
                # writer.add_scalar('agent_value_loss/Eval_1', stats[1], num_update)
                # writer.add_scalar('agent_value_loss/Eval_2', stats[2], num_update)
                # writer.add_scalar('agent_value_loss/Eval_3', stats[3], num_update)
                # writer.add_scalar('agent_value_loss/Eval_4', stats[4], num_update)
                # writer.add_scalar('agent_value_loss/Eval_5', stats[5], num_update)
                # writer.add_scalar('agent_value_loss/Eval_6', stats[6], num_update)
                # writer.add_scalar('agent_value_loss/Eval_7', stats[7], num_update)
                # writer.add_scalar('agent_value_loss/Eval_8', stats[8], num_update)
                # writer.add_scalar('agent_value_loss/Eval_9', stats[9], num_update)
        writer.close()



