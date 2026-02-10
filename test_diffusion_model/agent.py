import copy
import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F

from utils.utils import SinusoidalPosEmb
from diffusion import Diffusion


class MLP(nn.Module):
    def __init__(self,
                 state_dim,
                 action_dim,
                 device,
                 t_dim=16):

        super(MLP, self).__init__()
        self.device = device

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, t_dim),
        )

        action_input_dim = state_dim + action_dim + t_dim
        self.action_layer = nn.Sequential(nn.Linear(action_input_dim, 256),
                                          nn.Mish(),
                                          nn.Linear(256, 256),
                                          nn.Mish(),
                                          nn.Linear(256, 256),
                                          nn.Mish(),
                                          nn.Linear(256, action_dim))

        # self.final_layer = nn.Linear(256, action_dim)

        state_input_dim = state_dim + action_dim + state_dim + t_dim
        self.next_state_layer = nn.Sequential(nn.Linear(state_input_dim, 256),
                                              nn.Mish(),
                                              nn.Linear(256, 256),
                                              nn.Mish(),
                                              nn.Linear(256, 256),
                                              nn.Mish(),
                                              nn.Linear(256, state_dim))

    def forward(self, action_noise, time, state, action, s_noise):

        t = self.time_mlp(time)
        tmp = torch.cat([action_noise, t, state], dim=1).to(torch.float32)
        action_noise = self.action_layer(tmp)
        # a = self.final_layer(x)

        tmp = torch.cat([state, action, t, s_noise], dim=1).to(torch.float32)
        state_noise = self.next_state_layer(tmp)

        return action_noise, state_noise


# class MLP_state(nn.Module):
#     def __init__(self,
#                  state_dim,
#                  action_dim,
#                  device,
#                  t_dim=16):
#
#         super(MLP_state, self).__init__()
#         self.device = device
#
#         self.time_mlp = nn.Sequential(
#             SinusoidalPosEmb(t_dim),
#             nn.Linear(t_dim, t_dim * 2),
#             nn.Mish(),
#             nn.Linear(t_dim * 2, t_dim),
#         )
#
#         input_dim = state_dim + action_dim + t_dim
#         self.mid_layer = nn.Sequential(nn.Linear(input_dim, 256),
#                                        nn.Mish(),
#                                        nn.Linear(256, 256),
#                                        nn.Mish(),
#                                        nn.Linear(256, 256),
#                                        nn.Mish())
#
#         self.final_layer = nn.Linear(256, state_dim)
#
#     def forward(self, x, time, state):
#
#         t = self.time_mlp(time)
#         s = torch.cat([state, x, t], dim=1)
#         s = self.mid_layer(s)
#
#         return self.final_layer(s)


class Agent(object):
    def __init__(self, state_dim, action_dim, max_action, max_state, device, discount, tau, beta_schedule='linear', n_timesteps=100, lr=2e-4):
        self.model = MLP(state_dim=state_dim, action_dim=action_dim, device=device)

        self.actor = Diffusion(
            state_dim=state_dim, action_dim=action_dim, model=self.model, max_action=max_action,
            max_state=max_state, beta_schedule=beta_schedule, device=device, n_timesteps=n_timesteps
        ).to(device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.max_action = max_action
        self.max_state = max_state
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.device = device

    def sample_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        with torch.no_grad():
            action = self.actor.sample(state)
        return action.cpu().data.numpy().flatten()

    def sample_next_state(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        with torch.no_grad():
            action = self.actor.sample(state)
            next_state = self.actor.sample_state(state, action)
        return next_state.cpu().data.numpy().flatten()

    def sample_action_state(self, state, batch_size=1):
        state = torch.Tensor(state.reshape(batch_size, -1)).to(self.device).to(torch.float32)
        with torch.no_grad():
            action = self.actor.sample(state)
            next_state = self.actor.sample_state(state, action)
        return action.cpu().data.numpy().flatten(), next_state.cpu().data.numpy().flatten()

    def generate_next_state(self, state, action,batch_size=1):
        state = torch.Tensor(state.reshape(batch_size, -1)).to(self.device).to(torch.float32)
        action = torch.Tensor(action.reshape(batch_size, -1)).to(self.device).to(torch.float32)
        with torch.no_grad():
            # action = self.actor.sample(state)
            next_state = self.actor.sample_state(state, action)
        return next_state.cpu().data.numpy().flatten()

    def train(self, replay_buffer, iterations, batch_size=64, log_writer=None):
        metric = {'bc_loss': [], 'actor_loss': [], 'critic_loss': []}
        for i in range(iterations):
            state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

            # loss = self.actor.loss(action, state)
            loss = self.actor.loss(state, action, next_state).to(torch.float32)

            self.actor_optimizer.zero_grad()
            loss.backward()
            self.actor_optimizer.step()

            metric['bc_loss'].append(loss.item())
            metric['actor_loss'].append(loss.item())
            metric['critic_loss'].append(0.)

        return metric

    def save_model(self, dir, id=None):
        if id is not None:
            torch.save(self.actor.state_dict(), f'{dir}/actor_id.pth')
        else:
            torch.save(self.actor.state_dict(), f'{dir}/actor.pth')

    def load_model(self, dir, id=None):
        if id is not None:
            self.actor.load_state_dict(torch.load(f'{dir}/actor_{id}.pth'))
        else:
            self.actor.load_state_dict(torch.load(f'{dir}/actor.pth'))