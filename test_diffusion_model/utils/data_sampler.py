import torch
import numpy as np
import random
import os
# from .generate_diffusion_dataset import SimpleLayerNet
# import torch.optim as optim
import torch.nn.functional as F


import torch.nn as nn
import torch.optim as optim
# class Data_Sampler(object):
#     def __init__(self, data, device, reward_tune="no"):
#
#         self.state = torch.from_numpy(data['upper_states']).float()
#         self.action = torch.from_numpy(data['upper_action']).float()
#         self.next_state = torch.from_numpy(data['upper_next_states']).float()
#         reward = torch.from_numpy(data['upper_rewards']).view(-1, 1).float()
#         self.not_done = 1. - torch.from_numpy(data['upper_dones']).view(-1, 1).float()
#
#         self.size = self.state.shape[0]
#         self.state_dim = self.state.shape[1]
#         self.action_dim = self.action.shape[1]
#
#         self.device = device
#
#         if reward_tune == 'normalize':
#             reward = (reward - reward.mean()) / reward.std()
#
#         self.reward = reward
#
#     def sample(self, batch_size):
#         idx = torch.randint(0, self.size, size=(batch_size, ))
#
#         return(
#             self.state[idx].to(self.device),
#             self.action[idx].to(self.device),
#             self.next_state[idx].to(self.device),
#             self.reward[idx].to(self.device),
#             self.not_done[idx].to(self.device)
#         )

log_dir = "./{}".format("diffusion")
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
# np.save(os.path.join(log_dir, "eval"), eval)

from collections import deque
import random

class ReplayMemory(object):

    def __init__(self, capacity, device):
        self.memory = []
        self.device = device
        self.position = 0
        self.capacity = capacity

    def add(self, state, action, next_state, reward, done):
        """Save a transition"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        not_done = 1. - torch.from_numpy(done).view(-1, 1).float()
        self.memory[self.position] = (state, action, next_state, reward, not_done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        # print("@@@@@@@@@@@@@@@@@@@@@@@@@", self.capacity)
        batch = random.sample(self.memory, batch_size)
        # print(batch)
        state, action, next_state, reward, not_done = map(np.stack, zip(*batch))
        # return(
        #     self.memory[idx][0].to(self.device),
        #     self.memory[idx][1].to(self.device),
        #     self.memory[idx][2].to(self.device),
        #     self.memory[idx][3].to(self.device),
        #     self.memory[idx][4].to(self.device)
        # )

        # convert numpy array to tensor
        state = torch.from_numpy(state)
        action = torch.from_numpy(action)
        next_state = torch.from_numpy(next_state)
        reward = torch.from_numpy(reward)
        not_done = torch.from_numpy(not_done)
        return state, action, next_state, reward, not_done

    def __len__(self):
        return len(self.memory)

    def gen_data(self):
        np.random.seed(1)
        # data = ReplayMemory(capacity=100, device=device)
        for i in range(1000):
            state = torch.randn(5).cpu().data.numpy().clip(0, 2)
            # print("initial state", state)
            for j in range(5):
                a_0 = random.uniform(0, 8)
                a_1 = random.uniform(0, 8)
                a_2 = random.uniform(0, 8)

                action = [a_0, a_1, a_2]

                next_state_0 = state_position0(a_0, state)
                next_state_1 = state_position1(a_0, a_1, state)
                next_state_2 = state_position2(a_2, state)
                next_state_3 = state_position3(a_0, a_2, state)
                next_state_4 = state_position(a_1, state)

                next_state = [next_state_0, next_state_1, next_state_2, next_state_3, next_state_4]

                reward = np.mean(next_state) - np.mean(state)
                if j == 4:
                    done = torch.ones(1).cpu().data.numpy()
                else:
                    done = torch.zeros(1).cpu().data.numpy()
                # done = torch.bernoulli(torch.empty(1).uniform_(0, 1)).cpu().data.numpy()
                self.add(state, action, next_state, reward, done)
                state = next_state

device = torch.device("cuda")
if 'cuda' in device.type:
    torch.backends.cudnn.benchmark = True
    print('Using CUDA\n')

# for i in range(10):
#     state = torch.randn(5).cpu().data.numpy()
#     action = torch.randn(3).cpu().data.numpy()
#     next_state = torch.randn(5).cpu().data.numpy()
#     reward = torch.randn(1).cpu().data.numpy()
#     done = torch.bernoulli(torch.empty(1).uniform_(0, 1)).cpu().data.numpy()
#     data.add(state, action, next_state, reward, done)
import random
# a  = random.uniform(2, 3)
# a = np.clip(a, 0, 1)
# print(a)


def state_position0(a_0, state):
    state_0 = state[0]
    if ((2 * a_0 - state_0)>0) and (2 * a_0 - state_0)<=4:
        next_state_0 = state_0 + (2 * a_0 - state_0) + 1 + random.uniform(-0.3, 0.3)
    else:
        next_state_0 = state_0 + random.uniform(-1, 1)/(15+state_0-a_0) + random.uniform(-0.3, 0.3)
    next_state_0 = np.clip(next_state_0, 0, 13)
    return next_state_0


def state_position1(a_0, a_1, state):
    state_1 = state[1]
    if ((2*a_0-state_1+2)>0) and ((2*a_0-state_1+2)<2) and ((state_1-2*a_1+1)>-1) and (state_1-2*a_1+1)<3:
        next_state_1 = state_1 +1 + (2*a_0-state_1+2)*(state_1-2*a_1+1) + random.uniform(-0.3, 0.3)
    else:
        next_state_1 = state_1 + random.uniform(-2, 2)*1/((2*a_0-state_1+2)*(state_1-2*a_1+1) + 30) + random.uniform(-0.3, 0.3)
    next_state_1 = np.clip(next_state_1, 1, 15)
    return next_state_1


def state_position2(a_2, state):
    state_2 = state[2]
    if ((state_2-a_2)>0) and (state_2-a_2)<=2:
        next_state_2 = a_2 + 1 + random.uniform(-0.3, 0.3)
    else:
        next_state_2 = state_2 + 1/(10*a_2) + random.uniform(-0.3, 0.3)
    next_state_2 = np.clip(next_state_2, 2, 10)
    return next_state_2


def state_position3(a_0, a_2, state):
    state_3 = state[3]
    if ((state_3 - a_0*a_2/6-a_2*a_2/8)> -1) and (state_3 - a_0*a_2/6-a_2*a_2/8)<=3:
        next_state_3 = a_0*a_2/6+a_2*a_2/8+1+random.uniform(-0.3, 0.3)
    else:
        next_state_3 = state_3 + random.uniform(-1, 1)/(a_0 * a_2 / 6 + 10) + random.uniform(-1, 1)/(state_3 -a_0 * a_2 / 6 -a_2*a_2/8 +20) + random.uniform(-0.3, 0.3)
    next_state_3 = np.clip(next_state_3, 0, 20)
    return next_state_3


def state_position(a_1, state):
    state_4 = state[4]
    if ((state_4 - 5 * np.log(a_1))>-2) and ((state_4 - 5 * np.log(a_1)) < 2):
        next_state_4 = 5 * np.log(a_1 + 1) + random.uniform(-0.3, 0.3)
    else:
        next_state_4 = state_4 + 0.2*random.uniform(-1, 1)/(state_4 - 5 * np.log(a_1)) + random.uniform(-0.3, 0.3)
    next_state_4 = np.clip(next_state_4, 0, 20)
    return next_state_4


import pickle
class SimpleLayerNet_new(nn.Module):
    def __init__(self):
        super(SimpleLayerNet_new, self).__init__()
        self.fc1 = nn.Linear(8, 20)   
        self.fc2 = nn.Linear(20, 5)   

    def forward(self, x):
        x = F.relu(self.fc1(x)) 
        noise = torch.randn_like(x) * 3.0
        x = self.fc2(x+noise)
        return x

new_net = SimpleLayerNet_new()
new_net.load_state_dict(torch.load("simple_layer.pth"))

def gen_dataset(size):
    np.random.seed(1)
    data = ReplayMemory(capacity=size, device=device)
    for i in range(size//100):
        # state = torch.randn(5).cpu().data.numpy().clip(0, 2)
        state = (2*torch.randn(5).cpu().data.numpy()+2).clip(0, 10)
        # print("initial state", state)
        for j in range(100):
            a_1 = random.uniform(0, 6)
            a_2 = random.uniform(0, 6)
            a_3 = random.uniform(2, 8)
            action = [a_1, a_2, a_3]

            # next_state_0 = state_position0(a_0, state)
            # next_state_1 = state_position1(a_0, a_1, state)
            # next_state_2 = state_position2(a_2, state)
            # next_state_3 = state_position3(a_0, a_2, state)
            # next_state_4 = state_position(a_1, state)

            # next_state = [next_state_0, next_state_1, next_state_2, next_state_3, next_state_4]
            s_a = torch.cat((torch.as_tensor(state), torch.as_tensor(action)), dim=0)
            next_state = new_net(s_a).detach().numpy()
            # state = torch.randn(5).cpu().data.numpy()
            # action = torch.randn(3).cpu().data.numpy()
            # next_state = torch.randn(5).cpu().data.numpy()
            # reward = torch.randn(1).cpu().data.numpy()
            # reward = np.mean(next_state.detach().numpy()) - np.mean(state)
            reward = 10
            if j == 99:
                done = torch.ones(1).cpu().data.numpy()
            else:
                done = torch.zeros(1).cpu().data.numpy()
            # done = torch.bernoulli(torch.empty(1).uniform_(0, 1)).cpu().data.numpy()
            data.add(state, action, next_state, reward, done)
            state = next_state
    fn = 'dataset.pkl'
    with open(fn, 'wb') as f:
        picklestring = pickle.dump(data, f)
    return data