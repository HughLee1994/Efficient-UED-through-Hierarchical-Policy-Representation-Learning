import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# Define the actor network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_dim)
        self.max_action = max_action
        self.action_dim = action_dim
        self.max_action_tensor = torch.from_numpy(self.max_action).to(torch.float32)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = self.max_action_tensor * torch.tanh(self.fc3(x))
        # x = torch.from_numpy(self.max_action).to(torch.float32) * (torch.tanh(self.fc3(x))+1)/2
        # x = self.max_action * (torch.sigmoid(self.fc3(x)) + 0.0001*torch.ones(self.action_dim))
        # next_action = torch.clamp(x, self.min_values, self.max_values)
        return x


# Define the critic network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DDPG:
    def __init__(self, state_dim, action_dim, max_action, n_envs):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.001)
        self.max_action = max_action
        self.max_action_tensor = torch.from_numpy(self.max_action).to(torch.float32)
        self.action_dim = action_dim
        self.n_envs = n_envs

    # def select_action(self, state):
    #     state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    #     with torch.no_grad():
    #         action = self.actor(state)
    #
    #         min_action = torch.tensor([-0.01, 0.01, 0.01] * self.n_envs)
    #         max_values = torch.max(min_action, self.max_action_tensor)
    #         min_values = torch.min(min_action, self.max_action_tensor)
    #         next_action = torch.clamp(action[0], min_values, max_values).numpy()
    #     return next_action

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        # print("state shape", state.shape)
        with torch.no_grad():
            action = self.actor(state)
            next_action = action.numpy()[0]

            min_action = np.tile(np.array([-0.01, 0.01, 0.01]), self.n_envs)
            max_values = np.maximum(min_action, self.max_action)
            min_values = np.minimum(min_action, self.max_action)
            next_action = next_action.clip(min_values, max_values)
        return next_action

    def train(self, replay_buffer, batch_size=64, gamma=0.99, tau=0.001):
        state, action, next_state, _reward, done = replay_buffer.sample(batch_size)

        reward_mean = _reward.mean()
        reward_std = _reward.std()
        reward = (_reward - reward_mean) / (reward_std + 1e-6)
        # state = torch.tensor(state, dtype=torch.float32)
        # action = torch.tensor(action, dtype=torch.float32)
        # next_state = torch.tensor(next_state, dtype=torch.float32)
        # reward = torch.tensor(reward, dtype=torch.float32).unsqueeze(1)
        # done = torch.tensor(done, dtype=torch.float32).unsqueeze(1)

        target_action = self.actor_target(next_state)
        target_value = self.critic_target(next_state, target_action)
        target = reward + gamma * (1 - done) * target_value

        # Update critic
        current_value = self.critic(state, action)
        critic_loss = nn.MSELoss()(current_value, target.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        predicted_action = self.actor(state)
        actor_loss = -self.critic(state, predicted_action).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def train_with_diffusion(self, replay_buffer, synth_buffer, batch_size=64, ratio=0.3, gamma=0.99, tau=0.001):
        batch_size_real = int(batch_size * (1 - ratio))
        batch_size_syn = batch_size - batch_size_real

        state_real, action_real, next_state_real, _reward_real, done_real = replay_buffer.sample(batch_size_real)
        # state_syn, action_syn, next_state_syn, reward_syn, done_syn = synth_buffer.sample(1)
        reward_mean = _reward_real.mean()
        reward_std = _reward_real.std()
        reward_real = (_reward_real - reward_mean) / (reward_std + 1e-6)
        
        state_syn, action_syn, next_state_syn, _reward_syn, done_syn = synth_buffer.sample(batch_size_syn)
        # state = torch.tensor(list(state_real) + list(state_syn), dtype=torch.float32)
        reward_mean = _reward_syn.mean()
        reward_std = _reward_syn.std()
        reward_syn = (_reward_syn - reward_mean) / (reward_std + 1e-6)
        
        state = torch.cat((state_real, state_syn), dim=0)
        action = torch.cat((action_real, action_syn), dim=0)
        next_state = torch.cat((next_state_real, next_state_syn), dim=0)
        reward = torch.cat((reward_real, reward_syn), dim=0)
        done = torch.cat((done_real, done_syn), dim=0)
        # action = torch.tensor(list(action_real) + list(action_syn), dtype=torch.float32)
        # next_state = torch.tensor(list(next_state_real) + list(next_state_syn), dtype=torch.float32)
        # reward = torch.tensor(list(reward_real) + list(reward_syn), dtype=torch.float32).unsqueeze(1)
        # done = torch.tensor(list(done_real) + list(done_syn), dtype=torch.float32).unsqueeze(1)

        target_action = self.actor_target(next_state)
        target_value = self.critic_target(next_state, target_action)
        target = reward + gamma * (1 - done) * target_value

        # Update critic
        current_value = self.critic(state, action)
        critic_loss = nn.MSELoss()(current_value, target.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        predicted_action = self.actor(state)
        actor_loss = -self.critic(state, predicted_action).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.critic.state_dict(), filename + "_critic")

    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.critic.load_state_dict(torch.load(filename + "_critic"))


