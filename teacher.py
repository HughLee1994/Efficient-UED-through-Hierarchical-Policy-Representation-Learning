# TD3

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


# Define the Actor network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # x = torch.from_numpy(self.max_action).to(torch.float32) * (torch.tanh(self.fc3(x))+1)/2
        x = torch.from_numpy(self.max_action) * (torch.sigmoid(self.fc3(x)) + 0.0001)
        return x


# Define the Critic network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1).to(self.fc1.weight.dtype)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Define the TD3 agent
class TD3:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-3)

        self.critic_1 = Critic(state_dim, action_dim)
        self.critic_1_target = Critic(state_dim, action_dim)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=1e-3)

        self.critic_2 = Critic(state_dim, action_dim)
        self.critic_2_target = Critic(state_dim, action_dim)
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=1e-3)

        self.max_action = max_action
        self.max_action_tensor = torch.tensor(self.max_action)

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1))
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy().flatten()
        return action

    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.1, noise_clip=0.1, policy_freq=2):
        for it in range(iterations):
            batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = replay_buffer.sample(batch_size)
            state = torch.FloatTensor(batch_states)
            next_state = torch.FloatTensor(batch_next_states)
            action = torch.FloatTensor(batch_actions)
            reward = torch.FloatTensor(batch_rewards)
            done = torch.FloatTensor(batch_dones)

            next_action = self.actor_target(next_state)

            # Add noise to the next action for exploration
            noise = torch.FloatTensor(batch_actions).data.normal_(0, policy_noise)
            noise = noise.clamp(-noise_clip, noise_clip)
            # next_action = (next_action + noise).clamp(-self.max_action, self.max_action)
            min_action = torch.tensor([0.01] * len(self.max_action))
            max_values = torch.max(min_action, self.max_action_tensor)
            min_values = torch.min(min_action, self.max_action_tensor)
            next_action = torch.clamp((next_action + noise), min_values, max_values)

            # Compute the target Q value
            target_Q1 = self.critic_1_target(next_state, next_action)
            target_Q2 = self.critic_2_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + ((1 - done) * discount * target_Q).detach()

            # Compute the current Q estimates
            current_Q1 = self.critic_1(state, action)
            current_Q2 = self.critic_2(state, action)

            # Compute the critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            torch.autograd.set_detect_anomaly(True)
            # Optimize the critics
            self.critic_1_optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            self.critic_1_optimizer.step()

            self.critic_2_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_2_optimizer.step()

            # Delayed policy updates
            if it % policy_freq == 0:
                actor_loss = -self.critic_1(state, self.actor(state)).mean()

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Update the target networks
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.critic_1.state_dict(), filename + "_critic1")
        torch.save(self.critic_2.state_dict(), filename + "_critic2")

    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.critic_1.load_state_dict(torch.load(filename + "_critic1"))
        self.critic_2.load_state_dict(torch.load(filename + "_critic2"))


# Define the Q-network
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Define the policy network
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, device, hidden_dim=256):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        self.max_action = max_action
        self.max_action_tensor = torch.FloatTensor(max_action)
        self.device = device

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        return mean, log_std

    def sample_action(self, state):
        mean, log_std = self.forward(state)
        std = torch.exp(log_std).clamp(min=1e-6)
        normal = torch.distributions.Normal(mean, std)
        action = normal.rsample()

        min_action = torch.tensor([0.01] * len(self.max_action_tensor))
        max_values = torch.max(min_action, self.max_action_tensor)
        min_values = torch.min(min_action, self.max_action_tensor)
        next_action = torch.clamp(action, min_values, max_values).to(self.device)

        # return action.clamp(-1, 1)
        return next_action


class SAC:
    def __init__(self, state_dim, action_dim, device, max_action, hidden_dim=256, discount=0.99, tau=0.005, alpha=0.2):
        self.device = device
        self.q_net1 = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.q_net2 = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_q_net1 = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_q_net2 = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.policy_net = PolicyNetwork(state_dim, action_dim, max_action, device, hidden_dim).to(self.device)
        self.target_entropy = -action_dim
        self.log_alpha = torch.tensor(np.log(alpha)).to(self.device)
        self.alpha = self.log_alpha.exp()
        self.discount = discount
        self.tau = tau

        self.q_optimizer1 = optim.Adam(self.q_net1.parameters(), lr=3e-4)
        self.q_optimizer2 = optim.Adam(self.q_net2.parameters(), lr=3e-4)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=3e-4)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)

    def train(self, replay_buffer, batch_size):
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        with torch.no_grad():
            next_actions, next_log_probs = self.policy_net.sample_action(next_states)
            next_q1 = self.target_q_net1(next_states, next_actions)
            next_q2 = self.target_q_net2(next_states, next_actions)
            next_q_min = torch.min(next_q1, next_q2) - self.alpha * next_log_probs
            next_values = rewards + (1 - dones) * self.discount * next_q_min

        q1 = self.q_net1(states, actions)
        q2 = self.q_net2(states, actions)

        q1_loss = F.mse_loss(q1, next_values)
        q2_loss = F.mse_loss(q2, next_values)
        q_loss = q1_loss + q2_loss

        self.q_optimizer1.zero_grad()
        q_loss.backward()
        self.q_optimizer1.step()

        self.q_optimizer2.zero_grad()
        q_loss.backward()
        self.q_optimizer2.step()

        actions, log_probs = self.policy_net.sample_action(states)
        q1 = self.q_net1(states, actions)
        q2 = self.q_net2(states, actions)
        q_min = torch.min(q1, q2)

        policy_loss = (self.alpha * log_probs - q_min).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.alpha = self.log_alpha.exp()

        for target_param, param in zip(self.target_q_net1.parameters(), self.q_net1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.target_q_net2.parameters(), self.q_net2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.policy_net.sample_action(state)
        return action.cpu().numpy()[0]