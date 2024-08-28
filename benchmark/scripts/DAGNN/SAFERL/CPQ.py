import math
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import TanhTransform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MEAN_MIN = -9.0
MEAN_MAX = 9.0
LOG_STD_MIN = -5
LOG_STD_MAX = 2
EPS = 1e-7


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, log_std_multiplier=1.0, log_std_offset=-1.0, hidden_unit=256):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_unit)
        self.fc2 = nn.Linear(hidden_unit, hidden_unit)
        self.mu_head = nn.Linear(hidden_unit, action_dim)
        self.sigma_head = nn.Linear(hidden_unit, action_dim)

        self.log_sigma_multiplier = log_std_multiplier
        self.log_sigma_offset = log_std_offset

    def _get_outputs(self, state):
        a = F.relu(self.fc1(state))
        a = F.relu(self.fc2(a))
        mu = self.mu_head(a)
        mu = torch.clip(mu, MEAN_MIN, MEAN_MAX)
        log_sigma = self.sigma_head(a)
        # log_sigma = self.log_sigma_multiplier * log_sigma + self.log_sigma_offset

        log_sigma = torch.clip(log_sigma, LOG_STD_MIN, LOG_STD_MAX)
        sigma = torch.exp(log_sigma)

        a_distribution = TransformedDistribution(
            Normal(mu, sigma), TanhTransform(cache_size=1)
        )
        a_tanh_mode = torch.tanh(mu)
        return a_distribution, a_tanh_mode

    def forward(self, state):
        a_dist, a_tanh_mode = self._get_outputs(state)
        action = a_dist.rsample()
        logp_pi = a_dist.log_prob(action).sum(axis=-1)
        return action, logp_pi, a_tanh_mode

    def get_log_density(self, state, action):
        a_dist, _ = self._get_outputs(state)
        action_clip = torch.clip(action, -1. + EPS, 1. - EPS)
        logp_action = a_dist.log_prob(action_clip)
        return logp_action


class Double_Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_unit=256):
        super(Double_Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, hidden_unit)
        self.l2 = nn.Linear(hidden_unit, hidden_unit)
        self.l3 = nn.Linear(hidden_unit, 1)

        self.l4 = nn.Linear(state_dim + action_dim, hidden_unit)
        self.l5 = nn.Linear(hidden_unit, hidden_unit)
        self.l6 = nn.Linear(hidden_unit, 1)

    def forward(self, state, action):
        q1 = F.relu(self.l1(torch.cat([state, action], 1)))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(torch.cat([state, action], 1)))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def q1(self, state, action):
        q1 = F.relu(self.l1(torch.cat([state, action], 1)))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_unit=256):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, hidden_unit)
        self.l2 = nn.Linear(hidden_unit, hidden_unit)
        self.l3 = nn.Linear(hidden_unit, 1)

    def forward(self, state, action):
        q1 = F.relu(self.l1(torch.cat([state, action], 1)))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1

    def q1(self, state, action):
        q1 = F.relu(self.l1(torch.cat([state, action], 1)))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


# Vanilla Variational Auto-Encoder
class VAE(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim, max_action, hidden_unit=256):
        super(VAE, self).__init__()
        self.e1 = nn.Linear(state_dim + action_dim, hidden_unit)
        self.e2 = nn.Linear(hidden_unit, hidden_unit)

        self.mean = nn.Linear(hidden_unit, latent_dim)
        self.log_std = nn.Linear(hidden_unit, latent_dim)

        self.d1 = nn.Linear(state_dim + latent_dim, hidden_unit)
        self.d2 = nn.Linear(hidden_unit, hidden_unit)
        self.d3 = nn.Linear(hidden_unit, action_dim)

        self.max_action = max_action
        self.latent_dim = latent_dim

    def forward(self, state, action):
        z = F.relu(self.e1(torch.cat([state, action], 1)))
        z = F.relu(self.e2(z))

        mean = self.mean(z)
        # Clamped for numerical stability
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        z = mean + std * torch.randn_like(std)

        u = self.decode(state, z)

        return u, mean, std

    def decode(self, state, z=None):
        # When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
        if z is None:
            z = torch.randn((state.shape[0], self.latent_dim)).to(device).clamp(-0.5, 0.5)

        a = F.relu(self.d1(torch.cat([state, z], 1)))
        a = F.relu(self.d2(a))
        return self.max_action * torch.tanh(self.d3(a))


class CPQ(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            discount=0.99,
            tau=0.005,
            lmbda=0.75,
            threshold=30.0,
            alpha=1.0,
            n_actions=10,
            temp=1.0,
            cql_clip_diff_min=-np.inf,
            cql_clip_diff_max=np.inf,

    ):
        latent_dim = action_dim * 2

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.reward_critic = Double_Critic(state_dim, action_dim).to(device)
        self.reward_critic_target = copy.deepcopy(self.reward_critic)
        self.reward_critic_optimizer = torch.optim.Adam(self.reward_critic.parameters(), lr=3e-4)

        self.cost_critic = Critic(state_dim, action_dim).to(device)
        self.cost_critic_target = copy.deepcopy(self.cost_critic)
        self.cost_critic_optimizer = torch.optim.Adam(self.cost_critic.parameters(), lr=3e-4)

        self.vae = VAE(state_dim, action_dim, latent_dim, max_action).to(device)
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters())

        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.lmbda = lmbda

        self.alpha = alpha
        self.n_actions = n_actions
        self.temp = temp
        self.cql_clip_diff_min = cql_clip_diff_min
        self.cql_clip_diff_max = cql_clip_diff_max

        self.threshold = threshold
        self.log_lagrangian_weight = torch.zeros(1, requires_grad=True, device=device)
        self.lagrangian_weight_optimizer = torch.optim.Adam([self.log_lagrangian_weight], lr=1e-4)

        self.total_it = 0

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
            _, _, action = self.actor(state)
        return action.cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=100):
        self.total_it += 1

        # Sample replay buffer / batch
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        cost = torch.sum(torch.abs(action), axis=1).reshape(-1, 1)

        # Reward Critic Training
        with torch.no_grad():
            next_action, _, _ = self.actor(next_state)
            target_Qr1, target_Qr2 = self.reward_critic_target(next_state, next_action)
            # Soft Clipped Double Q-learning
            target_Qr = torch.min(target_Qr1, target_Qr2)
            target_Qc = self.cost_critic_target(next_state, next_action)
            weight = torch.where(target_Qc > self.threshold, 0.0, 1.0)
            target_Qr = reward + not_done * 0.99 * target_Qr * weight
        current_Qr1, current_Qr2 = self.reward_critic(state, action)

        td_qr_loss = F.mse_loss(current_Qr1, target_Qr) + F.mse_loss(current_Qr2, target_Qr)

        # CQL
        state_repeat = torch.repeat_interleave(state, self.n_actions, 0)
        next_state_repeat = torch.repeat_interleave(next_state, self.n_actions, 0)

        cql_random_actions = action.new_empty((batch_size*self.n_actions, self.action_dim), requires_grad=False).uniform_(-1, 1)
        cql_current_actions, cql_current_log_pis, _ = self.actor(state_repeat)
        cql_next_actions, cql_next_log_pis, _ = self.actor(next_state_repeat)
        cql_current_actions, cql_current_log_pis = cql_current_actions.detach(), cql_current_log_pis.detach()
        cql_next_actions, cql_next_log_pis = cql_next_actions.detach(), cql_next_log_pis.detach()

        cql_qr1_current_actions, cql_qr2_current_actions = self.reward_critic(state_repeat, cql_current_actions)
        cql_qr1_next_actions, cql_qr2_next_actions = self.reward_critic(next_state_repeat, cql_next_actions)
        cql_qr1_rand, cql_qr2_rand = self.reward_critic(state_repeat, cql_random_actions)

        cql_qr1_current_actions = cql_qr1_current_actions.reshape(-1, self.n_actions)
        cql_qr2_current_actions = cql_qr2_current_actions.reshape(-1, self.n_actions)
        cql_qr1_next_actions = cql_qr1_next_actions.reshape(-1, self.n_actions)
        cql_qr2_next_actions = cql_qr2_next_actions.reshape(-1, self.n_actions)
        cql_qr1_rand = cql_qr1_rand.reshape(-1, self.n_actions)
        cql_qr2_rand = cql_qr2_rand.reshape(-1, self.n_actions)

        cql_cat_qr1 = torch.cat([cql_qr1_rand, cql_qr1_next_actions, cql_qr1_current_actions], 1)
        cql_cat_qr2 = torch.cat([cql_qr2_rand, cql_qr2_next_actions, cql_qr2_current_actions], 1)

        cql_qr1_ood = torch.logsumexp(cql_cat_qr1 / self.temp, dim=1, keepdim=True) * self.temp
        cql_qr2_ood = torch.logsumexp(cql_cat_qr2 / self.temp, dim=1, keepdim=True) * self.temp

        """Subtract the log likelihood of data"""
        cql_qr1_diff = torch.clamp(
            cql_qr1_ood - current_Qr1,
            self.cql_clip_diff_min,
            self.cql_clip_diff_max,
            ).mean()
        cql_qr2_diff = torch.clamp(
            cql_qr2_ood - current_Qr2,
            self.cql_clip_diff_min,
            self.cql_clip_diff_max,
            ).mean()

        cql_qr_loss = self.alpha * (cql_qr1_diff + cql_qr2_diff)
        reward_critic_loss = td_qr_loss + cql_qr_loss

        self.reward_critic_optimizer.zero_grad()
        reward_critic_loss.backward()
        self.reward_critic_optimizer.step()

        # Cost Critic Training
        with torch.no_grad():
            next_action, _, _ = self.actor(next_state)
            target_Qc = self.cost_critic_target(next_state, next_action)
            target_Qc = cost + not_done * self.discount * target_Qc
        current_Qc = self.cost_critic(state, action)

        td_qc_loss = F.mse_loss(current_Qc, target_Qc)

        # CQL
        cql_qc_current_actions = self.cost_critic(state_repeat, cql_current_actions)
        cql_qc_next_actions = self.cost_critic(next_state_repeat, cql_next_actions)
        cql_qc_rand = self.cost_critic(state_repeat, cql_random_actions)

        cql_qc_current_actions = cql_qc_current_actions.reshape(-1, self.n_actions)
        cql_qc_next_actions = cql_qc_next_actions.reshape(-1, self.n_actions)
        cql_qc_rand = cql_qc_rand.reshape(-1, self.n_actions)

        cql_cat_qc = torch.cat([cql_qc_rand, cql_qc_next_actions, cql_qc_current_actions], 1)

        cql_qc_ood = torch.logsumexp(cql_cat_qc / self.temp, dim=1, keepdim=True) * self.temp

        """Subtract the log likelihood of data"""
        cql_qc_diff = torch.clamp(
            current_Qc - cql_qc_ood,
            self.cql_clip_diff_min,
            self.cql_clip_diff_max,
            ).mean()

        cql_qc_loss = self.alpha * cql_qc_diff
        # cost_critic_loss = td_qc_loss + cql_qc_loss
        cost_critic_loss = td_qc_loss

        self.cost_critic_optimizer.zero_grad()
        cost_critic_loss.backward()
        self.cost_critic_optimizer.step()

        # Compute policy loss
        pi, log_pi, _ = self.actor(state)
        qr1_pi, qr2_pi = self.reward_critic(state, pi)
        qr_pi = torch.squeeze(torch.min(qr1_pi, qr2_pi))
        qc_pi = self.cost_critic(state, pi)
        actor_loss = (-qr_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update Target Networks
        for param, target_param in zip(self.reward_critic.parameters(), self.reward_critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.cost_critic.parameters(), self.cost_critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        if self.total_it % 5000 == 0:
            print(f'mean qr value is {qr_pi.mean()}')
            print(f'mean qc value is {qc_pi.mean()}')
            print(f'mean weight is {weight.mean()}')