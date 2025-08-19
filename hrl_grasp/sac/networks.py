from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: Tuple[int, int] = (256, 256), out_dim: int = 1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden[0])
        self.fc2 = nn.Linear(hidden[0], hidden[1])
        self.fc3 = nn.Linear(hidden[1], out_dim)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class Actor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden=(256,256)):
        super().__init__()
        self.net = MLP(obs_dim, hidden, 2*action_dim)
        self.action_dim = action_dim
    def forward(self, obs):
        mu_logstd = self.net(obs)
        mu, log_std = mu_logstd[:, :self.action_dim], mu_logstd[:, self.action_dim:]
        log_std = torch.clamp(log_std, -5.0, 2.0)
        std = torch.exp(log_std)
        return mu, std

    def sample(self, obs):
        mu, std = self(obs)
        dist = torch.distributions.Normal(mu, std)
        x_t = dist.rsample()
        a = torch.tanh(x_t)  # bounded [-1, 1]
        log_prob = dist.log_prob(x_t) - torch.log(1 - a.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        mu_a = torch.tanh(mu)
        return a, log_prob, mu_a

class Critic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden=(256,256)):
        super().__init__()
        self.q1 = MLP(obs_dim + action_dim, hidden, 1)
        self.q2 = MLP(obs_dim + action_dim, hidden, 1)
    def forward(self, obs, act):
        xu = torch.cat([obs, act], dim=-1)
        return self.q1(xu), self.q2(xu)
