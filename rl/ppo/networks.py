"""PPO Actor and Critic networks (shared by PPO/TRPO/CPO)."""
import torch
import torch.nn as nn


class PPOActorNetwork(nn.Module):
    LOG_STD_MAX = 0.0
    LOG_STD_MIN = -3.0

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.mu_layer = nn.Linear(hidden, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, s):
        h = self.net(s)
        mu = self.mu_layer(h)
        log_std = torch.clamp(self.log_std.expand_as(mu),
                              self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mu, log_std

    def sample(self, s):
        mu, log_std = self.forward(s)
        std = log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        z = dist.rsample()
        action = torch.tanh(z)
        log_prob = (dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)).sum(-1)
        return action, log_prob

    def evaluate(self, s, a):
        mu, log_std = self.forward(s)
        std = log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        z = torch.atanh(torch.clamp(a, -0.999, 0.999))
        log_prob = (dist.log_prob(z) - torch.log(1 - a.pow(2) + 1e-6)).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy

    def get_dist(self, s):
        mu, log_std = self.forward(s)
        std = log_std.exp()
        return torch.distributions.Normal(mu, std)


class PPOCriticNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, s):
        return self.net(s)
