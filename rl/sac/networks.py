"""SAC Actor and Critic networks."""
import torch
import torch.nn as nn


class SACActorNetwork(nn.Module):
    LOG_STD_MAX =  2
    LOG_STD_MIN = -20

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
        )
        self.mu_layer      = nn.Linear(hidden, action_dim)
        self.log_std_layer = nn.Linear(hidden, action_dim)

    def forward(self, s: torch.Tensor):
        h       = self.net(s)
        mu      = self.mu_layer(h)
        log_std = self.log_std_layer(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mu, log_std

    def sample(self, s: torch.Tensor):
        mu, log_std = self.forward(s)
        std    = log_std.exp()
        dist   = torch.distributions.Normal(mu, std)
        z      = dist.rsample()
        action = torch.tanh(z)
        log_pi = (
            dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        ).sum(-1, keepdim=True)
        return action, log_pi, torch.tanh(mu)


class SACCriticNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()

        def _q():
            return nn.Sequential(
                nn.Linear(state_dim + action_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden),                 nn.ReLU(),
                nn.Linear(hidden, 1),
            )

        self.q1 = _q()
        self.q2 = _q()

    def forward(self, s: torch.Tensor, a: torch.Tensor):
        sa = torch.cat([s, a], dim=-1)
        return self.q1(sa), self.q2(sa)
