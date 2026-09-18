"""
rl/cpo/networks.py — Actor and Critic networks for CPO

Differences from SAC:
  · Actor is a Gaussian policy (log_prob used for REINFORCE gradient, no reparameterization needed)
  · Critic is split into two independent V networks: V^r (reward) and V^c (cost)
  · No Q network (CPO is on-policy; only state values are required)
"""
import torch
import torch.nn as nn
from torch.distributions import Normal


class CPOActorNetwork(nn.Module):
    """
    Gaussian policy π_θ(a|s).
    Outputs tanh-squashed actions and computes log-probabilities (for policy gradient and KL divergence).
    """
    LOG_STD_MAX =  2
    LOG_STD_MIN = -5

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),    nn.Tanh(),
        )
        self.mu_layer      = nn.Linear(hidden, action_dim)
        self.log_std_layer = nn.Linear(hidden, action_dim)

    def forward(self, s: torch.Tensor):
        h       = self.net(s)
        mu      = self.mu_layer(h)
        log_std = self.log_std_layer(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mu, log_std

    def get_dist(self, s: torch.Tensor) -> Normal:
        mu, log_std = self.forward(s)
        return Normal(mu, log_std.exp())

    def sample(self, s: torch.Tensor):
        """
        Sample an action, returning (action_tanh, log_prob, mean_tanh).
        log_prob includes tanh correction: log π(a|s) = log N(z) - Σ log(1-tanh²(z))
        """
        mu, log_std = self.forward(s)
        std  = log_std.exp()
        dist = Normal(mu, std)
        z    = dist.rsample()                                  # reparameterization (not required by CPO, but harmless)
        a    = torch.tanh(z)
        log_prob = (dist.log_prob(z)
                    - torch.log(1 - a.pow(2) + 1e-6)
                   ).sum(-1, keepdim=True)
        return a, log_prob, torch.tanh(mu)

    def log_prob(self, s: torch.Tensor, a_tanh: torch.Tensor) -> torch.Tensor:
        """
        Compute log-probability given a state and an existing action.
        Used for on-policy gradient: ∇_θ log π_θ(a|s)
        """
        mu, log_std = self.forward(s)
        std  = log_std.exp()
        dist = Normal(mu, std)
        # inverse tanh
        a_clamp = a_tanh.clamp(-1 + 1e-6, 1 - 1e-6)
        z       = torch.atanh(a_clamp)
        lp      = (dist.log_prob(z)
                   - torch.log(1 - a_tanh.pow(2) + 1e-6)
                  ).sum(-1, keepdim=True)
        return lp

    def kl_divergence(self, s: torch.Tensor, old_mu: torch.Tensor,
                      old_log_std: torch.Tensor) -> torch.Tensor:
        """
        KL( π_old(·|s) || π_θ(·|s) ), used for the Fisher-vector product.
        Note direction: old policy first (required for trust-region constraint).
        """
        mu, log_std = self.forward(s)
        std     = log_std.exp()
        old_std = old_log_std.exp()
        # KL(N(μ₁,σ₁) || N(μ₂,σ₂)) = log(σ₂/σ₁) + (σ₁²+(μ₁-μ₂)²)/(2σ₂²) - 1/2
        kl = (log_std - old_log_std
              + (old_std.pow(2) + (old_mu - mu).pow(2)) / (2 * std.pow(2) + 1e-8)
              - 0.5)
        return kl.sum(-1).mean()


class CPOValueNetwork(nn.Module):
    """
    State value network V(s).
    Used for reward V^r and cost V^c (created as two separate instances).
    """

    def __init__(self, state_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),    nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s)
