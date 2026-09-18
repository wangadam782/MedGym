"""Conditional Noise Model for stochastic PINN transitions.

Implements the stochastic component from the MedGym formulation (Eq. 2):

    z_nn(x, u, δt) = F_nn(x, u, δt) + W_c,    W_c ~ N(0, σ(x, δt)²)

The noise model learns per-feature standard deviations conditioned on the
current state x and elapsed time interval δt.
"""
import torch
import torch.nn as nn


class ConditionalNoiseModel(nn.Module):
    """Predicts per-feature noise std σ(x, δt) for stochastic transitions."""

    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.state_dim = state_dim
        input_dim = state_dim + 1  # state + δt

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, state_dim),
            nn.Softplus(),  # ensure σ > 0
        )

    def forward(self, x: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        if dt.dim() == 0:
            dt = dt.unsqueeze(0).unsqueeze(0).expand(x.shape[0], 1)
        elif dt.dim() == 1:
            dt = dt.unsqueeze(-1)
        inp = torch.cat([x, dt], dim=-1)
        return self.net(inp)

    def sample_noise(self, x: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """Sample W_c ~ N(0, σ(x, δt)²)."""
        sigma = self.forward(x, dt)
        return torch.randn_like(sigma) * sigma
