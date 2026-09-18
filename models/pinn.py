"""Physics-Informed Neural Network (PINN) world model.

Predicts dx/dt given current state and treatment actions.
Includes per-variable output scaling (Bilirubin dampened by default).
"""
import torch
import torch.nn as nn

from data.config import state_dim, action_dim


class PINN(nn.Module):
    def __init__(self, in_state_dim: int = state_dim, in_action_dim: int = action_dim):
        super().__init__()
        input_dim = in_state_dim + in_action_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, in_state_dim),
        )

        init_scale = torch.ones(in_state_dim)
        if in_state_dim > 2:
            init_scale[2] = 0.01  # Bilirubin
        self.log_scale = nn.Parameter(torch.log(init_scale))

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        raw = self.net(torch.cat([x, a], dim=1))
        scale = torch.nn.functional.softplus(self.log_scale).unsqueeze(0)
        return raw * scale

    def step(
        self,
        x: torch.Tensor,
        a: torch.Tensor,
        dt: float,
        state_min: torch.Tensor | None = None,
        state_max: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """One Euler integration step with optional physiological clamping."""
        x_next = x + self.forward(x, a) * dt
        if state_min is not None and state_max is not None:
            x_next = torch.max(torch.min(x_next, state_max), state_min)
        return x_next
