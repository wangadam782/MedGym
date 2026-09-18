"""Physics-Informed Neural Network world model for the hypotension benchmark.

Predicts dx/dt in normalised space, with step() and rollout() helpers.

Key differences from the sepsis PINN:
  1. Per-variable output scale driven by config (slow vars: Cr, Hep).
  2. step() defaults to n_substeps=8 because the fastest ODE time constant
     (tau_ce=0.25h) makes single-step Euler unstable at dt=1h.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from hypotension.data.config import SIDX, action_dim, state_dim


class PINN(nn.Module):
    def __init__(self, in_state=state_dim, in_action=action_dim, hidden=128,
                 slow_vars=("Cr", "Hep"), slow_scale=0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_state + in_action, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, in_state))
        s = torch.ones(in_state)
        for n in slow_vars:
            if n in SIDX:
                s[SIDX[n]] = slow_scale
        self.log_scale = nn.Parameter(torch.log(s))

    def forward(self, x, a):
        return self.net(torch.cat([x, a], 1)) * F.softplus(self.log_scale).unsqueeze(0)

    def step(self, x, a, dt=1.0, state_min=None, state_max=None, n_substeps=8):
        """Advance by dt. dt can be scalar or (B,) tensor (irregular sampling)."""
        if not torch.is_tensor(dt):
            dt = torch.as_tensor(float(dt), dtype=x.dtype, device=x.device)
        h = (dt / n_substeps).reshape(-1, 1) if dt.dim() else dt / n_substeps
        for _ in range(n_substeps):
            x = x + self.forward(x, a) * h
            if state_min is not None:
                x = torch.max(torch.min(x, state_max), state_min)
        return x

    def rollout(self, x0, actions, dt=1.0, **kw):
        """x0 (B, d_x), actions (B, K, d_u) -> (B, K+1, d_x)."""
        xs, x = [x0], x0
        for k in range(actions.shape[1]):
            x = self.step(x, actions[:, k], dt, **kw)
            xs.append(x)
        return torch.stack(xs, 1)
