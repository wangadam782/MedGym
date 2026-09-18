"""
rl/cpo/rollout_buffer.py — On-policy trajectory buffer (single-cost version)

Reverted to scalar cost interface (Urine cost only).
"""
import numpy as np
import torch

from data.config import device


class RolloutBuffer:
    """On-policy data buffer (CPO-specific)."""

    def __init__(self):
        self.clear()

    def clear(self):
        self.states    = []
        self.actions   = []
        self.log_probs = []
        self.rewards   = []
        self.costs     = []   # scalar float (urine cost)
        self.dones     = []
        self.values_r  = []
        self.values_c  = []   # scalar float
        self.dts       = []

        self.advantages_r = None
        self.advantages_c = None
        self.returns_r    = None
        self.returns_c    = None

    def push(self, state, action, log_prob, reward, cost,
             done, value_r, value_c, dt):
        self.states.append(state.copy())
        self.actions.append(action.copy())
        self.log_probs.append(float(log_prob))
        self.rewards.append(float(reward))
        self.costs.append(float(cost))
        self.dones.append(float(done))
        self.values_r.append(float(value_r))
        self.values_c.append(float(value_c))
        self.dts.append(float(dt))

    def compute_gae(
        self,
        last_value_r: float = 0.0,
        last_value_c: float = 0.0,
        gamma: float = 0.99,
        lam:   float = 0.95,
    ):
        N = len(self.rewards)
        adv_r = np.zeros(N, dtype=np.float32)
        adv_c = np.zeros(N, dtype=np.float32)
        ret_r = np.zeros(N, dtype=np.float32)
        ret_c = np.zeros(N, dtype=np.float32)

        gae_r = 0.0
        gae_c = 0.0

        for t in reversed(range(N)):
            dt_t    = self.dts[t]
            gamma_t = gamma ** dt_t
            mask    = 1.0 - self.dones[t]

            next_v_r = last_value_r if t == N - 1 else self.values_r[t + 1]
            next_v_c = last_value_c if t == N - 1 else self.values_c[t + 1]

            delta_r = self.rewards[t] + gamma_t * next_v_r * mask - self.values_r[t]
            delta_c = self.costs[t]   + gamma_t * next_v_c * mask - self.values_c[t]

            gae_r = delta_r + gamma_t * lam * mask * gae_r
            gae_c = delta_c + gamma_t * lam * mask * gae_c

            adv_r[t] = gae_r
            adv_c[t] = gae_c
            ret_r[t] = adv_r[t] + self.values_r[t]
            ret_c[t] = adv_c[t] + self.values_c[t]

        adv_r = (adv_r - adv_r.mean()) / (adv_r.std() + 1e-8)
        # cost advantage is not normalized

        self.advantages_r = torch.tensor(adv_r, dtype=torch.float32).to(device)
        self.advantages_c = torch.tensor(adv_c, dtype=torch.float32).to(device)
        self.returns_r    = torch.tensor(ret_r, dtype=torch.float32).to(device)
        self.returns_c    = torch.tensor(ret_c, dtype=torch.float32).to(device)

    def get_tensors(self):
        S  = torch.tensor(np.array(self.states),    dtype=torch.float32).to(device)
        A  = torch.tensor(np.array(self.actions),   dtype=torch.float32).to(device)
        LP = torch.tensor(self.log_probs,            dtype=torch.float32).to(device)
        return S, A, LP

    def __len__(self):
        return len(self.rewards)

    @property
    def mean_cost(self) -> float:
        if len(self.costs) == 0:
            return 0.0
        return float(np.mean(self.costs))

    @property
    def episode_cost(self) -> float:
        return float(np.sum(self.costs))
