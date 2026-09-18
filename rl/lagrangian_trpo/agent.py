"""LagrangianTrust Region Policy Optimization (Lagrangian TRPO) agent.

Combines TRPO's trust-region policy update with an adaptive Lagrange
multiplier for soft constraint enforcement.
"""
import torch
import torch.optim as optim
import numpy as np

from data.config import device
from rl.cpo.agent import CPOBuffer
from rl.ppo.networks import PPOCriticNetwork
from rl.trpo.agent import TRPO


class LagrangianTRPO(TRPO):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr_critic: float = 1e-3,
        lr_lagrange: float = 5e-2,
        gamma: float = 0.99,
        lam: float = 0.95,
        delta: float = 0.01,
        cost_limit: float = 0.1,
        init_lagrange_multiplier: float = 0.0,
        max_lagrange_multiplier: float | None = None,
        normalize_penalty: bool = True,
        cg_iters: int = 10,
        max_backtracks: int = 10,
        backtrack_coeff: float = 0.5,
        vf_train_iters: int = 5,
        hidden: int = 256,
    ):
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            lr_critic=lr_critic,
            gamma=gamma,
            lam=lam,
            delta=delta,
            cg_iters=cg_iters,
            max_backtracks=max_backtracks,
            backtrack_coeff=backtrack_coeff,
            vf_train_iters=vf_train_iters,
            hidden=hidden,
        )
        self.cost_limit = cost_limit
        self.lr_lagrange = lr_lagrange
        self.lagrange_multiplier = max(0.0, init_lagrange_multiplier)
        self.max_lagrange_multiplier = max_lagrange_multiplier
        self.normalize_penalty = normalize_penalty

        self.cost_critic = PPOCriticNetwork(state_dim, hidden).to(device)
        self.cost_critic_optim = optim.Adam(self.cost_critic.parameters(), lr=lr_critic)
        self.buffer = CPOBuffer()

    def select_action(self, state: np.ndarray, deterministic: bool = False):
        s = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            if deterministic:
                mu, _ = self.actor(s)
                return torch.tanh(mu).cpu().numpy()[0]

            action, log_prob = self.actor.sample(s)
            value = self.critic(s).item()
            cost_value = self.cost_critic(s).item()
            return action.cpu().numpy()[0], log_prob.item(), value, cost_value

    def store_transition(
        self,
        state,
        action,
        reward,
        cost,
        next_state,
        done,
        log_prob,
        value,
        cost_value,
    ):
        self.buffer.push(
            state, action, reward, cost, next_state, done, log_prob, value, cost_value
        )

    def _penalized_advantages(self, advantages, cost_advantages):
        penalty = self.lagrange_multiplier * cost_advantages
        penalized = advantages - penalty
        if self.normalize_penalty:
            penalized = penalized / (1.0 + self.lagrange_multiplier)
        return penalized

    def _update_lagrange_multiplier(self, mean_cost: float) -> float:
        constraint_violation = float(mean_cost - self.cost_limit)
        self.lagrange_multiplier += self.lr_lagrange * constraint_violation
        self.lagrange_multiplier = max(0.0, self.lagrange_multiplier)
        if self.max_lagrange_multiplier is not None:
            self.lagrange_multiplier = min(
                self.max_lagrange_multiplier, self.lagrange_multiplier
            )
        return constraint_violation

    def _get_rollout_data(self) -> dict:
        with torch.no_grad():
            last_state = torch.FloatTensor(self.buffer.next_states[-1]).unsqueeze(0).to(device)
            last_value = self.critic(last_state).item()
            last_cost_value = self.cost_critic(last_state).item()

        data = self.buffer.get(last_value, last_cost_value, self.gamma, self.lam)
        self.buffer.clear()
        return data

    def _build_update_tensors(self, data: dict) -> dict:
        tensors = super()._build_update_tensors(data)
        tensors["cost_advantages"] = torch.FloatTensor(data["cost_advantages"]).to(device)
        tensors["cost_returns"] = torch.FloatTensor(data["cost_returns"]).to(device)
        return tensors

    def _before_policy_update(self, tensors: dict, data: dict) -> dict:
        mean_cost = float(data["mean_cost"])
        constraint_violation = self._update_lagrange_multiplier(mean_cost)
        return {
            "mean_cost": mean_cost,
            "cost_limit": self.cost_limit,
            "constraint_violation": constraint_violation,
            "lagrange_multiplier": self.lagrange_multiplier,
        }

    def _policy_advantages(self, tensors: dict, data: dict):
        return self._penalized_advantages(
            tensors["advantages"],
            tensors["cost_advantages"],
        )

    def _update_value_functions(self, tensors: dict) -> dict:
        metrics = super()._update_value_functions(tensors)
        states = tensors["states"]
        cost_returns = tensors["cost_returns"]

        cost_vf_loss_total = 0.0
        for _ in range(self.vf_train_iters):
            cost_values = self.cost_critic(states).squeeze()
            cost_vf_loss = torch.nn.MSELoss()(cost_values, cost_returns)
            self.cost_critic_optim.zero_grad()
            cost_vf_loss.backward()
            self.cost_critic_optim.step()
            cost_vf_loss_total += cost_vf_loss.item()

        metrics["cost_vf_loss"] = cost_vf_loss_total / self.vf_train_iters
        return metrics

    def save(self, path: str) -> None:
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "cost_critic": self.cost_critic.state_dict(),
                "lagrange_multiplier": self.lagrange_multiplier,
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.cost_critic.load_state_dict(ckpt["cost_critic"])
        self.lagrange_multiplier = float(ckpt.get("lagrange_multiplier", 0.0))
