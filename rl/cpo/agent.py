"""Constrained Policy Optimization (CPO) agent.

Extends TRPO with safety constraints using the dual approach from
Achiam et al. (2017) "Constrained Policy Optimization".
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from data.config import device
from rl.base_agent import BaseAgent
from rl.ppo.networks import PPOActorNetwork, PPOCriticNetwork


class CPOBuffer:
    """Rollout buffer with cost tracking for constrained optimization."""

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.costs = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.cost_values = []

    def push(self, state, action, reward, cost, next_state, done, log_prob, value, cost_value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.costs.append(cost)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.cost_values.append(cost_value)

    def compute_gae(self, values_with_last, rewards, dones, gamma=0.99, lam=0.95):
        advantages = np.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values_with_last[t + 1] * (1 - dones[t]) - values_with_last[t]
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values_with_last[:-1]
        return advantages, returns

    def get(self, last_value, last_cost_value, gamma=0.99, lam=0.95):
        rewards = np.array(self.rewards)
        costs = np.array(self.costs)
        dones = np.array(self.dones)
        values = np.array(self.values + [last_value])
        cost_values = np.array(self.cost_values + [last_cost_value])

        advantages, returns = self.compute_gae(values, rewards, dones, gamma, lam)
        cost_advantages, cost_returns = self.compute_gae(cost_values, costs, dones, gamma, lam)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return {
            "states": np.array(self.states),
            "actions": np.array(self.actions),
            "log_probs": np.array(self.log_probs),
            "advantages": advantages,
            "returns": returns,
            "cost_advantages": cost_advantages,
            "cost_returns": cost_returns,
            "mean_cost": costs.mean(),
        }

    def clear(self):
        for attr in [self.states, self.actions, self.rewards, self.costs,
                     self.next_states, self.dones, self.log_probs,
                     self.values, self.cost_values]:
            attr.clear()

    def __len__(self):
        return len(self.states)


class CPO(BaseAgent):
    """CPO with dual critics (reward + cost) and analytic dual solution."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        lam: float = 0.95,
        delta: float = 0.01,
        cost_limit: float = 0.1,
        cg_iters: int = 10,
        max_backtracks: int = 10,
        backtrack_coeff: float = 0.5,
        vf_train_iters: int = 5,
        hidden: int = 256,
    ):
        self.gamma = gamma
        self.lam = lam
        self.delta = delta
        self.cost_limit = cost_limit
        self.cg_iters = cg_iters
        self.max_backtracks = max_backtracks
        self.backtrack_coeff = backtrack_coeff
        self.vf_train_iters = vf_train_iters

        self.actor = PPOActorNetwork(state_dim, action_dim, hidden).to(device)
        self.critic = PPOCriticNetwork(state_dim, hidden).to(device)
        self.cost_critic = PPOCriticNetwork(state_dim, hidden).to(device)

        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.cost_critic_optim = optim.Adam(self.cost_critic.parameters(), lr=lr_critic)
        self.buffer = CPOBuffer()

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        s = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            if deterministic:
                mu, _ = self.actor(s)
                return torch.tanh(mu).cpu().numpy()[0]
            else:
                action, log_prob = self.actor.sample(s)
                value = self.critic(s).item()
                cost_value = self.cost_critic(s).item()
                return action.cpu().numpy()[0], log_prob.item(), value, cost_value

    def store_transition(self, state, action, reward, cost, next_state, done,
                         log_prob, value, cost_value):
        self.buffer.push(state, action, reward, cost, next_state, done,
                         log_prob, value, cost_value)

    def _get_actor_params(self):
        return list(self.actor.parameters())

    def _flat_params(self):
        return torch.cat([p.data.flatten() for p in self._get_actor_params()])

    def _set_flat_params(self, flat_params):
        offset = 0
        for p in self._get_actor_params():
            n = p.numel()
            p.data.copy_(flat_params[offset:offset + n].view_as(p.data))
            offset += n

    def _flat_grad(self, grads):
        return torch.cat([g.flatten() for g in grads])

    def _hessian_vector_product(self, states, actions, old_log_probs, vector):
        new_log_probs, _ = self.actor.evaluate(states, actions)
        kl = (old_log_probs - new_log_probs).mean()
        params = self._get_actor_params()
        kl_grads = torch.autograd.grad(kl, params, create_graph=True, retain_graph=True)
        flat_kl_grads = self._flat_grad(kl_grads)
        kl_v = (flat_kl_grads * vector).sum()
        hvp_grads = torch.autograd.grad(kl_v, params, retain_graph=True)
        return self._flat_grad(hvp_grads) + 1e-2 * vector

    def _conjugate_gradient(self, states, actions, old_log_probs, b):
        x = torch.zeros_like(b)
        r = b.clone()
        p = r.clone()
        rdotr = r.dot(r)
        for _ in range(self.cg_iters):
            if rdotr < 1e-10:
                break
            Ap = self._hessian_vector_product(states, actions, old_log_probs, p)
            alpha = rdotr / (p.dot(Ap) + 1e-8)
            x += alpha * p
            r -= alpha * Ap
            new_rdotr = r.dot(r)
            p = r + (new_rdotr / (rdotr + 1e-8)) * p
            rdotr = new_rdotr
        return x

    def _surrogate(self, states, actions, old_log_probs, advantages):
        new_log_probs, _ = self.actor.evaluate(states, actions)
        ratio = torch.exp(new_log_probs - old_log_probs)
        return (ratio * advantages).mean()

    def update(self, replay_buffer=None, batch_size=None) -> dict:
        if len(self.buffer) == 0:
            return {}

        with torch.no_grad():
            last_s = torch.FloatTensor(self.buffer.next_states[-1]).unsqueeze(0).to(device)
            last_value = self.critic(last_s).item()
            last_cost_value = self.cost_critic(last_s).item()

        data = self.buffer.get(last_value, last_cost_value, self.gamma, self.lam)
        self.buffer.clear()

        states = torch.FloatTensor(data["states"]).to(device)
        actions = torch.FloatTensor(data["actions"]).to(device)
        old_log_probs = torch.FloatTensor(data["log_probs"]).to(device)
        advantages = torch.FloatTensor(data["advantages"]).to(device)
        returns = torch.FloatTensor(data["returns"]).to(device)
        cost_advantages = torch.FloatTensor(data["cost_advantages"]).to(device)
        cost_returns = torch.FloatTensor(data["cost_returns"]).to(device)
        mean_cost = data["mean_cost"]

        reward_loss = self._surrogate(states, actions, old_log_probs, advantages)
        params = self._get_actor_params()
        r_grads = torch.autograd.grad(reward_loss, params, retain_graph=True)
        g = self._flat_grad(r_grads)

        cost_loss = self._surrogate(states, actions, old_log_probs, cost_advantages)
        c_grads = torch.autograd.grad(cost_loss, params, retain_graph=True)
        b = self._flat_grad(c_grads)

        H_inv_g = self._conjugate_gradient(states, actions, old_log_probs, g)
        H_inv_b = self._conjugate_gradient(states, actions, old_log_probs, b)

        q = g.dot(H_inv_g)
        r_val = g.dot(H_inv_b)
        s_val = b.dot(H_inv_b)
        c = mean_cost - self.cost_limit
        eps_s = 1e-8

        if s_val > eps_s:
            if c > 0:
                discriminant = 2.0 * self.delta * s_val - c ** 2
                if discriminant >= 0:
                    lam_a = (r_val + torch.sqrt(torch.clamp(discriminant, min=0.0))) / (s_val + eps_s)
                    lam_b = (r_val - torch.sqrt(torch.clamp(discriminant, min=0.0))) / (s_val + eps_s)
                    lam_star = max(lam_a.item(), lam_b.item(), 0.0)
                else:
                    lam_star = max(0.0, r_val.item() / (s_val.item() + eps_s))
                denom = q - r_val ** 2 / (s_val + eps_s)
                nu_star = torch.sqrt(torch.clamp(2.0 * self.delta / (denom + eps_s), min=0.0))
                direction = nu_star * (H_inv_g - lam_star * H_inv_b)
            else:
                shs = 0.5 * g.dot(H_inv_g)
                step_size = torch.sqrt(shs / self.delta) if shs > 0 else torch.tensor(1.0)
                direction = H_inv_g / (step_size + eps_s)
        else:
            shs = 0.5 * g.dot(H_inv_g)
            step_size = torch.sqrt(shs / self.delta) if shs > 0 else torch.tensor(1.0)
            direction = H_inv_g / (step_size + eps_s)

        old_params = self._flat_params()
        old_obj = reward_loss.item()
        accepted = False
        kl = 0.

        for i in range(self.max_backtracks):
            step = (self.backtrack_coeff ** i) * direction
            self._set_flat_params(old_params + step)
            with torch.no_grad():
                new_obj = self._surrogate(states, actions, old_log_probs, advantages).item()
                new_log_probs, _ = self.actor.evaluate(states, actions)
                kl = (old_log_probs - new_log_probs).mean().item()
                new_cost_surr = self._surrogate(states, actions, old_log_probs, cost_advantages).item()
            projected_cost = mean_cost + new_cost_surr
            cost_ok = projected_cost <= self.cost_limit or projected_cost <= mean_cost
            if kl <= self.delta and new_obj >= old_obj and cost_ok:
                accepted = True
                break

        if not accepted:
            self._set_flat_params(old_params)
            kl = 0.

        vf_loss_total, cvf_loss_total = 0., 0.
        for _ in range(self.vf_train_iters):
            values = self.critic(states).squeeze()
            vf_loss = nn.MSELoss()(values, returns)
            self.critic_optim.zero_grad()
            vf_loss.backward()
            self.critic_optim.step()
            vf_loss_total += vf_loss.item()

            cost_values = self.cost_critic(states).squeeze()
            cvf_loss = nn.MSELoss()(cost_values, cost_returns)
            self.cost_critic_optim.zero_grad()
            cvf_loss.backward()
            self.cost_critic_optim.step()
            cvf_loss_total += cvf_loss.item()

        return {
            "pg_loss": -old_obj,
            "vf_loss": vf_loss_total / self.vf_train_iters,
            "cost_vf_loss": cvf_loss_total / self.vf_train_iters,
            "mean_cost": mean_cost,
            "kl": kl,
            "accepted": accepted,
        }

    def save(self, path: str) -> None:
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "cost_critic": self.cost_critic.state_dict(),
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.cost_critic.load_state_dict(ckpt["cost_critic"])
