"""Trust Region Policy Optimization (TRPO) agent.

Natural gradient with conjugate gradient and line search.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from data.config import device
from rl.base_agent import BaseAgent
from rl.ppo.networks import PPOActorNetwork, PPOCriticNetwork
from rl.ppo.agent import PPOBuffer


class TRPO(BaseAgent):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        lam: float = 0.95,
        delta: float = 0.01,
        cg_iters: int = 10,
        max_backtracks: int = 10,
        backtrack_coeff: float = 0.5,
        vf_train_iters: int = 5,
        hidden: int = 256,
    ):
        self.gamma = gamma
        self.lam = lam
        self.delta = delta
        self.cg_iters = cg_iters
        self.max_backtracks = max_backtracks
        self.backtrack_coeff = backtrack_coeff
        self.vf_train_iters = vf_train_iters

        self.actor = PPOActorNetwork(state_dim, action_dim, hidden).to(device)
        self.critic = PPOCriticNetwork(state_dim, hidden).to(device)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.buffer = PPOBuffer()

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        s = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            if deterministic:
                mu, _ = self.actor(s)
                return torch.tanh(mu).cpu().numpy()[0]
            else:
                action, log_prob = self.actor.sample(s)
                value = self.critic(s).item()
                return action.cpu().numpy()[0], log_prob.item(), value

    def store_transition(self, state, action, reward, next_state, done, log_prob, value):
        self.buffer.push(state, action, reward, next_state, done, log_prob, value)

    def _get_actor_params(self):
        return [p for p in self.actor.parameters()]

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
            x = x + alpha * p
            r = r - alpha * Ap
            new_rdotr = r.dot(r)
            p = r + (new_rdotr / (rdotr + 1e-8)) * p
            rdotr = new_rdotr
        return x

    def _surrogate_loss(self, states, actions, old_log_probs, advantages):
        new_log_probs, _ = self.actor.evaluate(states, actions)
        ratio = torch.exp(new_log_probs - old_log_probs)
        return (ratio * advantages).mean()

    def _get_rollout_data(self) -> dict:
        with torch.no_grad():
            last_state = torch.FloatTensor(self.buffer.next_states[-1]).unsqueeze(0).to(device)
            last_value = self.critic(last_state).item()

        data = self.buffer.get(last_value, self.gamma, self.lam)
        self.buffer.clear()
        return data

    def _build_update_tensors(self, data: dict) -> dict:
        return {
            "states": torch.FloatTensor(data["states"]).to(device),
            "actions": torch.FloatTensor(data["actions"]).to(device),
            "old_log_probs": torch.FloatTensor(data["log_probs"]).to(device),
            "advantages": torch.FloatTensor(data["advantages"]).to(device),
            "returns": torch.FloatTensor(data["returns"]).to(device),
        }

    def _before_policy_update(self, tensors: dict, data: dict) -> dict:
        return {}

    def _policy_advantages(self, tensors: dict, data: dict):
        return tensors["advantages"]

    def _trpo_policy_update(self, tensors: dict, advantages) -> dict:
        states = tensors["states"]
        actions = tensors["actions"]
        old_log_probs = tensors["old_log_probs"]

        loss = self._surrogate_loss(states, actions, old_log_probs, advantages)
        params = self._get_actor_params()
        grads = torch.autograd.grad(loss, params, retain_graph=True)
        g = self._flat_grad(grads)

        nat_grad = self._conjugate_gradient(states, actions, old_log_probs, g)
        shs = 0.5 * g.dot(nat_grad)
        step_size = torch.sqrt(shs / self.delta) if shs > 0 else torch.tensor(1.0)
        full_step = nat_grad / (step_size + 1e-8)

        old_params = self._flat_params()
        old_loss = loss.item()
        accepted = False
        kl = 0.

        for i in range(self.max_backtracks):
            step = (self.backtrack_coeff ** i) * full_step
            self._set_flat_params(old_params + step)
            with torch.no_grad():
                new_loss = self._surrogate_loss(states, actions, old_log_probs, advantages).item()
                new_log_probs, _ = self.actor.evaluate(states, actions)
                kl = (old_log_probs - new_log_probs).mean().item()
            if kl <= self.delta and new_loss >= old_loss:
                accepted = True
                break

        if not accepted:
            self._set_flat_params(old_params)

        return {
            "pg_loss": -old_loss,
            "kl": kl if accepted else 0.,
            "accepted": accepted,
        }

    def _update_value_functions(self, tensors: dict) -> dict:
        states = tensors["states"]
        returns = tensors["returns"]

        vf_loss_total = 0.
        for _ in range(self.vf_train_iters):
            values = self.critic(states).squeeze()
            vf_loss = nn.MSELoss()(values, returns)
            self.critic_optim.zero_grad()
            vf_loss.backward()
            self.critic_optim.step()
            vf_loss_total += vf_loss.item()

        return {
            "vf_loss": vf_loss_total / self.vf_train_iters,
        }

    def update(self, replay_buffer=None, batch_size=None) -> dict:
        if len(self.buffer) == 0:
            return {}

        data = self._get_rollout_data()
        tensors = self._build_update_tensors(data)
        extra_metrics = self._before_policy_update(tensors, data)
        advantages = self._policy_advantages(tensors, data)
        policy_metrics = self._trpo_policy_update(tensors, advantages)
        value_metrics = self._update_value_functions(tensors)

        return {
            **policy_metrics,
            **value_metrics,
            **extra_metrics,
        }

    def save(self, path: str) -> None:
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
