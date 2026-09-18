"""Proximal Policy Optimization (PPO) agent with GAE."""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from data.config import device
from rl.base_agent import BaseAgent
from rl.ppo.networks import PPOActorNetwork, PPOCriticNetwork


class PPOBuffer:
    """Rollout buffer for on-policy algorithms (PPO, TRPO, CPO)."""

    def __init__(self, max_size=4096):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.max_size = max_size

    def push(self, state, action, reward, next_state, done, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def compute_gae(self, last_value, gamma=0.99, lam=0.95):
        rewards = np.array(self.rewards)
        values = np.array(self.values + [last_value])
        dones = np.array(self.dones)

        advantages = np.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values[:-1]
        return advantages, returns

    def get(self, last_value, gamma=0.99, lam=0.95):
        advantages, returns = self.compute_gae(last_value, gamma, lam)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return {
            "states": np.array(self.states),
            "actions": np.array(self.actions),
            "log_probs": np.array(self.log_probs),
            "advantages": advantages,
            "returns": returns,
        }

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()

    def __len__(self):
        return len(self.states)


class PPO(BaseAgent):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        update_epochs: int = 10,
        mini_batch_size: int = 64,
        hidden: int = 256,
    ):
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.mini_batch_size = mini_batch_size

        self.actor = PPOActorNetwork(state_dim, action_dim, hidden).to(device)
        self.critic = PPOCriticNetwork(state_dim, hidden).to(device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr_actor)
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

    def _ppo_update_epoch(self, tensors: dict, indices: np.ndarray, advantages) -> dict:
        states = tensors["states"]
        actions = tensors["actions"]
        old_log_probs = tensors["old_log_probs"]
        returns = tensors["returns"]
        n = states.shape[0]
        total_pg_loss, total_vf_loss, total_entropy = 0., 0., 0.
        n_updates = 0

        for _ in range(self.update_epochs):
            indices = np.random.permutation(n)
            for start in range(0, n, self.mini_batch_size):
                end = min(start + self.mini_batch_size, n)
                idx = indices[start:end]

                mb_states = states[idx]
                mb_actions = actions[idx]
                mb_old_log_probs = old_log_probs[idx]
                mb_advantages = advantages[idx]
                mb_returns = returns[idx]

                new_log_probs, entropy = self.actor.evaluate(mb_states, mb_actions)
                ratio = torch.exp(new_log_probs - mb_old_log_probs)

                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * mb_advantages
                pg_loss = -torch.min(surr1, surr2).mean()

                values = self.critic(mb_states).squeeze()
                vf_loss = nn.MSELoss()(values, mb_returns)

                loss = pg_loss + self.vf_coef * vf_loss - self.entropy_coef * entropy.mean()

                self.actor_optim.zero_grad()
                self.critic_optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.actor_optim.step()
                self.critic_optim.step()

                total_pg_loss += pg_loss.item()
                total_vf_loss += vf_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1

        return {
            "pg_loss": total_pg_loss / max(n_updates, 1),
            "vf_loss": total_vf_loss / max(n_updates, 1),
            "entropy": total_entropy / max(n_updates, 1),
        }

    def _after_policy_update(self, tensors: dict) -> dict:
        return {}

    def update(self, replay_buffer=None, batch_size=None) -> dict:
        if len(self.buffer) == 0:
            return {}

        data = self._get_rollout_data()
        tensors = self._build_update_tensors(data)
        extra_metrics = self._before_policy_update(tensors, data)
        advantages = self._policy_advantages(tensors, data)
        indices = np.arange(tensors["states"].shape[0])
        update_metrics = self._ppo_update_epoch(tensors, indices, advantages)
        after_metrics = self._after_policy_update(tensors)

        return {
            **update_metrics,
            **after_metrics,
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
