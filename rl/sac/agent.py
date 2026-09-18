"""Soft Actor-Critic (SAC) agent with automatic entropy tuning."""
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from data.config      import device
from rl.base_agent    import BaseAgent
from rl.sac.networks  import SACActorNetwork, SACCriticNetwork
from rl.sac.replay_buffer import ReplayBuffer


class SAC(BaseAgent):

    def __init__(
        self,
        state_dim:  int,
        action_dim: int,
        gamma:  float = 0.997,
        tau:    float = 0.002,
        lr:     float = 1e-4,
        hidden: int   = 256,
    ):
        self.gamma = gamma
        self.tau   = tau

        self.actor         = SACActorNetwork(state_dim, action_dim, hidden).to(device)
        self.critic        = SACCriticNetwork(state_dim, action_dim, hidden).to(device)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optim  = optim.Adam(self.actor.parameters(),  lr=lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr)

        self.target_entropy = -float(action_dim)
        self.log_alpha      = torch.zeros(1, requires_grad=True, device=device)
        self.alpha          = self.log_alpha.exp().item()
        self.alpha_optim    = optim.Adam([self.log_alpha], lr=lr)

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        s = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            if deterministic:
                _, _, a = self.actor.sample(s)
            else:
                a, _, _ = self.actor.sample(s)
        return a.cpu().numpy()[0]

    def update(self, replay_buffer: ReplayBuffer, batch_size: int = 256) -> dict:
        if len(replay_buffer) < batch_size:
            return {}

        s, a, r, ns, d = replay_buffer.sample(batch_size)
        S  = torch.FloatTensor(s).to(device)
        A  = torch.FloatTensor(a).to(device)
        R  = torch.FloatTensor(r).unsqueeze(1).to(device)
        NS = torch.FloatTensor(ns).to(device)
        D  = torch.FloatTensor(d).unsqueeze(1).to(device)

        with torch.no_grad():
            na, log_pi_ns, _ = self.actor.sample(NS)
            q1_t, q2_t       = self.critic_target(NS, na)
            target_q         = R + (1 - D) * self.gamma * (
                torch.min(q1_t, q2_t) - self.alpha * log_pi_ns
            )

        q1, q2      = self.critic(S, A)
        critic_loss = nn.MSELoss()(q1, target_q) + nn.MSELoss()(q2, target_q)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        new_a, log_pi, _ = self.actor.sample(S)
        q1_n, q2_n       = self.critic(S, new_a)
        actor_loss       = (self.alpha * log_pi - torch.min(q1_n, q2_n)).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        alpha_loss = -(
            self.log_alpha * (log_pi + self.target_entropy).detach()
        ).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        self.alpha = self.log_alpha.exp().item()

        for p, tp in zip(self.critic.parameters(),
                         self.critic_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss":  actor_loss.item(),
            "alpha":       self.alpha,
        }

    def save(self, path: str) -> None:
        torch.save({
            "actor":  self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target = copy.deepcopy(self.critic)
