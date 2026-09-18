"""Abstract base class for RL agents.

All agents (SAC, PPO, TRPO, CPO) inherit from this class to ensure a
consistent interface for training and evaluation.
"""
from abc import ABC, abstractmethod
import numpy as np


class BaseAgent(ABC):

    @abstractmethod
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Given a state, return an action in the range [-1, 1]."""

    @abstractmethod
    def update(self, replay_buffer, batch_size: int) -> dict:
        """Update networks and return training metrics."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Save model weights to the specified path."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model weights from the specified path."""
