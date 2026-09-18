"""Experience replay buffer for off-policy algorithms."""
import random
import numpy as np
from collections import deque


class ReplayBuffer:
    def __init__(self, capacity: int = 200_000):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, ns, done):
        self.buf.append((
            s.copy(), a.copy(), float(r), ns.copy(), float(done),
        ))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (
            np.array(s,  dtype=np.float32),
            np.array(a,  dtype=np.float32),
            np.array(r,  dtype=np.float32),
            np.array(ns, dtype=np.float32),
            np.array(d,  dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buf)
