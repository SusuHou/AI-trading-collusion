"""Convergence criterion (page 26).

A session has converged once every speculator's greedy (argmax) action is
unchanged for `streak_target` consecutive periods (paper: 1,000,000; POC
configs may use a smaller value, plus a hard period cap where full
convergence is impractical).

Because the Q-update touches a single state row per period, the greedy
policy can only change at that row, so BatchQLearner.update's `changed`
flag makes this criterion exact.
"""
from __future__ import annotations

import numpy as np

PAPER_STREAK = 1_000_000


class ConvergenceTracker:
    def __init__(self, batch: int, streak_target: int = PAPER_STREAK):
        self.streak_target = int(streak_target)
        self.t = 0
        self.streak = np.zeros(batch, dtype=np.int64)
        self.best_streak = np.zeros(batch, dtype=np.int64)
        self.converged_at = np.full(batch, -1, dtype=np.int64)

    def update(self, changed: np.ndarray) -> np.ndarray:
        """changed: (batch,) bool for this period. Returns converged mask."""
        self.t += 1
        self.streak += 1
        self.streak[changed] = 0
        np.maximum(self.best_streak, self.streak, out=self.best_streak)
        newly = (self.streak >= self.streak_target) & (self.converged_at < 0)
        self.converged_at[newly] = self.t
        return self.converged

    @property
    def converged(self) -> np.ndarray:
        return self.converged_at >= 0

    @property
    def all_converged(self) -> bool:
        return bool(self.converged.all())

    def summary(self) -> dict:
        return {
            "periods_run": self.t,
            "n_converged": int(self.converged.sum()),
            "converged_at": self.converged_at.copy(),
            "best_streak": self.best_streak.copy(),
            "streak_target": self.streak_target,
        }
