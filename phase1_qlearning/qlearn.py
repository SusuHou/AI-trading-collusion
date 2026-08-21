"""Batched tabular Q-learning speculators (paper section 2 and 4.1-4.2).

State (page 22): s_t = (p_{t-1}, v_{t-1}, v_t), encoded as
    s = (p_idx * nv + vlag_idx) * nv + v_idx,   nS = np * nv * nv.
Action: index j into the nx-point multiplier grid; order flow
    x_{i,t} = (v_t - vbar) * c_j  (state-scaled action grid, page 24).

Implements:
  eq 2.4  Q_{t+1}(s_t, x_t) = (1-alpha) Q_t(s_t, x_t)
                              + alpha [pi_t + rho max_x' Q_t(s_{t+1}, x')]
  eq 2.6  epsilon-greedy action selection
  eq 4.3  epsilon_t(v) = exp(-beta n_t(v)), one visit counter per v grid
          point per session ("system" visits, shared across speculators)

Everything is vectorized over (batch, I): Q has shape (batch, I, nS, nx).
The greedy policy is cached per state and refreshed only at the single
(s_t, .) row an update touches, which makes the paper's convergence
criterion (argmax unchanged) exact and O(1) per period.
"""
from __future__ import annotations

import numpy as np

from env.benchmarks import Benchmarks, Params, q0_table


class BatchQLearner:
    def __init__(self, params: Params, bench: Benchmarks, batch: int):
        p = params
        self.params = params
        self.bench = bench
        self.batch = batch
        self.nS = p.np_ * p.nv * p.nv

        # Initial Q (page 25): rows depend on s only through its v component.
        q0 = q0_table(bench)                       # (nv, nx)
        v_of_s = np.arange(self.nS) % p.nv         # v is the fastest index
        q0_full = q0[v_of_s]                       # (nS, nx)
        self.Q = np.broadcast_to(q0_full, (batch, p.I, self.nS, p.nx)).copy()

        self.pol = self.Q.argmax(axis=3).astype(np.int8)   # greedy policy cache
        self.visits = np.zeros((batch, p.nv), dtype=np.int64)  # n_t(v), eq 4.3

        self._b = np.arange(batch)[:, None]        # (batch, 1)
        self._i = np.arange(p.I)[None, :]          # (1, I)

    # ------------------------------------------------------------------
    def state_index(self, p_idx, vlag_idx, v_idx):
        nv = self.params.nv
        return (p_idx * nv + vlag_idx) * nv + v_idx

    def epsilon(self, v_idx):
        """epsilon_t(v) = exp(-beta n_t(v)) for each session (eq 4.3)."""
        n = self.visits[np.arange(self.batch), v_idx]
        return np.exp(-self.params.beta * n)

    # ------------------------------------------------------------------
    def act(self, s, v_idx, unif, rand_a, explore_mask=None):
        """Select actions (batch, I) per eq 2.6 and count the v visit.

        unif, rand_a : pre-drawn U(0,1) and uniform action indices, (batch, I).
        explore_mask : optional (batch,) bool; sessions with False act greedily
                       (used to freeze converged sessions / evaluation).
        """
        greedy = self.pol[self._b, self._i, s[:, None]]        # (batch, I)
        eps = self.epsilon(v_idx)[:, None]                     # (batch, 1)
        explore = unif < eps
        if explore_mask is not None:
            explore &= explore_mask[:, None]
        a = np.where(explore, rand_a, greedy)
        self.visits[np.arange(self.batch), v_idx] += 1
        return a

    # ------------------------------------------------------------------
    def update(self, s, a, r, s_next, active=None):
        """Apply eq 2.4 at (s, a) for every session/speculator.

        s, s_next : (batch,) state indices;  a : (batch, I) action indices;
        r         : (batch, I) realized profits pi_{i,t};
        active    : optional (batch,) bool; inactive sessions are skipped.
        Returns changed : (batch,) bool -- True if any speculator's greedy
        action at s changed (drives the convergence criterion).
        """
        p = self.params
        b, i = self._b, self._i
        rows_next = self.Q[b, i, s_next[:, None], :]           # (batch, I, nx)
        target = r + p.rho * rows_next.max(axis=2)
        rows = self.Q[b, i, s[:, None], :]                     # copy (batch, I, nx)
        old = np.take_along_axis(rows, a[:, :, None], axis=2)[:, :, 0]
        newq = (1.0 - p.alpha) * old + p.alpha * target
        if active is not None:
            newq = np.where(active[:, None], newq, old)
        np.put_along_axis(rows, a[:, :, None], newq[:, :, None], axis=2)
        self.Q[b, i, s[:, None], :] = rows
        new_pol = rows.argmax(axis=2).astype(np.int8)
        old_pol = self.pol[b, i, s[:, None]]
        changed = (new_pol != old_pol).any(axis=1)
        self.pol[b, i, s[:, None]] = new_pol
        return changed
