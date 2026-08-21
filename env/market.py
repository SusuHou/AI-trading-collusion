"""Batched Kyle-style market environment (paper section 3-4).

Everything is vectorized over a leading `batch` dimension of independent
simulation sessions (one MarketEnv instance simulates `batch` parallel
sessions), so a POC run amortizes Python/NumPy overhead across sessions and
a future scale-out (SLURM / LLM rollouts) can reuse the same code path.

Per-period protocol implemented in `step()` (page 22-23; steps 2-4 of the
protocol -- action selection and Q-updates live in phase1_qlearning):
  y_t = sum_i x_{i,t} + u_t
  p_t = gamma0_hat_t + lambda_hat_t * y_t                     (eq 4.2)
  z_t = -xi (p_t - vbar)                                      (eq 3.2)
  pi_{i,t} = (v_t - p_t) x_{i,t}
  market maker appends (v_t, p_t, z_t, y_t) to its rolling window

Adaptive market maker (eq 4.1-4.2): two rolling-window OLS regressions,
  z on p  ->  z = xi0_hat - xi1_hat p    (slope of z on p is -xi1_hat)
  v on y  ->  v = gamma0_hat + gamma1_hat y
  lambda_hat = (theta gamma1_hat + xi1_hat) / (theta + xi1_hat^2)
maintained with O(1) running sufficient statistics over a circular buffer of
length Tm (with periodic exact re-summation to cancel floating-point drift).

Initialization judgment call (the paper does not specify the market maker's
first Tm periods): the rolling window is warm-started with Tm synthetic
observations generated from a theoretical benchmark (Nash by default), i.e.
v ~ uniform(v grid), u ~ N(0, sigma_u^2), y = I*chi*(v-vbar) + u,
p = vbar + lambda*y, z = -xi(p - vbar). The regressions then recover
xi1_hat = xi and lambda_hat = lambda exactly at t=0 and drift to the data as
real observations replace synthetic ones. Documented in README.
"""
from __future__ import annotations

import numpy as np

from env.benchmarks import Benchmarks, Params

# columns of the sufficient-statistics vector
_P, _PP, _Z, _PZ, _V, _Y, _YY, _VY = range(8)


class MarketEnv:
    def __init__(self, params: Params, bench: Benchmarks, batch: int,
                 rng: np.random.Generator, mm_init: str = "nash",
                 resync_every: int = 1 << 20):
        self.params = params
        self.bench = bench
        self.batch = batch
        self.rng = rng
        self.resync_every = resync_every

        Tm = params.Tm
        self.buf = np.empty((batch, Tm, 8), dtype=np.float64)
        self.ptr = 0
        self._pushes = 0
        self._fill_synthetic(mm_init)
        self.S = self.buf.sum(axis=1)  # (batch, 8) running sums

    # ------------------------------------------------------------------
    def _fill_synthetic(self, mode: str):
        """Warm-start the rolling window with benchmark-consistent data.

        Uses an exact balanced design when Tm % nv == 0 (default Tm=10000,
        nv=10): each v grid point appears Tm/nv times, crossed with a
        symmetric Gaussian-quantile u grid rescaled to sample std exactly
        sigma_u. Sample moments then equal population moments, so the MM's
        OLS regressions recover xi1_hat = xi, gamma0_hat = vbar and
        lambda_hat = lambda^{N or M} to machine precision at t=0.
        """
        from scipy.stats import norm as _norm
        p, b = self.params, self.bench
        if mode == "nash":
            lam, chi = b.lamN, b.chiN
        elif mode == "cartel":
            lam, chi = b.lamM, b.chiM
        else:
            raise ValueError(f"unknown mm_init {mode!r}")
        Tm = p.Tm
        if Tm % p.nv == 0:
            nu = Tm // p.nv
            j = np.arange(1, nu + 1)
            ug = _norm.ppf((2 * j - 1) / (2 * nu))
            ug = ug / np.sqrt(np.mean(ug * ug)) * p.sigma_u  # sample std == sigma_u
            v = np.repeat(b.vgrid, nu)                        # full factorial
            u = np.tile(ug, p.nv)
        else:  # fallback: random fill (approximate warm start)
            v = b.vgrid[self.rng.integers(0, p.nv, size=Tm)]
            u = self.rng.normal(0.0, p.sigma_u, size=Tm)
        v = np.broadcast_to(v, (self.batch, Tm))
        u = np.broadcast_to(u, (self.batch, Tm))
        y = p.I * chi * (v - p.vbar) + u
        pr = p.vbar + lam * y
        z = -p.xi * (pr - p.vbar)
        self._write_obs(slice(None), v, pr, z, y)

    def _write_obs(self, idx, v, pr, z, y):
        buf = self.buf
        buf[:, idx, _P] = pr
        buf[:, idx, _PP] = pr * pr
        buf[:, idx, _Z] = z
        buf[:, idx, _PZ] = pr * z
        buf[:, idx, _V] = v
        buf[:, idx, _Y] = y
        buf[:, idx, _YY] = y * y
        buf[:, idx, _VY] = v * y

    # ------------------------------------------------------------------
    def mm_coeffs(self):
        """(gamma0_hat, lambda_hat), each shape (batch,), from eq 4.1-4.2."""
        p = self.params
        n = float(p.Tm)
        S = self.S
        mp = S[:, _P] / n
        mz = S[:, _Z] / n
        mv = S[:, _V] / n
        my = S[:, _Y] / n
        var_p = S[:, _PP] / n - mp * mp
        cov_pz = S[:, _PZ] / n - mp * mz
        var_y = S[:, _YY] / n - my * my
        cov_vy = S[:, _VY] / n - mv * my
        # guard against degenerate windows (never triggered after warm start)
        var_p = np.maximum(var_p, 1e-300)
        var_y = np.maximum(var_y, 1e-300)
        xi1 = -cov_pz / var_p                # z = xi0 - xi1 p
        g1 = cov_vy / var_y                  # v = gamma0 + gamma1 y
        g0 = mv - g1 * my
        lam = (p.theta * g1 + xi1) / (p.theta + xi1 * xi1)
        return g0, lam

    # ------------------------------------------------------------------
    def step(self, x: np.ndarray, v: np.ndarray, u: np.ndarray,
             update_mm: bool = True):
        """One market period for all sessions.

        x : (batch, I) speculator order flows
        v : (batch,)   current fundamental values
        u : (batch,)   noise trader order flows

        Returns (p, pi, info): price (batch,), profits (batch, I), info dict
        with y, z, m, gamma0, lam_hat.
        """
        p = self.params
        y = x.sum(axis=1) + u
        g0, lam = self.mm_coeffs()
        price = g0 + lam * y
        z = -p.xi * (price - p.vbar)
        pi = (v - price)[:, None] * x

        if update_mm:
            self.S += -self.buf[:, self.ptr, :]
            self._write_obs(self.ptr, v, price, z, y)
            self.S += self.buf[:, self.ptr, :]
            self.ptr = (self.ptr + 1) % p.Tm
            self._pushes += 1
            if self._pushes % self.resync_every == 0:
                self.S = self.buf.sum(axis=1)  # cancel FP drift

        info = {"y": y, "z": z, "m": -(y + z), "gamma0": g0, "lam_hat": lam}
        return price, pi, info
