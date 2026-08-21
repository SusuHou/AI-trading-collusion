"""Theoretical benchmarks for Dou, Goldstein & Ji (2025), NBER WP 34054.

Implements, per docs/paper_spec.md:
  - the v grid (Gaussian-quantile discretization, page 24) and sigma_v_hat (footnote 24),
  - the lambda^N / lambda^M price-impact fixed points (spec section
    "Derived: lambda^N, lambda^M fixed points" -- a 1-D root-find on
        lambda = xi/(xi^2+theta) + [theta/(xi^2+theta)] * Omega*sv2 / (Omega^2*sv2 + su2)
    with Omega_N(lambda) = I/[(I+1) lambda] (Nash) and Omega_M(lambda) = I/(2 lambda)
    (cartel)),
  - chi^N = 1/[(I+1) lambda^N], chi^M = 1/(2 I lambda^M) (page 16-17; the algebraic
    identities chi^N*(I+1)*lambda^N = 1 and chi^M*2*I*lambda^M = 1 hold exactly),
  - expected per-period per-speculator profits pi^N, pi^M under each benchmark,
  - the action grid X (state-scaled: x = (v - vbar) * c_j, page 24) and price grid P,
  - the initial Q-matrix formula (page 25).

Everything here is solved once per parameter cell and cached in a `Benchmarks`
dataclass -- never re-solved inside the simulation loop.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Parameters (baseline values from paper_spec.md notation table / section 4.2)
# ---------------------------------------------------------------------------
@dataclass
class Params:
    I: int = 2               # number of informed speculators
    vbar: float = 1.0        # mean of fundamental value
    sigma_v: float = 1.0     # nominal std of v_t (grid built from this)
    sigma_u: float = 0.1     # std of noise trader order flow
    xi: float = 500.0        # info-insensitive investor demand slope (eq 3.2)
    theta: float = 0.1       # market maker's pricing-error weight (eq 3.3), fixed
    rho: float = 0.95        # speculators' subjective discount factor
    alpha: float = 0.01      # Q-learning rate (eq 2.4)
    beta: float = 5e-7       # exploration decay hyperparameter (eq 4.3)
    iota: float = 0.1        # grid-widening parameter
    nv: int = 10             # v grid size
    nx: int = 15             # action grid size
    np_: int = 31            # price grid size ("np" clashes with numpy)
    Tm: int = 10_000         # market maker rolling-window length

    @classmethod
    def from_dict(cls, d: dict) -> "Params":
        d = dict(d)
        if "np" in d:  # allow "np" key in YAML configs
            d["np_"] = d.pop("np")
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
# Discretization of v (page 24)
# ---------------------------------------------------------------------------
def v_grid(vbar: float, sigma_v: float, nv: int) -> np.ndarray:
    """v_k = vbar + sigma_v * Phi^{-1}((2k-1)/(2 nv)), k = 1..nv.

    Equal probability mass on each grid point.
    """
    k = np.arange(1, nv + 1)
    return vbar + sigma_v * norm.ppf((2 * k - 1) / (2 * nv))


def sigma_v_hat(vgrid: np.ndarray, vbar: float) -> float:
    """Footnote 24: sigma_v_hat = sqrt(nv^{-1} sum_k (v_k - vbar)^2).

    Used instead of the nominal sigma_v in all benchmark formulas.
    (= 0.938 at nv = 10, sigma_v = 1.)
    """
    return float(np.sqrt(np.mean((vgrid - vbar) ** 2)))


# ---------------------------------------------------------------------------
# lambda fixed points (spec: "Derived: lambda^N, lambda^M fixed points")
# ---------------------------------------------------------------------------
def _lambda_rhs(lam: float, omega_of_lam, xi: float, theta: float,
                sv2: float, su2: float) -> float:
    """RHS(lambda) of the fixed-point equation (eq 3.4 + Gaussian projection)."""
    omega = omega_of_lam(lam)
    proj = omega * sv2 / (omega * omega * sv2 + su2)   # E[v|y] slope
    denom = xi * xi + theta
    return xi / denom + (theta / denom) * proj


def solve_lambda(I: int, xi: float, theta: float, sigma_u: float,
                 sv_hat: float, mode: str) -> float:
    """Solve lambda = RHS(lambda) by brentq for mode in {"nash", "cartel"}.

    Omega(lambda) = I * chi(lambda) is the aggregate informed sensitivity
    (y = Omega (v - vbar) + u):
      Nash:   Omega_N = I chi^N = I / [(I+1) lambda]
      cartel: Omega_M = I chi^M = 1 / (2 lambda)
    Note: paper_spec.md's parenthetical "Omega_M = I/(2 lambda^M)" carries a
    spurious factor of I, contradicting the spec's own Omega = I*chi rule and
    chi^M = 1/(2 I lambda^M); we use the self-consistent Omega_M = 1/(2 lambda).
    """
    if mode == "nash":
        omega_of_lam = lambda lam: I / ((I + 1) * lam)
    elif mode == "cartel":
        omega_of_lam = lambda lam: 1.0 / (2.0 * lam)
    else:
        raise ValueError(f"unknown mode {mode!r}")

    sv2, su2 = sv_hat ** 2, sigma_u ** 2
    f = lambda lam: lam - _lambda_rhs(lam, omega_of_lam, xi, theta, sv2, su2)

    denom = xi * xi + theta
    # RHS is bounded above by xi/denom + (theta/denom) * sv_hat/(2 sigma_u)
    # (max of the projection term over Omega), so this bracket always works.
    hi = xi / denom + (theta / denom) * sv_hat / (2.0 * sigma_u) + 1.0
    lo = 1e-12
    assert f(lo) < 0 < f(hi), "fixed-point bracket failed"
    lam = float(brentq(f, lo, hi, xtol=1e-15, rtol=8.9e-16, maxiter=200))
    # Polish: RHS is a strong contraction near the root (|RHS'| << 1), so a few
    # fixed-point iterations drive the residual to machine precision.
    for _ in range(5):
        lam = _lambda_rhs(lam, omega_of_lam, xi, theta, sv2, su2)
    return lam


# ---------------------------------------------------------------------------
# Full benchmark bundle
# ---------------------------------------------------------------------------
@dataclass
class Benchmarks:
    params: Params
    vgrid: np.ndarray
    sv_hat: float
    lamN: float
    lamM: float
    chiN: float
    chiM: float
    piN: float        # expected per-period per-speculator Nash profit
    piM: float        # expected per-period per-speculator cartel profit
    c_grid: np.ndarray   # relative action multipliers, len nx: x = (v-vbar)*c_j
    p_grid: np.ndarray   # price grid, len np_

    # -- helpers ----------------------------------------------------------
    def x_values(self, v: np.ndarray) -> np.ndarray:
        """Action values for fundamental v: shape v.shape + (nx,)."""
        return (np.asarray(v)[..., None] - self.params.vbar) * self.c_grid

    def price_nash(self, y):
        return self.params.vbar + self.lamN * np.asarray(y)

    def price_cartel(self, y):
        return self.params.vbar + self.lamM * np.asarray(y)

    def path_profits(self, v: np.ndarray, u: np.ndarray, mode: str) -> np.ndarray:
        """Matched-path benchmark profits (OA eq IA.4.2 / IA.4.3).

        Per-period profit a Nash ("nash") or cartel ("cartel") benchmark
        speculator would earn on the SAME realized (v_t, u_t) path:
            pi_t = [v_t - vbar - lam (I chi (v_t - vbar) + u_t)] * chi (v_t - vbar)
        v, u broadcastable arrays; returns array of the broadcast shape.
        """
        if mode == "nash":
            lam, chi = self.lamN, self.chiN
        elif mode == "cartel":
            lam, chi = self.lamM, self.chiM
        else:
            raise ValueError(f"unknown mode {mode!r}")
        p = self.params
        dv = np.asarray(v) - p.vbar
        x = chi * dv
        price = p.vbar + lam * (p.I * x + np.asarray(u))
        return (np.asarray(v) - price) * x

    def nearest_p_idx(self, p: np.ndarray) -> np.ndarray:
        """Index of nearest price-grid point (grid is uniform)."""
        g = self.p_grid
        dp = g[1] - g[0]
        idx = np.rint((np.asarray(p) - g[0]) / dp).astype(np.int64)
        return np.clip(idx, 0, len(g) - 1)


def compute_benchmarks(params: Params) -> Benchmarks:
    p = params
    vg = v_grid(p.vbar, p.sigma_v, p.nv)
    svh = sigma_v_hat(vg, p.vbar)

    lamN = solve_lambda(p.I, p.xi, p.theta, p.sigma_u, svh, "nash")
    lamM = solve_lambda(p.I, p.xi, p.theta, p.sigma_u, svh, "cartel")
    chiN = 1.0 / ((p.I + 1) * lamN)
    chiM = 1.0 / (2.0 * p.I * lamM)

    # Expected per-period per-speculator profit with x_i = chi (v - vbar),
    # y = I chi (v - vbar) + u, p = vbar + lam y:
    #   E[(v - p) x_i] = chi (1 - lam I chi) sigma_v_hat^2
    sv2 = svh ** 2
    piN = chiN * (1.0 - lamN * p.I * chiN) * sv2   # = sv2 / ((I+1)^2 lamN)
    piM = chiM * (1.0 - lamM * p.I * chiM) * sv2   # = sv2 / (4 I lamM)

    # Action grid (page 24). x^N(v) = chiN (v-vbar), x^M(v) = chiM (v-vbar);
    # for v > vbar the interval is [x^M - iota (x^N - x^M), x^N + iota (x^N - x^M)]
    # and for v < vbar the mirrored one -- both equal (v - vbar) * [c_lo, c_hi]:
    d = chiN - chiM
    c_lo = chiM - p.iota * d
    c_hi = chiN + p.iota * d
    c = np.linspace(c_lo, c_hi, p.nx)

    # Price grid (page 24): bounds from the largest attainable informed flow
    # (+/- 1.96 sigma_u noise band), widened by iota.
    x_extreme = max(chiN, chiM) * float(np.max(np.abs(vg - p.vbar)))
    pH = p.vbar + lamN * (p.I * x_extreme + 1.96 * p.sigma_u)
    pL = p.vbar + lamN * (-p.I * x_extreme - 1.96 * p.sigma_u)
    lo = pL - p.iota * (pH - pL)
    hi = pH + p.iota * (pH - pL)
    pg = np.linspace(lo, hi, p.np_)

    return Benchmarks(params=p, vgrid=vg, sv_hat=svh, lamN=lamN, lamM=lamM,
                      chiN=chiN, chiM=chiM, piN=piN, piM=piM,
                      c_grid=c, p_grid=pg)


# ---------------------------------------------------------------------------
# Initial Q-matrix (page 25)
# ---------------------------------------------------------------------------
def q0_table(bench: Benchmarks) -> np.ndarray:
    """Initial Q values, shape (nv, nx): rows indexed by current-v grid point.

    Q_{i,0}(s, x) = 1/((1-rho) nx) * sum_{x_-i in X}
                        [ v - (vbar + lamN (x + (I-1) x_-i)) ] * x
    The RHS depends on s only through its current-v component; run code
    replicates each row across all (p_lag, v_lag) combinations.
    """
    p = bench.params
    dv = bench.vgrid - p.vbar                    # (nv,)
    xv = dv[:, None] * bench.c_grid[None, :]     # (nv, nx): own action x
    xo = dv[:, None] * bench.c_grid[None, :]     # (nv, nx): opponent action x_-i
    # payoff[v, j, m] = [dv - lamN (x_j + (I-1) x_m)] * x_j
    pay = (dv[:, None, None]
           - bench.lamN * (xv[:, :, None] + (p.I - 1) * xo[:, None, :])) \
        * xv[:, :, None]
    return pay.sum(axis=2) / ((1.0 - p.rho) * p.nx)
