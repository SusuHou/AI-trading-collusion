"""Simulation measures, exact Online Appendix section 4.1 formulas
(eq IA.4.1-IA.4.7), plus eq 3.6 / Definition 3.4 of the main text.

All measures are computed per session over the T-period measurement window
that starts at the session's convergence period Tc (paper: T = 100,000).
Batched series are time-major: shape (T, batch) or (T, batch, I).
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Delta^C -- matched-path comparison (eq IA.4.1-IA.4.3)
# ---------------------------------------------------------------------------
def delta_c_matched(pi_bar_i: np.ndarray, piN_bar: np.ndarray,
                    piM_bar: np.ndarray) -> np.ndarray:
    """Normalized profitability from matched-path window means.

    pi_bar_i : (batch, I) realized mean per-period profit of each speculator
    piN_bar  : (batch,) window mean of pi^N_t scored on the SAME (v_t, u_t)
               path via Benchmarks.path_profits (NOT the population mean)
    piM_bar  : (batch,) same for the cartel benchmark

    Delta^C_i = (pi_bar_i - piN_bar) / (piM_bar - piN_bar);
    Delta^C = mean_i Delta^C_i. Can fall below 0 if learning failed (OA p.52).
    """
    dci = (pi_bar_i - piN_bar[:, None]) / (piM_bar - piN_bar)[:, None]
    return dci.mean(axis=1)


def relative_profit_gain(pi_bar_i: np.ndarray, piN_bar: np.ndarray) -> np.ndarray:
    """sum_i pi_i / sum_i pi^N_i (OA 'Profit Gain Relative to Noncollusion')."""
    I = pi_bar_i.shape[1]
    return pi_bar_i.sum(axis=1) / (I * piN_bar)


# ---------------------------------------------------------------------------
# Trading policy estimate chi_hat (eq IA.4.4)
# ---------------------------------------------------------------------------
def chi_hat(x: np.ndarray, v: np.ndarray):
    """OLS x_{i,t} = chi_0 + chi_1 v_t + eps per speculator (eq IA.4.4).

    x : (T, batch, I) order flows;  v : (T, batch) fundamentals.
    Returns (chi_c, chi_i1): chi_c = mean_i chi_hat_{i,1} of shape (batch,),
    chi_i1 of shape (batch, I).
    """
    v = v[:, :, None]
    mv = v.mean(axis=0)
    mx = x.mean(axis=0)
    var_v = ((v - mv) ** 2).mean(axis=0)
    cov = ((v - mv) * (x - mx)).mean(axis=0)
    chi_i1 = cov / var_v                       # (batch, I)
    return chi_i1.mean(axis=1), chi_i1


# ---------------------------------------------------------------------------
# Price informativeness (eq IA.4.5)
# ---------------------------------------------------------------------------
def informativeness(chi_c: np.ndarray, I: int, sv_hat: float,
                    sigma_u: float) -> np.ndarray:
    """I^C = (I chi_hat^C)^2 (sigma_v_hat / sigma_u)^2."""
    return (I * chi_c) ** 2 * (sv_hat / sigma_u) ** 2


def informativeness_var(x_agg: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Direct var(x_t)/var(u_t) (Def 3.4) -- consistency check for IA.4.5."""
    return np.var(x_agg, axis=0) / np.var(u, axis=0)


# ---------------------------------------------------------------------------
# Market liquidity (eq IA.4.6)
# ---------------------------------------------------------------------------
def liquidity(lam_hat: np.ndarray, xi: float) -> np.ndarray:
    """L^C = mean_t 1 / |1 - xi lambda_hat_t|; lam_hat shape (T, batch)."""
    return (1.0 / np.abs(1.0 - xi * lam_hat)).mean(axis=0)


# ---------------------------------------------------------------------------
# Mispricing (eq IA.4.7)
# ---------------------------------------------------------------------------
def mispricing(lam_hat: np.ndarray, chi_c: np.ndarray, v: np.ndarray,
               I: int, vbar: float) -> np.ndarray:
    """E^C = mean_t |1 - lambda_hat_t I chi_hat^C| |v_t - vbar|.

    lam_hat, v : (T, batch);  chi_c : (batch,).
    """
    e_t = np.abs(1.0 - lam_hat * I * chi_c[None, :]) * np.abs(v - vbar)
    return e_t.mean(axis=0)
