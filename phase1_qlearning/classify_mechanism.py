"""Mechanism classification via calibrated impulse responses.

EXACT protocol from Online Appendix section 4.5 (page 62), per session:

1. Start from the session's converged steady state (frozen greedy policies).
2. Run an IRF window t = 1..9. At t = 3 inject an exogenous noise-trader
   shock u_shock, calibrated so the period-3 price deviation p_tilde_3 is
   exactly 1.2% (the paper's "medium deviation"; p_tilde is sign-adjusted by
   sgn(v_t - vbar) so a positive deviation means "price pushed away from vbar
   in the direction of the fundamental").
3. Measure each speculator's order-flow deviation at t = 4:
       x_tilde_{i,4} = (x_{i,4} - E[x_{i,4}]) / E[x_{i,4}]
4. Classify:
       price-trigger:  x_tilde_{i,4} > x_high      for BOTH i
       over-pruning:   |x_tilde_{i,4}| < x_low     for BOTH i
       unclassified:   otherwise
   OA thresholds: x_low = 5e-5, x_high = 10 * x_low = 5e-4 (both exposed as
   CLI parameters; the 10x relation is stated in the OA text).

Implementation notes (documented judgment calls):
- Expectations are estimated with paired common-random-number rollouts: a
  base path and a shock path share every (v_t, u_t) draw and differ only by
  u_shock at t = 3, replicated R times per session. Under CRN a policy that
  ignores the price state yields x_tilde = 0 exactly, which is what makes
  the tiny x_low = 5e-5 threshold meaningful.
- Order flows are sign-adjusted by sgn(v_t - vbar) before averaging (their
  unconditional mean is ~0 across v signs; the paper's E[x] is the long-run
  mean of the sign-adjusted flow, cf. Fig 3-6 percentage deviations).
- The market maker's pricing coefficients (gamma0_hat, lambda_hat) are held
  fixed at their end-of-session values during the 9-period IRF window (over
  9 periods a Tm = 10,000 rolling window would move them negligibly).
- u_shock = 0.012 * p_bar / lambda_hat * sgn(v_3 - vbar), which makes the
  paired price deviation at t = 3 exactly 1.2% of the long-run mean price
  p_bar (estimated from the burn-in window).

Usage:
  python -m phase1_qlearning.classify_mechanism results/poc_su0.1_seed11.npz \
      [more.npz ...] [--reps 1024] [--x-low 5e-5] [--x-high 5e-4]
Writes <input>_irf.npz alongside each input and prints a summary.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env.benchmarks import Params, compute_benchmarks

LABELS = {0: "over_pruning", 1: "price_trigger", 2: "unclassified"}


# ---------------------------------------------------------------------------
def _greedy_actions(pol, b_of, s):
    """pol: (batch, I, nS) int8; b_of, s: (B,) -> (B, I)."""
    return pol[b_of[:, None], np.arange(pol.shape[1])[None, :], s[:, None]]


def run_irf(npz_path: str, reps: int = 1024, burn: int = 50, T_irf: int = 9,
            shock_t: int = 3, target_dev: float = 0.012,
            x_low: float = 5e-5, x_high: float = 5e-4, seed: int = 0):
    d = np.load(npz_path, allow_pickle=False)
    cfg = json.loads(str(d["config"]))
    params = Params.from_dict(cfg["params"])
    bench = compute_benchmarks(params)
    pol = d["pol"]                       # (batch, I, nS)
    batch = pol.shape[0]
    g0_s, lh_s = d["gamma0"], d["lam_hat"]   # (batch,)

    rng = np.random.default_rng(seed)
    B = batch * reps
    b_of = np.repeat(np.arange(batch), reps)
    g0, lh = g0_s[b_of], lh_s[b_of]

    def nv_idx(n):
        return rng.integers(0, params.nv, size=n)

    def step(p_idx, vlag_idx, v_idx, u):
        """One frozen-policy, frozen-MM period. Returns (price, x, p_idx')."""
        s = (p_idx * params.nv + vlag_idx) * params.nv + v_idx
        a = _greedy_actions(pol, b_of, s)
        v = bench.vgrid[v_idx]
        x = (v - params.vbar)[:, None] * bench.c_grid[a]
        y = x.sum(axis=1) + u
        price = g0 + lh * y
        return price, x, bench.nearest_p_idx(price)

    # start every replication from its session's saved final state
    p_idx = d["state_p_idx"][b_of]
    vlag_idx = d["state_vlag_idx"][b_of]
    v_idx = d["state_v_idx"][b_of]

    # burn-in to steady state; estimate long-run mean price p_bar per session
    pbar_acc = np.zeros(B)
    for _ in range(burn):
        u = rng.normal(0.0, params.sigma_u, size=B)
        price, _, p_next = step(p_idx, vlag_idx, v_idx, u)
        pbar_acc += price
        p_idx, vlag_idx, v_idx = p_next, v_idx, nv_idx(B)
    p_bar = (pbar_acc / burn).reshape(batch, reps).mean(axis=1)  # (batch,)

    # paired IRF window: base and shock paths share all draws (CRN)
    st_b = (p_idx.copy(), vlag_idx.copy(), v_idx.copy())
    st_s = (p_idx.copy(), vlag_idx.copy(), v_idx.copy())
    P_b = np.empty((T_irf + 1, B)); P_s = np.empty((T_irf + 1, B))
    X_b = np.empty((T_irf + 1, B, params.I)); X_s = np.empty((T_irf + 1, B, params.I))
    SGN = np.empty((T_irf + 1, B))
    shock_sign = np.empty(B)

    for t in range(1, T_irf + 1):
        u = rng.normal(0.0, params.sigma_u, size=B)
        v_next = nv_idx(B)
        # both paths see the same v_t by construction of CRN (v is exogenous)
        assert np.array_equal(st_b[2], st_s[2])
        dv_sign = np.where(bench.vgrid[st_b[2]] >= params.vbar, 1.0, -1.0)
        SGN[t] = dv_sign
        u_s = u
        if t == shock_t:
            shock_sign = dv_sign
            u_shock = target_dev * p_bar[b_of] / lh * dv_sign
            u_s = u + u_shock
        pb, xb, pib = step(*st_b, u)
        ps, xs, pis = step(*st_s, u_s)
        P_b[t], X_b[t], P_s[t], X_s[t] = pb, xb, ps, xs
        st_b = (pib, st_b[2], v_next)
        st_s = (pis, st_s[2], v_next)

    # ---- deviations, averaged over replications ---------------------------
    shock_sign_r = shock_sign.reshape(batch, reps)

    # price IRF: paired difference, sign-adjusted by the shock sign, % of p_bar
    dp = (P_s - P_b).reshape(T_irf + 1, batch, reps) * shock_sign_r[None]
    p_tilde = dp.mean(axis=2) / p_bar[None, :]           # (T+1, batch)

    # order-flow IRF: sign-adjust by current sgn(v_t - vbar), normalize by the
    # base-path mean level E[x_hat] (long-run mean of the sign-adjusted flow)
    xh_b = X_b * SGN[:, :, None]
    xh_s = X_s * SGN[:, :, None]
    xh_b_r = xh_b.reshape(T_irf + 1, batch, reps, params.I)
    xh_s_r = xh_s.reshape(T_irf + 1, batch, reps, params.I)
    base_level = xh_b_r.mean(axis=(0, 2))                # (batch, I)
    base_level = np.where(np.abs(base_level) < 1e-12, np.nan, base_level)
    x_tilde = (xh_s_r - xh_b_r).mean(axis=2) / base_level[None]  # (T+1, batch, I)

    x4 = x_tilde[shock_t + 1]                            # (batch, I)
    price_trigger = (x4 > x_high).all(axis=1)
    over_pruning = (np.abs(x4) < x_low).all(axis=1)
    label = np.full(batch, 2, dtype=np.int64)
    label[over_pruning] = 0
    label[price_trigger] = 1                              # trigger wins if both

    out_path = npz_path.replace(".npz", "_irf.npz")
    np.savez_compressed(
        out_path, source=npz_path, sigma_u=params.sigma_u,
        p_tilde=p_tilde, x_tilde=x_tilde, x_tilde4=x4, label=label,
        p_bar=p_bar, reps=reps, burn=burn, shock_t=shock_t,
        target_dev=target_dev, x_low=x_low, x_high=x_high, seed=seed,
    )
    return out_path, label, x4, p_tilde


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="OA 4.5 IRF mechanism classifier")
    ap.add_argument("results", nargs="+", help="run_session .npz output files")
    ap.add_argument("--reps", type=int, default=1024)
    ap.add_argument("--burn", type=int, default=50)
    ap.add_argument("--x-low", type=float, default=5e-5)
    ap.add_argument("--x-high", type=float, default=5e-4)
    ap.add_argument("--target-dev", type=float, default=0.012)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    for path in args.results:
        out, label, x4, p_tilde = run_irf(
            path, reps=args.reps, burn=args.burn, x_low=args.x_low,
            x_high=args.x_high, target_dev=args.target_dev, seed=args.seed)
        n = len(label)
        counts = {name: int((label == code).sum()) for code, name in LABELS.items()}
        print(f"{os.path.basename(path)}: sessions={n}  "
              f"price_trigger={counts['price_trigger']}  "
              f"over_pruning={counts['over_pruning']}  "
              f"unclassified={counts['unclassified']}")
        print(f"  p_tilde@t3 = {np.nanmean(p_tilde[3]):+.4%} (calibration check, "
              f"target +1.2000%)")
        print(f"  x_tilde@t4: median {np.nanmedian(x4):+.3e}, "
              f"p90 {np.nanpercentile(x4, 90):+.3e}  -> {out}")


if __name__ == "__main__":
    main()
