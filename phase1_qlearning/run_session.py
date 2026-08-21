"""Run a batch of independent Q-learning sessions to convergence (or a period
cap for POC runs) and dump results to an .npz file.

Per-period protocol (page 22-23, exact order):
  1. each speculator picks explore/exploit per eq 2.6 with eps_t(v_t) (eq 4.3)
     using current state s_t = (p_{t-1}, v_{t-1}, v_t), submits x_{i,t}
  2. noise trader draws u_t ~ N(0, sigma_u^2)
  3. market maker prices p_t = gamma0_hat + lambda_hat * y_t (eq 4.2)
  4. z_t = -xi (p_t - vbar); profits pi_{i,t} = (v_t - p_t) x_{i,t}
  5. state transitions to (p_t, v_t, v_{t+1}); Q updated at (s_t, x_{i,t})
     (eq 2.4); MM appends (v_t, p_t, z_t, y_t) to its rolling window

Usage:
  python -m phase1_qlearning.run_session --config configs/poc.yaml \
      --sigma-u 0.1 --seed 1 --out results/poc_su0.1_seed1.npz

Sweep configs (a top-level `sweep:` mapping of param -> list) are resolved
with --cell-index (row-major over the cartesian product) for SLURM arrays.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env.benchmarks import Params, compute_benchmarks
from env.market import MarketEnv
from env import metrics as M
from phase1_qlearning.convergence import ConvergenceTracker
from phase1_qlearning.qlearn import BatchQLearner

RUN_DEFAULTS = {
    "batch": 40,             # independent sessions simulated in one NumPy batch
    "max_periods": 2_000_000,  # 0 = no cap (run until all sessions converge)
    "chunk_size": 20_000,    # periods per pre-drawn random block
    "conv_streak": 1_000_000,  # paper criterion; POC configs shrink this
    "eval_periods": 100_000,  # measurement window T after Tc (OA section 4.1)
    "freeze_on_convergence": True,
    "mm_init": "nash",
}


# ---------------------------------------------------------------------------
def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("params", {})
    cfg["run"] = {**RUN_DEFAULTS, **cfg.get("run", {})}
    return cfg


def sweep_cells(cfg: dict) -> list[dict]:
    """Cartesian product of the `sweep:` lists (row-major), as param dicts."""
    sweep = cfg.get("sweep") or {}
    if not sweep:
        return [{}]
    keys = list(sweep.keys())
    return [dict(zip(keys, combo)) for combo in itertools.product(*sweep.values())]


def resolve_cell(cfg: dict, cell_index: int | None) -> dict:
    cells = sweep_cells(cfg)
    if cell_index is None:
        if len(cells) > 1:
            raise SystemExit(f"config has {len(cells)} sweep cells; pass --cell-index")
        cell_index = 0
    cell = cells[cell_index]
    out = json.loads(json.dumps(cfg))  # deep copy
    out["params"].update(cell)
    out["cell"] = {"index": cell_index, "overrides": cell, "n_cells": len(cells)}
    out.pop("sweep", None)
    return out


# ---------------------------------------------------------------------------
def evaluate(params, bench, env, learner, state, rng, T_eval: int,
             update_mm: bool = True):
    """Measurement window (OA section 4.1): T_eval periods with frozen greedy
    policies (no exploration, no Q updates). The MM keeps adapting by default,
    matching the paper's post-convergence measurement environment.

    Computes the exact OA eq IA.4.1-IA.4.7 measures, including the
    matched-path Delta^C: pi^N_t / pi^M_t are re-scored on the SAME realized
    (v_t, u_t) path via Benchmarks.path_profits.
    Returns (metrics dict, final state).
    """
    batch = learner.batch
    p_idx, vlag_idx, v_idx = state
    P = np.empty((T_eval, batch))
    V = np.empty((T_eval, batch))
    X = np.empty((T_eval, batch, params.I))
    U = np.empty((T_eval, batch))
    LH = np.empty((T_eval, batch))
    G0 = np.empty((T_eval, batch))
    Pi = np.zeros((batch, params.I))
    piN_acc = np.zeros(batch)
    piM_acc = np.zeros(batch)
    no_explore = np.zeros(batch, dtype=bool)
    dummy_u01 = np.ones((batch, params.I))       # never < eps once masked off
    dummy_a = np.zeros((batch, params.I), dtype=np.int64)

    for t in range(T_eval):
        s = learner.state_index(p_idx, vlag_idx, v_idx)
        a = learner.act(s, v_idx, dummy_u01, dummy_a, explore_mask=no_explore)
        v = bench.vgrid[v_idx]
        x = (v - params.vbar)[:, None] * bench.c_grid[a]
        u = rng.normal(0.0, params.sigma_u, size=batch)
        price, pi, info = env.step(x, v, u, update_mm=update_mm)
        P[t], V[t], X[t], U[t] = price, v, x, u
        G0[t], LH[t] = info["gamma0"], info["lam_hat"]
        Pi += pi
        piN_acc += bench.path_profits(v, u, "nash")    # eq IA.4.2
        piM_acc += bench.path_profits(v, u, "cartel")  # eq IA.4.3
        p_idx = bench.nearest_p_idx(price)
        vlag_idx = v_idx
        v_idx = rng.integers(0, params.nv, size=batch)

    pi_bar_i = Pi / T_eval                        # (batch, I)
    piN_bar, piM_bar = piN_acc / T_eval, piM_acc / T_eval
    chi_c, chi_i1 = M.chi_hat(X, V)               # eq IA.4.4
    metrics = {
        "pi_mean": pi_bar_i,
        "piN_bar": piN_bar,
        "piM_bar": piM_bar,
        "delta_c": M.delta_c_matched(pi_bar_i, piN_bar, piM_bar),  # IA.4.1
        "profit_gain": M.relative_profit_gain(pi_bar_i, piN_bar),
        "chi_hat": chi_c,
        "chi_hat_i": chi_i1,
        "informativeness": M.informativeness(chi_c, params.I, bench.sv_hat,
                                             params.sigma_u),      # IA.4.5
        "informativeness_var": M.informativeness_var(X.sum(axis=2), U),
        "liquidity": M.liquidity(LH, params.xi),                   # IA.4.6
        "mispricing": M.mispricing(LH, chi_c, V, params.I, params.vbar),  # IA.4.7
        "mean_price": P.mean(axis=0),
        "gamma0_final": G0[-1],
        "lam_hat_final": LH[-1],
    }
    return metrics, (p_idx, vlag_idx, v_idx)


# ---------------------------------------------------------------------------
def run(cfg: dict, seed: int, out_path: str | None = None, quiet: bool = False,
        backend: str = "numpy"):
    """backend: "numpy" (default, reference implementation, runs anywhere) or
    "numba" (JIT-compiled hot loop; requires numba, e.g. .venv310 -- see
    README "Numba backend" section). Both backends consume the SAME rng
    calls in the same order per chunk, so outputs are directly comparable
    (tests/test_numba_parity.py asserts bit-for-bit equality on a toy cell).
    """
    if backend not in ("numpy", "numba"):
        raise ValueError(f"unknown backend {backend!r}")
    params = Params.from_dict(cfg["params"])
    rcfg = cfg["run"]
    batch = int(rcfg["batch"])
    rng = np.random.default_rng(seed)

    bench = compute_benchmarks(params)
    env = MarketEnv(params, bench, batch=batch, rng=rng, mm_init=rcfg["mm_init"])
    learner = BatchQLearner(params, bench, batch=batch)
    tracker = ConvergenceTracker(batch, streak_target=int(rcfg["conv_streak"]))

    if backend == "numba":
        from phase1_qlearning.qlearn_numba import run_chunk_numba

    # initial state: uniform over P x V x V (page 25)
    p_idx = rng.integers(0, params.np_, size=batch)
    vlag_idx = rng.integers(0, params.nv, size=batch)
    v_idx = rng.integers(0, params.nv, size=batch)

    max_periods = int(rcfg["max_periods"])
    chunk = int(rcfg["chunk_size"])
    freeze = bool(rcfg["freeze_on_convergence"])
    curve = []          # per-chunk mean per-speculator profit, (batch,)
    t0 = time.time()
    t = 0
    while (max_periods <= 0 or t < max_periods) and not tracker.all_converged:
        n = chunk if max_periods <= 0 else min(chunk, max_periods - t)
        v_next_blk = rng.integers(0, params.nv, size=(n, batch))
        u_blk = rng.normal(0.0, params.sigma_u, size=(n, batch))
        u01_blk = rng.random(size=(n, batch, params.I))
        ra_blk = rng.integers(0, params.nx, size=(n, batch, params.I))

        if backend == "numba":
            pi_acc = run_chunk_numba(params, bench, env, learner, tracker,
                                      (p_idx, vlag_idx, v_idx),
                                      v_next_blk, u_blk, u01_blk, ra_blk,
                                      freeze)
        else:
            pi_acc = np.zeros(batch)
            for k in range(n):
                active = ~tracker.converged if freeze else None
                s = learner.state_index(p_idx, vlag_idx, v_idx)
                a = learner.act(s, v_idx, u01_blk[k], ra_blk[k],
                                explore_mask=active)
                v = bench.vgrid[v_idx]
                x = (v - params.vbar)[:, None] * bench.c_grid[a]
                price, pi, _ = env.step(x, v, u_blk[k])
                p_idx_next = bench.nearest_p_idx(price)
                v_next = v_next_blk[k]
                s_next = learner.state_index(p_idx_next, v_idx, v_next)
                changed = learner.update(s, a, pi, s_next, active=active)
                tracker.update(changed)
                pi_acc += pi.mean(axis=1)
                p_idx, vlag_idx, v_idx = p_idx_next, v_idx, v_next
        t += n
        curve.append(pi_acc / n)
        if not quiet:
            eps = learner.epsilon(v_idx).mean()
            dc = (curve[-1].mean() - bench.piN) / (bench.piM - bench.piN)
            print(f"t={t:>10,}  eps~{eps:.3f}  conv {tracker.converged.sum():>3}"
                  f"/{batch}  block dC={dc:+.3f}  "
                  f"[{time.time()-t0:,.0f}s]", flush=True)

    metrics, final_state = evaluate(params, bench, env, learner,
                                    (p_idx, vlag_idx, v_idx), rng,
                                    int(rcfg["eval_periods"]))
    conv = tracker.summary()
    if not quiet:
        print(f"done: {conv['n_converged']}/{batch} converged "
              f"(target streak {conv['streak_target']:,}); "
              f"eval dC mean={metrics['delta_c'].mean():+.3f} "
              f"median={np.median(metrics['delta_c']):+.3f}")

    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        np.savez_compressed(
            out_path,
            config=json.dumps(cfg), seed=seed,
            sigma_u=params.sigma_u, I=params.I, rho=params.rho, xi=params.xi,
            lamN=bench.lamN, lamM=bench.lamM, chiN=bench.chiN, chiM=bench.chiM,
            piN=bench.piN, piM=bench.piM, sv_hat=bench.sv_hat,
            vgrid=bench.vgrid, c_grid=bench.c_grid, p_grid=bench.p_grid,
            Q=learner.Q.astype(np.float32), pol=learner.pol,
            visits=learner.visits,
            periods_run=conv["periods_run"],
            converged_at=conv["converged_at"], best_streak=conv["best_streak"],
            streak_target=conv["streak_target"],
            learn_curve=np.array(curve),
            pi_mean=metrics["pi_mean"], delta_c=metrics["delta_c"],
            piN_bar=metrics["piN_bar"], piM_bar=metrics["piM_bar"],
            profit_gain=metrics["profit_gain"],
            chi_hat=metrics["chi_hat"], chi_hat_i=metrics["chi_hat_i"],
            informativeness=metrics["informativeness"],
            informativeness_var=metrics["informativeness_var"],
            liquidity=metrics["liquidity"], mispricing=metrics["mispricing"],
            mean_price=metrics["mean_price"],
            gamma0=metrics["gamma0_final"], lam_hat=metrics["lam_hat_final"],
            state_p_idx=final_state[0], state_vlag_idx=final_state[1],
            state_v_idx=final_state[2],
        )
        if not quiet:
            print(f"saved -> {out_path}")
    return metrics, conv, (params, bench, env, learner, final_state)


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--cell-index", type=int, default=None,
                    help="index into the config's sweep cartesian product")
    ap.add_argument("--sigma-u", type=float, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--max-periods", type=int, default=None)
    ap.add_argument("--beta", type=float, default=None)
    ap.add_argument("--conv-streak", type=int, default=None)
    ap.add_argument("--eval-periods", type=int, default=None)
    ap.add_argument("--backend", choices=["numpy", "numba"], default="numpy",
                    help="numpy = pure-NumPy reference (runs on any Python; "
                         "default). numba = JIT-compiled hot loop, ~100x "
                         "faster per session but requires numba (e.g. this "
                         "machine's .venv310 -- see README 'Numba backend').")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    cfg = resolve_cell(load_config(args.config), args.cell_index)
    if args.sigma_u is not None:
        cfg["params"]["sigma_u"] = args.sigma_u
    if args.beta is not None:
        cfg["params"]["beta"] = args.beta
    for k in ("batch", "max_periods", "conv_streak", "eval_periods"):
        v = getattr(args, k)
        if v is not None:
            cfg["run"][k] = v
    run(cfg, seed=args.seed, out_path=args.out, quiet=args.quiet,
        backend=args.backend)


if __name__ == "__main__":
    main()
