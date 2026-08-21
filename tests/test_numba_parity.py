"""Correctness gate for the Numba backend (phase1_qlearning/qlearn_numba.py).

Runs the SAME toy config, seed, and rng call sequence through
run_session.run(..., backend="numpy") and run(..., backend="numba"), and
asserts the two produce numerically indistinguishable trajectories: the
Q-table, greedy-policy cache, visit counters, convergence bookkeeping (per-
session converged_at / best_streak), and the post-training evaluation
metrics.

Requires numba, which the project's default .venv (Python 3.14) does not
have -- run this file under .venv310 (see README "Numba backend"):

    .venv310/Scripts/python -m pytest tests/test_numba_parity.py -q

`pytest.importorskip` makes this a silent skip (not a failure) when numba
is unavailable, so `pytest tests/` continues to pass unmodified in the
default .venv.
"""
from __future__ import annotations

import copy

import numpy as np
import pytest

numba = pytest.importorskip("numba")

from phase1_qlearning.run_session import run

# Small toy config: short horizon, tiny grids, a convergence streak short
# enough that some sessions actually converge and exercise the "frozen
# session" (active=False) code path in both backends.
TOY_CFG = {
    "params": {
        "I": 2, "vbar": 1.0, "sigma_v": 1.0, "sigma_u": 0.5, "xi": 500.0,
        "theta": 0.1, "rho": 0.9, "alpha": 0.05, "beta": 2e-3, "iota": 0.1,
        "nv": 4, "nx": 5, "np": 9, "Tm": 200,
    },
    "run": {
        "batch": 6, "max_periods": 20_000, "chunk_size": 1_000,
        "conv_streak": 25, "eval_periods": 500,
        "freeze_on_convergence": True, "mm_init": "nash",
    },
}

# Same grid sizes as configs/baseline.yaml (nv=10, nx=15, np=31, Tm=10000,
# I=2, xi=500) -- the config the throughput benchmark in the README was
# measured on -- but with alpha/beta/conv_streak scaled up (like
# configs/poc.yaml does) so a short test still reaches the "some sessions
# converged" state within a few thousand periods instead of the tens of
# millions the paper's own beta=5e-7 would need.
BASELINE_SCALE_CFG = {
    "params": {
        "I": 2, "vbar": 1.0, "sigma_v": 1.0, "sigma_u": 0.5, "xi": 500.0,
        "theta": 0.1, "rho": 0.95, "alpha": 0.05, "beta": 2e-3, "iota": 0.1,
        "nv": 10, "nx": 15, "np": 31, "Tm": 10_000,
    },
    "run": {
        "batch": 4, "max_periods": 15_000, "chunk_size": 1_000,
        "conv_streak": 300, "eval_periods": 300,
        "freeze_on_convergence": True, "mm_init": "nash",
    },
}
SEED = 12345


def _run(cfg, backend, seed=SEED):
    metrics, conv, (params, bench, env, learner, final_state) = run(
        copy.deepcopy(cfg), seed=seed, out_path=None, quiet=True,
        backend=backend)
    return metrics, conv, learner, env, final_state


def _check_parity(cfg, require_convergence):
    metrics_np, conv_np, learner_np, env_np, state_np = _run(cfg, "numpy")
    metrics_nb, conv_nb, learner_nb, env_nb, state_nb = _run(cfg, "numba")

    if require_convergence:
        # confirms the test exercises the "frozen session" (active=False)
        # code path in both backends, not just the exploring-session path.
        assert (conv_np["converged_at"] >= 0).any(), \
            "config should converge some sessions within max_periods"

    # -- Q-learning state: bit-exact (same rng stream, mirrored arithmetic) -
    np.testing.assert_array_equal(learner_np.Q, learner_nb.Q)
    np.testing.assert_array_equal(learner_np.pol, learner_nb.pol)
    np.testing.assert_array_equal(learner_np.visits, learner_nb.visits)

    # -- convergence bookkeeping ---------------------------------------
    np.testing.assert_array_equal(conv_np["converged_at"], conv_nb["converged_at"])
    np.testing.assert_array_equal(conv_np["best_streak"], conv_nb["best_streak"])
    assert conv_np["periods_run"] == conv_nb["periods_run"]

    # -- market-maker rolling-window state ------------------------------
    np.testing.assert_array_equal(env_np.S, env_nb.S)
    np.testing.assert_array_equal(env_np.buf, env_nb.buf)
    assert env_np.ptr == env_nb.ptr

    # -- final (p_idx, vlag_idx, v_idx) state entering evaluate() --------
    for a, b in zip(state_np, state_nb):
        np.testing.assert_array_equal(a, b)

    # -- post-training evaluation metrics (both computed by the SAME
    #    pure-NumPy evaluate() -- should match to bounded float tolerance
    #    given bit-identical inputs and rng state) --------------------
    for key in ("pi_mean", "piN_bar", "piM_bar", "delta_c", "profit_gain",
                "chi_hat", "informativeness", "liquidity", "mispricing",
                "mean_price"):
        np.testing.assert_allclose(metrics_np[key], metrics_nb[key],
                                    rtol=0, atol=1e-12,
                                    err_msg=f"metric {key!r} diverged")


def test_numba_matches_numpy_reference_bit_exact_toy_grid():
    _check_parity(TOY_CFG, require_convergence=True)


def test_numba_matches_numpy_reference_bit_exact_baseline_grid():
    """Same check at the actual configs/baseline.yaml grid sizes (nv=10,
    nx=15, np=31, Tm=10000), so the parity guarantee isn't only established
    at toy scale. This is the grid the README benchmark numbers were
    measured on."""
    _check_parity(BASELINE_SCALE_CFG, require_convergence=False)
