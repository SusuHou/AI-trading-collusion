"""Toy-scale checks of the Q-learning machinery.

- eq 2.4 update arithmetic hand-traced on a tiny grid
- eq 4.3 exploration decay from per-v visit counters
- eq 2.6 epsilon-greedy branching (and explore-mask freezing)
- convergence.py detects stable vs unstable greedy policies
- a short end-to-end toy session runs and stays finite
"""
import numpy as np
import pytest

from env.benchmarks import Params, compute_benchmarks
from phase1_qlearning.convergence import ConvergenceTracker
from phase1_qlearning.qlearn import BatchQLearner

TOY = Params(sigma_u=1.0, nv=3, nx=4, np_=5, Tm=9, alpha=0.25, rho=0.9,
             beta=0.1)


@pytest.fixture
def learner():
    bench = compute_benchmarks(TOY)
    return BatchQLearner(TOY, bench, batch=2), bench


def test_q_update_matches_eq_2_4_by_hand(learner):
    ql, _ = learner
    p = ql.params
    rng = np.random.default_rng(0)
    ql.Q[:] = rng.normal(size=ql.Q.shape)  # arbitrary known Q
    ql.pol[:] = ql.Q.argmax(axis=3).astype(np.int8)
    Q_before = ql.Q.copy()

    s = np.array([7, 3])
    s_next = np.array([11, 40])
    a = np.array([[2, 0], [1, 3]])
    r = np.array([[1.5, -0.25], [0.0, 2.0]])

    ql.update(s, a, r, s_next)

    for b in range(2):
        for i in range(p.I):
            old = Q_before[b, i, s[b], a[b, i]]
            target = r[b, i] + p.rho * Q_before[b, i, s_next[b], :].max()
            expected = (1 - p.alpha) * old + p.alpha * target
            assert ql.Q[b, i, s[b], a[b, i]] == pytest.approx(expected, rel=1e-14)
            # all other cells of the visited row untouched
            mask = np.ones(p.nx, bool)
            mask[a[b, i]] = False
            assert np.array_equal(ql.Q[b, i, s[b], mask], Q_before[b, i, s[b], mask])
    # every non-visited state row untouched
    touched = np.zeros(ql.nS, bool)
    touched[s] = True
    assert np.array_equal(ql.Q[:, :, ~touched, :], Q_before[:, :, ~touched, :])


def test_update_changed_flag_and_policy_cache(learner):
    ql, _ = learner
    p = ql.params
    ql.Q[:] = 0.0
    ql.Q[:, :, :, 1] = 1.0            # greedy action = 1 everywhere
    ql.pol[:] = 1
    s = np.array([0, 0])
    # session 0: huge reward on action 3 flips the argmax; session 1: tiny one doesn't
    a = np.array([[3, 3], [3, 3]])
    r = np.array([[100.0, 100.0], [0.001, 0.001]])
    changed = ql.update(s, a, r, np.array([1, 1]))
    assert changed.tolist() == [True, False]
    assert ql.pol[0, :, 0].tolist() == [3, 3]
    assert ql.pol[1, :, 0].tolist() == [1, 1]


def test_epsilon_decay_eq_4_3(learner):
    ql, _ = learner
    p = ql.params
    ql.visits[0] = [0, 5, 50]
    ql.visits[1] = [2, 0, 0]
    for b, k in ((0, 0), (0, 1), (0, 2), (1, 0)):
        v_idx = np.array([k, k])
        assert ql.epsilon(v_idx)[b] == pytest.approx(np.exp(-p.beta * ql.visits[b, k]))


def test_act_epsilon_greedy_branches(learner):
    ql, _ = learner
    p = ql.params
    ql.Q[:] = 0.0
    ql.Q[:, :, :, 2] = 5.0
    ql.pol[:] = 2
    ql.visits[:] = 0                     # eps = exp(0) = 1 -> always explore
    s = np.zeros(2, dtype=np.int64)
    v_idx = np.zeros(2, dtype=np.int64)
    rand_a = np.array([[0, 3], [1, 1]])
    a = ql.act(s, v_idx, np.full((2, p.I), 0.5), rand_a)
    assert np.array_equal(a, rand_a)                   # exploring
    assert np.array_equal(ql.visits[:, 0], [1, 1])     # visit counted

    ql.visits[:, 0] = 10_000_000                        # eps ~ 0 -> greedy
    a = ql.act(s, v_idx, np.full((2, p.I), 0.5), rand_a)
    assert np.all(a == 2)

    ql.visits[:, 0] = 0                                 # eps = 1 but masked off
    a = ql.act(s, v_idx, np.full((2, p.I), 0.5), rand_a,
               explore_mask=np.array([False, True]))
    assert np.all(a[0] == 2) and np.array_equal(a[1], rand_a[1])


def test_convergence_tracker_stable_vs_unstable():
    tr = ConvergenceTracker(batch=3, streak_target=5)
    no_change = np.array([False, False, False])
    for t in range(4):
        tr.update(no_change)
    assert not tr.converged.any()
    # session 1 flips its policy at t=5, others hold
    tr.update(np.array([False, True, False]))
    assert tr.converged.tolist() == [True, False, True]
    assert tr.converged_at.tolist() == [5, -1, 5]
    for t in range(5):
        tr.update(no_change)
    assert tr.all_converged
    assert tr.converged_at.tolist() == [5, 10, 5]
    assert tr.best_streak[1] == 5


def test_toy_session_end_to_end():
    """A few thousand periods on the toy grids: runs, finite, and the
    frozen-policy evaluation produces finite metrics."""
    import phase1_qlearning.run_session as rs

    cfg = {"params": {"sigma_u": 1.0, "nv": 3, "nx": 4, "np": 5, "Tm": 9,
                      "beta": 1e-3},
           "run": {**rs.RUN_DEFAULTS, "batch": 4, "max_periods": 4000,
                   "conv_streak": 10_000, "eval_periods": 500,
                   "chunk_size": 1000}}
    metrics, conv, _ = rs.run(cfg, seed=0, quiet=True)
    assert conv["periods_run"] == 4000
    for key in ("delta_c", "informativeness", "liquidity", "mispricing"):
        assert np.all(np.isfinite(metrics[key])), key
    assert metrics["pi_mean"].shape == (4, 2)
