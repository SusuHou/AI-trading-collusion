"""Numba-accelerated training loop (run_session --backend numba).

Port of the per-period hot path of run_session.run() -- the sequence
MarketEnv.step() + BatchQLearner.act()/.update() + ConvergenceTracker.update()
-- into a single @njit(parallel=True) kernel that executes a whole pre-drawn
chunk of periods for all sessions. numba.prange runs over the batch dimension
(one thread per session); each session's periods remain strictly sequential
inside its thread, which is the constraint that makes the 50B-period
worst-case sessions the binding cost.

Parity contract with the pure-NumPy reference (tests/test_numba_parity.py
asserts EXACT equality of Q tables, greedy policies, visit counters,
convergence bookkeeping, MM window state and eval metrics):

  * random blocks are pre-drawn by run_session.run() with the SAME numpy
    Generator calls in the same order for both backends, so both consume an
    identical random stream;
  * every floating-point expression below mirrors the reference
    operation-for-operation (numpy's small-n sum starts from 0.0 and adds
    sequentially; np.maximum guard -> `if x < c`; np.rint -> np.rint; argmax
    keeps the first maximum via strict `>`), and numba compiles without
    fastmath, so no FMA contraction reorders the arithmetic;
  * the market maker's periodic exact re-summation (MarketEnv.resync_every)
    uses numpy pairwise summation, which a naive loop cannot reproduce
    bitwise -- so the wrapper splits kernel calls at resync boundaries and
    performs the resync OUTSIDE the kernel with the same buf.sum(axis=1).

One known portability caveat: eq 4.3 evaluates exp(-beta n). The kernel uses
the C library exp on scalars while numpy may use a SIMD exp on some CPUs
(AVX512 float64 paths); a 1-ulp difference there could flip a rare explore
decision and desynchronize trajectories. On this machine they agree bitwise
(the parity test would catch it if not).

This module imports numba at module level -- import it lazily (run_session
does) so the numpy backend keeps working on interpreters without numba
(e.g. the project's default Python 3.14 .venv; use .venv310, see README).
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

# columns of the sufficient-statistics vector (must match env.market)
_P, _PP, _Z, _PZ, _V, _Y, _YY, _VY = range(8)


@njit(parallel=True, cache=True)
def _run_chunk(Q, pol, visits, buf, S,
               p_idx, vlag_idx, v_idx,
               streak, best_streak, converged_at,
               v_next_blk, u_blk, u01_blk, ra_blk,
               vgrid, c_grid, p_lo, dp, n_p,
               vbar, xi, theta, alpha, rho, beta,
               ptr0, Tm, t0, streak_target, freeze,
               pi_acc):
    """Run n = u_blk.shape[0] periods for every session, mutating all
    per-session state arrays in place. Scalar port of, per period:
      BatchQLearner.act (eq 2.6 + eq 4.3)  ->  MarketEnv.step (eq 3.2/4.1/4.2)
      ->  BatchQLearner.update (eq 2.4)    ->  ConvergenceTracker.update.
    """
    n = u_blk.shape[0]
    batch = u_blk.shape[1]
    I = u01_blk.shape[2]
    nx = Q.shape[3]
    nv = vgrid.shape[0]
    nT = float(Tm)
    for b in prange(batch):
        # per-thread locals / scratch
        pi_ = p_idx[b]
        vl_ = vlag_idx[b]
        v_ = v_idx[b]
        a = np.empty(I, np.int64)
        x = np.empty(I, np.float64)
        pi_i = np.empty(I, np.float64)
        acc_pi = 0.0
        for k in range(n):
            active = (not freeze) or (converged_at[b] < 0)
            s = (pi_ * nv + vl_) * nv + v_

            # --- eq 2.6 epsilon-greedy, eq 4.3 state-dependent decay ------
            eps = np.exp(-beta * visits[b, v_])
            for i in range(I):
                if active and (u01_blk[k, b, i] < eps):
                    a[i] = ra_blk[k, b, i]
                else:
                    a[i] = pol[b, i, s]
            visits[b, v_] += 1

            # --- order flows (state-scaled action grid, page 24) ----------
            v = vgrid[v_]
            yacc = 0.0
            for i in range(I):
                x[i] = (v - vbar) * c_grid[a[i]]
                yacc += x[i]
            y = yacc + u_blk[k, b]

            # --- MM coefficients from S BEFORE this period's push ---------
            # (scalar transcription of MarketEnv.mm_coeffs, eq 4.1-4.2)
            mp = S[b, _P] / nT
            mz = S[b, _Z] / nT
            mv = S[b, _V] / nT
            my = S[b, _Y] / nT
            var_p = S[b, _PP] / nT - mp * mp
            cov_pz = S[b, _PZ] / nT - mp * mz
            var_y = S[b, _YY] / nT - my * my
            cov_vy = S[b, _VY] / nT - mv * my
            if var_p < 1e-300:
                var_p = 1e-300
            if var_y < 1e-300:
                var_y = 1e-300
            xi1 = -cov_pz / var_p                 # z = xi0 - xi1 p
            g1 = cov_vy / var_y                   # v = gamma0 + gamma1 y
            g0 = mv - g1 * my
            lam = (theta * g1 + xi1) / (theta + xi1 * xi1)

            # --- price, investor demand, profits (eq 4.2 / 3.2 / 3.4) -----
            price = g0 + lam * y
            z = -xi * (price - vbar)
            for i in range(I):
                pi_i[i] = (v - price) * x[i]

            # --- O(1) rolling-window update (MarketEnv.step order) --------
            ptr = (ptr0 + k) % Tm
            for c in range(8):
                S[b, c] = S[b, c] + (-buf[b, ptr, c])
            buf[b, ptr, _P] = price
            buf[b, ptr, _PP] = price * price
            buf[b, ptr, _Z] = z
            buf[b, ptr, _PZ] = price * z
            buf[b, ptr, _V] = v
            buf[b, ptr, _Y] = y
            buf[b, ptr, _YY] = y * y
            buf[b, ptr, _VY] = v * y
            for c in range(8):
                S[b, c] = S[b, c] + buf[b, ptr, c]

            # --- next state: snap price to grid (Benchmarks.nearest_p_idx)
            pn = int(np.rint((price - p_lo) / dp))
            if pn < 0:
                pn = 0
            if pn > n_p - 1:
                pn = n_p - 1
            v_next = v_next_blk[k, b]
            s_next = (pn * nv + v_) * nv + v_next

            # --- eq 2.4 Q update + greedy-policy cache refresh ------------
            changed = False
            for i in range(I):
                m = Q[b, i, s_next, 0]
                for j in range(1, nx):
                    q = Q[b, i, s_next, j]
                    if q > m:
                        m = q
                target = pi_i[i] + rho * m
                old = Q[b, i, s, a[i]]
                if active:
                    Q[b, i, s, a[i]] = (1.0 - alpha) * old + alpha * target
                bestj = 0                          # first-max argmax,
                bm = Q[b, i, s, 0]                 # matching np.argmax
                for j in range(1, nx):
                    q = Q[b, i, s, j]
                    if q > bm:
                        bm = q
                        bestj = j
                if bestj != pol[b, i, s]:
                    changed = True
                pol[b, i, s] = bestj

            # --- convergence streak (page 26 criterion) -------------------
            if changed:
                streak[b] = 0
            else:
                streak[b] += 1
            if streak[b] > best_streak[b]:
                best_streak[b] = streak[b]
            if streak[b] >= streak_target and converged_at[b] < 0:
                converged_at[b] = t0 + k + 1

            # --- learning-curve bookkeeping (pi.mean over speculators) ----
            pacc = 0.0
            for i in range(I):
                pacc += pi_i[i]
            acc_pi += pacc / I

            pi_ = pn
            vl_ = v_
            v_ = v_next

        p_idx[b] = pi_
        vlag_idx[b] = vl_
        v_idx[b] = v_
        pi_acc[b] = pi_acc[b] + acc_pi


def run_chunk_numba(params, bench, env, learner, tracker, state,
                    v_next_blk, u_blk, u01_blk, ra_blk, freeze):
    """Drive _run_chunk over one pre-drawn random block, mutating env /
    learner / tracker / state in place exactly as the per-period NumPy loop
    in run_session.run() would. Splits at MM resync boundaries so the exact
    numpy pairwise re-summation is preserved. Returns pi_acc (batch,).
    """
    p = params
    n = u_blk.shape[0]
    p_idx, vlag_idx, v_idx = state          # int64 (batch,) arrays, in place
    pi_acc = np.zeros(learner.batch)
    p_lo = float(bench.p_grid[0])
    dp = float(bench.p_grid[1] - bench.p_grid[0])
    done = 0
    while done < n:
        until_resync = env.resync_every - (env._pushes % env.resync_every)
        seg = min(n - done, until_resync)
        sl = slice(done, done + seg)
        _run_chunk(learner.Q, learner.pol, learner.visits, env.buf, env.S,
                   p_idx, vlag_idx, v_idx,
                   tracker.streak, tracker.best_streak, tracker.converged_at,
                   v_next_blk[sl], u_blk[sl], u01_blk[sl], ra_blk[sl],
                   bench.vgrid, bench.c_grid, p_lo, dp, p.np_,
                   p.vbar, p.xi, p.theta, p.alpha, p.rho, p.beta,
                   env.ptr, p.Tm, tracker.t, tracker.streak_target,
                   bool(freeze), pi_acc)
        env.ptr = (env.ptr + seg) % p.Tm
        env._pushes += seg
        tracker.t += seg
        if env._pushes % env.resync_every == 0:
            env.S = env.buf.sum(axis=1)     # cancel FP drift (MarketEnv.step)
        done += seg
    return pi_acc
