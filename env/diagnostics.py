"""
diagnostics.py
==============
Pre-flight checks for the Kyle / Q-learning algorithmic-collusion simulator.

Run these BEFORE launching the full SLURM sweep. Every one of them is cheap
(seconds), and each catches a class of silent failure that would otherwise
quietly corrupt chi_hat without ever raising an error.

    $ python diagnostics.py

Three checks:

  1. check_benchmarks(bm)      Is chi_hat measuring what we think it is?
  2. ClipMonitor               Are prices falling off the edge of p_grid?
  3. exploration_budget(...)   Is beta giving each Q-cell enough visits?

ASSUMPTIONS (adjust these two if your codebase differs):
  * `Params` and `Benchmarks` are importable from your module.
  * v_t is drawn uniformly from `bm.vgrid`, and u_t ~ N(0, sigma_u).
    If your sampler differs, pass your own via the `sampler` argument.
"""

from __future__ import annotations

import numpy as np

# --- EDIT THIS IMPORT to point at your module -------------------------------
# from model import Params, Benchmarks, build_benchmarks
# ---------------------------------------------------------------------------


# ===========================================================================
# 1. BENCHMARK CONSISTENCY
# ===========================================================================
# chi_hat = (agent - nash) / (cartel - nash).
# If `nash` or `cartel` is wrong, chi_hat is wrong -- and nothing downstream
# will ever tell you. These assertions make a bad benchmark fail LOUDLY.

def check_benchmarks(bm, n_draws: int = 200_000, seed: int = 0, tol: float = 0.02):
    """Verify the benchmark bundle is internally consistent.

    Raises AssertionError on failure; prints a report on success.
    """
    rng = np.random.default_rng(seed)
    p = bm.params
    ok = []

    def report(name: str, passed: bool, detail: str):
        ok.append(passed)
        mark = "PASS" if passed else "FAIL"
        print(f"  [{mark}] {name:<38} {detail}")

    print("\n=== 1. BENCHMARK CONSISTENCY ===")

    # --- 1a. Economic ordering -------------------------------------------
    # Collusion in a Kyle model means trading LESS aggressively.
    # So the cartel should have lower intensity, lower price impact,
    # and (the whole point) higher profit per speculator.
    report("chi_cartel < chi_nash (trade less)",
           bm.chiM < bm.chiN,
           f"chiM={bm.chiM:.4f}  chiN={bm.chiN:.4f}")

    report("lam_cartel < lam_nash (less impact)",
           bm.lamM < bm.lamN,
           f"lamM={bm.lamM:.4f}  lamN={bm.lamN:.4f}")

    report("pi_cartel > pi_nash (collusion pays)",
           bm.piM > bm.piN,
           f"piM={bm.piM:.6f}  piN={bm.piN:.6f}")

    # --- 1b. THE CRITICAL ONE: are the benchmarks REACHABLE? --------------
    # If chi^M lies outside c_grid, the agents literally cannot choose the
    # cartel action. chi_hat could never reach 1.0 no matter how collusive
    # they become -- and you would misread that as "partial collusion".
    lo, hi = float(bm.c_grid.min()), float(bm.c_grid.max())

    report("chi_nash inside c_grid",
           lo <= bm.chiN <= hi,
           f"c_grid=[{lo:.4f}, {hi:.4f}]  chiN={bm.chiN:.4f}")

    report("chi_cartel inside c_grid  <-- CRITICAL",
           lo <= bm.chiM <= hi,
           f"c_grid=[{lo:.4f}, {hi:.4f}]  chiM={bm.chiM:.4f}")

    # Not just inside -- comfortably inside, with room to go BEYOND the
    # cartel point. Otherwise you cannot detect super-collusive outcomes
    # (chi_hat > 1), and the grid edge acts as an artificial ceiling.
    margin_M = (bm.chiM - lo) / (hi - lo)
    report("chi_cartel not jammed against grid edge",
           margin_M > 0.05,
           f"chiM sits {margin_M:.1%} of the way into the grid (want >5%)")

    # --- 1c. Do the formulas agree with the simulator? --------------------
    # Replay many random (v, u) draws through path_profits and check the
    # empirical mean recovers the closed-form piN / piM. This catches the
    # classic bug of using nominal sigma_v where sv_hat was intended.
    v = rng.choice(bm.vgrid, size=n_draws)
    u = rng.normal(0.0, p.sigma_u, size=n_draws)

    emp_N = float(bm.path_profits(v, u, "nash").mean())
    emp_M = float(bm.path_profits(v, u, "cartel").mean())

    err_N = abs(emp_N - bm.piN) / max(abs(bm.piN), 1e-12)
    err_M = abs(emp_M - bm.piM) / max(abs(bm.piM), 1e-12)

    report("path_profits('nash') recovers piN",
           err_N < tol,
           f"empirical={emp_N:.6f}  closed-form={bm.piN:.6f}  err={err_N:.2%}")

    report("path_profits('cartel') recovers piM",
           err_M < tol,
           f"empirical={emp_M:.6f}  closed-form={bm.piM:.6f}  err={err_M:.2%}")

    # --- 1d. Is sv_hat actually being used? ------------------------------
    # sv_hat is the TRUE std of v on the discrete grid. If it differs from
    # the nominal sigma_v, then any benchmark computed with sigma_v is
    # subtly wrong. Flag the gap so you can confirm it was handled.
    sv_grid = float(np.std(bm.vgrid))
    gap = abs(sv_grid - p.sigma_v) / p.sigma_v
    print(f"  [INFO] grid std vs nominal sigma_v          "
          f"sv_hat={bm.sv_hat:.4f}  std(vgrid)={sv_grid:.4f}  "
          f"nominal={p.sigma_v:.4f}  gap={gap:.1%}")
    if gap > 0.01:
        print("         ^ discretisation shifts sigma_v by more than 1%. "
              "Confirm every benchmark uses sv_hat, NOT sigma_v.")

    # --- 1e. Sign sanity for chi_hat --------------------------------------
    # Because chiM < chiN, the denominator (chiM - chiN) is NEGATIVE.
    # A collusive outcome means chi went DOWN. If you ever see chi_hat < 0,
    # suspect a flipped benchmark rather than "anti-collusion".
    print(f"  [INFO] chi_hat denominator sign             "
          f"(chiM - chiN) = {bm.chiM - bm.chiN:+.4f}  (negative is expected)")

    assert all(ok), "BENCHMARK CHECK FAILED -- see [FAIL] lines above."
    print("  --> all benchmark checks passed.\n")
    return True


# ===========================================================================
# 2. PRICE-GRID CLIPPING MONITOR
# ===========================================================================
# nearest_p_idx() ends with np.clip(). That is SILENT: a price far above the
# grid and a price just above it both map to the same edge index. The agent
# cannot tell them apart, so its state is degraded exactly where the extreme
# (i.e. most interesting) behaviour lives.
#
# Drop a ClipMonitor into the simulation loop and watch the rate.

class ClipMonitor:
    """Track how often computed prices fall outside p_grid."""

    def __init__(self, benchmarks):
        self.bm = benchmarks
        self.lo = float(benchmarks.p_grid[0])
        self.hi = float(benchmarks.p_grid[-1])
        self.n_total = 0
        self.n_below = 0
        self.n_above = 0
        self.worst_below = np.inf
        self.worst_above = -np.inf

    def observe(self, prices) -> None:
        """Call this every period with the price(s) before discretisation."""
        p = np.atleast_1d(np.asarray(prices, dtype=float))
        below = p < self.lo
        above = p > self.hi
        self.n_total += p.size
        self.n_below += int(below.sum())
        self.n_above += int(above.sum())
        if below.any():
            self.worst_below = min(self.worst_below, float(p[below].min()))
        if above.any():
            self.worst_above = max(self.worst_above, float(p[above].max()))

    @property
    def rate(self) -> float:
        return (self.n_below + self.n_above) / max(self.n_total, 1)

    def report(self, warn_at: float = 0.01) -> float:
        print("\n=== 2. PRICE-GRID CLIPPING ===")
        print(f"  p_grid range         : [{self.lo:.4f}, {self.hi:.4f}]")
        print(f"  periods observed     : {self.n_total:,}")
        print(f"  clipped below        : {self.n_below:,}")
        print(f"  clipped above        : {self.n_above:,}")
        print(f"  CLIPPING RATE        : {self.rate:.3%}")
        if self.n_below:
            print(f"  worst price below    : {self.worst_below:.4f}")
        if self.n_above:
            print(f"  worst price above    : {self.worst_above:.4f}")

        if self.rate > warn_at:
            print(f"\n  *** WARNING: clipping rate exceeds {warn_at:.1%}. ***")
            print("  Distinct prices are collapsing onto the grid edges, so the")
            print("  agents' state is blind in exactly the region where extreme")
            print("  behaviour occurs. Widen p_grid (raise `iota`) and re-run.")
        else:
            print(f"  --> below the {warn_at:.1%} threshold. Grid is wide enough.\n")
        return self.rate


def clipping_dry_run(bm, n_periods: int = 200_000, seed: int = 0,
                     chi_used: float | None = None) -> float:
    """Estimate the clipping rate WITHOUT running Q-learning.

    Simulates order flow with all speculators trading at intensity `chi_used`
    (default: the most aggressive action available, i.e. the worst case) and
    reports how often the resulting price escapes p_grid.
    """
    rng = np.random.default_rng(seed)
    p = bm.params
    mon = ClipMonitor(bm)

    # Worst case = the most aggressive action the agents can actually pick.
    if chi_used is None:
        chi_used = float(np.max(np.abs(bm.c_grid)))

    v = rng.choice(bm.vgrid, size=n_periods)
    u = rng.normal(0.0, p.sigma_u, size=n_periods)

    dv = v - p.vbar
    x = chi_used * dv                       # each speculator's trade
    y = p.I * x + u                         # total order flow
    price = p.vbar + bm.lamN * y            # MM's price (Nash lambda)

    mon.observe(price)
    print(f"  (dry run at chi = {chi_used:.4f}, the most aggressive grid action)")
    return mon.report()


# ===========================================================================
# 3. EXPLORATION BUDGET  (is beta sane?)
# ===========================================================================
# beta is not a free knob. The convention in this literature is to pick it so
# that each (state, action) cell of the Q-table gets visited a target number
# of times, nu, by pure random exploration:
#
#     total exploration periods = integral_0^inf exp(-beta t) dt = 1 / beta
#     nu = (1 / beta) / (n_states * n_actions)
#
# Published baselines target nu on the order of 20-100.
#   nu ~ 3      -> Q-values are noise; "collusion" is meaningless.
#   nu ~ 50000  -> burning compute for nothing.

def exploration_budget(params, n_states: int, verbose: bool = True) -> float:
    """Compute nu, the expected random-exploration visits per Q-cell."""
    beta = params.beta
    n_actions = params.nx
    total_exploration = 1.0 / beta
    nu = total_exploration / (n_states * n_actions)

    if verbose:
        print("\n=== 3. EXPLORATION BUDGET ===")
        print(f"  beta                    : {beta:.3e}")
        print(f"  1 / beta (explore steps): {total_exploration:,.0f}")
        print(f"  n_states                : {n_states:,}")
        print(f"  n_actions (nx)          : {n_actions}")
        print(f"  Q-table cells           : {n_states * n_actions:,}")
        print(f"  nu (visits per cell)    : {nu:,.1f}")
        if nu < 10:
            print("  *** WARNING: nu < 10. Cells are undertrained -- most Q-values")
            print("      are noise. Lower beta (explore longer) or shrink the grids.")
        elif nu > 5_000:
            print("  *** NOTE: nu is very large. You may be wasting cluster time;")
            print("      a larger beta would converge sooner at no cost to quality.")
        else:
            print("  --> nu is in the healthy range used in the literature.\n")
    return nu


# ===========================================================================
# 4. EMPIRICAL VISIT COUNTS  (trust, but verify)
# ===========================================================================
# The nu formula above is a THEORETICAL estimate. Instrument the real run and
# check it. If the MINIMUM visit count across cells is tiny, those Q-values
# are garbage -- and any "collusion" that depends on them is an artifact.

class VisitCounter:
    """Count how often each (state, action) cell is actually visited."""

    def __init__(self, n_states: int, n_actions: int, n_agents: int = 1):
        self.counts = np.zeros((n_agents, n_states, n_actions), dtype=np.int64)

    def observe(self, agent: int, state_idx: int, action_idx: int) -> None:
        self.counts[agent, state_idx, action_idx] += 1

    def report(self, min_visits: int = 10) -> dict:
        print("\n=== 4. EMPIRICAL Q-CELL VISIT COUNTS ===")
        stats = {}
        for i in range(self.counts.shape[0]):
            c = self.counts[i]
            never = int((c == 0).sum())
            starved = int((c < min_visits).sum())
            total_cells = c.size
            print(f"  agent {i}:")
            print(f"    min visits    : {int(c.min()):,}")
            print(f"    median visits : {int(np.median(c)):,}")
            print(f"    max visits    : {int(c.max()):,}")
            print(f"    never visited : {never:,} / {total_cells:,} "
                  f"({never / total_cells:.1%})")
            print(f"    < {min_visits} visits  : {starved:,} / {total_cells:,} "
                  f"({starved / total_cells:.1%})")
            if never:
                print(f"    *** WARNING: {never} cells NEVER visited. Their Q-values")
                print("        are still at their initial values, not learned. ***")
            stats[i] = {"min": int(c.min()), "never": never, "starved": starved}
        print()
        return stats


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":
    print("=" * 72)
    print("PRE-FLIGHT DIAGNOSTICS -- run before the SLURM sweep")
    print("=" * 72)

    # ---- wire these up to your codebase -----------------------------------
    # p  = Params()
    # bm = build_benchmarks(p)
    #
    # check_benchmarks(bm)
    # clipping_dry_run(bm)
    #
    # # n_states depends on what your policy conditions on. Two examples:
    # #   state = (v,)             -> n_states = p.nv
    # #   state = (v, p_prev)      -> n_states = p.nv * p.np_
    # exploration_budget(p, n_states=p.nv * p.np_)
    # -----------------------------------------------------------------------

    print("\nEdit the import at the top and uncomment the block in __main__,")
    print("then run:  python diagnostics.py\n")
