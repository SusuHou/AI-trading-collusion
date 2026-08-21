"""Correctness gate for env/benchmarks.py and MarketEnv's pricing rule.

Verifies:
  1. lambda^N / lambda^M satisfy their defining fixed-point equations to high
     numerical precision (residual ~ 0) for both baseline sigma_u regimes.
  2. The algebraic identities chi^N (I+1) lambda^N = 1, chi^M 2 I lambda^M = 1.
  3. The speculators' FOCs hold at the symmetric equilibrium strategies.
  4. pi^N, pi^M match their closed forms; known magnitudes from the paper
     (each speculator earns ~54 in the cartel benchmark at xi=500) hold.
  5. Grids: v grid quantile formula, sigma_v_hat = 0.938 at nv=10, x grid
     contains the two benchmark strategies at every v, p grid symmetric.
  6. MarketEnv reproduces p^N / p^M and the expected benchmark profits when
     speculators are forced to play x^N / x^M with a Nash-warm-started
     market maker (zero learning).
"""
import numpy as np
import pytest

from env.benchmarks import (Benchmarks, Params, compute_benchmarks, q0_table,
                            sigma_v_hat, solve_lambda, v_grid, _lambda_rhs)


PARAM_CELLS = [
    Params(sigma_u=0.1),
    Params(sigma_u=100.0),
    Params(sigma_u=1.0, I=5, rho=0.5),
    Params(sigma_u=10.0, xi=5.0),      # low-xi regime
]


@pytest.fixture(params=PARAM_CELLS, ids=lambda p: f"I{p.I}_su{p.sigma_u}_xi{p.xi}")
def bench(request) -> Benchmarks:
    return compute_benchmarks(request.param)


# ---------------------------------------------------------------------------
# 1-2: fixed points and identities
# ---------------------------------------------------------------------------
def test_lambda_fixed_point_residual(bench):
    p = bench.params
    sv2, su2 = bench.sv_hat ** 2, p.sigma_u ** 2
    # Omega = I * chi(lambda): aggregate informed sensitivity in y = Omega dv + u
    omN = lambda lam: p.I / ((p.I + 1) * lam)
    omM = lambda lam: 1.0 / (2.0 * lam)
    resN = bench.lamN - _lambda_rhs(bench.lamN, omN, p.xi, p.theta, sv2, su2)
    resM = bench.lamM - _lambda_rhs(bench.lamM, omM, p.xi, p.theta, sv2, su2)
    assert abs(resN) < 1e-13 * max(1.0, bench.lamN)
    assert abs(resM) < 1e-13 * max(1.0, bench.lamM)
    assert bench.lamN > 0 and bench.lamM > 0


def test_chi_identities_exact(bench):
    p = bench.params
    assert bench.chiN * (p.I + 1) * bench.lamN == pytest.approx(1.0, abs=1e-14)
    assert bench.chiM * 2 * p.I * bench.lamM == pytest.approx(1.0, abs=1e-14)


# ---------------------------------------------------------------------------
# 3: first-order conditions at the symmetric strategies
# ---------------------------------------------------------------------------
def test_nash_foc(bench):
    """d/dx_i E[(v - vbar - lamN(x_i + (I-1) chiN dv + u)) x_i] = 0 at x_i = chiN dv."""
    p = bench.params
    for dv in (0.5, -1.3):
        xi_star = bench.chiN * dv
        foc = dv - bench.lamN * (p.I - 1) * bench.chiN * dv - 2 * bench.lamN * xi_star
        assert foc == pytest.approx(0.0, abs=1e-12 * abs(dv))


def test_cartel_foc(bench):
    """d/dx E[(v - vbar - lamM I x) I x] = 0 at x = chiM dv."""
    for dv in (0.5, -1.3):
        x_star = bench.chiM * dv
        foc = dv - 2 * bench.lamM * bench.params.I * x_star
        assert foc == pytest.approx(0.0, abs=1e-12 * abs(dv))


def test_mm_pricing_consistency(bench):
    """lam must equal the MM's eq-3.4 coefficient given rational E[v|y]."""
    p = bench.params
    for lam, omega in ((bench.lamN, p.I * bench.chiN), (bench.lamM, p.I * bench.chiM)):
        proj = omega * bench.sv_hat ** 2 / (omega ** 2 * bench.sv_hat ** 2 + p.sigma_u ** 2)
        implied = p.xi / (p.xi ** 2 + p.theta) + p.theta / (p.xi ** 2 + p.theta) * proj
        assert lam == pytest.approx(implied, rel=1e-12)


# ---------------------------------------------------------------------------
# 4: profits
# ---------------------------------------------------------------------------
def test_profit_closed_forms(bench):
    p = bench.params
    sv2 = bench.sv_hat ** 2
    assert bench.piN == pytest.approx(sv2 / ((p.I + 1) ** 2 * bench.lamN), rel=1e-12)
    assert bench.piM == pytest.approx(sv2 / (4 * p.I * bench.lamM), rel=1e-12)
    assert bench.piM > bench.piN > 0


def test_known_baseline_magnitudes():
    """Paper (spec 'Known illustrative numbers'): cartel profit ~54-55/speculator
    at xi=500 in both sigma_u regimes; lambda ~ 1/xi."""
    for su in (0.1, 100.0):
        b = compute_benchmarks(Params(sigma_u=su))
        assert 45 < b.piM < 65
        assert 0.5 / 500 < b.lamN < 5.0 / 500
    assert sigma_v_hat(v_grid(1.0, 1.0, 10), 1.0) == pytest.approx(0.938, abs=5e-4)


# ---------------------------------------------------------------------------
# 5: grids
# ---------------------------------------------------------------------------
def test_v_grid_formula():
    from scipy.stats import norm
    g = v_grid(1.0, 1.0, 10)
    assert len(g) == 10
    assert g[0] == pytest.approx(1.0 + norm.ppf(0.05))
    assert g[-1] == pytest.approx(1.0 + norm.ppf(0.95))
    assert np.allclose(g + g[::-1], 2.0)  # symmetric about vbar


def test_x_grid_contains_benchmarks(bench):
    """At every v grid point, both x^N(v) and x^M(v) lie on the action grid
    interval, and (for iota > 0) strictly inside it."""
    p = bench.params
    for v in bench.vgrid:
        xs = bench.x_values(np.array([v]))[0]
        lo, hi = min(xs[0], xs[-1]), max(xs[0], xs[-1])
        for chi in (bench.chiN, bench.chiM):
            assert lo < chi * (v - p.vbar) < hi or np.isclose(chi * (v - p.vbar), lo) \
                or np.isclose(chi * (v - p.vbar), hi)
    assert len(bench.c_grid) == p.nx
    # endpoints of the multiplier grid per page 24
    d = bench.chiN - bench.chiM
    assert bench.c_grid[0] == pytest.approx(bench.chiM - p.iota * d)
    assert bench.c_grid[-1] == pytest.approx(bench.chiN + p.iota * d)


def test_p_grid(bench):
    p = bench.params
    g = bench.p_grid
    assert len(g) == p.np_
    assert np.allclose(g + g[::-1], 2 * p.vbar)  # symmetric about vbar
    x_ext = max(bench.chiN, bench.chiM) * np.max(np.abs(bench.vgrid - p.vbar))
    pH = p.vbar + bench.lamN * (p.I * x_ext + 1.96 * p.sigma_u)
    assert g[-1] == pytest.approx(pH + p.iota * 2 * (pH - p.vbar))
    # nearest_p_idx round-trips grid points and clips out-of-range prices
    assert np.array_equal(bench.nearest_p_idx(g), np.arange(p.np_))
    assert bench.nearest_p_idx(np.array([g[0] - 99.0, g[-1] + 99.0])).tolist() \
        == [0, p.np_ - 1]


def test_lambda_oa_closed_form(bench):
    """OA eq IA.2.9/IA.2.12: lambda = (theta gamma + xi)/(theta + xi^2) with
    gamma = (I chi) / [(I chi)^2 + (sigma_u/sigma_v_hat)^2]."""
    p = bench.params
    for lam, chi in ((bench.lamN, bench.chiN), (bench.lamM, bench.chiM)):
        om = p.I * chi
        gamma = om / (om ** 2 + (p.sigma_u / bench.sv_hat) ** 2)
        assert lam == pytest.approx((p.theta * gamma + p.xi) / (p.theta + p.xi ** 2),
                                    rel=1e-12)


# ---------------------------------------------------------------------------
# Matched-path benchmark scoring (OA eq IA.4.2/IA.4.3)
# ---------------------------------------------------------------------------
def test_path_profits_conditional_exactness(bench):
    """With u = 0, pi_t must equal chi (1 - lam I chi) (v - vbar)^2 exactly."""
    p = bench.params
    v = bench.vgrid
    for mode, lam, chi in (("nash", bench.lamN, bench.chiN),
                           ("cartel", bench.lamM, bench.chiM)):
        got = bench.path_profits(v, np.zeros_like(v), mode)
        want = chi * (1 - lam * p.I * chi) * (v - p.vbar) ** 2
        assert np.allclose(got, want, rtol=1e-12)


def test_path_profits_mean_converges_to_expectation(bench):
    """Window mean of matched-path profits ~ pi^N / pi^M for long windows."""
    p = bench.params
    rng = np.random.default_rng(123)
    T = 400_000
    v = bench.vgrid[rng.integers(0, p.nv, size=T)]
    u = rng.normal(0.0, p.sigma_u, size=T)
    for mode, pi_th in (("nash", bench.piN), ("cartel", bench.piM)):
        assert bench.path_profits(v, u, mode).mean() == pytest.approx(pi_th, rel=0.02)


def test_matched_path_delta_c_endpoints(bench):
    """Scoring the Nash strategy itself must give Delta^C ~ 0, the cartel
    strategy Delta^C ~ 1 (matched-path normalization is exact per path)."""
    from env import metrics as M
    p = bench.params
    rng = np.random.default_rng(5)
    T = 50_000
    v = rng.choice(bench.vgrid, size=(T, 1))
    u = rng.normal(0.0, p.sigma_u, size=(T, 1))
    piN_bar = bench.path_profits(v, u, "nash").mean(axis=0)
    piM_bar = bench.path_profits(v, u, "cartel").mean(axis=0)
    for mode, target in (("nash", 0.0), ("cartel", 1.0)):
        pi_bar = np.repeat(bench.path_profits(v, u, mode).mean(axis=0)[:, None],
                           p.I, axis=1)
        dc = M.delta_c_matched(pi_bar, piN_bar, piM_bar)
        assert dc[0] == pytest.approx(target, abs=1e-12)


# ---------------------------------------------------------------------------
# Metric closed forms (OA eq IA.4.4-IA.4.7)
# ---------------------------------------------------------------------------
def test_metrics_closed_forms_on_benchmark_data(bench):
    from env import metrics as M
    p = bench.params
    rng = np.random.default_rng(11)
    T, batch = 200_000, 2
    v = rng.choice(bench.vgrid, size=(T, batch))
    u = rng.normal(0.0, p.sigma_u, size=(T, batch))
    x = np.repeat((bench.chiN * (v - p.vbar))[:, :, None], p.I, axis=2)

    chi_c, chi_i1 = M.chi_hat(x, v)
    assert np.allclose(chi_c, bench.chiN, rtol=1e-10)   # exact linear strategy
    assert np.allclose(chi_i1, bench.chiN, rtol=1e-10)

    # IA.4.5 closed form vs direct var ratio (agree up to sampling noise)
    ic = M.informativeness(chi_c, p.I, bench.sv_hat, p.sigma_u)
    ic_var = M.informativeness_var(x.sum(axis=2), u)
    assert np.allclose(ic, (p.I * bench.chiN) ** 2 * (bench.sv_hat / p.sigma_u) ** 2)
    assert np.allclose(ic_var, ic, rtol=0.05)

    # IA.4.6: with lam_hat_t == lamN for all t, L = 1/|1 - xi lamN|
    lam_hat = np.full((T, batch), bench.lamN)
    assert np.allclose(M.liquidity(lam_hat, p.xi),
                       1.0 / abs(1.0 - p.xi * bench.lamN))

    # IA.4.7: E = |1 - lamN I chiN| * mean|v - vbar|
    e = M.mispricing(lam_hat, chi_c, v, p.I, p.vbar)
    want = abs(1.0 - bench.lamN * p.I * bench.chiN) * np.abs(v - p.vbar).mean(axis=0)
    assert np.allclose(e, want, rtol=1e-10)


# ---------------------------------------------------------------------------
# Q0 table
# ---------------------------------------------------------------------------
def test_q0_matches_direct_sum(bench):
    """Recompute Q0 with explicit loops for a few cells (page 25 formula)."""
    p = bench.params
    q0 = q0_table(bench)
    assert q0.shape == (p.nv, p.nx)
    rng = np.random.default_rng(0)
    for _ in range(6):
        k = rng.integers(p.nv)
        j = rng.integers(p.nx)
        v = bench.vgrid[k]
        xj = (v - p.vbar) * bench.c_grid[j]
        total = 0.0
        for m in range(p.nx):
            xm = (v - p.vbar) * bench.c_grid[m]
            total += (v - (p.vbar + bench.lamN * (xj + (p.I - 1) * xm))) * xj
        expected = total / ((1 - p.rho) * p.nx)
        assert q0[k, j] == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------------------
# 6: MarketEnv reproduces the analytic benchmarks under forced strategies
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["nash", "cartel"])
@pytest.mark.parametrize("sigma_u", [0.1, 100.0])
def test_market_env_reproduces_benchmark(mode, sigma_u):
    from env.market import MarketEnv

    params = Params(sigma_u=sigma_u)
    bench = compute_benchmarks(params)
    lam = bench.lamN if mode == "nash" else bench.lamM
    chi = bench.chiN if mode == "nash" else bench.chiM
    pi_th = bench.piN if mode == "nash" else bench.piM

    batch = 4
    rng = np.random.default_rng(42)
    env = MarketEnv(params, bench, batch=batch, rng=rng, mm_init=mode)

    # With the MM warm-started on exact benchmark data, its regressions must
    # recover xi_hat = xi and lambda_hat = lam before any real trading.
    g0, lam_hat = env.mm_coeffs()
    assert np.allclose(lam_hat, lam, rtol=1e-10)
    assert np.allclose(g0, params.vbar, atol=1e-10)

    # Force x_i = chi (v - vbar); with u = 0 the price must be exactly
    # p = vbar + lam * I * chi * (v - vbar), and profits their conditional
    # means. MM updates frozen: zero learning, pure plug-in check.
    T = 2000
    prof = np.zeros(batch)
    for t in range(T):
        v = bench.vgrid[rng.integers(0, params.nv, size=batch)]
        x = np.repeat((chi * (v - params.vbar))[:, None], params.I, axis=1)
        u = np.zeros(batch)
        p_t, pi_t, info = env.step(x, v, u, update_mm=False)
        p_expected = params.vbar + lam * params.I * chi * (v - params.vbar)
        assert np.allclose(p_t, p_expected, rtol=1e-5, atol=1e-8)
        pi_expected = (v - p_expected) * chi * (v - params.vbar)
        assert np.allclose(pi_t, pi_expected[:, None], rtol=1e-5, atol=1e-10)
        prof += pi_t[:, 0]
    # Sample average of per-period profit -> chi (1 - lam I chi) * mean(dv^2);
    # with u=0 held out, compare against the theoretical pi using sample dv^2.
    # (v draws are uniform over the grid so E matches sigma_v_hat^2.)
    assert np.allclose(prof / T, pi_th, rtol=0.05)


def test_market_env_z_and_price_rule():
    """z_t = -xi (p_t - vbar) (eq 3.2) and p = gamma0_hat + lambda_hat y (eq 4.2)."""
    from env.market import MarketEnv

    params = Params(sigma_u=1.0)
    bench = compute_benchmarks(params)
    rng = np.random.default_rng(7)
    env = MarketEnv(params, bench, batch=3, rng=rng, mm_init="nash")
    g0, lam_hat = env.mm_coeffs()
    v = bench.vgrid[rng.integers(0, params.nv, size=3)]
    x = rng.normal(size=(3, params.I)) * 10
    u = rng.normal(size=3) * params.sigma_u
    p_t, pi_t, info = env.step(x, v, u)
    y = x.sum(axis=1) + u
    assert np.allclose(p_t, g0 + lam_hat * y, rtol=1e-12)
    assert np.allclose(info["z"], -params.xi * (p_t - params.vbar), rtol=1e-12)
    assert np.allclose(info["y"], y)
    assert np.allclose(pi_t, (v - p_t)[:, None] * x)
    assert np.allclose(info["m"], -(y + info["z"]))
