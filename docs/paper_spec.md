# Technical Spec Extracted From Dou, Goldstein & Ji (2025), NBER WP 34054

Source: `AI-POWERED TRADING, ALGORITHMIC COLLUSION, AND PRICE EFFICIENCY.pdf` (46 pages).
All equations below were re-extracted with PyMuPDF page-by-page after discovering that a
first-pass `pdftotext` extraction silently dropped every Greek symbol (ξ, θ, ρ, λ, χ, ε, α, β
all rendered as blank/garbled) — **do not trust any other paraphrase of this paper's math that
doesn't cite an equation number below; re-derive from here.**

**UPDATE**: the Online Appendix (`DGJ_AI_Trading_Market_Efficiency_V4_4_OA.pdf`, 77 pages) has
now been located and cleanly extracted to `docs/online_appendix_full_text.txt`. It confirms my
λᴺ/λᴹ re-derivation below is exactly correct (verified against OA eq. IA.2.9/IA.2.12), and
supplies the exact ΔC/liquidity/mispricing formulas and the exact mechanism-classification test
— all now incorporated below. Nothing in this spec is a heuristic proxy anymore except one
threshold-magnitude detail flagged explicitly in "Mechanism classification test" below (OCR
ambiguity in the OA's own PDF text, not something I'm guessing at the method level).

## Notation (confirmed against clean-extracted pages 10-29)

| Symbol | Meaning | Baseline value |
|---|---|---|
| `I` | number of informed speculators | 2 (swept 2..9 in Fig. 7) |
| `v̄` | mean of fundamental value `v_t` | 1 |
| `σ_v` | std of `v_t` | 1 (discretized: σ̂_v=0.938 at nv=10, use σ̂_v in all benchmark formulas per footnote 24) |
| `σ_u` | std of noise trader order flow `u_t ~ N(0,σ_u²)` | {0.1, 100} (swept, log-grid -5..5 in Fig. 2) |
| `ξ` (xi) | info-insensitive investor demand slope, `z_t = -ξ(p_t - v̄)`, eq (3.2) | 500 (environment param, swept — Fig. 1's y-axis) |
| `θ` | market maker's weight on pricing-error term, eq (3.3) | 0.1, **fixed constant, never swept** |
| `ρ` | informed speculators' subjective discount factor | 0.95 (swept 0.05..0.95 in Fig. 8) |
| `α` | Q-learning forgetting/learning rate, eq (2.4) | 0.01 |
| `β` | exploration-decay hyperparameter, eq (4.3) — **do not confuse with θ or any other symbol; this is its own hyperparameter** | 5e-7 |
| `ι` (iota) | grid-widening parameter for x/p discretization | 0.1 |
| `nv, nx, np` | grid sizes for v, x, p | 10, 15, 31 |
| `Tm` | market maker's rolling-window length | 10,000 |
| `Nsim` | independent simulation sessions per experiment | 1,000 |

## Core dynamics

**Bellman / Q-function** (eq 2.1-2.3):
`V_i(s) = max_x [E[π_i|s,x] + ρ E[V_i(s')|s,x]]`, `Q_i(s,x) = E[π_i|s,x] + ρE[V_i(s')|s,x]`,
`V_i(s) = max_x Q_i(s,x)`.

**Q-learning update** (eq 2.4):
```
Q_{i,t+1}(s_t, x_{i,t}) = (1-α) Q_{i,t}(s_t, x_{i,t}) + α [ π_{i,t} + ρ max_{x'} Q_{i,t}(s_{t+1}, x') ]
```
Only the visited `(s_t, x_{i,t})` cell updates; all others unchanged.

**ε-greedy action selection** (eq 2.6):
`x_{i,t} = argmax_x Q_{i,t}(s_t,x)` w.p. `1-ε_t`, else uniform-random over `X`.

**State-dependent exploration decay** (eq 4.3):
`ε_t(v) = exp(-β · n_t(v))`, where `n_t(v)` = number of times the CURRENT `v_t` grid value has
been visited so far in this session (i.e. one decay counter per `v` grid point, not a single
global counter). Note the paper's asymptotic-visits comment (footnote, §4.2): at baseline
`β=5e-7`, each `x∈X` is visited ≈`(nv/nx)·1/(1-exp(-β))≈1,333,333` times on average before
exploration "completes" for a given `v` grid point — useful as a unit test sanity check.

## Economic environment (eq 3.1-3.4, page 13-15)

- Per period: `v_t ~ N(v̄, σ_v²)`, informed speculator `i` observes `v_t` perfectly, chooses `x_{i,t}`.
- `u_t ~ N(0, σ_u²)` noise trader flow (independent of everything).
- `y_t ≡ Σ_i x_{i,t} + u_t` total order flow.
- `z_t = -ξ(p_t - v̄)` info-insensitive investor demand (eq 3.2).
- Market maker's problem (eq 3.3): `min_p E[(y_t+z_t)² + θ(p_t-v_t)² | y_t]`.
- **First-order condition / price rule (eq 3.4)**:
  ```
  p_t = [ξ/(ξ²+θ)]·y_t + [ξ²/(ξ²+θ)]·v̄ + [θ/(ξ²+θ)]·E[v_t|y_t]
  ```
- Speculator `i`'s profit: `π_{i,t} = (v_t - p_t)·x_{i,t}`.
- Speculator `i`'s problem (eq 3.1): `V_i(s_t) = max_x E[(v_t-p_t)x_{i,t} + ρV_i(s_{t+1}) | s_t, x_{i,t}]`.

## Theoretical benchmarks (page 16-17)

**Nash** (Benchmark I): each `i` solves `max_x E[(v_t-p^N(y_t))x | v_t]` given others play
`x^N(v_t)=χ^N(v_t-v̄)`, with `p^N(y)=v̄+λ^N y`, `y=x_i+(I-1)x^N+u_t`.

**Cartel** (Benchmark II): jointly `max_x E[(v_t-p^M(y_t))x|v_t]`, `p^M(y)=v̄+λ^M y`, `y=Ix+u_t`.

**λᴺ, λᴹ fixed points — CONFIRMED against Online Appendix eq. (IA.2.9), (IA.2.12)** (my
independent re-derivation from the main-text FOCs matched the OA's closed form exactly; use the
OA's notation below, it's authoritative):
```
χ^N = 1/[(I+1)λ^N],   λ^N = (θγ^N+ξ)/(θ+ξ²),   γ^N = (Iχ^N) / [(Iχ^N)² + (σ_u/σ̂_v)²]
χ^M = 1/(2Iλ^M),      λ^M = (θγ^M+ξ)/(θ+ξ²),   γ^M = (Iχ^M) / [(Iχ^M)² + (σ_u/σ̂_v)²]
```
Each pair is a circular/fixed-point system (λ↔γ↔χ) — solve numerically (e.g.
`scipy.optimize.brentq` on `f(λ)=λ-RHS(λ(γ(χ(λ))))` over bracket `(1e-8, 10)`), once per
`(I, ξ, θ, σ_u)` cell, cached — **not** re-solved every simulation period. Use `σ̂_v` (the
discretized-grid std, 0.938 at nv=10), not the nominal `σ_v=1`, per OA footnote matching main
text footnote 24.

**Expected profits in the two benchmarks** (OA Prop. IA.1/IA.2, useful test invariants):
```
π^N = σ̂_v² / [(I+1)² λ^N]        π^M = σ̂_v² / (4 I λ^M)
```
Both also equal `Π(χ,λ) = (1-λIχ)χσ̂_v²` evaluated at the respective `(χ,λ)` — check both formulas
agree in tests.

**As ξ→0 sanity check** (OA page 7, useful test): `χ^N→0`, `χ^M→0`, and
`p^N_t → v̄ + [I/(I+1)](v_t-v̄) + ξ⁻¹u_t`, `p^M_t → v̄ + (1/2)(v_t-v̄) + ξ⁻¹u_t`.

**Collusive equilibrium** (Def 3.1, page 17): (i) supra-competitive profits vs. Nash, (ii)
unilateral deviation possible but imposes costs on others. Normalized profitability (eq 3.6,
stated on page ~18): `Δ^C ≡ (π^C-π^N)/(π^M-π^N) ∈ (0,1]`.

## Simulation measures — EXACT formulas from Online Appendix §4.1 (eq. IA.4.1-IA.4.7)

Computed per session, over `T=100,000` periods immediately after that session's convergence
period `Tc` (i.e. periods `Tc..Tc+T`), then reported as the average/distribution across
`Nsim=1000` sessions.

**ΔC — matched-path comparison, NOT population averages** (eq IA.4.1-IA.4.3). For every period
`t` in the measurement window, recompute what Nash and Cartel benchmark speculators would have
earned given the SAME realized `v_t, u_t` draw the AI simulation actually saw:
```
π^N_t = [v_t - p^N(I·x^N(v_t)+u_t)] · x^N(v_t),   x^N(v_t)=χ^N(v_t-v̄),  p^N(y)=v̄+λ^N y
π^M_t = [v_t - p^M(I·x^M(v_t)+u_t)] · x^M(v_t),   x^M(v_t)=χ^M(v_t-v̄),  p^M(y)=v̄+λ^M y
π̄_i = mean_t(π_{i,t}) over the window (actual AI profit); π̄^N, π̄^M similarly from the formulas above
ΔC_i = (π̄_i - π̄^N) / (π̄^M - π̄^N),     ΔC = mean_i(ΔC_i)
```
This means: run the SAME shock path through the theoretical benchmark formulas, not just use
their unconditional expected values — implement `benchmarks.py` so it can score an arbitrary
`(v_t,u_t)` path under `x^N`/`x^M`, not just return the scalar `π^N`/`π^M` expectations.

**Trading policy estimate** `χ̂` (eq IA.4.4): OLS regress `x_{i,t} = χ̂_0 + χ̂_1 v_t + ε_t` over the
measurement window per speculator, average `χ̂_1` across speculators → `χ̂^C`.

**Price informativeness** (eq IA.4.5, closed form once `χ̂^C` is estimated — no need to compute
`var(x_t)` directly from simulated data, though that should also match as a consistency check):
```
I^C = (I·χ̂^C)² · (σ̂_v/σ_u)²
```

**Market liquidity** (eq IA.4.6 — simple closed form, NOT a numerical derivative):
```
L^C_t = 1 / |1 - ξ·λ̂_t|
```
where `λ̂_t` is the market maker's period-`t` adaptive price-impact estimate from eq (4.2)
(the rolling-regression `λ̂_t`, already computed every period for pricing — reuse it). Average
over the measurement window for `L^C`.

**Mispricing** (eq IA.4.7 — also closed form):
```
E^C_t = |1 - λ̂_t · I · χ̂^C| · |v_t - v̄|
```
Average over the window for `E^C`.

## Q-learning simulation setup (§4.1-4.2, page 22-26)

**State**: `s_t = {p_{t-1}, v_{t-1}, v_t}` (current signal + one-period memory of price & fundamental).

**Adaptive market maker** (eq 4.1-4.2): rolling window `D_t={v_{t-τ},p_{t-τ},z_{t-τ},y_{t-τ}}_{τ=1}^{Tm}`.
Two OLS regressions each period: `z_{t-τ} = ξ̂₀ - ξ̂₁ p_{t-τ} + ε_z`, `v_{t-τ} = γ̂₀ + γ̂₁ y_{t-τ} + ε_v`.
Plug-in price rule:
```
p̂_t(y) = γ̂₀,t + λ̂_t · y,   λ̂_t = (θ γ̂₁,t + ξ̂₁,t) / (θ + ξ̂₁,t²)
```
(θ is the fixed constant from eq 3.3, NOT re-estimated). Implement the rolling regressions with
running sufficient statistics (sums of squares/cross-products) updated incrementally each
period — O(1) per period, not an O(Tm) refit.

**Per-period protocol** (page 22-23, exact order matters):
1. Each speculator `i` picks exploration (prob `ε_t(v_t)`) or exploitation (prob `1-ε_t(v_t)`),
   submits `x_{i,t}` per eq (2.6), using CURRENT state `s_t={p_{t-1},v_{t-1},v_t}`.
2. Noise trader draws `u_t ~ N(0,σ_u²)`.
3. Market maker computes `p_t = p̂_t(y_t)` from eq (4.2), where `y_t=Σx_{i,t}+u_t`.
4. Info-insensitive investors submit `z_t=-ξ(p_t-v̄)`; each speculator realizes `π_{i,t}=(v_t-p_t)x_{i,t}`.
5. Next state `s_{t+1}={p_t,v_t,v_{t+1}}`, `v_{t+1}~N(v̄,σ_v²)` drawn independently; each `i` updates
   `Q_i` at `(s_t,x_{i,t})` via eq (2.4); market maker appends `(v_t,p_t,z_t,y_t)` to its rolling window.

**Discretization**:
- `v` grid (page 24): `v_k = v̄ + σ_v·Φ⁻¹((2k-1)/(2n_v))`, k=1..nv (Gaussian quantiles, equal
  probability mass per grid point). Compute σ̂_v from this grid and use σ̂_v (not the nominal
  σ_v=1) in χᴺ/χᴹ/λᴺ/λᴹ formulas per footnote 24.
- `x` grid: interval `[x^M - ι(x^N-x^M), x^N + ι(x^N-x^M)]` for `v̄`-relative-positive side,
  mirrored `[x^N - ι(x^M-x^N), x^M + ι(x^M-x^N)]` for the negative side, `nx` equally spaced points.
  (χᴺ, χᴹ, x^N, x^M all computed using the derived λᴺ, λᴹ above.)
- `p` grid: `p^H = v̄ + λ^N·(I·max(x^M,x^N) + 1.96σ_u)`, `p^L = v̄ + λ^N·(I·min(x^M,x^N) - 1.96σ_u)`,
  then `[p^L - ι(p^H-p^L), p^H + ι(p^H-p^L)]` into `np` points.

**Initial Q-matrix** (page 25): for each speculator `i`, state `s=(p,v_{lag},v)∈P×V×V`, action `x∈X`
(note: RHS below does not actually depend on `p` or `v_lag`, only on the current `v` component of
`s` — same value is replicated across all `(p,v_lag)` combinations for a given `v`):
```
Q_{i,0}(s,x) = [1/((1-ρ)·n_x)] · Σ_{x_{-i}∈X} [ v - (v̄ + λ^N·(x + (I-1)x_{-i})) ] · x
```
(expected discounted profit if the opponent(s) randomize uniformly over `X` and noise flow is
0, extended to infinite horizon via `1/(1-ρ)`). Initial states `s_0={p_{-1},v_{-1},v_0}` drawn
uniformly over `P×V×V`.

**Convergence criterion** (page 26): a session has converged once every speculator's argmax
action is unchanged for 1,000,000 consecutive periods, call this period `Tc`. Range across
experiments: 20M-50B periods. `Nsim=1000` independent sessions per experiment cell. Measures
(ΔC, χ̂, I, L, E — see "Simulation measures" above) are then computed over the NEXT `T=100,000`
periods after `Tc` (OA §4.1) — so a full session run is `Tc + 100,000` periods, not just `Tc`.

## Mechanism classification test — EXACT protocol from Online Appendix §4.5 (page 62)

This is the real test (not a heuristic). At `t=0` the session has converged (steady state,
period index resets to 0 for the IRF experiment). At `t=3`, inject a calibrated exogenous shock
`u_shock` to the noise trader's order flow — **calibrate its magnitude so the resulting price
deviation `p̃_t` at `t=3` equals exactly 1.2%** (this is the paper's "medium deviation" size;
`p̃_t ≡ (p_t-p̄_t)·sgn(v_t-v̄)`, `p̄_t` = long-run mean — same sign-adjustment as main-text Fig. 3).
Track `x̃_{i,t} ≡ (x_{i,t}-E[x_{i,t}])/E[x_{i,t}]` (order-flow deviation) at `t=4` for both
speculators `i=1,2`, averaged across replications (the paper reruns this shock across many
sample paths / uses the converged Q-tables directly rather than re-simulating from scratch each
time).

Classification (OA page 62, exact quote structure):
```
price-trigger:  x̃_{i,4} > x̄_high   for BOTH i=1,2   (aggressive "punishment" reaction)
over-pruning:   |x̃_{i,4}| < x̄_low  for BOTH i=1,2   (no reaction)
unclassified:   anything else (mixed/ambiguous signal)
```
Thresholds as stated in the OA text: `x̄_low = 5×10⁻⁵`. The OA's stated relation between the two
thresholds is garbled by an OCR artifact in the appendix PDF itself (renders as "x̄_high = 10x̄" —
the multiplier is unambiguous, its target variable isn't 100% clean in the extracted text).
Implement `x̄_high = 10 · x̄_low = 5×10⁻⁴` as the most natural reading (a threshold pair one order
of magnitude apart, both being small fractional deviations, matches the qualitative description
"sufficiently high" vs "sufficiently low" threshold and the scale of deviations shown in Fig 3-6
of the main text) — but expose both thresholds as config parameters so they're trivially
adjustable if cross-checking against reproduced figures suggests otherwise.

For qualitative validation while building this (matches main-text Figures 3-6, pages 30-35):
price-trigger sessions show reversion-to-mean for small shocks but BOTH speculators trading
aggressively at t+1 after medium/large shocks (mutual punishment, converging to similar
magnitude regardless of shock size, still elevated but decaying at t+2); over-pruning sessions
show near-zero response to shocks at all deviation sizes, and under a unilateral-deviation IRF
variant, the non-deviating speculator stays completely unresponsive even though the deviator
exploited it for a one-period gain (no punishment).

A secondary, more elaborate validity check exists in OA §4.2 (Fershtman-Pakes experience-based-
equilibrium bias test, eq IA.4.8-IA.4.13) — it confirms BOTH mechanisms are experience-based
equilibria (small ~0.16-0.19% bias) but does NOT distinguish between them, so it is not needed
for `classify_mechanism.py`; treat it as optional future robustness tooling, not required for
this build.

## Comparative statics (§6, page 40-41, for sweep configs)

- Vary `I` from 2 to 9 (Fig. 7), both `σ_u=0.1` and `σ_u=100`, baseline `ξ=500`.
- Vary `ρ` from 0.05 to 0.95 (Fig. 8), both `σ_u` levels, baseline `ξ=500`. At high `σ_u`, `ρ`
  should have little effect (over-pruning mechanism) — good regression-test invariant.

## Known illustrative numbers (for sanity-checking a working implementation)

- Baseline `ξ=500, σ_u=0.1`: `Δ^C≈0.75`, informed speculators ~10% above non-collusive profit,
  each earns ≈54 average profit; info-insensitive investors lose ≈108 total.
- Baseline `ξ=500, σ_u=100`: each speculator earns ≈54 average profit, from ≈88 loss by
  info-insensitive investors and ≈20 loss by noise traders.
