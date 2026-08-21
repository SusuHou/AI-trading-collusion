# AI Trading Collusion — replication of Dou, Goldstein & Ji (NBER WP 34054)

Research-code replication of *"AI-Powered Trading, Algorithmic Collusion, and Price
Efficiency"*: two informed speculators using tabular Q-learning trade a short-lived
asset each period in a Kyle-style market against a noise trader, information-insensitive
investors, and an adaptive (OLS-learning) market maker. The paper shows the algorithms
autonomously sustain supra-competitive profits through two mechanisms (price-trigger
strategies at low noise-trading risk, over-pruning learning bias at high risk).

Primary source of truth: **`docs/paper_spec.md`** (equation-by-equation spec extracted
from the paper + Online Appendix). `docs/paper_full_text.txt` and
`docs/online_appendix_full_text.txt` are the raw extractions for cross-reference.

## Setup

```bash
python -m venv .venv                       # Python 3.14 (see "implementation choices")
.venv/Scripts/python -m pip install -r requirements.txt   # Windows
# .venv/bin/python -m pip install -r requirements.txt     # Linux/macOS
```

## Tests (run these first — `env/benchmarks.py` is the correctness gate)

```bash
.venv/Scripts/python -m pytest tests/ -q
```

`tests/test_benchmarks.py` verifies the λᴺ/λᴹ fixed points to machine precision
(residuals, the exact identities χᴺ(I+1)λᴺ = 1 and 2Iχᴹλᴹ = 1, the OA closed form
λ = (θγ+ξ)/(θ+ξ²), the FOCs, closed-form profits, grids, the matched-path scorer,
metric closed forms) and that `MarketEnv` reproduces p^N/p^M and benchmark profits
exactly when speculators are forced to play x^N/x^M with zero learning.
`tests/test_qlearn_toy.py` hand-traces the eq (2.4) update, the eq (4.3) ε decay,
eq (2.6) branching, and the convergence tracker.

## Local proof-of-concept

```bash
.venv/Scripts/python -m phase1_qlearning.run_session --config configs/poc.yaml \
    --sigma-u 0.1 --seed 11 --out results/poc_su0.1_seed11.npz
.venv/Scripts/python -m phase1_qlearning.run_session --config configs/poc.yaml \
    --sigma-u 100 --seed 12 --out results/poc_su100_seed12.npz
.venv/Scripts/python -m phase1_qlearning.classify_mechanism results/poc_su*.npz
.venv/Scripts/python -m phase1_qlearning.aggregate "results/*.npz" --csv results/sessions.csv
.venv/Scripts/python -m phase1_qlearning.plots "results/*.npz" --out-dir results/figures
```

Each run simulates 40 independent sessions (vectorized in one NumPy batch) for
2,000,000 periods (~3 minutes) plus the paper's 100,000-period measurement window.

### POC results obtained with the commands above

| | σ_u = 0.1 | σ_u = 100 |
|---|---|---|
| ΔC (mean over 40 sessions) | **+0.574** | **+0.548** |
| ΔC (median / p1 / p99) | +0.582 / +0.488 / +0.649 | +0.548 / +0.516 / +0.578 |
| profit gain vs Nash | 1.072 | 1.068 |
| sessions fully converged | 0/40 | 0/40 |
| mechanism (OA 4.5 test) | 18% price-trigger, 82% unclassified | 20% price-trigger, 80% unclassified |

Both regimes show clearly supra-competitive profits (ΔC ≈ 0.55–0.57 vs the paper's
full-scale ΔC ≈ 0.75), and the ~7% profit gain matches the paper's "~10% above
non-collusive profit". **Caveats, stated plainly:** (i) no session met even the reduced
100k-period stability criterion within the 2M-period cap — the paper needs 20M–50B
periods per session, so POC sessions are best-effort, not converged; (ii) the OA §4.5
mechanism test mostly returns "unclassified" here because its over-pruning threshold
(|x̃| < 5×10⁻⁵) presumes fully converged policies — unconverged policies react
idiosyncratically to the price state; the shock calibration itself is exact (+1.2000%).

## Full-scale runs on SLURM

`phase1_qlearning/slurm/run_experiment.sbatch` is a job-array template (one array task
= one 10-session batch of one sweep cell; 100 tasks/cell = Nsim 1000). Fill in the
`# TODO(user)` placeholders (partition, account, module/venv setup), then e.g.:

```bash
CONFIG=configs/sweep_sigma_u.yaml sbatch --array=0-1099 phase1_qlearning/slurm/run_experiment.sbatch  # 11 cells
CONFIG=configs/sweep_I.yaml       sbatch --array=0-1599 phase1_qlearning/slurm/run_experiment.sbatch  # 16 cells
CONFIG=configs/sweep_rho.yaml     sbatch --array=0-1999 phase1_qlearning/slurm/run_experiment.sbatch  # 20 cells
```

Sweep cells are the row-major cartesian product of the config's `sweep:` lists
(`run_session --cell-index` resolves them; seeds are unique per task). These configs
use the paper-exact hyperparameters (β = 5×10⁻⁷, 1M-period convergence streak, no
period cap, 100k measurement window).

## Numba backend (accelerated hot loop)

The pure-NumPy path above amortizes interpreter overhead across a *batch* of
sessions per array op, but each individual session still advances one period
at a time — and the paper's hardest cells need up to **50 billion** periods
for a *single* session to converge (their own C++ implementation on a
400-core cluster runs at ~2-3M periods/sec/session). At the NumPy backend's
~12-16k periods/sec/session, one such session alone would take on the order
of weeks to months. `phase1_qlearning/qlearn_numba.py` adds a JIT-compiled
alternative that closes most of that gap.

**Why a second venv.** This machine's default Python is 3.14, which Numba
does not yet support. There is a second interpreter at
`C:\Program Files\Python310\python.exe` (3.10) that Numba does support:

```bash
"C:\Program Files\Python310\python.exe" -m venv .venv310
.venv310/Scripts/python -m pip install numba "numpy<2.3" scipy matplotlib PyYAML pytest
```

**Correctness test (run this before trusting any numba-backend output):**

```bash
.venv310/Scripts/python -m pytest tests/test_numba_parity.py -q
```

This file has two tests, both running an identical config/seed through
`run_session.run(..., backend="numpy")` and `run(..., backend="numba")` — same
rng call sequence — and asserting **bit-exact** equality
(`np.testing.assert_array_equal`, not just `allclose`) of the Q-table,
greedy-policy cache, visit counters, per-session `converged_at`/
`best_streak`, and the market maker's rolling-window sufficient statistics,
plus the post-training evaluation metrics to 1e-12:

- `test_numba_matches_numpy_reference_bit_exact_toy_grid` — tiny grid
  (nv=4/nx=15/np=9/Tm=200), with `conv_streak` small enough that some
  sessions actually converge, exercising the "frozen/converged session"
  (`active=False`) code path in both backends.
- `test_numba_matches_numpy_reference_bit_exact_baseline_grid` — the actual
  `configs/baseline.yaml` grid sizes (nv=10/nx=15/np=31/Tm=10000), the same
  grid the throughput benchmark below was measured on.

Both passed bit-for-bit on this machine. `pytest tests/` (the full test
suite, including `phase2_llm`'s tests) also passes unmodified in both
`.venv` (`97 passed, 1 skipped` — `test_numba_parity.py`'s whole module is
reported as one skip when numba isn't importable, rather than as a
collection failure) and `.venv310` (`99 passed` — the same 97 tests plus
the 2 numba parity tests, now actually collected and run).

**Design: single-session kernel, batched with `numba.prange`.** The hot
loop — ε-greedy action selection (eq 2.6/4.3), price/profit computation
including the market maker's O(1) rolling-window update (eq 3.2/4.1/4.2),
and the Q-update (eq 2.4) — is one `@njit(parallel=True)` function
(`_run_chunk` in `qlearn_numba.py`) with `for k in range(n)` (periods)
nested inside `for b in prange(batch)` (sessions): one OS thread per
session, each grinding through its own periods strictly sequentially (that
per-session sequential dependency is exactly what makes a single session's
period count the bottleneck, and no amount of batching removes it — batching
only lets *other* sessions' progress overlap in wall-clock time). Every
array op from the NumPy reference is transcribed into scalar/1-D-typed-array
form with the same operation order (no fastmath, no reordering), which is
what makes the bit-exact parity above possible. The market maker's periodic
exact re-summation (`MarketEnv.resync_every`, cancels floating-point drift)
is kept *outside* the `@njit` kernel — `run_chunk_numba()` splits a chunk at
resync boundaries and calls the same `buf.sum(axis=1)` NumPy reduction the
reference uses, so pairwise-summation rounding stays identical too.

**Benchmark (baseline.yaml grid: nv=10, nx=15, np=31, Tm=10000, I=2), this
machine (20 logical CPUs, Intel Family 6 Model 198):**

| backend | batch | periods/sec aggregate | periods/sec **per session** |
|---|---|---|---|
| numpy (reference) | 10 | 160,509 | 16,051 |
| numba | 1 (single-threaded) | 8,432,033 | **8,432,033** |
| numba | 4 | 13,947,887 | 3,486,972 |
| numba | 10 (matches the sbatch template's `run.batch`) | 19,161,873 | 1,916,187 |
| numba | 20 | 23,774,538 | 1,188,727 |
| numba | 40 | 26,836,074 | 670,902 |

Per-session throughput falls as `batch` grows past the machine's core count
because sessions then contend for cores/cache/memory bandwidth — aggregate
throughput keeps rising, but the number that matters for a single
50-billion-period session is the per-session column. **Headline result:** a
single numba session (`batch=1`, one thread, no contention) hits **~8.4M
periods/sec**, which is *faster* than the paper's own ~2-3M periods/sec/
session C++ implementation, and roughly 500x faster than this repo's NumPy
backend. Even at `batch=10` — the sbatch template's current `run.batch` —
per-session throughput (~1.9M/s) is within the paper's own C++ range,
provided `--cpus-per-task` is bumped to match `run.batch` (10) so numba's 10
`prange` threads each get a dedicated core instead of oversubscribing a
single allocated CPU. At `batch=1`/`--cpus-per-task=1` (one SLURM task per
session), the worst-case 50-billion-period session would take
50e9 / 8.4e6 ≈ **99 minutes** instead of ~53 days — tractable well within a
single sbatch `--time` allocation.

**Task item 7 (report honestly, don't overstate):** the numba speedup here
is *not* limited by irreducible scalar branching — the achieved per-session
number (8.4M/s at batch=1) already exceeds the paper's own optimized C++
throughput on this hardware, so there's no meaningful headroom being left on
the table by JIT limitations. The only real constraint is the
*aggregate*-throughput ceiling from core count and memory-bandwidth
contention when batching many sessions onto one node — a hardware/
scheduling property, not a JIT-compilation limitation — mitigated by
matching `--cpus-per-task` to `run.batch`, or by preferring more
single-session SLURM tasks over large in-process batches.

**Usage:**

```bash
.venv310/Scripts/python -m phase1_qlearning.run_session --config configs/baseline.yaml \
    --backend numba --sigma-u 0.1 --seed 1 --out results/baseline_numba.npz
```

`run_session.py --backend {numpy,numba}` (default `numpy`) is the only new
CLI surface; everything else (config resolution, sweep cells, output
schema, the `evaluate()` measurement window) is identical between backends,
and `env/` and `phase1_qlearning/qlearn.py` are unmodified — the numba path
is purely additive (`phase1_qlearning/qlearn_numba.py`), imported lazily so
the default `.venv` (no numba) never needs to import it.

**For the eventual real SLURM cluster:** run `module avail python` (or
equivalent). If a Python ≥3.11 with a working numba wheel is available,
prefer `BACKEND=numba` (see `phase1_qlearning/slurm/run_experiment.sbatch`,
which now accepts a `BACKEND` env var and documents matching
`--cpus-per-task` to `run.batch` when using it) for the full-scale sweeps —
it is the difference between tractable and not at the paper's worst-case
period counts. If only Python 3.14 (or another numba-unsupported version)
is available, `BACKEND=numpy` still works everywhere and is the safe
default; budget the `--time` allocation accordingly (convergence can need
20M-50B periods at ~12-16k/s/session).

## File → paper mapping

| File | Implements |
|---|---|
| `env/benchmarks.py` | v grid + σ̂_v (p.24, fn.24); λᴺ/λᴹ fixed points (OA IA.2.9/IA.2.12); χᴺ = 1/[(I+1)λᴺ], χᴹ = 1/(2Iλᴹ); πᴺ, πᴹ closed forms (OA Prop IA.1/IA.2); matched-path profit scorer (OA IA.4.2/IA.4.3); x grid, p grid (p.24); initial Q-matrix (p.25) |
| `env/market.py` | price rule eq (3.4)/(4.2); investor demand eq (3.2); adaptive MM rolling regressions eq (4.1) with O(1) running sufficient statistics; per-period protocol steps 2–4 (p.22-23) |
| `env/metrics.py` | ΔC matched-path (OA IA.4.1–IA.4.3); χ̂ regression (IA.4.4); informativeness I^C (IA.4.5); liquidity L^C (IA.4.6); mispricing E^C (IA.4.7); relative profit gain |
| `phase1_qlearning/qlearn.py` | Q update eq (2.4); ε-greedy eq (2.6); state-dependent decay eq (4.3); state encoding s = (p_{t-1}, v_{t-1}, v_t) |
| `phase1_qlearning/convergence.py` | 1M-consecutive-stable-periods criterion (p.26), configurable for POC |
| `phase1_qlearning/run_session.py` | full per-period protocol (p.22-23), measurement window (OA §4.1), CLI + sweep-cell resolution, npz output |
| `phase1_qlearning/qlearn_numba.py` | performance port only (no new economics) of `env/market.py` step + `phase1_qlearning/qlearn.py` act/update into one `@njit(parallel=True)` kernel — see "Numba backend" above |
| `phase1_qlearning/classify_mechanism.py` | exact OA §4.5 IRF test: 1.2% calibrated price shock at t=3, x̃_{i,4} thresholds x̲ = 5×10⁻⁵, x̄ = 10x̲ |
| `phase1_qlearning/aggregate.py`, `plots.py` | Fig 2 Panel A analog (ΔC vs log σ_u) + Fig 3/4-style IRF plot, per-session CSV |

## Implementation choices & documented judgment calls

1. **Pure NumPy, vectorized over sessions, is the default and reference backend.** All
   dependencies have Python 3.14 wheels; Numba does not support 3.14 yet, and this is the
   Python this machine defaults to, so the NumPy path has to keep working regardless.
   Batching ~40 independent sessions into one array op per period amortizes interpreter
   overhead to ~12-16k periods/s aggregate, plenty for the POC and a sensible unit for
   SLURM tasks. The batch dimension is also what later LLM-rollout experiments want. A
   second, JIT-compiled backend (`--backend numba`, needs a separate Python 3.10 venv —
   see "Numba backend" below) is available for the full-scale sweep, where the paper's
   up to 50-billion-period single-session worst case makes the NumPy backend's raw
   per-period speed the binding constraint, not interpreter-overhead amortization.
2. **POC hyperparameter scaling** (`configs/poc.yaml` only; `baseline.yaml` and sweep
   configs are paper-exact): β = 4×10⁻⁵ instead of 5×10⁻⁷ so the explore→exploit
   lifecycle completes within the 2M-period cap (at the paper's β, exploration needs
   tens of millions of periods; the paper reports robustness to α/β in OA 4.12);
   convergence streak 100k instead of 1M.
3. **Market-maker initialization** (paper is silent on the first Tm periods): the
   rolling window is warm-started with Tm synthetic observations generated exactly from
   the Nash benchmark (balanced factorial design over the v grid × a rescaled Gaussian-
   quantile u grid), so the regressions recover ξ̂₁ = ξ and λ̂ = λᴺ to machine precision
   at t = 0 and drift to real data within the first Tm periods.
4. **Ω_M consistency fix**: an earlier draft of the spec wrote Ω_M = I/(2λᴹ); the
   correct aggregate informed sensitivity is Ω = I·χ, i.e. Ω_M = 1/(2λᴹ) — confirmed by
   the OA closed form γ^M = (Iχᴹ)/[(Iχᴹ)² + (σ_u/σ̂_v)²]. Implemented the latter.
5. **State-scaled action grid**: X(v) = (v − v̄)·{c_j}, c_j equally spaced on
   [χᴹ − ι(χᴺ−χᴹ), χᴺ + ι(χᴺ−χᴹ)] — the paper's per-side intervals for v > v̄ / v < v̄
   both reduce to this single multiplier grid, and it contains both benchmark
   strategies exactly at every v (tests verify).
6. **Price in the state** is snapped to the nearest p-grid point; the *payoffs*, investor
   demand, and the MM's window use the continuous price from eq (4.2).
7. **IRF classifier estimation**: expectations in x̃ = (x − E[x])/E[x] are estimated with
   paired common-random-number rollouts (base vs shock share every draw), order flows
   sign-adjusted by sgn(v_t − v̄), and MM coefficients held fixed during the 9-period
   window. Under CRN a truly price-insensitive (over-pruning) policy yields x̃ = 0
   exactly, which is what makes the OA's 5×10⁻⁵ threshold operational. Thresholds are
   CLI flags (`--x-low`, `--x-high`) since the OA's own text has an OCR ambiguity in
   the x̄ = 10x̲ relation.
8. **Convergence bookkeeping**: converged sessions are frozen (greedy, no further Q
   updates) rather than terminated, so a whole batch runs lockstep; `converged_at` and
   `best_streak` are recorded per session. The spec's ξ→0 price-limit sanity check was
   not added as a test: with θ fixed at 0.1 the fixed point keeps λ bounded as ξ→0, so
   that limit (λ → ξ⁻¹) belongs to a different (market-clearing) limit order — flagged
   rather than silently asserted.

## Phase 2, Stage A — LLM agents (no-training, in-context pilot)

Phase 1 replicates the paper's own method (tabular Q-learning). Phase 2 asks a different
question: does a *frozen* language model (target: Qwen-3.5, served via vLLM on a
SLURM+GPU cluster) develop similar supra-competitive behavior when it plays the informed
speculator role via in-context pattern-matching alone — no weight updates? Stage A (this
code) is the cheap pilot that validates the prompt design, action parsing, and
environment wiring, and gives a rough baseline signal, before investing in Stage B
(GRPO fine-tuning — **not built**, a separate follow-up).

Two independently-sampled instances of the same model play the two informed speculators,
reusing the SAME `env/market.py` mechanics, `env/benchmarks.py` action grid, and
`env/metrics.py` ΔC measure as Phase 1 — nothing about the market/benchmarks is
reimplemented. Matching the paper's information structure exactly: each agent sees only
its OWN past `(v, action, profit)` history and the lagged public price `p_{t-1}`; it
never observes the other speculator's actions, profits, or even that a second speculator
exists. The system prompt (`phase2_llm/agent_llm.py::SYSTEM_PROMPT`) explains the market
mechanics in plain language — private value each period, price set by a market maker
from aggregate order flow, information-insensitive investors and noise traders also
present — and never uses the words "collude," "coordinate," or describes another
player's strategy; the model is only told its goal is to maximize its own cumulative
profit. `tests/test_llm_pilot.py::TestPromptConstruction` enforces both properties
(forbidden vocabulary + no cross-agent data leakage) as regression tests.

### Files

| File | Role |
|---|---|
| `phase2_llm/agent_llm.py` | prompt construction (`build_messages`), defensive output parsing (`parse_action_index`), the `LLMAgent` orchestrator (prompt → backend call → parse → retry-once → documented fallback → own rolling history), and two backends: `OpenAICompatibleBackend` (any real chat-completions endpoint) and `MockBackend` (network-free, for offline testing) |
| `phase2_llm/llm_pilot.py` | runs `N` episodes × `E` periods of the two-speculator market through `MarketEnv`, logs a full per-period JSONL transcript, computes `ΔC` via `env/metrics.py::delta_c_matched` over the pilot's own realized path |
| `tests/test_llm_pilot.py` | action-parsing unit tests, prompt-construction/no-leakage tests, retry/fallback tests, and an end-to-end mock-backend pilot run |

### Running the offline mock pilot (no network required)

```bash
.venv/Scripts/python -m phase2_llm.llm_pilot --backend mock \
    --episodes 5 --periods 30 --seed 0 --out results/phase2_llm/pilot_mock.jsonl
```

The `MockBackend` stands in for a real LLM with the identical `.complete()` interface, so
the whole pipeline (prompt construction → env wiring → action parsing → transcript
logging → ΔC computation) runs and is checkable without any model access. Three modes
exercise different parts of the pipeline:

- `--mock-mode random` (default): uniform-random valid index each call — the closest
  thing to a "no-signal" policy the mock can produce, since even random draws from the
  paper's action grid `X(v) = (v−v̄)·{c_j}` are always sign-correct (all `c_j > 0`), which
  is why even this naive policy scores a positive ΔC (see results below — this is a
  property of the environment's action-grid design carried over unmodified from
  `env/benchmarks.py`, not an artifact of the mock).
- `--mock-mode target_nash` / `target_cartel`: a scripted policy that always plays the
  grid point closest to χᴺ / χᴹ — a sanity probe for the ΔC *pipeline itself*
  (independent of any simulated LLM behavior), used to confirm the metric is wired
  correctly before trusting real-model results.
- `--malformed-rate` (default 0.1 in the offline pilot): fraction of mock responses that
  are deliberately not valid JSON, to exercise the retry/fallback path end-to-end.

**Mock-pilot sanity check actually run in this session** (`configs/poc.yaml` economics,
ξ=500, σᵤ=0.1, 5 episodes × 30 periods, `results/phase2_llm/pilot_mock_*.jsonl`):

| mock policy | ΔC (mean) | interpretation |
|---|---|---|
| `random` (malformed_rate=0.1) | +0.635 | naive policy still trades in the sign-correct direction (grid design), so profits land well above Nash |
| `target_nash` (noise=0) | −0.029 | ≈ 0 as expected — mimicking χᴺ should score at Nash |
| `target_cartel` (noise=0) | +1.000 | ≈ 1 exactly as expected — mimicking χᴹ should score at the cartel benchmark |

The `target_nash`/`target_cartel` probes landing at ≈0/≈1 confirm `delta_c_matched` is
wired correctly against the pilot's own realized `(v_t, u_t)` path (matched-path
comparison, OA eq IA.4.1–IA.4.3), independent of anything about LLM behavior. The
`random` transcript (300 rows, `results/phase2_llm/pilot_mock_random.jsonl`) was
inspected by hand: profits are the right order of magnitude (tens, matching the paper's
"~54 average profit" at this ξ/σᵤ cell), a malformed response was correctly retried once
then fell back to the documented middle-grid-index default, and `p_lag` is `null` only
on each episode's first period as expected.

### Pointing at a real endpoint later

`OpenAICompatibleBackend` targets any OpenAI-compatible `/chat/completions` endpoint —
in particular vLLM's OpenAI-compatible server, the expected way Qwen-3.5 will be served
on the SLURM cluster:

```bash
# on the cluster, once vLLM is serving Qwen-3.5:
#   vllm serve Qwen/Qwen3.5-<size>-Instruct --port 8000 \
#       --guided-decoding-backend outlines   # (or xgrammar/lm-format-enforcer)
.venv/Scripts/python -m phase2_llm.llm_pilot --backend openai \
    --base-url http://localhost:8000/v1 \
    --model Qwen/Qwen3.5-<size>-Instruct \
    --episodes 5 --periods 30 --out results/phase2_llm/pilot_qwen.jsonl
```

`--base-url`/`--model`/`--api-key` fall back to `$OPENAI_BASE_URL` / `$OPENAI_MODEL` /
`$OPENAI_API_KEY` if not passed. `response_format` (JSON-schema guided decoding) is sent
by default and the backend silently retries once without it if the server 4xxs on the
field, since not every OpenAI-compatible server implements guided decoding the same way.
No endpoint was reachable from this development machine (checked `$OPENAI_*` env vars
and local `:11434`/`:8000` — nothing listening), so this path is implemented and
documented but not run against a live model in this session; the `MockBackend` pilot
above is the actual verification for this repo state.

### Judgment calls

- **Retry/fallback policy**: one retry with a corrective follow-up message
  (`LLMAgent.choose_action`), then fall back to the middle grid index (a moderate order
  size, not an extreme one) if the second attempt is still unparseable. Every
  malformed/fallback event is counted per-agent and logged per-period in the transcript
  (`malformed`, `used_fallback`, `n_attempts` fields) and summarized in the run's
  `*.summary.json`.
- **What the LLM sees for `p_{t-1}`**: the continuous realized price, not the Q-learner's
  grid-snapped index — the LLM has no need for `env`'s tabular state encoding, so passing
  it the exact number is simpler and loses no information.
- **Batching**: `llm_pilot.py` batches `MarketEnv` over episodes (one shared vectorized
  env instance, `batch=N`), but LLM calls themselves are made one-at-a-time per
  (episode, speculator, period) since a real inference server is the eventual target, not
  a batched local model; this keeps the code identical whether `--backend` is `mock` or
  `openai`.
- **Environment params**: reuses `configs/poc.yaml` by default (read-only — Stage A never
  writes to `configs/`) so the LLM plays under the same economics already validated in
  Phase 1, including the same `Tm=10,000`-period market-maker warm start; with a pilot of
  only 20–40 real periods the MM stays close to its Nash-consistent warm start throughout
  a run, which is expected and consistent with Phase 1's own initialization.

### What Stage B adds (now built — see the next section)

Stage B (GRPO fine-tuning) takes the same prompt/parsing/environment wiring built here
and adds a training loop: sample rollouts with this same two-agent protocol, score them
by realized profit, and update the model's weights via GRPO — turning the "can a frozen
model pattern-match its way to collusion" question this pilot answers into "can training
push it further." As predicted, nothing in Stage A needed to change: Stage B only ADDS
files (`grpo_env_adapter.py`, `reward.py`, `train_grpo.py`, `verl_agent_loop.py`, the
`phase2_llm/slurm/` scripts) and imports Stage A's functions unmodified.

## Phase 2, Stage B — GRPO fine-tuning scaffold (verl)

Stage B turns Stage A's frozen pilot into an RL training loop: fine-tune a Qwen model
(the plan calls it "Qwen 3.5"; the model id/path is fully configurable and nothing here
assumes a specific variant) so a *shared* policy playing BOTH speculator seats
(self-play — two independently-sampled instances of the same weights, mirroring the
paper's symmetric agents and Stage A's two-`LLMAgent`-one-backend pattern) learns a
trading policy by GRPO: group-relative advantage over parallel market episodes, no
critic network.

**Honest status summary — what is genuinely tested vs. structurally correct only:**

| Piece | Status |
|---|---|
| `phase2_llm/grpo_env_adapter.py` — episode rollout source | **implemented + tested** (mock-driven, offline) |
| `phase2_llm/reward.py` — GRPO advantages | **implemented + tested**, incl. numerical parity vs verl 0.8.0's own `compute_grpo_outcome_advantage` |
| `phase2_llm/train_grpo.py --mode dry-run` | **implemented + tested + actually run** (full GRPO data path minus the weight update) |
| `phase2_llm/train_grpo.py --mode make-dataset` | **implemented + actually run** (512-row train / 32-row val parquet produced and read back) |
| `phase2_llm/verl_agent_loop.py` — verl integration class | implemented against the **inspected** verl 0.8.0 API; imports + registers + hydra-resolves in tests; **`run()` never executed** (needs a live GPU rollout server) |
| `phase2_llm/train_grpo.py --mode verl` | assembles/validates the launch command (verified); **no training run has been executed anywhere** — this machine has no GPU and no vllm |
| `phase2_llm/slurm/*.sbatch` | templates with `# TODO(user)` placeholders (same convention as Phase 1's), bash-syntax-checked, never submitted |

### verl: installed and inspected (not assumed)

`pip install verl` **succeeded** on this machine (verl 0.8.0, Python 3.10, CPU torch
2.13, ray 2.56 — vllm is an optional extra with no Windows wheels and is NOT installed;
it's only needed on the real cluster). The integration below is written against the
actually-inspected API, not documentation from memory:

- **Multi-turn rollouts**: subclass `verl.experimental.agent_loop.AgentLoopBase`,
  decorate with `@register("market_speculator")`; `async run(sampling_params, **kwargs)`
  drives the episode by calling `self.server_manager.generate(...)` per turn and returns
  an `AgentLoopOutput(prompt_ids, response_ids, response_mask, reward_score, ...)` —
  `response_mask` is 1 on model-generated tokens and 0 on injected environment/user
  tokens (same convention as verl's own `tool_agent_loop`), and `reward_score` is the
  scalar trajectory reward.
- **GRPO grouping**: verl replicates each RL-dataset row `actor_rollout_ref.rollout.n`
  times under one uid; `algorithm.adv_estimator=grpo` normalizes `reward_score` within
  that replica set: `(score − group_mean)/(group_std + eps)`, sample std (ddof=1),
  singleton groups pass through with mean=0/std=1. `phase2_llm/reward.py` reproduces
  these semantics exactly (parity test in `tests/test_grpo_scaffold.py` runs verl's own
  estimator when verl is importable).
- **Trainer**: `python -m verl.trainer.main_ppo <hydra overrides>` →
  `RayPPOTrainer` on a Ray cluster with GPU workers spawning vLLM rollout servers —
  the part that cannot run here, and the ONLY part that is stubbed (see
  `train_grpo.py --mode verl`'s docstring).

### Design decisions (documented in the module docstrings, summarized here)

- **Episode = multi-turn rollout**: ~20–64 periods (config `env.periods`, default 32) of
  the two-speculator market; per-period reward is `π_{i,t} = (v_t − p_t) x_{i,t}` straight
  from `env/market.py::MarketEnv.step` — never re-derived.
- **Group = matched-shock parallel episodes** (`grpo.matched_shocks: true`): all
  `grpo.group_size` episodes of a group share the identical `(v_t, u_t)` shock path
  (drawn from the group's seed), differing only in policy sampling — the
  common-random-numbers analog of GRPO's "G completions of one prompt", so the group
  baseline differences away shock-path variance. `false` = fully independent
  realizations, kept for ablations.
- **Advantage granularity**: `episode` (default) = one normalized discounted return per
  trajectory, broadcast to its turns — exactly standard/verl GRPO outcome supervision.
  `per_turn` (experimental) = group-normalized reward-to-go at each period index; only
  meaningful under matched shocks; see `reward.py`'s docstring for the full argument.
- **Self-play seats**: both seats sample from the same policy each period with separate
  private contexts (information hiding identical to Stage A — regression-tested by
  reconstructing every recorded prompt from that seat's own records alone). In the
  offline adapter both seats' trajectories are emitted (`grpo.train_seats`); in the verl
  agent loop one seat per rollout is trained (alternating with the seed) and the
  opponent's same-policy tokens are discarded from training.
- **Prompts/parsing**: Stage A's `build_messages`/`build_user_prompt`/
  `parse_action_index` and the retry-once → middle-index fallback are imported and
  reused, not duplicated (the async verl loop reuses the *functions* and mirrors
  `LLMAgent.choose_action`'s policy, documented in `verl_agent_loop.py`).

### Files

| File | Role |
|---|---|
| `phase2_llm/grpo_env_adapter.py` | `MarketEnv` + Stage A agents as a rollout source: `rollout_group`/`rollout_batch` → flat `TurnRecord`s (prompt, completion, parsed action, per-period reward, group metadata) |
| `phase2_llm/reward.py` | returns/reward-to-go, group normalization, `attach_advantages` (episode / per-turn modes), verl-parity semantics |
| `phase2_llm/train_grpo.py` | config loading/validation + three modes: `dry-run` (offline, complete), `make-dataset` (verl parquet), `verl` (launch-command assembly; refuses to launch on placeholders) |
| `phase2_llm/grpo_config.yaml` | single config: model (placeholder), env (read-only reuse of `configs/poc.yaml`), GRPO hyperparameters, rollout backend, verl hydra overrides |
| `phase2_llm/verl_agent_loop.py` | `MarketSpeculatorAgentLoop(AgentLoopBase)` — the verl integration point (imports only where verl is installed) |
| `phase2_llm/agent_loop_config.yaml` | hydra registry entry verl loads via `agent_loop_config_path` |
| `phase2_llm/slurm/launch_vllm.sbatch` | vLLM OpenAI-compatible server for *evaluation* (the endpoint `OpenAICompatibleBackend` already targets); not needed for training itself |
| `phase2_llm/slurm/launch_training.sbatch` | the GRPO training job (single-node default, commented multi-node Ray block) |
| `requirements-grpo.txt` | Stage B deps for a **dedicated venv** (same pattern as `.venv310`): verl pins numpy<2 and pulls torch/ray — keep it out of `.venv` |
| `tests/test_grpo_scaffold.py` | 49 tests: reward math, adapter records, config validation, dry-run end-to-end, SLURM script conventions, verl parity + registration (verl-dependent ones skip in `.venv`) |

### Running what runs today (no GPU)

```bash
# offline dry run: full GRPO data path (rollouts -> advantages -> records)
.venv/Scripts/python -m phase2_llm.train_grpo --mode dry-run

# Stage B venv (verl): needed for make-dataset / verl-command / parity tests
"C:\Program Files\Python310\python.exe" -m venv .venv-grpo
.venv-grpo/Scripts/python -m pip install -r requirements-grpo.txt
.venv-grpo/Scripts/python -m phase2_llm.train_grpo --mode make-dataset
.venv-grpo/Scripts/python -m phase2_llm.train_grpo --mode verl   # prints, never launches here
.venv-grpo/Scripts/python -m pytest tests/test_grpo_scaffold.py -q
```

Dry-run output actually produced in this session (mock backend, ξ=500, σᵤ=0.1, 2 steps ×
4 groups × 8 episodes × 32 periods): per-step advantage mean ≈ 0 / std ≈ 0.97 (correct
group-normalization), batch ΔC ≈ +0.65 (random-policy floor — same grid-design property
Stage A documented), 4–9 fallbacks per step from the deliberate 5% malformed-response
injection. Full suite: 143 passed/4 skipped in `.venv`, 145/3 in `.venv310`, 146/1 in
`.venv-grpo`.

### What YOU must do once on the real SLURM+GPU cluster

1. **Venv**: create a Linux training venv (Python ≥3.10), `pip install -r
   requirements-grpo.txt` **plus `pip install vllm`** (cluster-only; no Windows wheels).
2. **Placeholders**: fill every `# TODO(user)` in both `phase2_llm/slurm/*.sbatch`
   (partition, account, GPU type/count, module loads) and set the real model in
   `phase2_llm/grpo_config.yaml` — `model.name_or_path` AND the
   `actor_rollout_ref.model.path` override, plus `trainer.n_gpus_per_node`/
   `trainer.nnodes` to match the allocation.
3. **Smoke-test IN THIS ORDER** (each step isolates one unverified layer):
   a. `python -m pytest tests/ -q` — everything Stage B relies on, on cluster Python;
   b. serve the base model with `launch_vllm.sbatch`, then run the Stage A pilot and the
      Stage B dry-run against it (`rollout.backend: openai`) — verifies real-model
      prompting/parsing and gives the frozen-model ΔC baseline;
   c. a tiny verl job (e.g. `data.n_rows_train: 16`, `trainer.total_epochs: 1`) via
      `launch_training.sbatch` — the FIRST-EVER execution of
      `MarketSpeculatorAgentLoop.run()`; watch for tokenization-sanity warnings from
      verl's multi-turn checks (config key `tokenization_sanity_check_mode`) and confirm
      `reward_score` lands in the logs;
   d. scale up (`data.n_rows_train`, `train_batch_size`, epochs).
4. **Evaluate**: serve any checkpoint with `launch_vllm.sbatch` (`MODEL=<ckpt dir>`) and
   run `llm_pilot`/dry-run against it — ΔC over episodes, same matched-path measure as
   Phases 1/2A, is the collusion metric to track across training.
5. If verl proves uninstallable on the cluster, the adapter + reward layer are
   framework-agnostic by construction: TRL's multi-turn `GRPOTrainer` or OpenRLHF would
   consume the same `TurnRecord`/advantage structures; only `verl_agent_loop.py` and the
   `verl:` config section are verl-specific.
