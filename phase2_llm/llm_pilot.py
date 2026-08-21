"""Phase 2 Stage A pilot: two frozen LLM instances play the informed
speculator role in `env/market.py`, in-context only (no weight updates).

Reuses the SAME environment, grid, and metrics as Phase 1:
  - `env.benchmarks.compute_benchmarks` for the action grid / Nash & cartel
    benchmarks (paper-exact, see docs/paper_spec.md),
  - `env.market.MarketEnv` for the actual per-period price/profit mechanics
    (eq 3.4/4.1-4.2), batched over episodes,
  - `env.metrics.delta_c_matched` for the same matched-path Delta^C measure
    Phase 1 reports, computed over the pilot's own realized (v_t, u_t) path.

Per-period protocol (mirrors phase1_qlearning/run_session.py's loop, minus
Q-learning: both speculators are LLMAgents, price/profit come from
MarketEnv.step, and each agent's action index comes from an LLM call
instead of an epsilon-greedy Q-table lookup):
  1. v_t drawn from the same discretized grid used everywhere else in this
     project (bench.vgrid, equal-probability quantiles); each speculator's
     LLM call sees ONLY its own private v_t, own rolling history, and the
     lagged public price p_{t-1} (continuous, not grid-snapped -- the LLM
     has no need for the Q-table's index encoding).
  2. noise trader draws u_t ~ N(0, sigma_u^2).
  3. MarketEnv.step prices the period and realizes both speculators' profits.
  4. each LLMAgent.observe() appends (v_t, own action, own profit) to ITS
     OWN history only -- never the other speculator's.

Usage (offline, no network -- the acceptance bar for this session):
  .venv/Scripts/python -m phase2_llm.llm_pilot --backend mock \
      --episodes 5 --periods 30 --out results/phase2_llm/pilot_mock.jsonl

Usage against a real OpenAI-compatible endpoint (e.g. vLLM serving
Qwen-3.5 on the SLURM cluster later -- see README.md "Phase 2" section):
  .venv/Scripts/python -m phase2_llm.llm_pilot --backend openai \
      --base-url http://localhost:8000/v1 --model Qwen/Qwen3.5-XXB-Instruct \
      --episodes 5 --periods 30 --out results/phase2_llm/pilot_qwen.jsonl
"""
from __future__ import annotations

import argparse
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

from phase2_llm.agent_llm import (
    LLMAgent, MockBackend, OpenAICompatibleBackend,
    DEFAULT_HISTORY_LEN, DEFAULT_MAX_RETRIES, DEFAULT_FALLBACK,
)

DEFAULT_PARAMS_CONFIG = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "configs", "poc.yaml"))


# ---------------------------------------------------------------------------
def load_params(config_path: str | None) -> Params:
    """Reads env params from an existing configs/*.yaml (read-only -- this
    module never writes to configs/). Falls back to Params() defaults (which
    match the paper's baseline, see env/benchmarks.py) if no config given."""
    if not config_path:
        return Params()
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return Params.from_dict(cfg.get("params", {}))


def build_backend(args, bench):
    if args.backend == "mock":
        target_chi = None
        mode = args.mock_mode
        if mode == "target_nash":
            target_chi, mode = bench.chiN, "target_chi"
        elif mode == "target_cartel":
            target_chi, mode = bench.chiM, "target_chi"
        return MockBackend(mode=mode, c_grid=bench.c_grid, target_chi=target_chi,
                           noise_sd=args.mock_noise_sd,
                           malformed_rate=args.malformed_rate, seed=args.seed)
    if args.backend == "openai":
        base_url = args.base_url or os.environ.get("OPENAI_BASE_URL",
                                                    "http://localhost:8000/v1")
        model = args.model or os.environ.get("OPENAI_MODEL", "Qwen/Qwen3.5-Instruct")
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
        return OpenAICompatibleBackend(base_url=base_url, model=model,
                                       api_key=api_key, temperature=args.temperature)
    raise ValueError(f"unknown backend {args.backend!r}")


# ---------------------------------------------------------------------------
def run(args) -> dict:
    params = load_params(args.config)
    if args.sigma_u is not None:
        params.sigma_u = args.sigma_u
    if args.xi is not None:
        params.xi = args.xi
    params.I = 2  # Stage A pilot is exactly the paper's two-speculator setup

    bench = compute_benchmarks(params)
    rng = np.random.default_rng(args.seed)

    n_ep, n_per, I = args.episodes, args.periods, params.I
    env = MarketEnv(params, bench, batch=n_ep, rng=rng, mm_init="nash")
    backend = build_backend(args, bench)

    agents = [[LLMAgent(backend, agent_name=f"ep{e}_spec{i}",
                        history_len=args.history_len,
                        max_retries=args.max_retries,
                        fallback=args.fallback)
              for i in range(I)] for e in range(n_ep)]

    v_idx = rng.integers(0, params.nv, size=n_ep)
    p_lag = np.full(n_ep, np.nan)  # NaN sentinel == "not available yet"

    V = np.empty((n_per, n_ep))
    U = np.empty((n_per, n_ep))
    X = np.empty((n_per, n_ep, I))
    Pi = np.empty((n_per, n_ep, I))
    P = np.empty((n_per, n_ep))

    out_path = args.out or os.path.join(
        "results", "phase2_llm", f"pilot_{args.backend}_seed{args.seed}.jsonl")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    t0 = time.time()
    with open(out_path, "w", encoding="utf-8") as f_out:
        for t in range(n_per):
            v = bench.vgrid[v_idx]                       # (n_ep,)
            grid = bench.x_values(v)                      # (n_ep, nx)
            a_idx = np.empty((n_ep, I), dtype=np.int64)
            metas = [[None] * I for _ in range(n_ep)]

            for e in range(n_ep):
                plag_e = None if np.isnan(p_lag[e]) else float(p_lag[e])
                for i in range(I):
                    idx, meta = agents[e][i].choose_action(
                        period=t, v_t=float(v[e]), p_lag=plag_e,
                        action_values=grid[e])
                    a_idx[e, i] = idx
                    metas[e][i] = meta

            grid_b = np.broadcast_to(grid[:, None, :], (n_ep, I, params.nx))
            x = np.take_along_axis(grid_b, a_idx[:, :, None], axis=2)[:, :, 0]
            u = rng.normal(0.0, params.sigma_u, size=n_ep)
            price, pi, info = env.step(x, v, u, update_mm=True)

            V[t], U[t], X[t], Pi[t], P[t] = v, u, x, pi, price

            for e in range(n_ep):
                plag_e = None if np.isnan(p_lag[e]) else float(p_lag[e])
                row = {"episode": e, "period": t, "v": float(v[e]),
                      "p_lag": plag_e, "price": float(price[e]),
                      "u": float(u[e]), "agents": []}
                for i in range(I):
                    agents[e][i].observe(period=t, v_t=float(v[e]), p_lag=plag_e,
                                         action_index=int(a_idx[e, i]),
                                         action_value=float(x[e, i]),
                                         profit=float(pi[e, i]))
                    meta = metas[e][i]
                    agent_row = {
                        "action_index": int(a_idx[e, i]),
                        "action_value": float(x[e, i]),
                        "profit": float(pi[e, i]),
                        "n_attempts": meta.n_attempts,
                        "malformed": meta.malformed,
                        "used_fallback": meta.used_fallback,
                    }
                    if args.log_prompts:
                        agent_row["raw_response"] = meta.raw_response
                        agent_row["user_prompt"] = meta.prompt_messages[-1]["content"][
                            :args.log_prompt_chars]
                    row["agents"].append(agent_row)
                f_out.write(json.dumps(row) + "\n")

            p_lag = price
            v_idx = rng.integers(0, params.nv, size=n_ep)

    elapsed = time.time() - t0

    # -- metrics: same matched-path Delta^C Phase 1 reports (OA IA.4.1-4.3) --
    piN_path = bench.path_profits(V, U, "nash")     # (n_per, n_ep)
    piM_path = bench.path_profits(V, U, "cartel")
    piN_bar = piN_path.mean(axis=0)                   # (n_ep,)
    piM_bar = piM_path.mean(axis=0)
    pi_bar_i = Pi.mean(axis=0)                         # (n_ep, I)
    delta_c = M.delta_c_matched(pi_bar_i, piN_bar, piM_bar)   # (n_ep,)
    profit_gain = M.relative_profit_gain(pi_bar_i, piN_bar)   # (n_ep,)

    n_malformed = sum(a.n_malformed for row in agents for a in row)
    n_fallback = sum(a.n_fallback for row in agents for a in row)
    n_calls = n_ep * n_per * I

    summary = {
        "backend": args.backend,
        "episodes": n_ep, "periods": n_per, "I": I,
        "params": {"sigma_u": params.sigma_u, "xi": params.xi, "rho": params.rho,
                  "nv": params.nv, "nx": params.nx, "np": params.np_},
        "benchmarks": {"lamN": bench.lamN, "lamM": bench.lamM,
                      "chiN": bench.chiN, "chiM": bench.chiM,
                      "piN": bench.piN, "piM": bench.piM},
        "delta_c_per_episode": delta_c.tolist(),
        "delta_c_mean": float(delta_c.mean()),
        "delta_c_median": float(np.median(delta_c)),
        "profit_gain_per_episode": profit_gain.tolist(),
        "profit_gain_mean": float(profit_gain.mean()),
        "pi_bar_i": pi_bar_i.tolist(),
        "n_llm_calls": n_calls,
        "n_malformed": n_malformed,
        "n_fallback": n_fallback,
        "elapsed_sec": elapsed,
        "transcript_path": out_path,
    }
    summary_path = out_path + ".summary.json"
    with open(summary_path, "w", encoding="utf-8") as f_sum:
        json.dump(summary, f_sum, indent=2)
    summary["summary_path"] = summary_path

    if not args.quiet:
        print(f"phase2_llm pilot: {n_ep} episodes x {n_per} periods x {I} "
              f"speculators = {n_calls} LLM calls in {elapsed:.1f}s "
              f"({args.backend} backend)")
        print(f"  malformed responses: {n_malformed}/{n_calls} "
              f"({n_fallback} triggered the documented fallback)")
        print(f"  Delta^C: mean={summary['delta_c_mean']:+.3f} "
              f"median={summary['delta_c_median']:+.3f} "
              f"per-episode={['%+.3f' % d for d in delta_c]}")
        print(f"  profit gain vs Nash (mean): {summary['profit_gain_mean']:.3f}")
        print(f"  transcript -> {out_path}")
        print(f"  summary    -> {summary_path}")

    return summary


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=DEFAULT_PARAMS_CONFIG,
                    help="existing configs/*.yaml to read env params from "
                         "(read-only; pass '' to use Params() defaults)")
    ap.add_argument("--episodes", "-N", type=int, default=5,
                    help="number of independent episodes (pilot default: small)")
    ap.add_argument("--periods", "-E", type=int, default=30,
                    help="periods per episode (pilot default: small)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sigma-u", type=float, default=None)
    ap.add_argument("--xi", type=float, default=None)
    ap.add_argument("--out", default=None, help="transcript .jsonl path")
    ap.add_argument("--quiet", action="store_true")

    ap.add_argument("--backend", choices=["mock", "openai"], default="mock")
    ap.add_argument("--mock-mode",
                    choices=["random", "fixed", "target_nash", "target_cartel"],
                    default="random",
                    help="MockBackend policy (see agent_llm.MockBackend docstring)")
    ap.add_argument("--mock-noise-sd", type=float, default=1.0,
                    help="index jitter for --mock-mode target_nash/target_cartel")
    ap.add_argument("--malformed-rate", type=float, default=0.1,
                    help="fraction of mock responses that are deliberately "
                         "malformed, to exercise the retry/fallback path "
                         "(default 0.1 so the offline pilot actually tests it)")

    ap.add_argument("--base-url", default=None,
                    help="OpenAI-compatible base URL, e.g. http://localhost:8000/v1 "
                         "(vLLM). Falls back to $OPENAI_BASE_URL.")
    ap.add_argument("--model", default=None,
                    help="model name as served by the endpoint. "
                         "Falls back to $OPENAI_MODEL.")
    ap.add_argument("--api-key", default=None,
                    help="falls back to $OPENAI_API_KEY (often unused for local vLLM).")
    ap.add_argument("--temperature", type=float, default=0.7)

    ap.add_argument("--history-len", type=int, default=DEFAULT_HISTORY_LEN)
    ap.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    ap.add_argument("--fallback", choices=["middle", "random"], default=DEFAULT_FALLBACK)
    ap.add_argument("--log-prompts", action="store_true",
                    help="include raw responses + truncated user prompts in the transcript")
    ap.add_argument("--log-prompt-chars", type=int, default=600)

    args = ap.parse_args(argv)
    if args.config == "":
        args.config = None
    return run(args)


if __name__ == "__main__":
    main()
