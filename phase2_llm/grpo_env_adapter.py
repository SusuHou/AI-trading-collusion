"""Phase 2 Stage B: MarketEnv + Stage A prompts/parsing as a GRPO rollout source.

Framework-agnostic by design (plain Python + NumPy, no torch/verl imports):
given any policy that satisfies Stage A's `LLMBackend` protocol -- a real
vLLM-served model via `agent_llm.OpenAICompatibleBackend`, or
`agent_llm.MockBackend` for offline testing -- this module produces full
multi-turn episode rollouts of the two-speculator market, structured as flat
per-turn records carrying everything an RL trainer needs: the exact prompt
messages, the raw sampled completion, the parsed action, the per-period
reward pi_{i,t} = (v_t - p_t) x_{i,t} (computed by env/market.py, NOT
re-derived here), and the (group_id, episode, seat, period) metadata needed
to reconstruct GRPO group structure for `reward.attach_advantages`.

Reuse, not duplication (task requirement):
  - prompts:   `agent_llm.build_messages` (via `LLMAgent.choose_action`)
  - parsing:   `agent_llm.parse_action_index` + LLMAgent's retry-once ->
               documented-middle-fallback policy, unchanged from Stage A
  - market:    `env.market.MarketEnv.step` (eq 3.2/3.4/4.1-4.2), batched with
               batch = group_size, exactly like `llm_pilot.run` batches over
               episodes
  - grids:     `env.benchmarks.compute_benchmarks` action grid X(v)
  - info hiding: each seat is its own `LLMAgent` with its OWN history; the
               two seats share one backend object (= one set of weights,
               independently sampled) -- the same self-play pattern Stage A
               established and the paper's symmetric-agents design requires.

GRPO group structure -- the documented judgment call
----------------------------------------------------
A "group" = `group_size` parallel episodes sampled under the SAME policy.
With `matched_shocks=True` (default) all episodes in a group share the
identical (v_t, u_t) shock path (drawn once from the group's seed and
broadcast across the batch); only the policy's own sampling differs across
members. This is the common-random-numbers analog of GRPO's "G completions
of the same prompt": the group-mean baseline then differences away the
shock-path component of return variance, which at sigma_u/sigma_v scales
would otherwise dominate the advantage signal. `matched_shocks=False` gives
fully independent realizations (the group baseline is still unbiased, just
much noisier) and is kept for ablations.

Both seats' trajectories are emitted and tagged with `seat`; by symmetry of
the shared policy they are identically distributed, so a trainer may use
both (default in train_grpo) or filter to one seat. Note seats within one
episode are NOT independent (their profits interact through the price) --
that correlation is fine for a baseline but is why `group_size` counts
episodes, not trajectories.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from env.benchmarks import Benchmarks, Params
from env.market import MarketEnv
from phase2_llm.agent_llm import (
    LLMAgent, LLMBackend,
    DEFAULT_HISTORY_LEN, DEFAULT_MAX_RETRIES, DEFAULT_FALLBACK,
)


# ---------------------------------------------------------------------------
# Rollout records
# ---------------------------------------------------------------------------
@dataclass
class TurnRecord:
    """One (episode, seat, period) decision: prompt -> completion -> reward."""
    group_id: str          # GRPO group key (shared by all episodes of a group)
    episode: int           # group-member index, 0..group_size-1
    seat: int              # speculator index, 0..I-1 (both seats = one policy)
    period: int            # 0..periods-1
    v: float               # fundamental this period (this seat's private signal)
    p_lag: Optional[float] # lagged public price (None on period 0)
    u: float               # realized noise-trader flow (env-side, NOT in prompt)
    price: float           # realized market price p_t
    prompt_messages: list  # exact chat messages sent (Stage A build_messages)
    raw_response: Optional[str]  # raw completion text (last attempt)
    action_index: int
    action_value: float
    reward: float          # pi_{i,t} = (v_t - p_t) x_{i,t}, from MarketEnv.step
    malformed: bool
    used_fallback: bool
    n_attempts: int
    advantage: Optional[float] = None  # filled by reward.attach_advantages

    def to_dict(self, include_prompts: bool = True,
                prompt_chars: Optional[int] = None) -> dict:
        d = dataclasses.asdict(self)
        if not include_prompts:
            d.pop("prompt_messages")
            d.pop("raw_response")
        elif prompt_chars is not None:
            d["prompt_messages"] = [
                {"role": m["role"], "content": m["content"][:prompt_chars]}
                for m in d["prompt_messages"]]
        return d


@dataclass
class GroupRollout:
    """One GRPO group: `group_size` parallel episodes under one policy."""
    group_id: str
    group_size: int
    periods: int
    n_seats: int
    matched_shocks: bool
    seed: int
    turns: list          # flat list[TurnRecord], ordered (period, episode, seat)
    v_path: np.ndarray   # (periods, group_size) realized fundamentals
    u_path: np.ndarray   # (periods, group_size) realized noise flows
    prices: np.ndarray   # (periods, group_size) realized prices
    profits: np.ndarray  # (periods, group_size, n_seats) realized profits

    def trajectories(self) -> dict:
        """{(episode, seat): [TurnRecord sorted by period]}."""
        out: dict = {}
        for rec in self.turns:
            out.setdefault((rec.episode, rec.seat), []).append(rec)
        for k in out:
            out[k].sort(key=lambda r: r.period)
        return out


# ---------------------------------------------------------------------------
# Rollout generation
# ---------------------------------------------------------------------------
def rollout_group(backend: LLMBackend, params: Params, bench: Benchmarks, *,
                  group_size: int, periods: int, seed: int,
                  matched_shocks: bool = True,
                  group_id: Optional[str] = None,
                  history_len: int = DEFAULT_HISTORY_LEN,
                  max_retries: int = DEFAULT_MAX_RETRIES,
                  fallback: str = DEFAULT_FALLBACK) -> GroupRollout:
    """Sample one GRPO group of `group_size` parallel market episodes.

    The per-period protocol is llm_pilot.run's loop verbatim (which itself
    mirrors phase1_qlearning/run_session.py minus Q-learning): agents choose
    from bench.x_values(v) via LLM calls, MarketEnv.step prices the period
    and realizes profits, each agent observes only its OWN outcome.

    `seed` drives ONLY the environment shocks (v path, u path, initial v);
    policy stochasticity lives in the backend (server-side sampling for a
    real model, MockBackend's own rng offline). With matched_shocks the
    (v, u) path is drawn once ((periods,) vectors) and shared by all group
    members; otherwise each member gets an independent path.
    """
    if group_size < 1:
        raise ValueError("group_size must be >= 1")
    if group_size < 2:
        # reward.group_normalize's singleton convention is verl parity only;
        # a real GRPO group needs >= 2 members to have a baseline at all.
        import warnings
        warnings.warn("group_size=1 gives GRPO no baseline; use >= 2 "
                      "(allowed only for debugging)", stacklevel=2)
    gid = group_id if group_id is not None else f"group_seed{seed}"
    G, I = group_size, params.I

    shock_rng = np.random.default_rng(seed)
    env = MarketEnv(params, bench, batch=G, rng=shock_rng, mm_init="nash")

    if matched_shocks:
        v_idx = np.broadcast_to(
            shock_rng.integers(0, params.nv, size=(periods, 1)), (periods, G))
        u_path = np.broadcast_to(
            shock_rng.normal(0.0, params.sigma_u, size=(periods, 1)),
            (periods, G)).copy()
    else:
        v_idx = shock_rng.integers(0, params.nv, size=(periods, G))
        u_path = shock_rng.normal(0.0, params.sigma_u, size=(periods, G))

    agents = [[LLMAgent(backend, agent_name=f"{gid}_ep{e}_seat{i}",
                        history_len=history_len, max_retries=max_retries,
                        fallback=fallback)
               for i in range(I)] for e in range(G)]

    turns: list[TurnRecord] = []
    v_path = np.empty((periods, G))
    prices = np.empty((periods, G))
    profits = np.empty((periods, G, I))
    p_lag = np.full(G, np.nan)

    for t in range(periods):
        v = bench.vgrid[v_idx[t]]          # (G,)
        grid = bench.x_values(v)           # (G, nx)
        a_idx = np.empty((G, I), dtype=np.int64)
        metas = [[None] * I for _ in range(G)]

        for e in range(G):
            plag_e = None if np.isnan(p_lag[e]) else float(p_lag[e])
            for i in range(I):
                idx, meta = agents[e][i].choose_action(
                    period=t, v_t=float(v[e]), p_lag=plag_e,
                    action_values=grid[e])
                a_idx[e, i] = idx
                metas[e][i] = meta

        grid_b = np.broadcast_to(grid[:, None, :], (G, I, params.nx))
        x = np.take_along_axis(grid_b, a_idx[:, :, None], axis=2)[:, :, 0]
        price, pi, _info = env.step(x, v, u_path[t], update_mm=True)

        v_path[t], prices[t], profits[t] = v, price, pi

        for e in range(G):
            plag_e = None if np.isnan(p_lag[e]) else float(p_lag[e])
            for i in range(I):
                agents[e][i].observe(period=t, v_t=float(v[e]), p_lag=plag_e,
                                     action_index=int(a_idx[e, i]),
                                     action_value=float(x[e, i]),
                                     profit=float(pi[e, i]))
                meta = metas[e][i]
                turns.append(TurnRecord(
                    group_id=gid, episode=e, seat=i, period=t,
                    v=float(v[e]), p_lag=plag_e, u=float(u_path[t, e]),
                    price=float(price[e]),
                    prompt_messages=meta.prompt_messages,
                    raw_response=meta.raw_response,
                    action_index=int(a_idx[e, i]),
                    action_value=float(x[e, i]),
                    reward=float(pi[e, i]),
                    malformed=meta.malformed,
                    used_fallback=meta.used_fallback,
                    n_attempts=meta.n_attempts,
                ))
        p_lag = price

    return GroupRollout(group_id=gid, group_size=G, periods=periods,
                        n_seats=I, matched_shocks=matched_shocks, seed=seed,
                        turns=turns, v_path=v_path, u_path=np.asarray(u_path),
                        prices=prices, profits=profits)


def rollout_batch(backend: LLMBackend, params: Params, bench: Benchmarks, *,
                  n_groups: int, group_size: int, periods: int,
                  base_seed: int, matched_shocks: bool = True,
                  **agent_kwargs) -> list[GroupRollout]:
    """One GRPO training step's worth of data: `n_groups` independent groups
    (distinct shock seeds), each of `group_size` parallel episodes, all
    sampled from the same `backend` (= the same policy snapshot)."""
    return [
        rollout_group(backend, params, bench, group_size=group_size,
                      periods=periods, seed=base_seed + k,
                      matched_shocks=matched_shocks,
                      group_id=f"g{k}_seed{base_seed + k}", **agent_kwargs)
        for k in range(n_groups)
    ]


def flatten_turns(groups: Sequence[GroupRollout]) -> list:
    """All TurnRecords of a batch, in a deterministic order."""
    return [rec for g in groups for rec in g.turns]
