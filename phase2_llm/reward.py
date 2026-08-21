"""Phase 2 Stage B: GRPO group-relative advantages from per-period trading profits.

Pure NumPy, no torch/verl dependency -- runs (and is tested) in the repo's
default `.venv`. The semantics deliberately match verl 0.8.0's
`verl.trainer.ppo.core_algos.compute_grpo_outcome_advantage` (inspected from
the installed package in `.venv-grpo`, see README "Phase 2, Stage B"), so the
framework-agnostic dry-run path and an eventual verl training run compute the
same numbers:

  - trajectory score = sum of that trajectory's rewards (optionally
    rho-discounted here; verl sums token-level rewards, which for our
    one-scalar-per-trajectory `reward_score` is the same thing),
  - advantage = (score - group_mean) / (group_std + eps),
  - group_std is the SAMPLE std (ddof=1 -- torch.std's default `unbiased=True`),
  - a singleton group uses mean=0, std=1 (verl's convention: the raw score
    passes through scaled by 1/(1+eps)). Singleton groups should never occur
    in a real GRPO run (group_size >= 2 by construction); this branch exists
    only for exact verl parity, and callers get a warning-shaped docstring
    note instead of a silent surprise.

Advantage granularity -- the documented judgment call
-----------------------------------------------------
GRPO has no critic, so the only baseline available is the group statistic.
Two granularities are implemented:

1. `mode="episode"` (DEFAULT, recommended): one scalar advantage per
   trajectory = normalized (discounted) episode return, broadcast to every
   turn of that trajectory. This is exactly standard/verl GRPO ("outcome
   supervision"): the group members are parallel market realizations sampled
   under the current policy, and -- when the adapter's `matched_shocks=True`
   default is used -- they share the identical (v_t, u_t) shock path, so the
   group mean differences away the (large) shock-path component of return
   variance, exactly like GRPO's same-prompt grouping differences away
   prompt difficulty. Per-period credit assignment is left to the model via
   the discount inside the return; that is how GRPO handles multi-step tasks
   (one outcome score per rollout) and requires no assumption that period-t
   states are comparable across group members.

2. `mode="per_turn"` (experimental, off by default): advantage at turn t =
   group-normalized reward-to-go G_t = sum_{s>=t} gamma^{s-t} r_s across the
   group members' SAME period index t. This is a denser signal, but it is
   only meaningful under matched shocks (all members face the same v_t, u_t
   at each t) and even then period-t *states* (p_{t-1}, own history) drift
   apart across members as their action histories diverge, so the cross-
   member baseline at t is an approximation. Kept because dense per-turn
   advantages are what a token-level trainer would want if episode-level
   learning proves too slow; clearly marked experimental.

All array functions are shape-documented and tested on synthetic data in
`tests/test_grpo_scaffold.py`, including a direct numerical parity test
against verl's own `compute_grpo_outcome_advantage` when verl+torch are
importable (they are in `.venv-grpo`; the test skips in the default venv).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence

import numpy as np

DEFAULT_EPS = 1e-6


# ---------------------------------------------------------------------------
# Return / reward-to-go primitives
# ---------------------------------------------------------------------------
def episode_return(rewards: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Discounted return sum_t gamma^t r_t along axis 0.

    rewards : (T,) or (T, ...) -- time-major, like everything in env/metrics.
    Returns scalar (for 1-D input) or array of the trailing shape.
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    T = rewards.shape[0]
    disc = gamma ** np.arange(T, dtype=np.float64)
    out = np.tensordot(disc, rewards, axes=(0, 0))
    return float(out) if np.ndim(out) == 0 else out


def reward_to_go(rewards: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """G_t = sum_{s>=t} gamma^{s-t} r_s, same shape as `rewards` ((T, ...))."""
    rewards = np.asarray(rewards, dtype=np.float64)
    out = np.empty_like(rewards)
    acc = np.zeros(rewards.shape[1:], dtype=np.float64)
    for t in range(rewards.shape[0] - 1, -1, -1):
        acc = rewards[t] + gamma * acc
        out[t] = acc
    return out


# ---------------------------------------------------------------------------
# Group-relative normalization (the GRPO advantage itself)
# ---------------------------------------------------------------------------
def group_normalize(scores: np.ndarray, eps: float = DEFAULT_EPS,
                    ddof: int = 1) -> np.ndarray:
    """(score - mean) / (sample_std + eps) over a single group.

    ddof=1 (sample std) matches torch.std's default and therefore verl's
    compute_grpo_outcome_advantage. A single-element group returns
    score / (1 + eps) -- verl's singleton convention (mean=0, std=1); real
    GRPO groups must have >= 2 members, see module docstring.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size == 1:
        return scores / (1.0 + eps)
    return (scores - scores.mean()) / (scores.std(ddof=ddof) + eps)


def grpo_advantages(returns: np.ndarray, group_ids: Sequence,
                    eps: float = DEFAULT_EPS) -> np.ndarray:
    """Episode-level GRPO advantages: normalize `returns` within each group.

    returns   : (n,) one (discounted) return per trajectory
    group_ids : length-n sequence of hashable group keys (trajectories with
                equal keys were sampled as one GRPO group -- same shock path
                under matched_shocks, same policy snapshot)
    Returns (n,) advantages.
    """
    returns = np.asarray(returns, dtype=np.float64)
    if returns.shape[0] != len(group_ids):
        raise ValueError(f"returns has {returns.shape[0]} entries but "
                         f"group_ids has {len(group_ids)}")
    out = np.empty_like(returns)
    members: dict = defaultdict(list)
    for j, g in enumerate(group_ids):
        members[g].append(j)
    for g, idx in members.items():
        out[idx] = group_normalize(returns[idx], eps=eps)
    return out


def turn_level_advantages(rewards: np.ndarray, group_ids: Sequence,
                          gamma: float = 1.0,
                          eps: float = DEFAULT_EPS) -> np.ndarray:
    """[experimental] Per-turn advantages: normalize reward-to-go across the
    group at each period index t (see module docstring for caveats).

    rewards   : (T, n) per-period rewards, column j = trajectory j
    group_ids : length-n group keys (as in `grpo_advantages`)
    Returns (T, n) advantages.
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    if rewards.ndim != 2:
        raise ValueError(f"rewards must be (T, n), got shape {rewards.shape}")
    if rewards.shape[1] != len(group_ids):
        raise ValueError(f"rewards has {rewards.shape[1]} trajectories but "
                         f"group_ids has {len(group_ids)}")
    g2g = reward_to_go(rewards, gamma=gamma)          # (T, n)
    out = np.empty_like(g2g)
    members: dict = defaultdict(list)
    for j, g in enumerate(group_ids):
        members[g].append(j)
    for g, idx in members.items():
        for t in range(rewards.shape[0]):
            out[t, idx] = group_normalize(g2g[t, idx], eps=eps)
    return out


# ---------------------------------------------------------------------------
# Convenience: annotate the env adapter's TurnRecords in place
# ---------------------------------------------------------------------------
def attach_advantages(turns: Iterable, mode: str = "episode",
                      gamma: float = 1.0, eps: float = DEFAULT_EPS) -> dict:
    """Compute advantages for a flat list of TurnRecord-like objects and set
    each record's `.advantage` in place.

    Duck-typed on the attributes `group_id`, `episode`, `seat`, `period`,
    `reward`, `advantage` (see grpo_env_adapter.TurnRecord) so this module
    stays import-free of the adapter. A trajectory = all turns sharing
    (group_id, episode, seat); its GRPO group = all trajectories sharing
    group_id (both seats of the shared policy are group members -- they are
    identically distributed by symmetry; see README "Phase 2, Stage B").

    Returns a diagnostics dict: per-trajectory returns/advantages plus the
    group count, useful for dry-run logging and tests.
    """
    if mode not in ("episode", "per_turn"):
        raise ValueError(f"unknown advantage mode {mode!r}")

    turns = list(turns)
    trajs: dict = defaultdict(list)
    for rec in turns:
        trajs[(rec.group_id, rec.episode, rec.seat)].append(rec)

    keys = sorted(trajs.keys(), key=lambda k: (str(k[0]), k[1], k[2]))
    for k in keys:
        trajs[k].sort(key=lambda r: r.period)
        periods = [r.period for r in trajs[k]]
        if periods != list(range(len(periods))):
            raise ValueError(f"trajectory {k} has non-contiguous periods {periods}")

    lengths = {len(trajs[k]) for k in keys}
    group_ids = [k[0] for k in keys]

    if mode == "episode":
        returns = np.array([episode_return(np.array([r.reward for r in trajs[k]]),
                                           gamma=gamma) for k in keys])
        advs = grpo_advantages(returns, group_ids, eps=eps)
        for k, a in zip(keys, advs):
            for rec in trajs[k]:
                rec.advantage = float(a)
        adv_per_traj = advs
    else:  # per_turn
        if len(lengths) != 1:
            raise ValueError("per_turn mode requires equal-length trajectories, "
                             f"got lengths {sorted(lengths)}")
        rewards = np.array([[r.reward for r in trajs[k]] for k in keys]).T  # (T, n)
        returns = np.array([episode_return(rewards[:, j], gamma=gamma)
                            for j in range(rewards.shape[1])])
        advs_t = turn_level_advantages(rewards, group_ids, gamma=gamma, eps=eps)
        for j, k in enumerate(keys):
            for t, rec in enumerate(trajs[k]):
                rec.advantage = float(advs_t[t, j])
        adv_per_traj = advs_t.mean(axis=0)

    return {
        "n_trajectories": len(keys),
        "n_groups": len(set(group_ids)),
        "trajectory_keys": keys,
        "returns": returns,
        "advantages_per_trajectory": adv_per_traj,
    }
