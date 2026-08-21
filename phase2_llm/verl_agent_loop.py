"""Phase 2 Stage B: verl agent-loop integration for the two-speculator market.

STATUS -- read this honestly before trusting anything here:
  * verl 0.8.0 WAS successfully pip-installed on this dev machine (CPU torch,
    ray 2.56, no vllm -- vllm has no Windows wheels and is an optional verl
    extra), and this module is written against the ACTUAL inspected API of
    `verl.experimental.agent_loop` (AgentLoopBase / AgentLoopOutput /
    register, verl 0.8.0). `tests/test_grpo_scaffold.py` verifies (when verl
    is importable, i.e. in `.venv-grpo`) that this module imports, that the
    loop class registers under "market_speculator", and that its
    reward/advantage conventions match verl's own GRPO estimator.
  * It has NOT been executed end-to-end: `AgentLoopBase.run` needs a live
    rollout server (`self.server_manager.generate` -> vLLM/sglang worker on
    GPU) inside a Ray cluster driven by `verl.trainer.main_ppo`. Neither GPU
    nor vllm exists in this environment. The `run()` body below is therefore
    structurally correct against the inspected interfaces but unverified at
    runtime -- treat it as the integration point to smoke-test FIRST on the
    real cluster (see README "Phase 2, Stage B" for the exact smoke-test
    order).

How this maps onto verl GRPO
----------------------------
  * One RL-dataset row  = one episode spec: `extra_info.shock_seed` (+ env
    overrides). verl replicates each row `actor_rollout_ref.rollout.n` times
    with a shared uid -> those n replicas ARE the GRPO group, and because
    they share the row's shock_seed, they share the shock path: exactly
    `grpo_env_adapter.rollout_group(matched_shocks=True)`'s group semantics.
  * One `run()` call = one full multi-turn episode = one trajectory. The
    TRAINED seat's conversation becomes the token sequence (assistant tokens
    mask=1, injected user/env turns mask=0 -- the same convention as verl's
    own tool_agent_loop); the OPPONENT seat is sampled from the same server
    (same weights -- self-play) in its own separate conversation whose
    tokens are discarded from training. Which seat is trained alternates
    with the seed so both seats' data distributions are (symmetrically)
    covered across the dataset.
  * `reward_score` = discounted episode return sum_t gamma^t pi_t of the
    trained seat; with `algorithm.adv_estimator: grpo` verl turns that into
    (score - group_mean)/(group_std + eps) -- numerically identical to
    `phase2_llm.reward.grpo_advantages` (parity-tested).
  * Prompts/parsing are Stage A's, unchanged: each period appends a user
    message built by `agent_llm.build_user_prompt` (system prompt once, at
    the start) and the model answers with the same {"action_index": i} JSON,
    parsed by `agent_llm.parse_action_index` with the same retry-once ->
    middle-index fallback. Each seat still only ever sees its own history.

Registration with verl: point `actor_rollout_ref.rollout.agent.
agent_loop_config_path` at a YAML list like
    - name: market_speculator
      _target_: phase2_llm.verl_agent_loop.MarketSpeculatorAgentLoop
      params_config: configs/poc.yaml
      periods: 32
      gamma: 1.0
(hydra instantiates this class with those kwargs; see
`phase2_llm/grpo_config.yaml` and `train_grpo.py --mode verl`).
"""
from __future__ import annotations

import json
from typing import Any, Optional
from uuid import uuid4

import numpy as np

# Hard requirement: this module only imports where verl is installed
# (.venv-grpo here; the training venv on the cluster). train_grpo.py and the
# tests import it lazily/optionally.
from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase, AgentLoopMetrics, AgentLoopOutput, register,
)

from env.benchmarks import Params, compute_benchmarks
from env.market import MarketEnv
from phase2_llm.agent_llm import (
    SYSTEM_PROMPT, build_user_prompt, parse_action_index,
    DEFAULT_HISTORY_LEN, PeriodRecord,
)


@register("market_speculator")
class MarketSpeculatorAgentLoop(AgentLoopBase):
    """One rollout = one multi-turn episode of the two-speculator market,
    trained on one seat, self-play opponent sampled from the same server."""

    def __init__(self, *args,
                 params_config: Optional[str] = None,
                 sigma_u: Optional[float] = None,
                 xi: Optional[float] = None,
                 periods: int = 32,
                 gamma: float = 1.0,
                 history_len: int = DEFAULT_HISTORY_LEN,
                 max_parse_retries: int = 1,
                 **kwargs):
        super().__init__(*args, **kwargs)
        # Env params resolved once per AgentLoop instance, same read-only
        # config pattern as llm_pilot.load_params.
        if params_config:
            import yaml
            with open(params_config, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            params = Params.from_dict(cfg.get("params", {}))
        else:
            params = Params()
        if sigma_u is not None:
            params.sigma_u = sigma_u
        if xi is not None:
            params.xi = xi
        params.I = 2
        self.params = params
        self.bench = compute_benchmarks(params)
        self.periods = periods
        self.gamma = gamma
        self.history_len = history_len
        self.max_parse_retries = max_parse_retries
        self.response_length = self.rollout_config.response_length

    # -- helpers ----------------------------------------------------------
    async def _sample_action(self, prompt_ids: list[int], messages: list[dict],
                             n_actions: int, sampling_params: dict,
                             collect: bool):
        """One seat's one-period decision: generate -> parse -> retry once ->
        middle-index fallback (mirrors agent_llm.LLMAgent.choose_action, but
        async and token-level -- LLMAgent's sync `.complete()` protocol can't
        wrap `server_manager.generate` without blocking the event loop, so
        the retry/fallback POLICY is reused rather than the class; any change
        here must stay in lockstep with LLMAgent, enforced by comment + the
        shared parse_action_index/fallback convention tests).

        Returns (action_index, gen_token_ids, gen_mask, raw_text, used_fallback).
        gen_token_ids/gen_mask cover ALL attempts when `collect` (trained
        seat): retried garbage stays in the sequence (mask=1 -- the policy
        really did emit it) with the corrective user message masked 0.
        """
        out_ids: list[int] = []
        out_mask: list[int] = []
        raw = None
        idx = None
        for attempt in range(self.max_parse_retries + 1):
            output = await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=prompt_ids + out_ids,
                sampling_params=sampling_params,
            )
            gen = list(output.token_ids)
            raw = self.tokenizer.decode(gen)
            if collect:
                out_ids += gen
                out_mask += [1] * len(gen)
            idx = parse_action_index(raw, n_actions)
            if idx is not None or attempt >= self.max_parse_retries:
                break
            corrective = [{"role": "user", "content": (
                "That response was not a valid choice. Reply with ONLY a JSON "
                f'object of the form {{"action_index": <integer 0..{n_actions - 1}>}}, '
                "choosing one index from the numbered list above.")}]
            messages += [{"role": "assistant", "content": raw}] + corrective
            if collect:
                corr_ids = await self.apply_chat_template(corrective)
                out_ids += corr_ids
                out_mask += [0] * len(corr_ids)
        used_fallback = idx is None
        if used_fallback:
            idx = n_actions // 2   # Stage A's documented "middle" fallback
        return idx, out_ids, out_mask, raw, used_fallback

    # -- the rollout ------------------------------------------------------
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        extra = kwargs.get("extra_info") or {}
        seed = int(extra.get("shock_seed", 0))
        periods = int(extra.get("periods", self.periods))
        train_seat = int(extra.get("train_seat", seed % 2))

        params, bench = self.params, self.bench
        I = params.I
        shock_rng = np.random.default_rng(seed)
        env = MarketEnv(params, bench, batch=1, rng=shock_rng, mm_init="nash")
        # Same draw order as grpo_env_adapter.rollout_group(matched_shocks=True)
        # so a verl group (n replicas of this row) and an offline dry-run group
        # see the same shock path for the same seed.
        v_idx = shock_rng.integers(0, params.nv, size=(periods, 1))
        u_path = shock_rng.normal(0.0, params.sigma_u, size=(periods, 1))

        # Per-seat state: messages + Stage A rolling history records.
        messages = [[{"role": "system", "content": SYSTEM_PROMPT}] for _ in range(I)]
        history: list[list[PeriodRecord]] = [[] for _ in range(I)]

        # Trained seat's running token sequence (tool_agent_loop convention).
        prompt_ids: list[int] = []
        response_mask: list[int] = []
        num_llm_turns = 0
        rewards = np.zeros((periods, I))
        p_lag: Optional[float] = None
        n_fallback = 0

        for t in range(periods):
            v_t = float(bench.vgrid[v_idx[t, 0]])
            grid = bench.x_values(np.array([v_t]))[0]     # (nx,)
            n_actions = len(grid)
            x = np.empty((1, I))
            a_idx = np.empty(I, dtype=np.int64)
            raws: list[Optional[str]] = [None] * I

            for i in range(I):
                user_msg = {"role": "user", "content": build_user_prompt(
                    period=t, v_t=v_t, p_lag=p_lag, action_values=grid,
                    history=history[i], history_len=self.history_len)}
                messages[i].append(user_msg)
                collect = (i == train_seat)
                if collect:
                    if t == 0:
                        prompt_ids = await self.apply_chat_template(messages[i])
                    else:
                        turn_ids = await self.apply_chat_template([user_msg])
                        prompt_ids += turn_ids
                        response_mask += [0] * len(turn_ids)
                    seat_prompt_ids = prompt_ids
                else:
                    seat_prompt_ids = await self.apply_chat_template(messages[i])
                idx, gen_ids, gen_mask, raw, fb = await self._sample_action(
                    seat_prompt_ids, messages[i], n_actions, sampling_params,
                    collect=collect)
                if collect:
                    prompt_ids += gen_ids
                    response_mask += gen_mask
                    num_llm_turns += 1
                messages[i].append({"role": "assistant", "content": raw or ""})
                a_idx[i] = idx
                x[0, i] = grid[idx]
                raws[i] = raw
                n_fallback += int(fb)

            price, pi, _ = env.step(x, np.array([v_t]), u_path[t], update_mm=True)
            rewards[t] = pi[0]
            for i in range(I):
                history[i].append(PeriodRecord(
                    period=t, v=v_t, p_lag=p_lag, action_index=int(a_idx[i]),
                    action_value=float(x[0, i]), profit=float(pi[0, i])))
            p_lag = float(price[0])

            if len(response_mask) >= self.response_length:
                break   # token budget exhausted; reward covers periods played

        disc = self.gamma ** np.arange(rewards.shape[0])
        reward_score = float((disc * rewards[:, train_seat]).sum())

        response_ids = prompt_ids[-len(response_mask):] if response_mask else []
        initial_prompt_ids = prompt_ids[:len(prompt_ids) - len(response_mask)]
        return AgentLoopOutput(
            prompt_ids=initial_prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            reward_score=reward_score,
            num_turns=num_llm_turns,
            metrics=AgentLoopMetrics(),
            extra_fields={"train_seat": train_seat, "shock_seed": seed,
                          "n_fallback": n_fallback,
                          "episode_profits": json.dumps(rewards[:, train_seat].tolist())},
        )
