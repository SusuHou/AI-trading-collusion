"""Tests for Phase 2 Stage B (GRPO scaffold) -- everything verifiable
without a GPU or a real model:

  1. reward.py advantage math on synthetic data (returns, reward-to-go,
     group normalization, episode/per-turn attachment), including a direct
     numerical parity test against verl's own GRPO estimator (runs only
     where verl+torch are importable, i.e. `.venv-grpo`; skips in `.venv`).
  2. grpo_env_adapter.py rollout generation driven by Stage A's MockBackend:
     record shapes/fields, matched-shock group semantics, reward == the
     MarketEnv profit, prompt reuse & information hiding carried over from
     Stage A, end-to-end batch + advantage attachment.
  3. train_grpo.py config parsing/validation, the dry-run loop end-to-end,
     and the assembled verl launch command (never launched here).
  4. SLURM script structure: TODO(user) placeholder convention (same as
     phase1_qlearning/slurm/run_experiment.sbatch), bash syntax check when a
     bash executable is available.
  5. verl integration (skips without verl): phase2_llm/verl_agent_loop.py
     imports and registers "market_speculator" against the real
     AgentLoopBase registry.
"""
import json
import os
import shutil
import subprocess

import numpy as np
import pytest
import yaml

from env.benchmarks import Params, compute_benchmarks
from phase2_llm import reward as R
from phase2_llm import train_grpo
from phase2_llm.agent_llm import MockBackend
from phase2_llm.grpo_env_adapter import (
    TurnRecord, flatten_turns, rollout_batch, rollout_group,
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SLURM_DIR = os.path.join(REPO_ROOT, "phase2_llm", "slurm")


def small_setup(sigma_u=0.1):
    params = Params.from_dict({"sigma_u": sigma_u, "Tm": 200, "nv": 4})
    return params, compute_benchmarks(params)


# ---------------------------------------------------------------------------
# 1. reward.py math
# ---------------------------------------------------------------------------
class TestReturns:
    def test_episode_return_undiscounted(self):
        assert R.episode_return(np.array([1.0, 2.0, 3.0])) == pytest.approx(6.0)

    def test_episode_return_discounted_hand_computed(self):
        r = np.array([1.0, 2.0, 4.0])
        assert R.episode_return(r, gamma=0.5) == pytest.approx(1 + 1.0 + 1.0)

    def test_episode_return_batched(self):
        r = np.array([[1.0, 10.0], [2.0, 20.0]])
        out = R.episode_return(r, gamma=1.0)
        np.testing.assert_allclose(out, [3.0, 30.0])

    def test_reward_to_go_hand_computed(self):
        r = np.array([[1.0], [2.0], [4.0]])
        g = R.reward_to_go(r, gamma=0.5)
        np.testing.assert_allclose(g[:, 0], [1 + 1 + 1, 2 + 2, 4])

    def test_reward_to_go_gamma1_is_reverse_cumsum(self):
        r = np.arange(12, dtype=float).reshape(6, 2)
        g = R.reward_to_go(r, gamma=1.0)
        np.testing.assert_allclose(g, np.cumsum(r[::-1], axis=0)[::-1])


class TestGroupNormalize:
    def test_zero_mean_unit_std(self):
        rng = np.random.default_rng(0)
        s = rng.normal(3.0, 5.0, size=64)
        a = R.group_normalize(s, eps=0.0)
        assert a.mean() == pytest.approx(0.0, abs=1e-12)
        assert a.std(ddof=1) == pytest.approx(1.0, rel=1e-12)

    def test_uses_sample_std_ddof1(self):
        s = np.array([0.0, 2.0])   # mean 1, sample std sqrt(2), pop std 1
        a = R.group_normalize(s, eps=0.0)
        np.testing.assert_allclose(a, [-1 / np.sqrt(2), 1 / np.sqrt(2)])

    def test_singleton_matches_verl_convention(self):
        # verl: mean=0, std=1 for a group of one -> score/(1+eps)
        a = R.group_normalize(np.array([7.0]), eps=1e-6)
        assert a[0] == pytest.approx(7.0 / (1 + 1e-6))

    def test_constant_group_gives_zero(self):
        a = R.group_normalize(np.array([2.0, 2.0, 2.0]))
        np.testing.assert_allclose(a, 0.0)


class TestGrpoAdvantages:
    def test_within_group_normalization(self):
        returns = np.array([1.0, 3.0, 100.0, 300.0])
        gids = ["a", "a", "b", "b"]
        adv = R.grpo_advantages(returns, gids, eps=0.0)
        # groups are normalized independently -> identical z-scores
        np.testing.assert_allclose(adv[:2], adv[2:])
        assert adv[:2].mean() == pytest.approx(0.0, abs=1e-12)

    def test_group_constant_shift_invariance(self):
        rng = np.random.default_rng(1)
        returns = rng.normal(size=8)
        gids = ["g0"] * 4 + ["g1"] * 4
        adv1 = R.grpo_advantages(returns, gids)
        shifted = returns + np.array([10.0] * 4 + [-5.0] * 4)
        adv2 = R.grpo_advantages(shifted, gids)
        np.testing.assert_allclose(adv1, adv2, atol=1e-12)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            R.grpo_advantages(np.zeros(3), ["a", "a"])

    def test_turn_level_shape_and_per_t_mean_zero(self):
        rng = np.random.default_rng(2)
        rewards = rng.normal(size=(5, 6))                # T=5, n=6
        gids = ["a"] * 3 + ["b"] * 3
        adv = R.turn_level_advantages(rewards, gids, gamma=0.9, eps=0.0)
        assert adv.shape == (5, 6)
        for t in range(5):
            assert adv[t, :3].mean() == pytest.approx(0.0, abs=1e-12)
            assert adv[t, 3:].mean() == pytest.approx(0.0, abs=1e-12)


def _make_turns(rewards_by_traj, group_of_traj):
    """Synthetic TurnRecords: {traj_key: [r_0, r_1, ...]}."""
    turns = []
    for (ep, seat), rs in rewards_by_traj.items():
        for t, r in enumerate(rs):
            turns.append(TurnRecord(
                group_id=group_of_traj[(ep, seat)], episode=ep, seat=seat,
                period=t, v=1.0, p_lag=None, u=0.0, price=1.0,
                prompt_messages=[], raw_response="{}", action_index=0,
                action_value=0.0, reward=r, malformed=False,
                used_fallback=False, n_attempts=1))
    return turns


class TestAttachAdvantages:
    def test_episode_mode_broadcasts_one_advantage_per_trajectory(self):
        turns = _make_turns(
            {(0, 0): [1.0, 1.0], (1, 0): [3.0, 3.0]},
            {(0, 0): "g", (1, 0): "g"})
        diag = R.attach_advantages(turns, mode="episode", eps=0.0)
        assert diag["n_trajectories"] == 2 and diag["n_groups"] == 1
        by_traj = {}
        for rec in turns:
            by_traj.setdefault((rec.episode, rec.seat), set()).add(rec.advantage)
        assert all(len(v) == 1 for v in by_traj.values())   # constant per traj
        advs = sorted(a for s in by_traj.values() for a in s)
        np.testing.assert_allclose(advs, [-1 / np.sqrt(2), 1 / np.sqrt(2)])

    def test_episode_mode_allows_ragged_trajectories(self):
        turns = _make_turns(
            {(0, 0): [1.0, 1.0, 1.0], (1, 0): [3.0]},
            {(0, 0): "g", (1, 0): "g"})
        diag = R.attach_advantages(turns, mode="episode")
        np.testing.assert_allclose(sorted(diag["returns"]), [3.0, 3.0])
        # equal returns -> zero advantage (std=0, saved by the default eps)
        assert all(rec.advantage == pytest.approx(0.0, abs=1e-9) for rec in turns)

    def test_per_turn_mode_varies_within_trajectory(self):
        turns = _make_turns(
            {(0, 0): [1.0, 5.0], (1, 0): [2.0, 1.0]},
            {(0, 0): "g", (1, 0): "g"})
        R.attach_advantages(turns, mode="per_turn", gamma=1.0, eps=0.0)
        rec = {(r.episode, r.period): r.advantage for r in turns}
        # t=0: G = [6, 3]; t=1: G = [5, 1] -> signs (+,-) both periods
        assert rec[(0, 0)] > 0 > rec[(1, 0)]
        assert rec[(0, 1)] > 0 > rec[(1, 1)]

    def test_per_turn_mode_rejects_ragged(self):
        turns = _make_turns({(0, 0): [1.0, 1.0], (1, 0): [1.0]},
                            {(0, 0): "g", (1, 0): "g"})
        with pytest.raises(ValueError, match="equal-length"):
            R.attach_advantages(turns, mode="per_turn")

    def test_unknown_mode_rejected(self):
        with pytest.raises(ValueError, match="unknown advantage mode"):
            R.attach_advantages([], mode="ppo")

    def test_non_contiguous_periods_rejected(self):
        turns = _make_turns({(0, 0): [1.0, 1.0]}, {(0, 0): "g"})
        turns[1].period = 5
        with pytest.raises(ValueError, match="non-contiguous"):
            R.attach_advantages(turns, mode="episode")


class TestVerlParity:
    def test_matches_verl_grpo_estimator(self):
        """grpo_advantages must equal verl 0.8.0's
        compute_grpo_outcome_advantage on random scores/groups."""
        torch = pytest.importorskip("torch")
        core_algos = pytest.importorskip("verl.trainer.ppo.core_algos")

        rng = np.random.default_rng(42)
        n, L, eps = 12, 7, 1e-6
        scores = rng.normal(50.0, 20.0, size=n)          # profit-scale returns
        index = np.array(["g0"] * 4 + ["g1"] * 4 + ["g2"] * 3 + ["g3"])  # incl. singleton

        # verl: scalar outcome reward on the last response token
        tlr = torch.zeros((n, L), dtype=torch.float64)
        tlr[:, -1] = torch.from_numpy(scores)
        mask = torch.ones((n, L), dtype=torch.float64)
        adv_verl, _ = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=tlr, response_mask=mask, index=index,
            epsilon=eps, norm_adv_by_std_in_grpo=True)
        # verl broadcasts the scalar advantage over response tokens
        adv_verl = adv_verl[:, 0].numpy()

        adv_ours = R.grpo_advantages(scores, list(index), eps=eps)
        np.testing.assert_allclose(adv_ours, adv_verl, rtol=1e-6, atol=1e-8)


# ---------------------------------------------------------------------------
# 2. env adapter with MockBackend
# ---------------------------------------------------------------------------
class TestRolloutGroup:
    def test_shapes_and_fields(self):
        params, bench = small_setup()
        backend = MockBackend(mode="random", seed=0)
        grp = rollout_group(backend, params, bench, group_size=3, periods=4,
                            seed=7)
        assert len(grp.turns) == 3 * 4 * params.I
        assert grp.v_path.shape == (4, 3)
        assert grp.profits.shape == (4, 3, params.I)
        rec = grp.turns[0]
        assert rec.period == 0 and rec.p_lag is None
        assert [m["role"] for m in rec.prompt_messages[:2]] == ["system", "user"]
        assert 0 <= rec.action_index < params.nx
        assert np.isfinite(rec.reward)
        # later periods carry a real lagged price
        later = [r for r in grp.turns if r.period > 0]
        assert all(r.p_lag is not None for r in later)

    def test_matched_shocks_share_v_u_across_group(self):
        params, bench = small_setup()
        grp = rollout_group(MockBackend(mode="random", seed=0), params, bench,
                            group_size=4, periods=6, seed=3,
                            matched_shocks=True)
        for t in range(6):
            assert len(set(grp.v_path[t])) == 1
            assert len(set(grp.u_path[t])) == 1

    def test_independent_shocks_differ_across_group(self):
        params, bench = small_setup()
        grp = rollout_group(MockBackend(mode="random", seed=0), params, bench,
                            group_size=4, periods=8, seed=3,
                            matched_shocks=False)
        assert any(len(set(grp.u_path[t])) > 1 for t in range(8))

    def test_same_seed_reproduces_shock_path(self):
        params, bench = small_setup()
        g1 = rollout_group(MockBackend(mode="random", seed=0), params, bench,
                           group_size=2, periods=5, seed=11)
        g2 = rollout_group(MockBackend(mode="random", seed=99), params, bench,
                           group_size=2, periods=5, seed=11)
        np.testing.assert_array_equal(g1.v_path, g2.v_path)
        np.testing.assert_array_equal(g1.u_path, g2.u_path)

    def test_reward_is_market_profit(self):
        """reward must equal (v - p) * x from MarketEnv.step, not re-derived."""
        params, bench = small_setup()
        grp = rollout_group(MockBackend(mode="random", seed=1), params, bench,
                            group_size=2, periods=5, seed=5)
        for rec in grp.turns:
            assert rec.reward == pytest.approx(
                (rec.v - rec.price) * rec.action_value, rel=1e-12)
            assert rec.reward == pytest.approx(
                grp.profits[rec.period, rec.episode, rec.seat], rel=1e-12)

    def test_information_hiding_carried_over_from_stage_a(self):
        """Every recorded prompt must be EXACTLY what Stage A's
        build_user_prompt produces from that seat's OWN prior turns alone --
        proving prompts are a function of own-seat data only (no opponent
        actions/profits, no shared state) and that the adapter reuses Stage
        A's construction rather than rolling its own."""
        from phase2_llm.agent_llm import (
            SYSTEM_PROMPT, PeriodRecord, build_user_prompt,
        )
        params, bench = small_setup()
        grp = rollout_group(MockBackend(mode="random", seed=2), params, bench,
                            group_size=2, periods=6, seed=9)
        by_key = {(r.episode, r.seat, r.period): r for r in grp.turns}
        for rec in grp.turns:
            own_prior = [by_key[(rec.episode, rec.seat, t)]
                         for t in range(rec.period)]
            own_hist = [PeriodRecord(period=r.period, v=r.v, p_lag=r.p_lag,
                                     action_index=r.action_index,
                                     action_value=r.action_value,
                                     profit=r.reward) for r in own_prior]
            grid = bench.x_values(np.array([rec.v]))[0]
            expected = build_user_prompt(period=rec.period, v_t=rec.v,
                                         p_lag=rec.p_lag, action_values=grid,
                                         history=own_hist)
            assert rec.prompt_messages[0]["content"] == SYSTEM_PROMPT
            assert rec.prompt_messages[1]["content"] == expected

    def test_group_size_one_warns(self):
        params, bench = small_setup()
        with pytest.warns(UserWarning, match="no baseline"):
            rollout_group(MockBackend(seed=0), params, bench, group_size=1,
                          periods=2, seed=0)

    def test_to_dict_prompt_controls(self):
        params, bench = small_setup()
        grp = rollout_group(MockBackend(seed=0), params, bench, group_size=2,
                            periods=2, seed=0)
        d_full = grp.turns[0].to_dict()
        assert "prompt_messages" in d_full and json.dumps(d_full)
        d_trunc = grp.turns[0].to_dict(prompt_chars=10)
        assert all(len(m["content"]) <= 10 for m in d_trunc["prompt_messages"])
        d_slim = grp.turns[0].to_dict(include_prompts=False)
        assert "prompt_messages" not in d_slim and "raw_response" not in d_slim


class TestBatchPlusAdvantages:
    def test_end_to_end_batch_advantage_stats(self):
        params, bench = small_setup()
        backend = MockBackend(mode="random", seed=0, malformed_rate=0.1)
        groups = rollout_batch(backend, params, bench, n_groups=2,
                               group_size=4, periods=5, base_seed=100)
        assert [g.group_id for g in groups] == ["g0_seed100", "g1_seed101"]
        turns = flatten_turns(groups)
        diag = R.attach_advantages(turns, mode="episode")
        # 2 groups x 4 episodes x 2 seats trajectories
        assert diag["n_trajectories"] == 16 and diag["n_groups"] == 2
        assert all(rec.advantage is not None for rec in turns)
        advs = np.asarray(diag["advantages_per_trajectory"])
        # within each group of 8 trajectories: mean 0, sample std ~1
        for g in groups:
            grp_advs = advs[[i for i, k in enumerate(diag["trajectory_keys"])
                             if k[0] == g.group_id]]
            assert grp_advs.mean() == pytest.approx(0.0, abs=1e-9)
            assert grp_advs.std(ddof=1) == pytest.approx(1.0, rel=1e-3)

    def test_deterministic_scripted_policy_gives_zero_advantage(self):
        """All group members playing the identical cartel policy under
        matched shocks earn identical returns -> advantages exactly 0."""
        params, bench = small_setup()
        backend = MockBackend(mode="target_chi", c_grid=bench.c_grid,
                              target_chi=bench.chiM, noise_sd=0.0, seed=0)
        grp = rollout_group(backend, params, bench, group_size=3, periods=4,
                            seed=1)
        diag = R.attach_advantages(grp.turns, mode="episode")
        np.testing.assert_allclose(np.asarray(diag["advantages_per_trajectory"]),
                                   0.0, atol=1e-9)


# ---------------------------------------------------------------------------
# 3. train_grpo config parsing + dry run + verl command assembly
# ---------------------------------------------------------------------------
GRPO_CONFIG = os.path.join(REPO_ROOT, "phase2_llm", "grpo_config.yaml")


class TestConfig:
    def test_shipped_config_loads_and_validates(self):
        cfg = train_grpo.load_config(GRPO_CONFIG)
        assert cfg["grpo"]["group_size"] >= 2
        assert cfg["grpo"]["advantage_mode"] in ("episode", "per_turn")
        # model stays a configurable placeholder, never a hardcoded variant
        assert "CHANGE_ME" in cfg["model"]["name_or_path"]

    def _write_cfg(self, tmp_path, mutate):
        cfg = yaml.safe_load(open(GRPO_CONFIG, encoding="utf-8"))
        mutate(cfg)
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        return str(p)

    def test_group_size_one_rejected(self, tmp_path):
        p = self._write_cfg(tmp_path, lambda c: c["grpo"].update(group_size=1))
        with pytest.raises(ValueError, match="group_size"):
            train_grpo.load_config(p)

    def test_bad_advantage_mode_rejected(self, tmp_path):
        p = self._write_cfg(tmp_path,
                            lambda c: c["grpo"].update(advantage_mode="ppo"))
        with pytest.raises(ValueError, match="advantage_mode"):
            train_grpo.load_config(p)

    def test_missing_section_rejected(self, tmp_path):
        p = self._write_cfg(tmp_path, lambda c: c.pop("grpo"))
        with pytest.raises(ValueError, match="missing config sections"):
            train_grpo.load_config(p)

    def test_bad_backend_rejected(self, tmp_path):
        p = self._write_cfg(tmp_path,
                            lambda c: c["rollout"].update(backend="anthropic"))
        with pytest.raises(ValueError, match="backend"):
            train_grpo.load_config(p)


class TestDryRun:
    def test_dry_run_end_to_end_mock(self, tmp_path):
        cfg = train_grpo.load_config(GRPO_CONFIG)
        cfg["env"].update(periods=3)
        cfg["grpo"].update(group_size=2, groups_per_step=2)
        cfg["train"].update(steps=2, out_dir=str(tmp_path))
        cfg["rollout"].update(backend="mock", malformed_rate=0.2)
        summary = train_grpo.run_dry_run(cfg, quiet=True)

        assert len(summary["steps"]) == 2
        for s in summary["steps"]:
            assert s["n_trajectories"] == 2 * 2 * 2   # groups x episodes x seats
            assert s["n_turns"] == s["n_trajectories"] * 3
            assert abs(s["advantage_mean"]) < 1e-6
            assert np.isfinite(s["delta_c_batch"])

        with open(summary["records_path"], encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        assert len(rows) == sum(s["n_turns"] for s in summary["steps"])
        r = rows[0]
        for field in ("group_id", "episode", "seat", "period", "reward",
                      "advantage", "prompt_messages", "raw_response", "step"):
            assert field in r, f"missing {field}"
        assert r["advantage"] is not None
        assert os.path.exists(summary["summary_path"])

    def test_dry_run_distinct_seeds_across_steps(self, tmp_path):
        cfg = train_grpo.load_config(GRPO_CONFIG)
        cfg["env"].update(periods=2)
        cfg["grpo"].update(group_size=2, groups_per_step=2)
        cfg["train"].update(steps=2, out_dir=str(tmp_path))
        summary = train_grpo.run_dry_run(cfg, quiet=True)
        with open(summary["records_path"], encoding="utf-8") as f:
            gids = {(json.loads(l)["step"], json.loads(l)["group_id"])
                    for l in f}
        assert len({g for _, g in gids}) == 4   # no seed collisions


class TestVerlCommand:
    def test_command_contains_grpo_wiring(self):
        cfg = train_grpo.load_config(GRPO_CONFIG)
        cmd = train_grpo.build_verl_command(cfg)
        joined = " ".join(cmd)
        assert "verl.trainer.main_ppo" in joined
        assert "algorithm.adv_estimator=grpo" in joined
        assert "actor_rollout_ref.rollout.agent.default_agent_loop=market_speculator" in joined
        assert f"actor_rollout_ref.rollout.n={cfg['grpo']['group_size']}" in joined

    def test_run_verl_reports_placeholder_and_does_not_launch(self):
        cfg = train_grpo.load_config(GRPO_CONFIG)
        out = train_grpo.run_verl(cfg, launch=False, quiet=True)
        assert any("placeholder" in p for p in out["problems"])

    def test_model_path_synced_from_model_section(self, tmp_path):
        cfg = train_grpo.load_config(GRPO_CONFIG)
        cfg["model"]["name_or_path"] = "Qwen/RealModel"
        del cfg["verl"]["overrides"]["actor_rollout_ref.model.path"]
        cmd = train_grpo.build_verl_command(cfg)
        assert "actor_rollout_ref.model.path=Qwen/RealModel" in cmd


# ---------------------------------------------------------------------------
# 4. SLURM scripts
# ---------------------------------------------------------------------------
class TestSlurmScripts:
    SCRIPTS = ["launch_vllm.sbatch", "launch_training.sbatch"]

    @pytest.mark.parametrize("name", SCRIPTS)
    def test_placeholder_convention(self, name):
        path = os.path.join(SLURM_DIR, name)
        assert os.path.exists(path), path
        text = open(path, encoding="utf-8").read()
        assert text.startswith("#!/bin/bash")
        # same TODO(user)/CHANGE_ME convention as phase1's run_experiment.sbatch
        assert "TODO(user)" in text
        for directive in ("--job-name", "--partition=CHANGE_ME",
                          "--account=CHANGE_ME", "--gres=gpu:CHANGE_ME",
                          "--output="):
            assert directive in text, f"{name} missing #SBATCH {directive}"

    def test_vllm_script_serves_openai_compatible_endpoint(self):
        text = open(os.path.join(SLURM_DIR, "launch_vllm.sbatch"),
                    encoding="utf-8").read()
        assert "vllm serve" in text
        assert "OpenAICompatibleBackend" in text   # documents the reuse

    def test_training_script_launches_via_train_grpo(self):
        text = open(os.path.join(SLURM_DIR, "launch_training.sbatch"),
                    encoding="utf-8").read()
        assert "--mode verl --launch" in text
        assert "make-dataset" in text
        assert "PYTHONPATH" in text   # ray workers must import phase2_llm/env

    @pytest.mark.parametrize("name", SCRIPTS + ["../../phase1_qlearning/slurm/run_experiment.sbatch"])
    def test_bash_syntax(self, name):
        bash = shutil.which("bash")
        if bash is None:
            pytest.skip("no bash available")
        path = os.path.normpath(os.path.join(SLURM_DIR, name))
        proc = subprocess.run([bash, "-n", path], capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr


# ---------------------------------------------------------------------------
# 5. verl integration (skips wherever verl isn't installed, e.g. .venv)
# ---------------------------------------------------------------------------
class TestVerlAgentLoop:
    def test_agent_loop_imports_and_registers(self):
        pytest.importorskip("verl")
        from verl.experimental.agent_loop.agent_loop import (
            AgentLoopBase, _agent_loop_registry,
        )
        from phase2_llm.verl_agent_loop import MarketSpeculatorAgentLoop

        assert issubclass(MarketSpeculatorAgentLoop, AgentLoopBase)
        assert "market_speculator" in _agent_loop_registry
        target = _agent_loop_registry["market_speculator"]["_target_"]
        assert target.endswith("verl_agent_loop.MarketSpeculatorAgentLoop")
        # run() is the abstract hook verl's AgentLoopWorker awaits
        import inspect
        assert inspect.iscoroutinefunction(MarketSpeculatorAgentLoop.run)

    def test_agent_loop_config_yaml_target_resolves_via_hydra(self):
        """phase2_llm/agent_loop_config.yaml is what verl's AgentLoopWorker
        feeds to hydra.utils.instantiate -- its _target_ must resolve to the
        registered class and its kwargs must match __init__'s signature."""
        pytest.importorskip("verl")
        import inspect
        import hydra.utils
        from phase2_llm.verl_agent_loop import MarketSpeculatorAgentLoop

        cfg_path = os.path.join(REPO_ROOT, "phase2_llm", "agent_loop_config.yaml")
        entries = yaml.safe_load(open(cfg_path, encoding="utf-8"))
        entry = {e["name"]: e for e in entries}["market_speculator"]
        cls = hydra.utils.get_class(entry["_target_"])
        assert cls is MarketSpeculatorAgentLoop
        init_params = inspect.signature(cls.__init__).parameters
        for k in entry:
            if k in ("name", "_target_"):
                continue
            assert k in init_params, f"agent_loop_config kwarg {k!r} not in __init__"
        # the referenced env config must exist and be the read-only Stage A one
        assert os.path.exists(os.path.join(REPO_ROOT, entry["params_config"]))
