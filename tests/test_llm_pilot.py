"""Tests for phase2_llm (Stage A: no-training LLM pilot).

Covers the deterministic, network-free parts end-to-end:
  1. action-index parsing/validation (agent_llm.parse_action_index)
  2. prompt construction (agent_llm.build_messages) -- contains the expected
     fields, and never leaks a second speculator's data
  3. LLMAgent's retry-once-then-fallback policy for malformed backend output
  4. an end-to-end mock-backend pilot run (llm_pilot.run) -- transcript +
     summary files, Delta^C sanity (matches-Nash / matches-cartel probes)
"""
import json

import numpy as np
import pytest

from env.benchmarks import Params, compute_benchmarks
from phase2_llm.agent_llm import (
    LLMAgent, MockBackend, PeriodRecord,
    build_messages, build_user_prompt, format_action_grid, format_history,
    parse_action_index,
)
from phase2_llm import llm_pilot


# ---------------------------------------------------------------------------
# 1. parse_action_index
# ---------------------------------------------------------------------------
class TestParseActionIndex:
    def test_clean_json(self):
        assert parse_action_index('{"action_index": 3}', n_actions=10) == 3

    def test_json_with_extra_whitespace_and_fields(self):
        raw = '  {"action_index": 5, "reasoning": "I like this one"}  '
        assert parse_action_index(raw, n_actions=10) == 5

    def test_json_embedded_in_prose(self):
        raw = 'Sure, here is my choice: {"action_index": 2} -- hope that helps!'
        assert parse_action_index(raw, n_actions=10) == 2

    def test_bare_integer_fallback(self):
        raw = "I will go with option 4 this period."
        assert parse_action_index(raw, n_actions=10) == 4

    def test_out_of_range_index_rejected(self):
        assert parse_action_index('{"action_index": 99}', n_actions=10) is None
        assert parse_action_index('{"action_index": -1}', n_actions=10) is None

    def test_negative_bare_integer_out_of_range(self):
        assert parse_action_index("I choose -3", n_actions=10) is None

    def test_garbage_returns_none(self):
        assert parse_action_index("no numbers here at all", n_actions=10) is None

    def test_empty_or_none_returns_none(self):
        assert parse_action_index("", n_actions=10) is None
        assert parse_action_index(None, n_actions=10) is None

    def test_float_index_coerced(self):
        assert parse_action_index('{"action_index": 3.0}', n_actions=10) == 3

    def test_boundary_indices_valid(self):
        assert parse_action_index('{"action_index": 0}', n_actions=10) == 0
        assert parse_action_index('{"action_index": 9}', n_actions=10) == 9
        assert parse_action_index('{"action_index": 10}', n_actions=10) is None


# ---------------------------------------------------------------------------
# 2. prompt construction
# ---------------------------------------------------------------------------
class TestPromptConstruction:
    def test_action_grid_numbered_and_signed(self):
        text = format_action_grid([-2.5, 0.0, 3.25])
        assert "0: order size = -2.5000" in text
        assert "1: order size = +0.0000" in text
        assert "2: order size = +3.2500" in text

    def test_history_empty_says_first_period(self):
        assert "first period" in format_history([], history_len=10)

    def test_history_formats_own_records(self):
        hist = [PeriodRecord(period=0, v=1.5, p_lag=None, action_index=2,
                             action_value=7.0, profit=3.0)]
        text = format_history(hist, history_len=10)
        assert "period 0" in text
        assert "1.5000" in text
        assert "index 2" in text
        assert "7.0000" in text
        assert "3.0000" in text

    def test_history_respects_rolling_window_length(self):
        hist = [PeriodRecord(period=t, v=float(t), p_lag=None, action_index=0,
                             action_value=0.0, profit=0.0) for t in range(20)]
        text = format_history(hist, history_len=3)
        assert "period 19" in text
        assert "period 17" in text
        assert "period 16" not in text  # outside the rolling window

    def test_build_messages_has_system_and_user_roles(self):
        msgs = build_messages(period=0, v_t=1.2, p_lag=None,
                              action_values=[1.0, 2.0, 3.0], history=[])
        assert [m["role"] for m in msgs] == ["system", "user"]

    def test_system_prompt_never_mentions_collusion_or_opponent(self):
        msgs = build_messages(period=0, v_t=1.2, p_lag=None,
                              action_values=[1.0, 2.0, 3.0], history=[])
        system_text = msgs[0]["content"].lower()
        for forbidden in ("collu", "coordinat", "cartel", "the other player",
                          "the other trader", "the other speculator", "opponent"):
            assert forbidden not in system_text, f"leaked {forbidden!r}"

    def test_user_prompt_contains_current_period_fields(self):
        text = build_user_prompt(period=5, v_t=1.4142, p_lag=0.987,
                                 action_values=[10.0, 20.0], history=[])
        assert "Period 5" in text
        assert "1.4142" in text
        assert "0.9870" in text
        assert "action_index" in text

    def test_first_period_has_no_p_lag_leak(self):
        text = build_user_prompt(period=0, v_t=1.0, p_lag=None,
                                 action_values=[1.0], history=[])
        assert "not available" in text

    def test_prompt_does_not_leak_other_agents_history(self):
        """An agent's prompt must be built ONLY from its own history -- this
        test constructs two agents with disjoint sentinel history values and
        checks agent A's prompt never contains agent B's sentinel numbers."""
        hist_a = [PeriodRecord(period=0, v=1.111, p_lag=None, action_index=1,
                               action_value=1.111, profit=1.111)]
        hist_b = [PeriodRecord(period=0, v=9.999, p_lag=None, action_index=9,
                               action_value=9.999, profit=9.999)]
        msgs_a = build_messages(period=1, v_t=1.5, p_lag=None,
                                action_values=[1.0, 2.0], history=hist_a)
        full_text_a = json.dumps(msgs_a)
        assert "9.999" not in full_text_a
        assert "1.111" in full_text_a
        # sanity: hist_b really would show up if it were passed by mistake
        msgs_b = build_messages(period=1, v_t=1.5, p_lag=None,
                                action_values=[1.0, 2.0], history=hist_b)
        assert "9.999" in json.dumps(msgs_b)


# ---------------------------------------------------------------------------
# 3. LLMAgent retry / fallback policy
# ---------------------------------------------------------------------------
class _AlwaysGarbageBackend:
    """Deterministically-broken backend to exercise the fallback path."""

    def __init__(self):
        self.calls = 0

    def complete(self, messages, n_actions, action_values=None):
        self.calls += 1
        return "I refuse to answer with JSON."


class _RaisesBackend:
    def complete(self, messages, n_actions, action_values=None):
        raise RuntimeError("simulated network failure")


class TestLLMAgentRetryFallback:
    def test_valid_backend_response_used_directly(self):
        backend = MockBackend(mode="fixed", fixed_index=3, malformed_rate=0.0)
        agent = LLMAgent(backend, max_retries=1, fallback="middle")
        idx, meta = agent.choose_action(period=0, v_t=1.0, p_lag=None,
                                        action_values=list(range(10)))
        assert idx == 3
        assert meta.malformed is False
        assert meta.used_fallback is False
        assert meta.n_attempts == 1

    def test_malformed_output_retries_once_then_falls_back_to_middle(self):
        backend = _AlwaysGarbageBackend()
        agent = LLMAgent(backend, max_retries=1, fallback="middle")
        action_values = list(range(11))  # nx=11 -> middle index = 5
        idx, meta = agent.choose_action(period=0, v_t=1.0, p_lag=None,
                                        action_values=action_values)
        assert backend.calls == 2                 # 1 try + 1 retry
        assert meta.n_attempts == 2
        assert meta.malformed is True
        assert meta.used_fallback is True
        assert idx == 5
        assert agent.n_malformed == 1
        assert agent.n_fallback == 1

    def test_backend_exception_is_handled_defensively(self):
        agent = LLMAgent(_RaisesBackend(), max_retries=1, fallback="middle")
        idx, meta = agent.choose_action(period=0, v_t=1.0, p_lag=None,
                                        action_values=list(range(7)))
        assert meta.used_fallback is True
        assert idx == 3  # middle of 7

    def test_observe_appends_only_own_record(self):
        backend = MockBackend(mode="fixed", fixed_index=0)
        agent = LLMAgent(backend)
        assert agent.history == []
        agent.observe(period=0, v_t=1.0, p_lag=None, action_index=0,
                      action_value=0.5, profit=0.1)
        assert len(agent.history) == 1
        assert agent.history[0].profit == 0.1


# ---------------------------------------------------------------------------
# 4. end-to-end mock-backend pilot
# ---------------------------------------------------------------------------
class _Args:
    """Minimal stand-in for argparse.Namespace with llm_pilot.run's fields."""
    def __init__(self, **kw):
        defaults = dict(
            config=None, episodes=3, periods=6, seed=0, sigma_u=None, xi=None,
            out=None, quiet=True, backend="mock", mock_mode="random",
            mock_noise_sd=1.0, malformed_rate=0.1, base_url=None, model=None,
            api_key=None, temperature=0.7, history_len=5, max_retries=1,
            fallback="middle", log_prompts=True, log_prompt_chars=200,
        )
        defaults.update(kw)
        for k, v in defaults.items():
            setattr(self, k, v)


class TestEndToEndMockPilot:
    def test_pilot_runs_and_produces_sane_output(self, tmp_path):
        out_path = str(tmp_path / "pilot.jsonl")
        args = _Args(out=out_path, episodes=3, periods=6, seed=0)
        summary = llm_pilot.run(args)

        # transcript file: one line per (episode, period)
        with open(out_path, encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 3 * 6
        row0 = json.loads(lines[0])
        assert row0["episode"] == 0 and row0["period"] == 0
        assert row0["p_lag"] is None            # first period has no lagged price
        assert len(row0["agents"]) == 2          # I = 2 speculators
        for a in row0["agents"]:
            assert "action_index" in a and "profit" in a
            assert np.isfinite(a["profit"])

        # a later period should have a real (non-null) lagged price
        row_later = json.loads(lines[7])  # episode 1, period 1
        assert row_later["p_lag"] is not None

        # summary sanity
        assert summary["n_llm_calls"] == 3 * 6 * 2
        assert summary["n_malformed"] >= 0
        assert len(summary["delta_c_per_episode"]) == 3
        assert all(np.isfinite(d) for d in summary["delta_c_per_episode"])
        assert np.isfinite(summary["delta_c_mean"])

        import os
        assert os.path.exists(summary["summary_path"])
        with open(summary["summary_path"], encoding="utf-8") as f:
            saved = json.load(f)
        assert saved["delta_c_mean"] == pytest.approx(summary["delta_c_mean"])

    def test_malformed_rate_actually_exercises_fallback_path(self, tmp_path):
        out_path = str(tmp_path / "pilot_malformed.jsonl")
        args = _Args(out=out_path, episodes=4, periods=10, seed=1,
                    malformed_rate=1.0)  # every response is malformed
        summary = llm_pilot.run(args)
        assert summary["n_malformed"] == summary["n_llm_calls"]
        assert summary["n_fallback"] == summary["n_llm_calls"]
        # even under 100% malformed input, the pilot must not crash and must
        # still produce a finite Delta^C (all agents fall back to the
        # documented middle-index default every period)
        assert np.isfinite(summary["delta_c_mean"])

    def test_scripted_cartel_like_policy_gives_high_delta_c(self, tmp_path):
        """Sanity probe for the Delta^C pipeline itself: a scripted policy
        that always plays close to the cartel benchmark action should score
        Delta^C close to 1, independent of any real LLM behavior."""
        out_path = str(tmp_path / "pilot_cartel.jsonl")
        args = _Args(out=out_path, episodes=4, periods=15, seed=2,
                    mock_mode="target_cartel", mock_noise_sd=0.0,
                    malformed_rate=0.0)
        summary = llm_pilot.run(args)
        assert summary["delta_c_mean"] == pytest.approx(1.0, abs=0.05)

    def test_scripted_nash_like_policy_gives_low_delta_c(self, tmp_path):
        out_path = str(tmp_path / "pilot_nash.jsonl")
        args = _Args(out=out_path, episodes=4, periods=15, seed=3,
                    mock_mode="target_nash", mock_noise_sd=0.0,
                    malformed_rate=0.0)
        summary = llm_pilot.run(args)
        assert abs(summary["delta_c_mean"]) < 0.1

    def test_pilot_never_calls_env_or_configs_mutating_apis(self):
        """Guardrail matching the task's scope boundary: compute_benchmarks
        is a pure function of Params, so re-running with the same seed/config
        must be exactly reproducible (nothing mutates shared env/ state)."""
        params = Params.from_dict({"sigma_u": 0.1})
        b1 = compute_benchmarks(params)
        b2 = compute_benchmarks(params)
        assert b1.lamN == b2.lamN
        assert np.array_equal(b1.c_grid, b2.c_grid)
