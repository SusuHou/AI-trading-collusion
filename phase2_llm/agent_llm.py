"""Phase 2 Stage A: LLM-agent variant of the informed speculator.

Two independently-sampled instances of the SAME language model play the
informed-speculator role in `env/market.py`'s Kyle-style market, in place of
the tabular Q-learners of `phase1_qlearning/`. No weight updates happen here
-- this is a frozen, in-context-only pilot (see `llm_pilot.py` for the
episode loop and `README.md`'s Phase 2 section for the full description).

Information structure (must match docs/paper_spec.md's state
s_t = {p_{t-1}, v_{t-1}, v_t} exactly): each agent only ever sees its OWN
past (v, action, profit) history and the lagged public price. It never sees
the other speculator's actions, profits, or any hint that a second
speculator exists -- `build_messages` below takes only one agent's history
as input, and the system prompt never mentions collusion, coordination, or
another player's strategy (judgment call, see README).

This module has two independent pieces:
  1. Prompt construction + output parsing (`build_messages`,
     `parse_action_index`) -- pure functions, no network, fully unit-testable.
  2. `LLMAgent` (orchestrates prompt -> backend call -> parse -> retry ->
     fallback -> rolling history) driven by any `LLMBackend`:
       - `OpenAICompatibleBackend`: talks to a real OpenAI-compatible
         chat-completions endpoint (vLLM, Ollama, etc.) via `requests`.
       - `MockBackend`: syntactically-valid, network-free stand-in used by
         `llm_pilot.py`'s offline pilot and by the tests in this repo, since
         no real Qwen-3.5 endpoint is available in this environment.
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from typing import Optional, Protocol, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Defaults (documented judgment calls, overridable by callers / CLI flags)
# ---------------------------------------------------------------------------
DEFAULT_HISTORY_LEN = 10     # periods of own rolling transcript kept in-prompt
DEFAULT_MAX_RETRIES = 1      # "retry once" per the task spec
DEFAULT_FALLBACK = "middle"  # fallback policy name, see LLMAgent._fallback_index


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are an independent trader participating in a repeated trading game over many periods.

Each period works as follows:
1. A risky asset has a "fundamental value" for that period, which is redrawn at random each \
period. You privately observe this period's fundamental value before you act. Nobody else \
sees the value you were shown.
2. You choose an order: a signed quantity of the asset to trade this period, picked from a \
numbered list of allowed order sizes you are given each period. A positive order size means \
buying that many units; a negative order size means selling that many units.
3. A market maker sets this period's traded price based on the total order flow it receives \
from all sources that period (your order, plus other order flow you do not observe \
individually -- the market maker does not reveal who submitted what). The market maker does \
not know the fundamental value directly; it only estimates it statistically from the total \
order flow it sees. There are also information-insensitive investors, whose trading pushes \
the price back toward its long-run average level, and noise traders, whose order flow is \
random and unrelated to the fundamental value.
4. Your profit for the period is (fundamental value - traded price) x (your order size). You \
profit when you bought and the price ends up below the fundamental value, or when you sold \
and the price ends up above the fundamental value. Larger orders tend to move the price \
further against you, all else equal.
5. The game then repeats with a new period: a new fundamental value is drawn, and the cycle \
continues. You keep a private memory of your own past periods -- the fundamental value you \
saw, the order size you chose, and the profit you earned -- to inform future decisions.

Your goal is to choose, each period, the order size that maximizes your own cumulative profit \
over the course of the game, using your own private history of past periods. Respond only in \
the required JSON format, with no other text.
"""


def format_action_grid(action_values: Sequence[float]) -> str:
    """Numbered list of this period's order-flow choices, e.g. '0: order size = +1.2345'."""
    return "\n".join(
        f"{i}: order size = {val:+.4f}" for i, val in enumerate(action_values)
    )


def format_history(history: Sequence["PeriodRecord"], history_len: int) -> str:
    """This agent's OWN rolling transcript only -- never the other agent's."""
    trimmed = list(history)[-history_len:]
    if not trimmed:
        return "(no past periods yet -- this is your first period)"
    lines = [
        f"period {rec.period}: fundamental value = {rec.v:+.4f}, you chose order size "
        f"{rec.action_value:+.4f} (index {rec.action_index}), your profit = {rec.profit:+.4f}"
        for rec in trimmed
    ]
    return "\n".join(lines)


def build_user_prompt(period: int, v_t: float, p_lag: Optional[float],
                       action_values: Sequence[float],
                       history: Sequence["PeriodRecord"],
                       history_len: int = DEFAULT_HISTORY_LEN) -> str:
    p_lag_str = "not available (this is the first period)" if p_lag is None else f"{p_lag:+.4f}"
    n = len(action_values)
    return f"""Period {period}.
Your private fundamental value this period: {v_t:+.4f}
Previous period's public traded price: {p_lag_str}

Your private history of past periods (yours only):
{format_history(history, history_len)}

Choose ONE order size for this period from the list below (index: order size):
{format_action_grid(action_values)}

Respond with a JSON object of exactly this form, choosing one index from 0 to {n - 1}:
{{"action_index": <integer index>}}"""


def build_messages(period: int, v_t: float, p_lag: Optional[float],
                    action_values: Sequence[float],
                    history: Sequence["PeriodRecord"],
                    history_len: int = DEFAULT_HISTORY_LEN) -> list[dict]:
    """[system, user] chat messages for one agent's one-period decision."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(
            period, v_t, p_lag, action_values, history, history_len)},
    ]


# ---------------------------------------------------------------------------
# Output parsing (defensive: guided decoding is not guaranteed on every backend)
# ---------------------------------------------------------------------------
_JSON_OBJ_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)
_BARE_INT_RE = re.compile(r"-?\d+")


def _validate_index(idx, n_actions: int) -> Optional[int]:
    try:
        idx = int(idx)
    except (TypeError, ValueError):
        return None
    if 0 <= idx < n_actions:
        return idx
    return None


def _try_json_object(text: str, n_actions: int) -> Optional[int]:
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if isinstance(obj, dict) and "action_index" in obj:
        return _validate_index(obj["action_index"], n_actions)
    if isinstance(obj, (int, float)):
        return _validate_index(obj, n_actions)
    return None


def parse_action_index(raw_text: Optional[str], n_actions: int) -> Optional[int]:
    """Best-effort extraction of a valid action index from raw model output.

    Tries, in order: (1) the whole response as JSON, (2) the first {...}
    substring as JSON, (3) the first bare integer in the text. Returns None
    if nothing yields a valid in-range index -- callers are responsible for
    the retry/fallback policy (see `LLMAgent.choose_action`).
    """
    if not raw_text:
        return None
    idx = _try_json_object(raw_text.strip(), n_actions)
    if idx is not None:
        return idx
    m = _JSON_OBJ_RE.search(raw_text)
    if m:
        idx = _try_json_object(m.group(0), n_actions)
        if idx is not None:
            return idx
    m2 = _BARE_INT_RE.search(raw_text)
    if m2:
        idx = _validate_index(m2.group(0), n_actions)
        if idx is not None:
            return idx
    return None


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------
class LLMBackendError(RuntimeError):
    """Raised when a backend cannot produce any response (network/HTTP error)."""


class LLMBackend(Protocol):
    def complete(self, messages: list[dict], n_actions: int,
                 action_values: Optional[Sequence[float]] = None) -> str:
        ...


ACTION_JSON_SCHEMA = {
    "type": "object",
    "properties": {"action_index": {"type": "integer"}},
    "required": ["action_index"],
    "additionalProperties": False,
}


class OpenAICompatibleBackend:
    """Talks to any OpenAI-compatible /chat/completions endpoint.

    Targets vLLM's OpenAI-compatible server (the eventual Qwen-3.5 SLURM
    deployment) but works against any compatible backend (Ollama, LM Studio,
    the real OpenAI API, ...). Uses `response_format` guided decoding when
    the server supports it (vLLM: `--guided-decoding-backend`), and silently
    retries once without it if the server rejects the field -- some
    OpenAI-compatible servers 4xx on unrecognized request keys.
    """

    def __init__(self, base_url: str, model: str, api_key: Optional[str] = None,
                 temperature: float = 0.7, max_tokens: int = 200,
                 timeout: float = 60.0, extra_headers: Optional[dict] = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.extra_headers = extra_headers or {}

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json", **self.extra_headers}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def complete(self, messages: list[dict], n_actions: int,
                 action_values: Optional[Sequence[float]] = None) -> str:
        import requests  # local import: only needed if this backend is used

        url = f"{self.base_url}/chat/completions"
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "action_choice", "strict": True,
                                "schema": ACTION_JSON_SCHEMA},
            },
        }
        try:
            resp = requests.post(url, headers=self._headers(), json=body,
                                 timeout=self.timeout)
            resp.raise_for_status()
        except requests.RequestException:
            body.pop("response_format", None)  # some servers 4xx on this key
            try:
                resp = requests.post(url, headers=self._headers(), json=body,
                                     timeout=self.timeout)
                resp.raise_for_status()
            except requests.RequestException as exc2:
                raise LLMBackendError(f"request to {url} failed: {exc2}") from exc2
        try:
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError) as exc:
            raise LLMBackendError(f"unexpected response shape from {url}: {exc}") from exc


class MockBackend:
    """Network-free stand-in with the same `.complete` interface.

    Used for (a) the offline pilot that validates the whole pipeline without
    a real Qwen endpoint, and (b) tests. Modes:
      - "random"     : uniform-random valid index each call (default --
                       a naive/no-signal policy, the primary offline check).
      - "fixed"      : always the same index (`fixed_index`, default the
                       middle of the grid).
      - "target_chi" : aims at a configured relative-action multiplier
                       (e.g. bench.chiM for a "plays like the cartel
                       benchmark" probe, bench.chiN for "plays like Nash"),
                       by picking the closest `c_grid` entry, with optional
                       Gaussian index jitter (`noise_sd`). This is a scripted
                       sanity probe for the Delta^C pipeline itself -- NOT a
                       simulated LLM policy -- so that a "mimics cartel" mock
                       run can be checked to produce Delta^C near 1 and a
                       "mimics Nash" run near 0 before trusting real-model
                       results.
    `malformed_rate` injects a fraction of syntactically-invalid responses so
    the retry/fallback path in `LLMAgent` is actually exercised offline.
    """

    def __init__(self, mode: str = "random", c_grid: Optional[np.ndarray] = None,
                 target_chi: Optional[float] = None, noise_sd: float = 1.0,
                 fixed_index: Optional[int] = None, malformed_rate: float = 0.0,
                 seed: int = 0):
        if mode not in ("random", "fixed", "target_chi"):
            raise ValueError(f"unknown MockBackend mode {mode!r}")
        if mode == "target_chi" and (c_grid is None or target_chi is None):
            raise ValueError("mode='target_chi' requires c_grid and target_chi")
        self.mode = mode
        self.c_grid = None if c_grid is None else np.asarray(c_grid)
        self.target_chi = target_chi
        self.noise_sd = noise_sd
        self.fixed_index = fixed_index
        self.malformed_rate = malformed_rate
        self._rng = random.Random(seed)
        self.n_calls = 0
        self.n_malformed_emitted = 0

    def complete(self, messages: list[dict], n_actions: int,
                 action_values: Optional[Sequence[float]] = None) -> str:
        self.n_calls += 1
        if self.malformed_rate > 0 and self._rng.random() < self.malformed_rate:
            self.n_malformed_emitted += 1
            return "sorry, I cannot decide right now (no JSON here)"

        if self.mode == "random":
            idx = self._rng.randrange(n_actions)
        elif self.mode == "fixed":
            base = self.fixed_index if self.fixed_index is not None else n_actions // 2
            idx = min(max(base, 0), n_actions - 1)
        else:  # target_chi
            base = int(np.argmin(np.abs(self.c_grid - self.target_chi)))
            jitter = round(self._rng.gauss(0.0, self.noise_sd))
            idx = min(max(base + jitter, 0), n_actions - 1)
        return json.dumps({"action_index": idx})


# ---------------------------------------------------------------------------
# LLMAgent: orchestrates prompt -> backend -> parse -> retry -> fallback
# ---------------------------------------------------------------------------
@dataclass
class PeriodRecord:
    period: int
    v: float
    p_lag: Optional[float]
    action_index: int
    action_value: float
    profit: float


@dataclass
class DecisionMeta:
    prompt_messages: list[dict]
    raw_response: Optional[str]
    n_attempts: int
    malformed: bool     # True if every attempt failed to parse
    used_fallback: bool


class LLMAgent:
    """One in-context speculator: owns its rolling history and retry policy.

    Never receives the other speculator's data -- callers of `choose_action`
    pass only this agent's own `history` (built up via `observe`).
    """

    def __init__(self, backend: LLMBackend, agent_name: str = "speculator",
                 history_len: int = DEFAULT_HISTORY_LEN,
                 max_retries: int = DEFAULT_MAX_RETRIES,
                 fallback: str = DEFAULT_FALLBACK):
        self.backend = backend
        self.agent_name = agent_name
        self.history_len = history_len
        self.max_retries = max_retries
        self.fallback = fallback
        self.history: list[PeriodRecord] = []
        self.n_malformed = 0
        self.n_fallback = 0

    def _fallback_index(self, n_actions: int) -> int:
        """Documented default when the backend never returns a valid index.

        "middle" (default): the middle grid index -- a moderate order size,
        neither the largest available buy nor sell, so a mis-parsed period
        does not inject an extreme/outlier order into the market. "random":
        uniform-random index, useful for stress-testing the pipeline.
        """
        if self.fallback == "middle":
            return n_actions // 2
        if self.fallback == "random":
            return random.randrange(n_actions)
        raise ValueError(f"unknown fallback policy {self.fallback!r}")

    def choose_action(self, period: int, v_t: float, p_lag: Optional[float],
                       action_values: Sequence[float]) -> tuple[int, DecisionMeta]:
        n = len(action_values)
        messages = build_messages(period, v_t, p_lag, action_values,
                                  self.history, self.history_len)
        raw = None
        idx = None
        attempts = 0
        for attempt in range(self.max_retries + 1):
            attempts = attempt + 1
            try:
                raw = self.backend.complete(messages, n, action_values=action_values)
            except Exception as exc:  # backend/network failure -> treat as malformed
                raw = f"<backend error: {exc}>"
                idx = None
            else:
                idx = parse_action_index(raw, n)
            if idx is not None:
                break
            if attempt < self.max_retries:
                messages = messages + [
                    {"role": "assistant", "content": raw or ""},
                    {"role": "user", "content": (
                        "That response was not a valid choice. Reply with ONLY a JSON "
                        f'object of the form {{"action_index": <integer 0..{n - 1}>}}, '
                        "choosing one index from the numbered list above."
                    )},
                ]

        used_fallback = idx is None
        malformed = used_fallback
        if used_fallback:
            self.n_malformed += 1
            self.n_fallback += 1
            idx = self._fallback_index(n)

        meta = DecisionMeta(prompt_messages=messages, raw_response=raw,
                            n_attempts=attempts, malformed=malformed,
                            used_fallback=used_fallback)
        return idx, meta

    def observe(self, period: int, v_t: float, p_lag: Optional[float],
                action_index: int, action_value: float, profit: float) -> None:
        """Append this period's OWN outcome to the private rolling history."""
        self.history.append(PeriodRecord(period, v_t, p_lag, action_index,
                                         action_value, profit))
