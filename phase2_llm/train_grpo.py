"""Phase 2 Stage B: GRPO training-loop wiring for the two-speculator market.

Three modes (all driven by phase2_llm/grpo_config.yaml):

  --mode dry-run  (default; fully implemented and tested, no GPU/verl needed)
      Runs the complete GRPO DATA path -- everything except the weight
      update: sample `train.steps` steps of `grpo.groups_per_step` rollout
      groups x `grpo.group_size` episodes from the configured backend
      (MockBackend offline, or a real vLLM endpoint via --backend openai),
      compute group-relative advantages (phase2_llm/reward.py), write the
      per-turn training records to JSONL, and print per-step diagnostics
      (mean return, advantage mean~0/std~1 sanity, Delta^C of the sampled
      batch via env/metrics.py for collusion monitoring during training).

  --mode make-dataset  (implemented; needs pandas+pyarrow -> run in .venv-grpo)
      Emits the verl RL-dataset parquet files: one row per episode spec
      (shock_seed + env overrides in extra_info, Stage A system prompt as
      the raw chat). verl replicates each row rollout.n times under one uid
      -- that replica set is the GRPO group, matching the dry-run's
      matched-shocks group semantics (same seed -> same shock path).

  --mode verl  (wiring implemented; the actual training run is NOT possible
      in this dev environment and was NOT run -- be honest about this)
      Assembles the `python -m verl.trainer.main_ppo <hydra overrides>`
      command from the config's verl.overrides section, validates the
      referenced files exist, and prints (or with --launch, exec's) it.
      WHAT'S MISSING AND WHY: verl's RayPPOTrainer needs a Ray cluster with
      GPU workers and a vLLM (or sglang) rollout server for
      `AgentLoopBase.server_manager.generate`; this machine has neither a
      GPU nor a vllm install (no Windows wheels; vllm is an optional verl
      extra). The pieces that ARE fully implemented and mock-tested are the
      rollout source (grpo_env_adapter.py), the advantage math (reward.py,
      parity-tested against verl's own GRPO estimator), and the verl
      integration class (verl_agent_loop.py, import/registration-tested in
      .venv-grpo). On the cluster, run the smoke-test sequence in README
      "Phase 2, Stage B" before a full job.

Offline usage:
  .venv/Scripts/python -m phase2_llm.train_grpo --mode dry-run
  .venv-grpo/Scripts/python -m phase2_llm.train_grpo --mode make-dataset
  .venv-grpo/Scripts/python -m phase2_llm.train_grpo --mode verl
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
from env import metrics as M
from phase2_llm import reward as R
from phase2_llm.agent_llm import MockBackend, OpenAICompatibleBackend
from phase2_llm.grpo_env_adapter import flatten_turns, rollout_batch

DEFAULT_CONFIG = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                              "grpo_config.yaml"))
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_REQUIRED_SECTIONS = ("model", "env", "grpo", "rollout", "train")


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    missing = [s for s in _REQUIRED_SECTIONS if s not in cfg]
    if missing:
        raise ValueError(f"{path}: missing config sections {missing}")
    g = cfg["grpo"]
    if int(g["group_size"]) < 2:
        raise ValueError("grpo.group_size must be >= 2 (a GRPO group of 1 "
                         "has no baseline)")
    if g.get("advantage_mode", "episode") not in ("episode", "per_turn"):
        raise ValueError(f"unknown grpo.advantage_mode {g['advantage_mode']!r}")
    if cfg["rollout"].get("backend", "mock") not in ("mock", "openai"):
        raise ValueError(f"unknown rollout.backend {cfg['rollout']['backend']!r}")
    return cfg


def build_env(cfg: dict):
    env_cfg = cfg["env"]
    path = env_cfg.get("params_config")
    if path:
        path = os.path.join(_REPO_ROOT, path) if not os.path.isabs(path) else path
        with open(path, "r", encoding="utf-8") as f:
            params = Params.from_dict(yaml.safe_load(f).get("params", {}))
    else:
        params = Params()
    if env_cfg.get("sigma_u") is not None:
        params.sigma_u = float(env_cfg["sigma_u"])
    if env_cfg.get("xi") is not None:
        params.xi = float(env_cfg["xi"])
    params.I = 2
    return params, compute_benchmarks(params)


def build_backend(cfg: dict, bench, seed: int):
    r = cfg["rollout"]
    if r.get("backend", "mock") == "mock":
        mode, target_chi = r.get("mock_mode", "random"), None
        if mode == "target_nash":
            mode, target_chi = "target_chi", bench.chiN
        elif mode == "target_cartel":
            mode, target_chi = "target_chi", bench.chiM
        return MockBackend(mode=mode, c_grid=bench.c_grid, target_chi=target_chi,
                           malformed_rate=float(r.get("malformed_rate", 0.0)),
                           seed=seed)
    base_url = r.get("base_url") or os.environ.get("OPENAI_BASE_URL",
                                                   "http://localhost:8000/v1")
    model = cfg["model"]["name_or_path"]
    return OpenAICompatibleBackend(base_url=base_url, model=model,
                                   api_key=os.environ.get("OPENAI_API_KEY"),
                                   temperature=float(r.get("temperature", 0.7)))


# ---------------------------------------------------------------------------
# --mode dry-run: the full GRPO data path, minus the weight update
# ---------------------------------------------------------------------------
def run_dry_run(cfg: dict, quiet: bool = False) -> dict:
    params, bench = build_env(cfg)
    g, tr, r = cfg["grpo"], cfg["train"], cfg["rollout"]
    backend = build_backend(cfg, bench, seed=int(tr.get("base_seed", 0)))
    out_dir = os.path.join(_REPO_ROOT, tr.get("out_dir", "results/phase2_llm/grpo"))
    os.makedirs(out_dir, exist_ok=True)
    records_path = os.path.join(out_dir, "dry_run_records.jsonl")

    train_seats = set(g.get("train_seats", [0, 1]))
    steps_out = []
    t0 = time.time()
    with open(records_path, "w", encoding="utf-8") as f_out:
        for step in range(int(tr["steps"])):
            base_seed = int(tr.get("base_seed", 0)) + step * int(g["groups_per_step"])
            groups = rollout_batch(
                backend, params, bench,
                n_groups=int(g["groups_per_step"]),
                group_size=int(g["group_size"]),
                periods=int(cfg["env"]["periods"]),
                base_seed=base_seed,
                matched_shocks=bool(g.get("matched_shocks", True)),
                history_len=int(r.get("history_len", 10)),
                max_retries=int(r.get("max_retries", 1)),
                fallback=r.get("fallback", "middle"),
            )
            turns = [rec for rec in flatten_turns(groups)
                     if rec.seat in train_seats]
            diag = R.attach_advantages(turns,
                                       mode=g.get("advantage_mode", "episode"),
                                       gamma=float(g.get("gamma", 1.0)),
                                       eps=float(g.get("eps", R.DEFAULT_EPS)))
            for rec in turns:
                row = rec.to_dict(include_prompts=True, prompt_chars=600)
                row["step"] = step
                f_out.write(json.dumps(row) + "\n")

            # collusion monitoring: Delta^C of this step's sampled batch,
            # matched-path (OA IA.4.1-4.3), same computation as llm_pilot.
            dcs = []
            for grp in groups:
                piN = bench.path_profits(grp.v_path, grp.u_path, "nash").mean(axis=0)
                piM = bench.path_profits(grp.v_path, grp.u_path, "cartel").mean(axis=0)
                dcs.append(M.delta_c_matched(grp.profits.mean(axis=0), piN, piM))
            delta_c = float(np.concatenate(dcs).mean())

            adv = np.array([a for a in diag["advantages_per_trajectory"]]).ravel()
            n_fallback = sum(rec.used_fallback for rec in turns)
            step_summary = {
                "step": step,
                "n_trajectories": diag["n_trajectories"],
                "n_groups": diag["n_groups"],
                "n_turns": len(turns),
                "n_fallback": n_fallback,
                "return_mean": float(np.mean(diag["returns"])),
                "return_std": float(np.std(diag["returns"])),
                "advantage_mean": float(adv.mean()),
                "advantage_std": float(adv.std()),
                "delta_c_batch": delta_c,
            }
            steps_out.append(step_summary)
            if not quiet:
                print(f"[step {step}] traj={step_summary['n_trajectories']} "
                      f"return={step_summary['return_mean']:+.3f}"
                      f"±{step_summary['return_std']:.3f} "
                      f"adv(mean,std)=({step_summary['advantage_mean']:+.3f},"
                      f"{step_summary['advantage_std']:.3f}) "
                      f"Delta^C={delta_c:+.3f} fallback={n_fallback}")

    summary = {
        "mode": "dry-run",
        "backend": cfg["rollout"].get("backend", "mock"),
        "model": cfg["model"]["name_or_path"],
        "params": {"sigma_u": params.sigma_u, "xi": params.xi},
        "grpo": {k: g.get(k) for k in ("group_size", "groups_per_step",
                                       "advantage_mode", "gamma",
                                       "matched_shocks")},
        "steps": steps_out,
        "elapsed_sec": time.time() - t0,
        "records_path": records_path,
    }
    summary_path = os.path.join(out_dir, "dry_run_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    summary["summary_path"] = summary_path
    if not quiet:
        print(f"records -> {records_path}\nsummary -> {summary_path}")
    return summary


# ---------------------------------------------------------------------------
# --mode make-dataset: verl RL dataset (episode-spec rows)
# ---------------------------------------------------------------------------
def make_dataset(cfg: dict, quiet: bool = False) -> dict:
    try:
        import pandas as pd  # noqa: F401  (pandas+pyarrow live in .venv-grpo)
    except ImportError as exc:
        raise SystemExit(
            "make-dataset needs pandas+pyarrow (installed with verl in "
            ".venv-grpo; not added to the default .venv on purpose): "
            f"{exc}") from exc
    from phase2_llm.agent_llm import SYSTEM_PROMPT

    d = cfg.get("data", {})
    env_cfg = cfg["env"]
    outs = {}
    for split, n_key, out_key, seed0 in (
            ("train", "n_rows_train", "out_train", 0),
            ("val", "n_rows_val", "out_val", 10_000_000)):
        n = int(d.get(n_key, 0))
        if n <= 0:
            continue
        rows = []
        for k in range(n):
            seed = seed0 + k
            rows.append({
                # verl RLHFDataset chat format; the agent loop rebuilds the
                # real per-period prompts itself (Stage A build_user_prompt) --
                # this row-level prompt only seeds tokenization plumbing.
                "prompt": [{"role": "system", "content": SYSTEM_PROMPT}],
                "agent_name": "market_speculator",
                "data_source": "ai-trading-collusion",
                "extra_info": {
                    "shock_seed": seed,
                    "periods": int(env_cfg["periods"]),
                    "train_seat": seed % 2,
                },
            })
        out_path = os.path.join(_REPO_ROOT, d[out_key])
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        pd.DataFrame(rows).to_parquet(out_path, index=False)
        outs[split] = {"path": out_path, "rows": n}
        if not quiet:
            print(f"{split}: {n} rows -> {out_path}")
    return outs


# ---------------------------------------------------------------------------
# --mode verl: assemble/validate the verl launch command
# ---------------------------------------------------------------------------
def build_verl_command(cfg: dict) -> list[str]:
    overrides = dict(cfg.get("verl", {}).get("overrides", {}))
    if not overrides:
        raise ValueError("config has no verl.overrides section")
    # keep model path in sync unless the user overrode it explicitly
    overrides.setdefault("actor_rollout_ref.model.path", cfg["model"]["name_or_path"])
    overrides.setdefault("actor_rollout_ref.rollout.n", cfg["grpo"]["group_size"])
    cmd = [sys.executable, "-m", "verl.trainer.main_ppo"]
    for k, v in overrides.items():
        if isinstance(v, bool):
            v = str(v).lower()
        cmd.append(f"{k}={v}")
    return cmd


def run_verl(cfg: dict, launch: bool = False, quiet: bool = False) -> dict:
    problems = []
    try:
        import verl  # noqa: F401
        verl_version = verl.__version__
    except ImportError:
        verl_version = None
        problems.append("verl is not importable in this venv (use .venv-grpo "
                        "locally, or the cluster training venv)")
    model = cfg["model"]["name_or_path"]
    if "CHANGE_ME" in model:
        problems.append(f"model.name_or_path is still a placeholder: {model}")
    alc = cfg.get("verl", {}).get("overrides", {}).get(
        "actor_rollout_ref.rollout.agent.agent_loop_config_path")
    if alc and not os.path.exists(os.path.join(_REPO_ROOT, alc)):
        problems.append(f"agent_loop_config_path not found: {alc}")

    cmd = build_verl_command(cfg)
    if not quiet:
        print("verl launch command (verl", verl_version or "NOT INSTALLED", "):")
        print("  " + " \\\n    ".join(cmd))
        for p in problems:
            print(f"  !! {p}")
        if not launch:
            print("\nNOT launched (pass --launch on a Ray+GPU cluster; see "
                  "phase2_llm/slurm/launch_training.sbatch). This dev "
                  "environment has no GPU/vllm, so no training run has been "
                  "executed or verified here.")
    if launch:
        if problems:
            raise SystemExit("refusing to launch:\n  - " + "\n  - ".join(problems))
        os.execv(sys.executable, cmd)
    return {"cmd": cmd, "problems": problems, "verl_version": verl_version}


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--mode", choices=["dry-run", "make-dataset", "verl"],
                    default="dry-run")
    ap.add_argument("--launch", action="store_true",
                    help="verl mode only: actually exec the trainer (cluster only)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.mode == "dry-run":
        return run_dry_run(cfg, quiet=args.quiet)
    if args.mode == "make-dataset":
        return make_dataset(cfg, quiet=args.quiet)
    return run_verl(cfg, launch=args.launch, quiet=args.quiet)


if __name__ == "__main__":
    main()
