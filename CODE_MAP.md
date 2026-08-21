# AI Trading Collusion Code Map

This file is a practical reading and execution guide. Do not start by reading every line of code. Use this map to understand the project in layers.

## What This Project Does

The project replicates the trading-collusion paper in two stages:

1. `phase1_qlearning`: tabular Q-learning replication of the paper's original experiment.
2. `phase2_llm`: LLM-agent pilot and GRPO fine-tuning scaffold built on the same market environment.

The core market logic is shared in `env/`.

## First Files To Read

Read these in this order:

1. `README.md`
   - Overall explanation, run commands, caveats, and SLURM notes.
2. `docs/paper_spec.md`
   - Equation-by-equation paper specification.
3. `configs/poc.yaml`
   - Small proof-of-concept settings.
4. `phase1_qlearning/run_session.py`
   - Main entry point for Q-learning experiments.
5. `env/market.py`
   - The market environment and per-period mechanics.
6. `phase1_qlearning/qlearn.py`
   - Q-learning action selection and update rule.

Only read `qlearn_numba.py` after the NumPy version is clear.

## Main Entry Points

### Phase 1: Q-learning replication

Run one local proof-of-concept session:

```bash
python -m phase1_qlearning.run_session --config configs/poc.yaml --sigma-u 0.1 --seed 11 --out results/poc_su0.1_seed11.npz
```

Run tests:

```bash
python -m pytest tests/ -q
```

Aggregate results:

```bash
python -m phase1_qlearning.aggregate "results/*.npz" --csv results/sessions.csv
```

### Phase 2: LLM pilot

Offline mock pilot:

```bash
python -m phase2_llm.llm_pilot --backend mock --episodes 5 --periods 30 --seed 0 --out results/phase2_llm/pilot_mock.jsonl
```

GRPO dry run:

```bash
python -m phase2_llm.train_grpo --mode dry-run
```

## Environment Files

- `requirements.txt`: lightweight environment for Phase 1 and Phase 2 mock/openai pilot.
- `requirements-grpo.txt`: heavier GRPO/verl environment. Use a separate environment.
- `.venv`: existing local environment for general runs.
- `.venv310`: existing local Python 3.10 environment for Numba.
- `.venv-grpo`: existing local GRPO/verl environment.

On a Linux cluster, create fresh environments rather than copying the Windows virtual environments.

## SLURM Files

### Phase 1

`phase1_qlearning/slurm/run_experiment.sbatch`

Current status: template exists, but it still has cluster-specific placeholders.

Must fill:

- `#SBATCH --partition=CHANGE_ME`
- `#SBATCH --account=CHANGE_ME`
- Python module loading
- virtual environment path
- `--cpus-per-task`, especially if using `BACKEND=numba`

Example intended submission pattern:

```bash
CONFIG=configs/sweep_sigma_u.yaml BACKEND=numba sbatch --array=0-1099 phase1_qlearning/slurm/run_experiment.sbatch
```

### Phase 2

- `phase2_llm/slurm/launch_vllm.sbatch`
- `phase2_llm/slurm/launch_training.sbatch`

These are for serving a model and running GRPO training. They require GPU cluster details and a real model path.

## What To Verify Before Full Runs

Do this order on any new machine or cluster:

1. Install environment.
2. Run tests:

```bash
python -m pytest tests/ -q
```

3. Run one small Phase 1 POC:

```bash
python -m phase1_qlearning.run_session --config configs/poc.yaml --sigma-u 0.1 --seed 11 --out results/smoke_test.npz
```

4. Confirm `results/smoke_test.npz` exists.
5. Run aggregation on that one output.
6. Only then submit a small SLURM array.
7. Only after the small SLURM array works, submit full sweeps.

## What Not To Do First

- Do not read all files line by line.
- Do not start with GRPO.
- Do not start with the Numba backend unless Phase 1 NumPy tests already pass.
- Do not submit full sweep jobs before a one-task SLURM smoke test works.

## Minimal Understanding Goal

For now, you only need to understand:

1. `env/market.py`: how one market period works.
2. `phase1_qlearning/qlearn.py`: how agents choose and update actions.
3. `phase1_qlearning/run_session.py`: how many periods/sessions are run and saved.
4. `configs/*.yaml`: what experiment settings control.
5. `phase1_qlearning/slurm/run_experiment.sbatch`: how local runs become cluster jobs.

That is enough to become operational.
