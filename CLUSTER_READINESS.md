# Cluster Readiness Checklist

Use this checklist before running the project on a supercomputer.

## 1. Identify Cluster Basics

Fill these in first:

```text
Cluster name:
Scheduler: SLURM / other
Login command:
Project/account name:
CPU partition:
GPU partition:
Python module name:
CUDA module name, if needed:
Storage path for project:
Storage path for results:
```

## 2. Copy Or Clone Project

Preferred:

```bash
git clone <repo-url> ai-trading-collusion
cd ai-trading-collusion
```

If there is no remote repository yet, copy the folder to the cluster with the cluster's recommended transfer method.

Do not copy Windows virtual environments such as `.venv`, `.venv310`, or `.venv-grpo`.

## 3. Create Phase 1 Environment

On the cluster:

```bash
module load python/<version>
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Then test:

```bash
python -m pytest tests/ -q
```

## 4. Run One Local Smoke Test On Login Or Debug Node

Use a tiny proof-of-concept run before SLURM:

```bash
python -m phase1_qlearning.run_session --config configs/poc.yaml --sigma-u 0.1 --seed 11 --out results/cluster_smoke_test.npz
```

Expected result:

```text
results/cluster_smoke_test.npz exists
no Python import error
no missing package error
```

## 5. Edit Phase 1 SLURM Script

Open:

```text
phase1_qlearning/slurm/run_experiment.sbatch
```

Replace:

```text
#SBATCH --partition=CHANGE_ME
#SBATCH --account=CHANGE_ME
source .venv/bin/activate
```

Also decide:

```text
BACKEND=numpy: safer, slower
BACKEND=numba: faster, requires numba-compatible Python
```

For `BACKEND=numba`, set:

```text
#SBATCH --cpus-per-task=<run.batch from config>
```

For the current sweep configs, `run.batch` is likely 10. Confirm in the config before submitting.

## 6. Submit One SLURM Task First

Do not start with the full array.

```bash
CONFIG=configs/sweep_sigma_u.yaml BACKEND=numpy sbatch --array=0-0 phase1_qlearning/slurm/run_experiment.sbatch
```

Check:

```bash
squeue -u $USER
cat logs/ai-collusion_<jobid>_0.out
ls results/sweep_sigma_u/
```

Expected result:

```text
one .npz file created
no import error
no permission error
no missing module error
```

## 7. Submit A Small Array

After one task works:

```bash
CONFIG=configs/sweep_sigma_u.yaml BACKEND=numpy sbatch --array=0-9 phase1_qlearning/slurm/run_experiment.sbatch
```

Only after this works, use the full array.

## 8. Full Phase 1 Sweeps

From the README:

```bash
CONFIG=configs/sweep_sigma_u.yaml sbatch --array=0-1099 phase1_qlearning/slurm/run_experiment.sbatch
CONFIG=configs/sweep_I.yaml       sbatch --array=0-1599 phase1_qlearning/slurm/run_experiment.sbatch
CONFIG=configs/sweep_rho.yaml     sbatch --array=0-1999 phase1_qlearning/slurm/run_experiment.sbatch
```

Use `BACKEND=numba` only after Numba passes tests on the cluster.

## 9. Postprocess Results

After jobs finish:

```bash
python -m phase1_qlearning.classify_mechanism "results/sweep_sigma_u"/*.npz
python -m phase1_qlearning.aggregate "results/sweep_sigma_u/*.npz" --csv results/sweep_sigma_u/sessions.csv
python -m phase1_qlearning.plots "results/sweep_sigma_u/*.npz" --out-dir results/sweep_sigma_u/figures
```

Repeat for other sweep folders.

## 10. Phase 2 Is Later

Do Phase 2 only after Phase 1 is stable.

Phase 2 requires:

- GPU partition
- model path
- vLLM
- GRPO/verl environment
- edited `phase2_llm/grpo_config.yaml`
- edited `phase2_llm/slurm/*.sbatch`

Phase 2 is not the first operational milestone.

## Immediate Next Action

Today, do only this:

```bash
python -m pytest tests/ -q
python -m phase1_qlearning.run_session --config configs/poc.yaml --sigma-u 0.1 --seed 11 --out results/local_smoke_test.npz
```

If those pass locally, the project is real enough to move to the cluster.
