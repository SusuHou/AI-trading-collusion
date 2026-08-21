#!/bin/bash
#
# Phase 2 Stage B: launch the verl GRPO fine-tuning job (Ray head + workers).
#
# Prerequisites on the cluster (see README "Phase 2, Stage B" and
# requirements-grpo.txt):
#   1. a training venv with: verl (pip install verl), vllm, and this repo's
#      requirements.txt -- verl pulls torch/ray/transformers/peft itself.
#   2. the RL dataset parquet files:
#        python -m phase2_llm.train_grpo --mode make-dataset
#   3. placeholders below + in phase2_llm/grpo_config.yaml filled in
#      (model path, GPU counts).
#   4. the smoke-test sequence in README run once interactively (agent-loop
#      registration, then a tiny 1-step training job) before a full run.
#
# Usage (after filling in the TODOs below):
#   sbatch phase2_llm/slurm/launch_training.sbatch
#
# Single-node: Ray is started implicitly by verl -- no ray start needed.
# Multi-node: uncomment the ray head/worker block below (standard verl/Ray
# SLURM pattern) and set --nodes accordingly.
#
#SBATCH --job-name=grpo-collusion
#SBATCH --partition=CHANGE_ME        # TODO(user): set this (GPU partition)
#SBATCH --account=CHANGE_ME          # TODO(user): set this (allocation account)
#SBATCH --nodes=1                    # TODO(user): >1 needs the Ray block below
#SBATCH --gres=gpu:CHANGE_ME:4      # TODO(user): GPU type/count; must match
                                     #   trainer.n_gpus_per_node in grpo_config.yaml's verl overrides
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G                   # TODO(user): size to the model
#SBATCH --time=24:00:00              # TODO(user)
#SBATCH --output=logs/grpo_%j.out

set -euo pipefail

# TODO(user): adapt environment setup to the cluster (module system, venv).
# module load cuda/12.4 python/3.11
# source /path/to/venv-verl/bin/activate

# The agent loop (phase2_llm/verl_agent_loop.py) and env/ must be importable
# from the Ray workers:
export PYTHONPATH="${SLURM_SUBMIT_DIR:-$PWD}:${PYTHONPATH:-}"
mkdir -p logs results/phase2_llm/grpo

# ---- multi-node Ray cluster (uncomment for --nodes > 1) --------------------
# head_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
# head_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
# port=6379
# export RAY_ADDRESS="$head_ip:$port"
# srun --nodes=1 --ntasks=1 -w "$head_node" \
#     ray start --head --node-ip-address="$head_ip" --port=$port --block &
# sleep 15
# srun --nodes=$((SLURM_JOB_NUM_NODES - 1)) --ntasks=$((SLURM_JOB_NUM_NODES - 1)) \
#     --exclude="$head_node" ray start --address="$RAY_ADDRESS" --block &
# sleep 15
# -----------------------------------------------------------------------------

# Regenerate the dataset if it does not exist yet (idempotent, cheap).
[ -f results/phase2_llm/grpo/train.parquet ] || \
    python -m phase2_llm.train_grpo --mode make-dataset

# train_grpo --mode verl assembles the `python -m verl.trainer.main_ppo
# <hydra overrides>` command from phase2_llm/grpo_config.yaml (verl.overrides
# section -- model path, GRPO group size = rollout.n, the market_speculator
# agent loop, GPU counts) and exec's it with --launch. Equivalent to calling
# verl.trainer.main_ppo directly; going through train_grpo keeps the single
# config file authoritative and refuses to launch on placeholder values.
python -m phase2_llm.train_grpo \
    --config phase2_llm/grpo_config.yaml \
    --mode verl --launch

# Afterwards, evaluate a checkpoint by serving it (launch_vllm.sbatch with
# MODEL=<checkpoint dir>) and running the Stage A pilot / Stage B dry-run
# against it to get Delta^C:
#   python -m phase2_llm.llm_pilot --backend openai --base-url http://<node>:8000/v1 \
#       --model <checkpoint> --episodes 20 --periods 32
