#!/bin/bash
#
# Phase 2: serve the Qwen model as a vLLM OpenAI-compatible endpoint.
#
# This is the endpoint `phase2_llm/agent_llm.py::OpenAICompatibleBackend`
# already targets -- used by BOTH:
#   - Stage A's frozen pilot:  python -m phase2_llm.llm_pilot --backend openai
#       --base-url http://$(hostname):8000/v1 --model $MODEL
#   - Stage B's rollout dry-run against a real (frozen or checkpointed)
#     model: python -m phase2_llm.train_grpo --mode dry-run  with
#     rollout.backend: openai in phase2_llm/grpo_config.yaml.
# NOTE: the actual verl GRPO training job does NOT need this server -- verl
# spawns its own vLLM rollout workers inside the Ray cluster (see
# launch_training.sbatch). This script is for inference-only evaluation runs.
#
# Usage (after filling in the TODOs below):
#   MODEL=Qwen/Qwen3.5-<size>-Instruct sbatch phase2_llm/slurm/launch_vllm.sbatch
# then read the node/port from logs/vllm_<jobid>.out and point --base-url at it.
#
#SBATCH --job-name=vllm-qwen
#SBATCH --partition=CHANGE_ME        # TODO(user): set this (GPU partition)
#SBATCH --account=CHANGE_ME          # TODO(user): set this (allocation account)
#SBATCH --gres=gpu:CHANGE_ME:1       # TODO(user): GPU type/count (e.g. gpu:a100:1;
                                     #   bump count + --tensor-parallel-size for large models)
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G                    # TODO(user): size to the model
#SBATCH --time=08:00:00              # TODO(user): server lifetime
#SBATCH --output=logs/vllm_%j.out

set -euo pipefail

# TODO(user): set the HF model id or local checkpoint path (kept configurable
# on purpose -- nothing in this repo assumes a specific Qwen-3.5 variant).
MODEL="${MODEL:?set MODEL=<HF id or checkpoint path, e.g. Qwen/Qwen3.5-XX-Instruct>}"
PORT="${PORT:-8000}"

# TODO(user): adapt environment setup to the cluster (module system, venv).
# vLLM has no Windows wheels, so it is NOT in this repo's requirements files;
# install it on the cluster into a dedicated venv (see requirements-grpo.txt):
# module load cuda/12.4 python/3.11
# source /path/to/venv-vllm/bin/activate   # pip install vllm

mkdir -p logs
echo "vLLM serving ${MODEL} on http://$(hostname):${PORT}/v1"

# --guided-decoding-backend: agent_llm.OpenAICompatibleBackend sends a
# json_schema response_format; vLLM enforces it with outlines/xgrammar (and
# the backend silently retries without it if unsupported, so this flag is a
# nice-to-have, not a hard requirement).
srun vllm serve "${MODEL}" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --guided-decoding-backend outlines \
    --tensor-parallel-size "${TENSOR_PARALLEL:-1}"   # TODO(user): match GPU count
