#!/usr/bin/env bash
set -euo pipefail

# Example:
# sbatch --nodes=2 --ntasks-per-node=8 examples/slurm/multi_node_collective.sh

export MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-${SLURM_GPUS_ON_NODE:-8}}"

srun torchrun \
  --nnodes="$SLURM_JOB_NUM_NODES" \
  --nproc-per-node="$NPROC_PER_NODE" \
  --node-rank="$SLURM_NODEID" \
  --master-addr="$MASTER_ADDR" \
  --master-port="$MASTER_PORT" \
  -m teamredbench run /root/TeamRedBench/configs/suites/full.yaml
