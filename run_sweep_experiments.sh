#!/bin/bash
set -euo pipefail

# Usage:
#   bash run_sweep_experiments.sh [input_dir]
#
# Assumptions:
#   - runAll_sweep.sh exists in the project root
#   - main1.py reads SOLVER, TIME_LIMIT, SEED, and CONFIG_NAME from env vars
#   - logs are written to ./experiment_logs/

INPUT_DIR="${1:-input}"
LOG_DIR="experiment_logs"
mkdir -p "$LOG_DIR"

TIMES=(10 30 60 100)

CONFIGS=(
  default
  diversified
  intensified
  balanced
  offset24 offset48 offset72
  constructfrac25 constructfrac35 constructfrac45
  constructionpass1 constructionpass3 constructionpass5
  noreverse notieshuffle forwardonly
  elite3 elite6 elite8
  intensify10 intensify20 intensify30
  destroy_conservative destroy_medium destroy_aggressive
  topk4 topk8 topk12
  lspass2 lspass4 lspass6
  restarttrigger8 restarttrigger12 restarttrigger16 restarttrigger20
  accept10 accept20 accept30
  perturb_random perturb_route perturb_shaw
  fastscreen
)

for T in "${TIMES[@]}"; do
  for CFG in "${CONFIGS[@]}"; do
    LOG_FILE="${LOG_DIR}/results_sweep_${CFG}_${T}s.log"
    SHELL_TIMEOUT=$((T + 20))
    echo "Running sweep config=${CFG} time=${T}s shell_timeout=${SHELL_TIMEOUT}s -> ${LOG_FILE}"
    SOLVER="sweep" TIME_LIMIT="${T}" SEED="0" CONFIG_NAME="${CFG}"       ./runAll_sweep.sh "$INPUT_DIR" "$SHELL_TIMEOUT" "$LOG_FILE"
  done
done

echo "Done. Logs written to ${LOG_DIR}/"
